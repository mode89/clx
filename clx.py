#!/usr/bin/env python3

from bisect import bisect_left
from collections import namedtuple
import functools
from pathlib import Path
import re
import sys
import threading

import pyrsistent as pr

from pprint import pprint

def main():
    text = slurp("example.clj")
    ctx = EvaluationContext(
        globals=functools.reduce(
            lambda acc, kv: assoc(acc, symbol(kv[0]), Var(kv[1])),
            _builtins().items(),
            hash_map()),
        locals=hash_map())
    ctx = _load_core(ctx)
    try:
        result, ctx = _eval(read_string(text), ctx)
        print("Result:", pr_str_(result, True))
    except Exception as exc:
        print("Error:", exc)

def _builtins():
    return {
        "+": lambda *args: sum(args),
        "-": lambda *args: args[0] - sum(args[1:]),
        "*": _mul,
        "/": _div,
        "=": _eq,
        "not": lambda x: not x,
        "some": some,
        "with-meta": with_meta,
        "meta": meta,
        "str": str_,
        "list": list_,
        "vec": vec,
        "vector": vector,
        "hash-map": hash_map,
        "count": count,
        "first": first,
        "rest": rest,
        "nth": nth,
        "concat": concat,
        "destructure1": destructure1,
        "gensym": gensym,
        "print": print_,
        "println": println,
        "slurp": slurp,
        "throw": throw,
    }

def _load_core(ctx):
    path = Path(__file__).parent / "core.clj"
    text = slurp(path)
    _, ctx = _eval(read_string(text), ctx)
    return ctx

def _conditions(pre=None, post=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if callable(pre):
                if not pre(*args, **kwargs):
                    raise Exception("Precondition failed")
            elif isinstance(pre, list):
                for pre_func in pre:
                    if not pre_func(*args, **kwargs):
                        raise Exception("Precondition failed")
            elif pre is not None:
                raise Exception("Precondition must be a function or a list")
            result = func(*args, **kwargs)
            if callable(post):
                if not post(result):
                    raise Exception("Postcondition failed")
            elif isinstance(post, list):
                for post_func in post:
                    if not post_func(result):
                        raise Exception("Postcondition failed")
            elif post is not None:
                raise Exception("Postcondition must be a function or a list")
            return result
        return wrapper
    return decorator

def _with_attrs(**kwargs):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        for key, value in kwargs.items():
            setattr(wrapper, key, value)
        return wrapper
    return decorator

#************************************************************
# Reader
#************************************************************

TOKEN_RE = re.compile(
    r"[\s,]*"
    "("
        r"~@" "|"
        r"[\[\]{}()'`~^@]" "|"
        r"\"(?:[\\]." "|" r"[^\\\"])*\"?" "|"
        r";.*" "|"
        r"[^\s\[\]{}()'\"`@,;]+"
    ")")

INT_RE = re.compile(r"-?[0-9]+$")
FLOAT_RE = re.compile(r"-?[0-9]+\.[0-9]+$")
STRING_RE = re.compile(r"\"(?:[\\].|[^\\\"])*\"$")

Token = namedtuple("Token", ["value", "line", "column"])

def read_string(text):
    return read_string1(f"(do {text})")

def read_string1(text):
    tokens = list(tokenize(text))
    form, _rest = read_form(tokens)
    return form

def tokenize(text):
    lines = text.splitlines()
    line_starts = []
    line_start = 0
    for line in lines:
        line_starts.append(line_start)
        line_start += (len(line) + 1)
    for match in TOKEN_RE.finditer(text):
        token = match.group(1)
        if token[0] == ";":
            continue
        line = bisect_left(line_starts, match.start(1) + 1)
        column = match.start(1) - line_starts[line - 1] + 1
        yield Token(value=token, line=line, column=column)

def read_form(tokens):
    token = tokens[0]
    tvalue = token.value
    if tvalue == "'":
        form, _rest = read_form(tokens[1:])
        return list_(_S_QUOTE, form), _rest
    elif tvalue == "`":
        form, _rest = read_form(tokens[1:])
        return quasiquote(form), _rest
    elif tvalue == "~":
        form, _rest = read_form(tokens[1:])
        return list_(_S_UNQUOTE, form), _rest
    elif tvalue == "~@":
        form, _rest = read_form(tokens[1:])
        return list_(_S_SPLICE_UNQUOTE, form), _rest
    elif tvalue == "^":
        _meta, rest1 = read_form(tokens[1:])
        form, rest2 = read_form(rest1)
        return list_(_S_WITH_META, form, _meta), rest2
    elif tvalue == "(":
        return read_collection(tokens, list_, "(", ")")
    elif tvalue == "[":
        return read_collection(tokens, vector, "[", "]")
    elif tvalue == "{":
        return read_collection(tokens, hash_map, "{", "}")
    else:
        return read_atom(token), tokens[1:]

def read_atom(token):
    tvalue = token.value
    if re.match(INT_RE, tvalue):
        return int(tvalue)
    elif re.match(FLOAT_RE, tvalue):
        return float(tvalue)
    elif re.match(STRING_RE, tvalue):
        return unescape(tvalue[1:-1])
    elif tvalue[0] == "\"":
        raise Exception("Expected end of string")
    elif tvalue == "true":
        return True
    elif tvalue == "false":
        return False
    elif tvalue == "nil":
        return None
    elif tvalue[0] == ":":
        return keyword(tvalue[1:])
    else:
        return with_meta(
            symbol(tvalue),
            hash_map(
                _K_LINE, token.line,
                _K_COLUMN, token.column))

def unescape(text):
    return text \
        .replace(r"\\", "\\") \
        .replace(r"\"", "\"") \
        .replace(r"\n", "\n")

def read_collection(tokens, ctor, start, end):
    token0 = tokens[0]
    assert token0.value == start, f"Expected '{start}'"
    tokens = tokens[1:]
    sequence = []
    while tokens[0].value != end:
        element, tokens = read_form(tokens)
        sequence.append(element)
        if len(tokens) == 0:
            raise Exception(f"Expected '{end}'")
    return with_meta(
        ctor(*sequence),
        hash_map(
            _K_LINE, token0.line,
            _K_COLUMN, token0.column)), \
        tokens[1:]

def quasiquote(form):
    if is_list(form) and len(form) > 0:
        head = form[0]
        if head == _S_UNQUOTE:
            assert len(form) == 2, "unquote expects 1 argument"
            return form[1]
        elif head == _S_SPLICE_UNQUOTE:
            raise Exception("splice-unquote not in list")
        else:
            return _quasiquote_sequence(form)
    if is_vector(form) and len(form) > 0:
        return list_(_S_VEC, _quasiquote_sequence(form))
    elif is_symbol(form):
        return list_(_S_QUOTE, form)
    else:
        return form

def _quasiquote_sequence(form):
    def entry(f): # pylint: disable=invalid-name
        if is_list(f) and len(f) > 0 and f[0] == _S_UNQUOTE:
            assert len(f) == 2, "unquote expects 1 argument"
            return list_(_S_LIST, f[1])
        elif is_list(f) and len(f) > 0 and f[0] == _S_SPLICE_UNQUOTE:
            assert len(f) == 2, "splice-unquote expects 1 argument"
            return f[1]
        else:
            return list_(_S_LIST, quasiquote(f))
    return cons(_S_CONCAT, list_(*map(entry, form)))

#************************************************************
# Printer
#************************************************************

def pr_str_(obj, readably=False):
    if is_nil(obj):
        return "nil"
    if is_bool(obj):
        return "true" if obj else "false"
    if is_string(obj):
        if readably:
            return "\"" + escape(obj) + "\""
        return obj
    if is_keyword(obj):
        return ":" + obj.name
    if is_symbol(obj):
        return obj.name
    if is_list(obj):
        return "(" + \
            " ".join(map(lambda x: pr_str_(x, readably), obj)) + \
            ")"
    if is_vector(obj):
        return "[" + \
            " ".join(map(lambda x: pr_str_(x, readably), obj)) + \
            "]"
    if is_hash_map(obj):
        ret = functools.reduce(
            lambda acc, kv:
                cons(pr_str_(kv[0], readably),
                    cons(pr_str_(kv[1], readably), acc)),
            obj.items(),
            list_())
        return "{" + " ".join(ret) + "}"
    return str(obj)

def escape(text):
    return text \
        .replace("\\", r"\\") \
        .replace("\"", r"\"") \
        .replace("\n", r"\n")

#************************************************************
# Eval
#************************************************************

@_conditions(
    pre=lambda form, ctx: isinstance(ctx, EvaluationContext),
    post=[
        lambda result: isinstance(result, tuple),
        lambda result: isinstance(result[1], EvaluationContext),
    ])
def _eval(form0, ctx):
    form = _macroexpand(form0, ctx)
    if is_list(form):
        head = form[0]
        if head == _S_DEF:
            return _eval_def(form, ctx)
        elif head == _S_LET_STAR:
            return _eval_let(form, ctx)
        elif head == _S_DO:
            return _eval_do(form, ctx)
        elif head == _S_IF:
            return _eval_if(form, ctx)
        elif head == _S_FN_STAR:
            return _eval_fn(form, ctx)
        elif head == _S_QUOTE:
            return _eval_quote(form, ctx)
        elif head == _S_EVAL:
            return _eval_eval(form, ctx)
        elif head == _S_EVAL_BANG:
            return _eval_eval_bang(form, ctx)
        elif head == _S_GLOBALS:
            return _eval_globals(form, ctx)
        elif head == _S_REQUIRE_PYTHON_INTEROP:
            return _eval_require_python_interop(form, ctx)
        else:
            return _eval_call(form, ctx)
    elif is_vector(form):
        return _eval_vector(form, ctx)
    elif is_hash_map(form):
        return _eval_hash_map(form, ctx)
    elif is_symbol(form):
        value = _resolve_symbol(form, ctx)
        if value is _UNDEFINED:
            raise Exception(f"Symbol '{form.name}' not found")
        return value, ctx
    else:
        return form, ctx

def _eval_def(form, ctx):
    assert len(form) == 3, "def expects 2 arguments"
    _name, value_form = form[1:]
    assert is_symbol(_name), \
        "def expects a symbol as the first argument"
    value, _ctx = _eval(value_form, ctx)
    return value, assoc_in(_ctx, [_K_GLOBALS, _name], Var(value))

def _eval_let(form, ctx):
    assert len(form) > 1, "let* expects at least 1 argument"
    assert len(form) < 4, "let* expects at most 2 arguments"
    bindings = form[1]
    body = form[2] if len(form) == 3 else None
    assert is_vector(bindings), \
        "let* expects a vector as the first argument"
    assert len(bindings) % 2 == 0, \
        "bindings of let* must have even number of elements"
    _ctx = ctx
    for _name, value_form in zip(bindings[::2], bindings[1::2]):
        assert is_symbol(_name), \
            "first element of each binding pair must be a symbol"
        value, _ctx = _eval(value_form, _ctx)
        _ctx = assoc_in(_ctx, [_K_LOCALS, _name], value)
    result, _ctx = _eval(body, _ctx)
    return result, assoc(_ctx, _K_LOCALS, ctx.locals)

def _eval_do(form, ctx):
    if len(form) == 1:
        return None, ctx
    else:
        _ctx = ctx
        for _form in form[1:-1]:
            _, _ctx = _eval(_form, _ctx)
        return _eval(form[-1], _ctx)

def _eval_if(form, ctx):
    assert len(form) == 4, "if expects 3 arguments"
    cond, _ctx = _eval(form[1], ctx)
    if cond:
        return _eval(form[2], _ctx)
    else:
        return _eval(form[3], _ctx)

def _eval_fn(form, ctx):
    assert len(form) == 3, "fn* expects 2 arguments"
    params = form[1]
    body = form[2]
    assert is_symbol(params), "first argument of fn* must be a symbol"
    return Function(ctx, params, body, None), ctx

def _eval_quote(form, ctx):
    assert len(form) == 2, "quote expects 1 argument"
    return form[1], ctx

def _eval_eval(form, ctx):
    assert len(form) == 3, "eval expects 2 arguments"
    cfg, _ctx = _eval(form[1], ctx)
    assert is_hash_map(cfg), \
        "eval expects a hash-map as the first argument"
    _form, _ctx = _eval(form[2], _ctx)
    result, _ = _eval(
        _form,
        EvaluationContext(
            globals=functools.reduce(
                lambda acc, kv: assoc(acc, kv[0], Var(kv[1])),
                get(cfg, _K_GLOBALS, hash_map()).items(),
                hash_map()),
            locals=get(cfg, _K_LOCALS, hash_map())))
    return result, _ctx

def _eval_eval_bang(form, ctx):
    assert len(form) == 2, "eval! expects 1 argument"
    _form, _ctx = _eval(form[1], ctx)
    return _eval(_form, _ctx)

def _eval_globals(form, ctx):
    assert len(form) == 1, "globals expects no arguments"
    return get(ctx, _K_GLOBALS), ctx

def _eval_require_python_interop(form, ctx):
    assert len(form) == 1, \
        "require-python-interop expects no arguments"
    return None, update(ctx, _K_GLOBALS, assoc,
        symbol("python/exec"), Var(exec),
        symbol("python/eval"), Var(eval),
        symbol("python/import"), Var(__import__),
        symbol("python/getattr"), Var(getattr),
        symbol("python/setattr"), Var(setattr))

def _eval_call(form, ctx):
    head = form[0]
    func, _ctx = _eval(head, ctx)
    assert callable(func), "Expected a callable object"
    args = []
    for arg in form[1:]:
        value, _ctx = _eval(arg, _ctx)
        args.append(value)
    return func(*args), _ctx

def _eval_vector(form, ctx):
    _ctx = ctx
    elements = []
    for element in form:
        value, _ctx = _eval(element, _ctx)
        elements.append(value)
    return vector(*elements), _ctx

def _eval_hash_map(form, ctx):
    _ctx = ctx
    kvs = []
    for key, value in form.items():
        _key, _ctx = _eval(key, _ctx)
        kvs.append(_key)
        _value, _ctx = _eval(value, _ctx)
        kvs.append(_value)
    return hash_map(*kvs), _ctx

def _macroexpand(form, ctx):
    while True:
        form1 = _macroexpand1(form, ctx)
        if form1 is form:
            return form
        else:
            form = form1

def _macroexpand1(form, ctx):
    if is_list(form) and len(form) > 0:
        head = form[0]
        if is_symbol(head):
            value = _resolve_symbol(head, ctx)
            if is_macro(value):
                return value(*form[1:])
    return form

def _resolve_symbol(sym, ctx):
    assert is_symbol(sym), "Expected a symbol"
    _locals = get(ctx, _K_LOCALS)
    if sym in _locals:
        return _locals[sym]
    else:
        _globals = get(ctx, _K_GLOBALS)
        if sym in _globals:
            return get(_globals, sym).deref()
        else:
            return _UNDEFINED

#************************************************************
# Core
#************************************************************

_UNDEFINED = object()

def is_nil(obj):
    return obj is None

def is_bool(obj):
    return isinstance(obj, bool)

def some(obj):
    return obj is not None

def str_(*args):
    return "".join(
        map(lambda x: pr_str_(x, False),
            filter(some, args)))

def is_string(obj):
    return isinstance(obj, str)

class Function:
    def __init__(self, ctx, params, body, _meta):
        self._ctx = ctx
        self._params = params
        self._body = body
        self.__meta__ = _meta
    def __call__(self, *args):
        ctx1 = assoc_in(self._ctx, [_K_LOCALS, self._params], PyTuple(args))
        result, ctx2 = _eval(self._body, ctx1)
        assert ctx1 == ctx2, "Function body must not change context"
        return result
    def _with_meta(self, _meta):
        return Function(self._ctx, self._params, self._body, _meta)

def is_macro(obj):
    if callable(obj):
        _meta = meta(obj)
        return is_hash_map(_meta) and get(_meta, _K_MACRO, False)
    return False

def with_meta(obj, _meta):
    assert is_hash_map(_meta), "Expected a hash-map"
    return obj._with_meta(_meta) # pylint: disable=protected-access

def meta(obj):
    return getattr(obj, "__meta__", None)

def define_record(name, *fields):
    cls = namedtuple(name, fields)
    def _get(self, key, default=None):
        assert is_keyword(key), "Expected a keyword"
        return getattr(self, key.name, default)
    cls._get = _get # pylint: disable=protected-access
    def _assoc(self, key, value):
        assert is_keyword(key), "Expected a keyword"
        return self._replace(**{key.name: value})
    cls._assoc = _assoc # pylint: disable=protected-access
    return cls

EvaluationContext = define_record("EvaluationContext", "globals", "locals")

Keyword = namedtuple("Keyword", ["name"])

@_with_attrs(table={}, lock=threading.Lock())
def keyword(_name):
    with keyword.lock:
        table = keyword.table
        if _name in table:
            return table[_name]
        else:
            new_kw = Keyword(_name)
            table[_name] = new_kw
            return new_kw

def is_keyword(obj):
    return isinstance(obj, Keyword)

_K_LINE = keyword("line")
_K_COLUMN = keyword("column")
_K_GLOBALS = keyword("globals")
_K_LOCALS = keyword("locals")
_K_MACRO = keyword("macro")
_K_AS = keyword("as")

class Symbol:
    def __init__(self, _name, _hash, _meta):
        self.name = sys.intern(_name)
        self._hash = _hash
        self.__meta__ = _meta
    def __eq__(self, other):
        return isinstance(other, Symbol) and self.name is other.name
    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self.name)
        return self._hash
    def _with_meta(self, _meta):
        return Symbol(self.name, self._hash, _meta)

def symbol(_name):
    return Symbol(_name, None, None)

def is_symbol(obj):
    return isinstance(obj, Symbol)

_S_AMPER = symbol("&")
_S_DEF = symbol("def")
_S_LET_STAR = symbol("let*")
_S_DO = symbol("do")
_S_IF = symbol("if")
_S_FN_STAR = symbol("fn*")
_S_QUOTE = symbol("quote")
_S_UNQUOTE = symbol("unquote")
_S_SPLICE_UNQUOTE = symbol("splice-unquote")
_S_WITH_META = symbol("with-meta")
_S_GLOBALS = symbol("globals")
_S_REQUIRE_PYTHON_INTEROP = symbol("require-python-interop")
_S_EVAL = symbol("eval")
_S_EVAL_BANG = symbol("eval!")
_S_VEC = symbol("vec")
_S_LIST = symbol("list")
_S_CONCAT = symbol("concat")
_S_FIRST = symbol("first")
_S_REST = symbol("rest")

class Var:
    def __init__(self, value):
        self.storage = threading.local()
        self.storage.value = value
    def deref(self):
        return self.storage.value

class Atom:
    def __init__(self, value):
        self._value = value
        self._lock = threading.Lock()
    def reset(self, value):
        with self._lock:
            self._value = value
            return value
    def swap(self, func, *args):
        with self._lock:
            self._value = func(self._value, *args)
            return self._value
    def deref(self):
        with self._lock:
            return self._value

def atom(value):
    return Atom(value)

def is_atom(obj):
    return isinstance(obj, Atom)

class PyTuple:
    def __init__(self, impl):
        assert isinstance(impl, tuple), "Expected a tuple"
        self._impl = impl
    def __eq__(self, other):
        return isinstance(other, PyTuple) and self._impl == other._impl
    def __len__(self):
        return len(self._impl)
    def __getitem__(self, index):
        return self._impl[index]
    def __iter__(self):
        return iter(self._impl)
    def _first(self):
        return self._impl[0]
    def _rest(self):
        return PyTuple(self._impl[1:])

class PyList:
    def __init__(self, impl):
        assert isinstance(impl, list), "Expected a list"
        self._impl = impl
    def __eq__(self, other):
        return isinstance(other, PyList) and self._impl == other._impl
    def __len__(self):
        return len(self._impl)
    def __getitem__(self, index):
        return self._impl[index]

class List:
    def __init__(self, impl, _meta):
        self._impl = impl
        self.__meta__ = _meta
    def __eq__(self, other):
        return isinstance(other, List) and self._impl == other._impl
    def __hash__(self):
        return hash(self._impl)
    def __len__(self):
        return len(self._impl)
    def __getitem__(self, index):
        return self._impl[index]
    def _with_meta(self, _meta):
        return List(self._impl, _meta)
    def _cons(self, value):
        return List(self._impl.cons(value), _meta=None)

def list_(*elements):
    return List(pr.plist(elements), _meta=None)

def is_list(obj):
    return isinstance(obj, List)

class Vector:
    def __init__(self, impl, _meta):
        assert isinstance(impl, pr.PVector), "Expected a PVector"
        self._impl = impl
        self.__meta__ = _meta
    def __eq__(self, other):
        return isinstance(other, Vector) and self._impl == other._impl
    def __len__(self):
        return len(self._impl)
    def __getitem__(self, index):
        return self._impl[index]
    def _with_meta(self, _meta):
        return Vector(self._impl, _meta)
    def _first(self):
        return self._impl[0]
    def _rest(self):
        return Vector(self._impl[1:], _meta=None)

def vec(coll):
    return Vector(pr.pvector(coll), _meta=None)

def vector(*elements):
    return vec(elements)

def is_vector(obj):
    return isinstance(obj, Vector)

class HashMap:
    def __init__(self, impl, _meta):
        assert isinstance(impl, pr.PMap), "Expected a PMap"
        self._impl = impl
        self.__meta__ = _meta
    def __eq__(self, other):
        return isinstance(other, HashMap) and self._impl == other._impl
    def __len__(self):
        return len(self._impl)
    def __iter__(self):
        return iter(self._impl)
    def __getitem__(self, key):
        return self._impl[key]
    def items(self):
        return self._impl.items()
    def _with_meta(self, _meta):
        return HashMap(self._impl, _meta)
    def _get(self, key, default=None):
        return self._impl.get(key, default)
    def _assoc(self, key, value):
        return HashMap(self._impl.set(key, value), _meta=None)

def hash_map(*elements):
    assert len(elements) % 2 == 0, "hash-map expects even number of elements"
    return HashMap(
        pr.pmap(dict(zip(elements[::2], elements[1::2]))),
        _meta=None)

def is_hash_map(obj):
    return isinstance(obj, HashMap)

def _mul(*args):
    result = 1
    for arg in args:
        result *= arg
    return result

def _div(*args):
    result = args[0]
    for arg in args[1:]:
        result /= arg
    return result

def _eq(*args):
    arg0 = args[0]
    for arg in args[1:]:
        if not equals(arg0, arg):
            return False
    return True

def equals(a, b): # pylint: disable=invalid-name
    return a == b

def count(coll):
    return len(coll)

def get(coll, key, default=None):
    return coll._get(key, default) # pylint: disable=protected-access

def assoc1(coll, key, value):
    return coll._assoc(key, value) # pylint: disable=protected-access

def assoc(coll, *kvs):
    for i in range(0, len(kvs), 2):
        key = kvs[i]
        value = kvs[i + 1]
        coll = assoc1(coll, key, value)
    return coll

def assoc_in(coll, path, value):
    if path:
        key = path[0]
        return assoc1(coll, key, assoc_in(get(coll, key), path[1:], value))
    else:
        return value

def update(coll, key, func, *args):
    return assoc1(coll, key, func(get(coll, key), *args))

def dissoc(coll, key):
    return coll._dissoc(key) # pylint: disable=protected-access

def cons(val, coll):
    return coll._cons(val) # pylint: disable=protected-access

def first(coll):
    return coll._first() # pylint: disable=protected-access

def rest(coll):
    return coll._rest() # pylint: disable=protected-access

def nth(coll, index, default=_UNDEFINED):
    try:
        return coll[index]
    except IndexError:
        if default is _UNDEFINED:
            raise
        return default

def _partition(num, coll):
    part = []
    for val in coll:
        part.append(val)
        if len(part) == num:
            yield part
            part = []

def destructure1(bform, value):
    return vec(_destructure1(bform, value))

def _destructure1(bform, value):
    if is_symbol(bform):
        return [bform, value]
    elif is_vector(bform):
        return _destructure_sequence(bform, value)
    elif is_hash_map(bform):
        raise NotImplementedError() # TODO: associative destructuring
    else:
        raise Exception("Unsupported binding form")

def _destructure_sequence(bform, value):
    if len(bform) == 0:
        return []
    if len(bform) == 1:
        sfirst = gensym()
        return [sfirst, list_(_S_FIRST, value)] + \
            _destructure1(bform[0], sfirst)
    elif bform[-2] is _K_AS:
        name = bform[-1]
        assert is_symbol(name), "Expected a symbol after :as"
        return [name, value] + _destructure1(bform[:-2], value)
    elif bform[0] == _S_AMPER:
        assert len(bform) == 2, "Expected exactly 1 entry after &"
        if is_hash_map(bform[1]): # TODO: keyword arguments
            raise NotImplementedError()
        return _destructure1(bform[1], value)
    else:
        sfirst = gensym()
        srest = gensym()
        return [
            sfirst, list_(_S_FIRST, value),
            srest, list_(_S_REST, value)
        ] + _destructure1(bform[0], sfirst) + \
            _destructure_sequence(bform[1:], srest)

def concat(*colls):
    result = []
    for coll in colls:
        for val in coll:
            result.append(val)
    return List(pr.plist(result), _meta=None)

@_with_attrs(counter=Atom(0))
def gensym(prefix="__G__"):
    index = gensym.counter.swap(lambda x: x + 1)
    return symbol(prefix + str(index))

def slurp(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

def print_(*args):
    print(" ".join(map(lambda x: pr_str_(x, False), args)), end="")

def println(*args):
    print_(*args)
    print_("\n")

def throw(obj):
    raise Exception(obj)

if __name__ == '__main__':
    main()
