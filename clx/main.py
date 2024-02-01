# pylint: disable=too-many-lines
from abc import abstractmethod, ABC
import ast
from bisect import bisect_left
from collections import namedtuple
from collections.abc import Hashable, Iterator, Iterable, Mapping, Sequence
import functools
import re
import sys
import threading
import types

import pyrsistent as pr

_UNDEFINED = object()

#************************************************************
# Utilities
#************************************************************

_MUNGE_TABLE = {
    "-": "_",
    "_": "_USCORE_",
    ".": "_DOT_",
    ":": "_COLON_",
    "+": "_PLUS_",
    "*": "_STAR_",
    "&": "_AMPER_",
    ">": "_GT_",
    "<": "_LT_",
    "=": "_EQ_",
    "%": "_PERCENT_",
    "#": "_SHARP_",
    "!": "_BANG_",
    "?": "_QMARK_",
    "'": "_SQUOTE_",
    "|": "_BAR_",
    "/": "_SLASH_",
    "$": "_DOLLAR_"
}

def munge(chars):
    return "".join(_MUNGE_TABLE.get(c, c) for c in chars)

#************************************************************
# Types
#************************************************************

class IMeta(ABC):
    @abstractmethod
    def with_meta(self, _meta):
        raise NotImplementedError()

class ICounted(ABC):
    @abstractmethod
    def count_(self):
        raise NotImplementedError()

class ISeqable(ABC):
    @abstractmethod
    def seq(self):
        raise NotImplementedError()

class ISeq(ISeqable, ABC):
    @abstractmethod
    def first(self):
        raise NotImplementedError()
    @abstractmethod
    def next(self):
        raise NotImplementedError()
    @abstractmethod
    def rest(self):
        raise NotImplementedError()

class IAssociative(ABC):
    @abstractmethod
    def lookup(self, key, not_found):
        raise NotImplementedError()
    @abstractmethod
    def assoc(self, key, value):
        raise NotImplementedError()

class ICollection(ABC):
    @abstractmethod
    def conj(self, value):
        raise NotImplementedError()

class ISequential(ABC):
    pass

class Symbol(Hashable, IMeta):
    def __init__(self, _namespace, _name, _meta):
        self.name = sys.intern(_name)
        self.namespace = sys.intern(_namespace) if _namespace else None
        self._hash = hash((self.namespace, self.name))
        self.__meta__ = _meta
    def __str__(self):
        return f"Symbol({self.namespace}, {self.name})"
    def __eq__(self, other):
        return isinstance(other, Symbol) \
            and self.name is other.name \
            and self.namespace is other.namespace
    def __hash__(self):
        return self._hash
    def with_meta(self, _meta):
        return Symbol(self.namespace, self.name, _meta)

def symbol(arg1, arg2=None):
    if arg2 is None:
        if isinstance(arg1, str):
            if "/" in arg1:
                _ns, name = arg1.split("/", 1)
                return Symbol(_ns, name, None)
            else:
                return Symbol(None, arg1, None)
        elif isinstance(arg1, Symbol):
            return arg1
        else:
            raise Exception(
                "symbol expects a string or a Symbol as the first argument")
    else:
        return Symbol(arg1, arg2, None)

def is_symbol(obj):
    return isinstance(obj, Symbol)

def is_simple_symbol(obj):
    return isinstance(obj, Symbol) and obj.namespace is None

class Keyword(Hashable):
    def __init__(self, _namespace, _name):
        self.name = sys.intern(_name)
        if _namespace is None:
            self.namespace = None
            self.munged = munge(_name)
        else:
            self.namespace = sys.intern(_namespace)
            self.munged = munge(_namespace + "/" + _name)
        self._hash = hash((_namespace, _name))
    def __eq__(self, other):
        return self is other
    def __hash__(self):
        return self._hash
    def __str__(self):
        return f"Keyword({self.namespace}, {self.name})"

KEYWORD_TABLE = {}
KEYWORD_TABLE_LOCK = threading.Lock()

def keyword(arg1, arg2=None):
    if arg2 is None:
        if isinstance(arg1, str):
            if "/" in arg1:
                _ns, _name = arg1.split("/", 1)
                qname = sys.intern(arg1)
            else:
                _ns = None
                _name = arg1
                qname = sys.intern(_name)
        elif isinstance(arg1, Symbol):
            _ns = arg1.namespace
            _name = arg1.name
            qname = f"{_ns}/{_name}" if _ns else _name
            qname = sys.intern((_ns + "/" + _name) if _ns else _name)
        elif isinstance(arg1, Keyword):
            return arg1
        else:
            raise Exception(
                "keyword expects a string, a Keyword, or a Symbol as the "
                "first argument")
    else:
        if arg1 is None:
            _ns = None
            _name = arg2
            qname = sys.intern(_name)
        else:
            _ns = arg1
            _name = arg2
            qname = sys.intern(_ns + "/" + _name)
    with KEYWORD_TABLE_LOCK:
        if qname in KEYWORD_TABLE:
            return KEYWORD_TABLE[qname]
        else:
            new_kw = Keyword(_ns, _name)
            KEYWORD_TABLE[qname] = new_kw
            return new_kw

def is_keyword(obj):
    return isinstance(obj, Keyword)

def is_simple_keyword(obj):
    return isinstance(obj, Keyword) and obj.namespace is None

class PersistentList(
        Hashable,
        Sequence,
        IMeta,
        ICounted,
        ISeq,
        ISequential,
        ICollection):
    def __init__(self, _first, _rest, length, _meta):
        self._first = _first
        self._rest = _rest
        self._length = length
        self.__meta__ = _meta
    def __eq__(self, other):
        if self is other:
            return True
        elif type(other) is PersistentList:
            if self._length != other._length:
                return False
            else:
                lst1, lst2 = self, other
                while lst1._length > 0:
                    if lst1._first != lst2._first:
                        return False
                    lst1, lst2 = lst1._rest, lst2._rest
                return True
        else:
            return _equiv_sequential(self, other)
    def __hash__(self):
        raise NotImplementedError()
    def __len__(self):
        return self._length
    def __iter__(self):
        lst = self
        while lst._length > 0:
            yield lst._first
            lst = lst._rest
    def __getitem__(self, index):
        raise NotImplementedError()
    def with_meta(self, _meta):
        return PersistentList(self._first, self._rest, self._length, _meta)
    def count_(self):
        return self._length
    def first(self):
        return self._first
    def next(self):
        if self._length <= 1:
            return None
        return self._rest
    def rest(self):
        return self._rest
    def seq(self):
        return self
    def conj(self, value):
        return PersistentList(value, self, self._length + 1, None)

_EMPTY_LIST = PersistentList(None, None, 0, None)
_EMPTY_LIST.rest = lambda: _EMPTY_LIST
_EMPTY_LIST.seq = lambda: None

def list_(*elements):
    return _list_from_iterable(elements)

def _list_from_iterable(iterable):
    _list = iterable if isinstance(iterable, list) else list(iterable)
    result = _EMPTY_LIST
    for elem in reversed(_list):
        result = PersistentList(
            elem, result, result._length + 1, None) # pylint: disable=protected-access
    return result

def is_list(obj):
    return type(obj) is PersistentList

class PersistentVector(
        Hashable,
        Sequence,
        IMeta,
        ICounted,
        ISeqable,
        ISequential):
    def __init__(self, impl, _meta):
        self._impl = impl
        self.__meta__ = _meta
    def __eq__(self, other):
        return self is other \
            or (type(other) is PersistentVector \
                and self._impl == other._impl) \
            or _equiv_sequential(self, other)
    def __hash__(self):
        return hash(self._impl)
    def __len__(self):
        return len(self._impl)
    def __iter__(self):
        return iter(self._impl)
    def __getitem__(self, index):
        if type(index) is slice:
            raise NotImplementedError() # PVector's slice is slow
        return self._impl[index]
    def with_meta(self, _meta):
        return PersistentVector(self._impl, _meta)
    def count_(self):
        return len(self._impl)
    def seq(self):
        return _list_from_iterable(self._impl)

_EMPTY_VECTOR = PersistentVector(pr.pvector(), _meta=None)
_EMPTY_VECTOR.seq = lambda: None

def vec(coll):
    if not coll:
        return _EMPTY_VECTOR
    return PersistentVector(pr.pvector(coll), None)

def vector(*elements):
    return vec(elements)

def is_vector(obj):
    return type(obj) is PersistentVector

class PersistentMap(
        Hashable,
        Mapping,
        IMeta,
        ICounted,
        ISeqable,
        IAssociative):
    def __init__(self, impl, _meta):
        assert isinstance(impl, pr.PMap), "Expected a PMap"
        self._impl = impl
        self.__meta__ = _meta
    def __eq__(self, other):
        return isinstance(other, PersistentMap) \
            and self._impl == other._impl
    def __hash__(self):
        return hash(self._impl)
    def __len__(self):
        return len(self._impl)
    def __iter__(self):
        return iter(self._impl)
    def __getitem__(self, key):
        return self._impl[key]
    def with_meta(self, _meta):
        return PersistentMap(self._impl, _meta)
    def count_(self):
        return len(self._impl)
    def seq(self):
        if len(self._impl) == 0:
            return None
        raise NotImplementedError()
    def lookup(self, key, not_found):
        return self._impl.get(key, not_found)
    def assoc(self, key, value):
        return PersistentMap(self._impl.set(key, value), _meta=None)
    def merge(self, *other):
        return PersistentMap(self._impl.update(*other), _meta=None)

_EMPTY_MAP = PersistentMap(pr.pmap(), _meta=None)

def hash_map(*elements):
    assert len(elements) % 2 == 0, "hash-map expects even number of elements"
    if len(elements) == 0:
        return _EMPTY_MAP
    else:
        return PersistentMap(
            pr.pmap(dict(zip(elements[::2], elements[1::2]))),
            _meta=None)

def is_hash_map(obj):
    return isinstance(obj, PersistentMap)

class IRecord(IAssociative, ABC):
    pass

def define_record(name, *fields):
    for field in fields:
        assert is_simple_keyword(field), \
            "field names must be simple keywords"
    init_args = ", ".join([f.munged for f in fields])
    init_fields = "; ".join(
        [f"self.{f.munged} = {f.munged}" for f in fields])
    def assoc_method(field):
        params = ", ".join(
            ["value" if f is field else f"self.{f.munged}" for f in fields])
        return f"lambda self, value: {name}({params})"
    assoc_methods = [f"kw_{f.munged}: {assoc_method(f)}" for f in fields]
    _ns = {
        **{f"kw_{munge(f.name)}": keyword(f) for f in fields},
        "IRecord": IRecord,
    }
    exec( # pylint: disable=exec-used
        f"""
assoc_methods = {{
  {", ".join(assoc_methods)}
}}

class {name}(IRecord):
  def __init__(self, {init_args}):
    {init_fields}
  def lookup(self, field, not_found):
    return getattr(self, field.munged, not_found)
  def assoc(self, field, value):
    return assoc_methods[field](self, value)
cls = {name}
        """,
        _ns)
    return _ns["cls"]

class Cons(Hashable, Sequence, IMeta, ISeq, ISequential):
    def __init__(self, _first, _rest, _meta):
        assert isinstance(_rest, ISeq), "rest of Cons must be a seq"
        self._first = _first
        self._rest = _rest
        self.__meta__ = _meta
    def __bool__(self):
        return True
    def __len__(self):
        raise NotImplementedError()
    def __hash__(self):
        raise NotImplementedError()
    def __eq__(self, other):
        return self is other \
            or (isinstance(other, Cons) \
                and self._first == other._first \
                and self._rest == other._rest) \
            or _equiv_sequential(self, other)
    def __iter__(self):
        raise NotImplementedError()
    def __getitem__(self, index):
        raise NotImplementedError()
    def with_meta(self, _meta):
        return Cons(self._first, self._rest, _meta)
    def first(self):
        return self._first
    def next(self):
        return self._rest.seq()
    def rest(self):
        return self._rest
    def seq(self):
        return self

class LazySeq(Hashable, Sequence, IMeta, ISeq, ISequential):
    def __init__(self, _func, _seq, _meta):
        self._lock = threading.Lock()
        self._func = _func
        self._seq = _seq
        self.__meta__ = _meta
    def __eq__(self, other):
        return self is other \
            or _equiv_sequential(self, other)
    def __bool__(self):
        return bool(self.seq())
    def __len__(self):
        raise NotImplementedError()
    def __hash__(self):
        raise NotImplementedError()
    def __iter__(self):
        raise NotImplementedError()
    def __getitem__(self, index):
        raise NotImplementedError()
    def with_meta(self, _meta):
        return LazySeq(self._func, self._seq, _meta)
    def first(self):
        return first(self._force())
    def next(self):
        return next_(self._force())
    def rest(self):
        return rest(self._force())
    def seq(self):
        return seq(self._force())
    def _force(self):
        with self._lock:
            if self._func is not None:
                self._seq = self._func()
                self._func = None
            return self._seq

def _equiv_sequential(x, y): # pylint: disable=invalid-name
    assert isinstance(x, ISequential), "expected a sequential"
    if isinstance(y, ISequential):
        x, y = seq(x), seq(y)
        while True:
            if x is None:
                return y is None
            elif y is None:
                return False
            elif x is y:
                return True
            elif first(x) != first(y):
                return False
            else:
                x, y = next_(x), next_(y)
    else:
        return False

#************************************************************
# Constants
#************************************************************

_S_QUOTE = symbol("quote")
_S_UNQUOTE = symbol("unquote")
_S_SPLICE_UNQUOTE = symbol("splice-unquote")
_S_WITH_META = symbol("with-meta")
_S_VEC = symbol("vec")
_S_VECTOR = symbol("vector")
_S_LIST = symbol("list")
_S_HASH_MAP = symbol("hash-map")
_S_CONCAT = symbol("concat")
_S_DEF = symbol("def")
_S_DO = symbol("do")
_S_LET_STAR = symbol("let*")
_S_IF = symbol("if")
_S_FN_STAR = symbol("fn*")
_S_PYTHON = symbol("___python")
_S_AMPER = symbol("&")
_S_KEYWORD = symbol("keyword")
_S_SYMBOL = symbol("symbol")

_K_LINE = keyword("line")
_K_COLUMN = keyword("column")
_K_CURRENT_NS = keyword("current-ns")
_K_NAMESPACES = keyword("namespaces")
_K_LOCALS = keyword("locals")
_K_COUNTER = keyword("counter")
_K_PY_NAME = keyword("py-name")
_K_BINDINGS = keyword("bindings")

#************************************************************
# Reader
#************************************************************

Token = namedtuple("Token", ["string", "line", "column"])

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

def read_string(text):
    tokens = seq(tokenize(text))
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
        line_id = bisect_left(line_starts, match.start(1) + 1)
        column = match.start(1) - line_starts[line_id - 1] + 1
        yield Token(string=token, line=line_id, column=column)

def read_form(tokens):
    token = first(tokens)
    assert isinstance(token, Token), "expected a token"
    rtokens = rest(tokens)
    tstring = token.string
    if tstring == "'":
        form, _rest = read_form(rtokens)
        return \
            with_meta(
                list_(_S_QUOTE, form),
                hash_map(
                    _K_LINE, token.line,
                    _K_COLUMN, token.column)), \
            _rest
    elif tstring == "`":
        form, _rest = read_form(rtokens)
        return quasiquote(form), _rest
    elif tstring == "~":
        form, _rest = read_form(rtokens)
        return list_(_S_UNQUOTE, form), _rest
    elif tstring == "~@":
        form, _rest = read_form(rtokens)
        return list_(_S_SPLICE_UNQUOTE, form), _rest
    elif tstring == "^":
        _meta, rest1 = read_form(rtokens)
        form, rest2 = read_form(rest1)
        return vary_meta(form, merge, _meta), rest2
    elif tstring == "(":
        return read_collection(tokens, list_, "(", ")")
    elif tstring == "[":
        return read_collection(tokens, vector, "[", "]")
    elif tstring == "{":
        return read_collection(tokens, hash_map, "{", "}")
    else:
        return read_atom(token), rtokens

def read_atom(token):
    assert isinstance(token, Token), "expected a token"
    tstring = token.string
    if re.match(INT_RE, tstring):
        return int(tstring)
    elif re.match(FLOAT_RE, tstring):
        return float(tstring)
    elif re.match(STRING_RE, tstring):
        return unescape(tstring[1:-1])
    elif tstring[0] == "\"":
        raise Exception("Unterminated string")
    elif tstring == "true":
        return True
    elif tstring == "false":
        return False
    elif tstring == "nil":
        return None
    elif tstring[0] == ":":
        return keyword(tstring[1:])
    else:
        return with_meta(
            symbol(tstring),
            hash_map(
                _K_LINE, token.line,
                _K_COLUMN, token.column))

def unescape(text):
    return text \
        .replace(r"\\", "\\") \
        .replace(r"\"", "\"") \
        .replace(r"\n", "\n")

def read_collection(
        tokens,
        ctor,
        start,
        end):
    token0 = first(tokens)
    assert isinstance(token0, Token), "expected a token"
    assert token0.string == start, f"Expected '{start}'"
    tokens = rest(tokens)
    elements = []
    while True:
        token = first(tokens)
        if token is None:
            raise Exception(f"Expected '{end}'")
        assert isinstance(token, Token), "expected a token"
        if token.string == end:
            return with_meta(
                ctor(*elements),
                hash_map(
                    _K_LINE, token0.line,
                    _K_COLUMN, token0.column)), \
                rest(tokens)
        element, tokens = read_form(tokens)
        elements.append(element)

def quasiquote(form):
    if is_list(form) and len(form) > 0:
        head = form.first()
        if head == _S_UNQUOTE:
            assert len(form) == 2, "unquote expects 1 argument"
            return second(form)
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
    def entry(_f):
        if is_list(_f) and len(_f) > 0 and _f.first() == _S_UNQUOTE:
            assert len(_f) == 2, "unquote expects 1 argument"
            return list_(_S_LIST, second(_f))
        elif is_list(_f) and len(_f) > 0 and _f.first() == _S_SPLICE_UNQUOTE:
            assert len(_f) == 2, "splice-unquote expects 1 argument"
            return second(_f)
        else:
            return list_(_S_LIST, quasiquote(_f))
    return cons(_S_CONCAT, list_(*map(entry, form)))

#************************************************************
# Printer
#************************************************************

def pr_str(obj, readably=False):
    if obj is None:
        return "nil"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, str):
        if readably:
            return "\"" + escape(obj) + "\""
        return obj
    if isinstance(obj, Keyword):
        _ns = f"{obj.namespace}/" if obj.namespace else ""
        return f":{_ns}{obj.name}"
    if isinstance(obj, Symbol):
        _ns = f"{obj.namespace}/" if obj.namespace else ""
        return f"{_ns}{obj.name}"
    if isinstance(obj, PersistentList):
        return "(" + \
            " ".join(map(lambda x: pr_str(x, readably), obj)) + \
            ")"
    if isinstance(obj, PersistentVector):
        return "[" + \
            " ".join(map(lambda x: pr_str(x, readably), obj)) + \
            "]"
    if isinstance(obj, PersistentMap):
        ret = functools.reduce(
            lambda acc, kv:
                acc.conj(pr_str(kv[1], readably))
                    .conj(pr_str(kv[0], readably)),
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
# Evaluation
#************************************************************

Context = define_record("Context",
    _K_CURRENT_NS,
    _K_NAMESPACES,
    _K_LOCALS,
    _K_COUNTER) # for generating unique names
Namespace = define_record("Namespace", _K_BINDINGS)
Binding = define_record("Binding", _K_PY_NAME)

def eval_string(text):
    tokens = list(tokenize(text))
    _bindings, _globals = _basic_bindings()

    def _def(name, value):
        _globals[name] = value
        return value
    _globals["___def"] = _def

    ctx = Context(
        current_ns="user",
        namespaces=hash_map("user", Namespace(bindings=_bindings)),
        locals=hash_map(),
        counter=1000)

    while tokens:
        form, tokens = read_form(tokens)
        result, body, ctx = _compile(form, ctx)
        body = ast.Module(body, type_ignores=[])
        result = ast.Expression(result, type_ignores=[])
        _transform_ast(ctx, body, result)
        exec(compile(body, "<none>", "exec"), _globals) # pylint: disable=exec-used
        result = eval(compile(result, "<none>", "eval"), _globals) # pylint: disable=eval-used

    return result, ctx, _globals

def _compile(form, ctx):
    if isinstance(form, PersistentList):
        head = form.first()
        if head == _S_DEF:
            return _compile_def(form, ctx)
        if head == _S_DO:
            return _compile_do(form, ctx)
        if head == _S_LET_STAR:
            return _compile_let(form, ctx)
        if head == _S_IF:
            return _compile_if(form, ctx)
        if head == _S_FN_STAR:
            return _compile_fn(form, ctx)
        if head == _S_QUOTE:
            assert len(form) == 2, "quote expects exactly 1 argument"
            return _node(ast.Constant, form, second(form)), [], ctx
        if head == _S_PYTHON:
            return _compile_python(form, ctx)
        else:
            return _compile_call(form, ctx)
    elif isinstance(form, PersistentVector):
        return _compile_vector(form, ctx)
    elif isinstance(form, PersistentMap):
        return _compile_map(form, ctx)
    elif isinstance(form, Symbol):
        py_name = _resolve_symbol(ctx, form).py_name
        return _node(ast.Name, form, py_name, ast.Load()), [], ctx
    else:
        # Location information is not available for constants
        return ast.Constant(form, lineno=0, col_offset=0), [], ctx

def _compile_def(form, ctx):
    assert len(form) == 3, "def expects 2 arguments"
    name = second(form)
    assert is_simple_symbol(name), \
        "def expects a simple symbol as the first argument"
    value, body, ctx = _compile(third(form), ctx)
    _ns = ctx.current_ns
    py_name = munge(f"{_ns}/{name.name}")
    return \
        _node(ast.Call, form,
            _node(ast.Name, form, "___def", ast.Load()),
            [_node(ast.Constant, form, py_name), value], []), \
        body, \
        assoc_in(ctx,
            list_(_K_NAMESPACES, _ns, _K_BINDINGS, name.name),
            Binding(py_name))

def _compile_do(form, ctx):
    stmts = []
    forms = form.rest()
    result_name = None
    while forms:
        _f, forms = forms.first(), forms.rest()
        f_result, f_stmts, ctx = _compile(_f, ctx)
        stmts.extend(f_stmts)
        result_name, ctx = _gen_name(ctx)
        stmts.append(
            _node(ast.Assign, form,
                [_node(ast.Name, form, result_name, ast.Store())],
                f_result))
    return \
        _node(ast.Name, form, result_name, ast.Load()) \
            if result_name is not None \
            else _node(ast.Constant, form, None), \
        stmts, ctx

def _compile_let(form, ctx):
    assert len(form) > 1, "let* expects at least 1 argument"
    assert len(form) < 4, "let* expects at most 2 arguments"
    bindings = second(form)
    assert is_vector(bindings), \
        "let* expects a vector as the first argument"
    assert len(bindings) % 2 == 0, \
        "bindings of let* must have even number of elements"
    old_locals = ctx.locals
    body = []
    for i in range(0, len(bindings), 2):
        _name, value = bindings[i], bindings[i + 1]
        assert is_simple_symbol(_name), \
            "first element of each binding pair must be a symbol"
        value_expr, value_stmts, ctx = _compile(value, ctx)
        munged = munge(_name.name)
        py_name, ctx = _gen_name(ctx, f"{munged}_")
        body.extend(value_stmts)
        body.append(
            _node(ast.Assign, _name,
                [_node(ast.Name, _name, py_name, ast.Store())], value_expr))
        ctx = assoc_in(ctx, list_(_K_LOCALS, _name.name), Binding(py_name))
    if len(form) == 3:
        body_expr, body_stmts, ctx = _compile(third(form), ctx)
        body.extend(body_stmts)
    else:
        body_expr = _node(ast.Constant, form, None)
    return body_expr, body, ctx.assoc(_K_LOCALS, old_locals)

def _compile_if(form, ctx):
    assert len(form) == 4, "if expects 3 arguments"
    test, then, else_ = second(form), third(form), fourth(form)
    test_expr, test_stmts, ctx = _compile(test, ctx)
    then_expr, then_stmts, ctx = _compile(then, ctx)
    else_expr, else_stmts, ctx = _compile(else_, ctx)
    result_name, ctx = _gen_name(ctx, "___if_result_")
    result_store = _node(ast.Name, form, result_name, ast.Store())
    result_load = _node(ast.Name, form, result_name, ast.Load())
    return \
        result_load, \
        test_stmts + [
            _node(ast.If, form,
                test_expr,
                then_stmts + [
                    _node(ast.Assign, form, [result_store], then_expr)
                ],
                else_stmts + [
                    _node(ast.Assign, form, [result_store], else_expr)
                ]
            ),
        ], \
        ctx

def _compile_fn(form, ctx):
    assert len(form) > 1, "fn* expects at least 1 argument"
    assert len(form) < 4, "fn* expects at most 2 arguments"
    params_form = second(form)
    assert isinstance(params_form, PersistentVector), \
        "fn* expects a vector of parameters as the first argument"
    fname, ctx = _gen_name(ctx, "___fn_")
    old_locals = ctx.locals

    pos_params = []
    rest_param = None
    while params_form:
        param = first(params_form)
        assert is_simple_symbol(param), \
            "parameters of fn* must be simple symbols"
        if param == _S_AMPER:
            assert next_(next_(params_form)) is None, \
                "fn* expects a single symbol after &"
            rest_param = second(params_form)
            assert is_simple_symbol(rest_param), \
                "rest parameter of fn* must be a simple symbol"
            ctx = assoc_in(ctx,
                list_(_K_LOCALS, rest_param.name),
                Binding(munge(rest_param.name)))
            break
        pos_params.append(param)
        ctx = assoc_in(ctx,
            list_(_K_LOCALS, param.name),
            Binding(munge(param.name)))
        params_form = rest(params_form)

    if len(form) == 3:
        body_form = third(form)
        body_expr, body_stmts, ctx = _compile(body_form, ctx)
        body = body_stmts + [_node(ast.Return, form, body_expr)]
    else:
        body = [_node(ast.Pass, form)]

    def _arg(_p):
        return _node(ast.arg, _p, munge(_p.name))

    return \
        _node(ast.Name, form, fname, ast.Load()), \
        [_node(ast.FunctionDef, form,
            fname,
            ast.arguments(
                posonlyargs=[_arg(p) for p in pos_params],
                args=[],
                vararg=_arg(rest_param) if rest_param else None,
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body,
            [])], \
        ctx.assoc(_K_LOCALS, old_locals)

def _compile_python(form, ctx):
    def _eval_entry(entry):
        if isinstance(entry, Symbol):
            return _resolve_symbol(ctx, entry).py_name
        elif isinstance(entry, str):
            return entry
        else:
            raise Exception("___python expects strings or symbols")
    source = "".join(map(_eval_entry, form.rest()))
    module = ast.parse(source)
    if module.body and isinstance(module.body[-1], ast.Expr):
        stmts = module.body[:-1]
        result = module.body[-1].value
    else:
        stmts = module.body
        result = _node(ast.Constant, form, None)
    return result, stmts, ctx

def _compile_call(form, ctx):
    args = []
    body = []
    for arg in form.rest():
        arg_expr, arg_body, ctx = _compile(arg, ctx)
        args.append(arg_expr)
        body.extend(arg_body)
    _f, f_body, ctx = _compile(form.first(), ctx)
    body.extend(f_body)
    return _node(ast.Call, form, _f, args, []), body, ctx

def _compile_vector(form, ctx):
    el_exprs = []
    stmts = []
    for elm in form:
        el_expr, el_stmts, ctx = _compile(elm, ctx)
        el_exprs.append(el_expr)
        stmts.extend(el_stmts)
    py_vector = _resolve_symbol(ctx, _S_VECTOR).py_name
    return \
        _node(ast.Call, form,
            _node(ast.Name, form, py_vector, ast.Load()),
            el_exprs,
            []), \
        stmts, \
        ctx

def _compile_map(form, ctx):
    args = []
    stmts = []
    for key, value in form.items():
        key_expr, key_stmts, ctx = _compile(key, ctx)
        args.append(key_expr)
        stmts.extend(key_stmts)
        value_expr, value_stmts, ctx = _compile(value, ctx)
        args.append(value_expr)
        stmts.extend(value_stmts)
    py_hash_map = _resolve_symbol(ctx, _S_HASH_MAP).py_name
    return \
        _node(ast.Call, form,
            _node(ast.Name, form, py_hash_map, ast.Load()),
            args,
            []), \
        stmts, \
        ctx

def _resolve_symbol(ctx, sym):
    if is_simple_symbol(sym):
        result = ctx.locals.lookup(sym.name, None)
        if result is not None:
            return result
        else:
            result = ctx \
                .namespaces.lookup(ctx.current_ns, None) \
                .bindings.lookup(sym.name, None)
            if result is not None:
                return result
    else:
        namespace = ctx.namespaces.lookup(sym.namespace, None)
        if namespace is not None:
            result = namespace.bindings.lookup(sym.name, None)
            if result is not None:
                return result
        else:
            raise Exception(f"Namespace '{sym.namespace}' not found")
    raise Exception(f"Symbol '{pr_str(sym)}' not found")

def _gen_name(ctx, base="___gen_"):
    counter = ctx.counter
    ctx = assoc(ctx, _K_COUNTER, counter + 1)
    return f"{base}{counter}", ctx

def _node(type_, form, *args):
    _n = type_(*args)
    _meta = meta(form)
    _n.lineno = _meta.lookup(_K_LINE, None)
    _n.col_offset = _meta.lookup(_K_COLUMN, None)
    return _n

def _transform_ast(ctx, body, result):
    _fix_constants(ctx, body, result)

CONSTANT_COUNTER = 1000

def _fix_constants(ctx, body, result):
    consts = {}

    def const_name(index):
        return f"___const_{index}"

    class Transformer(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, (
                    Keyword,
                    Symbol,
                    PersistentList,
                    PersistentVector,
                    PersistentMap)):
                global CONSTANT_COUNTER # pylint: disable=global-statement
                index = CONSTANT_COUNTER
                CONSTANT_COUNTER += 1
                consts[index] = _compile_quote(ctx, node.value)
                return ast.Name(const_name(index), ast.Load(),
                    lineno=0, col_offset=0)
            else:
                self.generic_visit(node)
                return node

    Transformer().visit(body)
    Transformer().visit(result)

    body.body[:0] = [
        ast.Assign(
            [ast.Name(const_name(index),
                ast.Store(), lineno=0, col_offset=0)],
            value,
            lineno=0, col_offset=0)
        for index, value in consts.items()
    ]

def _compile_quote(ctx, value):
    if isinstance(value, (Keyword, Symbol)):
        ctor = _resolve_symbol(ctx,
            _S_KEYWORD if isinstance(value, Keyword) else _S_SYMBOL).py_name
        _ns = ast.Constant(value.namespace, lineno=0, col_offset=0)
        _name = ast.Constant(value.name, lineno=0, col_offset=0)
        return ast.Call(
            ast.Name(ctor, ast.Load(), lineno=0, col_offset=0),
            [_ns, _name], [], lineno=0, col_offset=0)
    elif isinstance(value, (PersistentList, PersistentVector)):
        ctor = _resolve_symbol(ctx,
            _S_LIST if isinstance(value, PersistentList) else _S_VECTOR) \
            .py_name
        return ast.Call(
            ast.Name(ctor, ast.Load(), lineno=0, col_offset=0),
            [_compile_quote(ctx, v) for v in value],
            [], lineno=0, col_offset=0)
    elif isinstance(value, PersistentMap):
        ctor = _resolve_symbol(ctx, _S_HASH_MAP).py_name
        args = []
        for key, val in value.items():
            args.append(_compile_quote(ctx, key))
            args.append(_compile_quote(ctx, val))
        return ast.Call(
            ast.Name(ctor, ast.Load(), lineno=0, col_offset=0),
            args, [], lineno=0, col_offset=0)
    return ast.Constant(value, lineno=0, col_offset=0)

def _basic_bindings():
    bindings = {
        "+": lambda *args: sum(args),
        "even?": lambda x: x % 2 == 0,
        "odd?": lambda x: x % 2 == 1,
        "with-meta": with_meta,
        "apply": apply,
        "keyword": keyword,
        "symbol": symbol,
        "list": list_,
        "vector": vector,
        "hash-map": hash_map,
    }
    _bindings = hash_map()
    _globals = {}
    for name, value in bindings.items():
        munged = munge(f"user/{name}")
        _bindings = _bindings.assoc(name, Binding(munged))
        _globals[munged] = value
    return _bindings, _globals

#************************************************************
# Core
#************************************************************

def apply(func, *args):
    assert callable(func), "apply expects a function as the first argument"
    assert len(args) > 0, "apply expects at least 2 arguments"
    last_arg = args[-1]
    assert isinstance(last_arg, Iterable), \
        "last argument of apply must be iterable"
    return func(*args[:-1], *last_arg)

def meta(obj):
    return getattr(obj, "__meta__", None)

def with_meta(obj, _meta):
    assert isinstance(_meta, PersistentMap), \
        "with-meta expects a PersistentMap as the second argument"
    if isinstance(obj, IMeta):
        return obj.with_meta(_meta)
    elif isinstance(obj, types.FunctionType):
        clone = types.FunctionType(
            obj.__code__,
            obj.__globals__,
            obj.__name__,
            obj.__defaults__,
            obj.__closure__)
        clone.__meta__ = _meta
        return clone
    else:
        raise Exception("object does not support metadata")

def vary_meta(obj, f, *args): # pylint: disable=invalid-name
    return with_meta(obj, f(meta(obj), *args))

def is_counted(x): # pylint: disable=invalid-name
    return isinstance(x, ICounted)

def count(x): # pylint: disable=invalid-name
    return x.count_()

def cons(obj, coll):
    if coll is None:
        return list_(obj)
    elif isinstance(coll, ISeq):
        return Cons(obj, coll, _meta=None)
    else:
        return Cons(obj, seq(coll), _meta=None)

def lazy_seq(func):
    return LazySeq(func, None, _meta=None)

def seq(coll):
    if isinstance(coll, ISeqable):
        return coll.seq()
    elif coll is None:
        return None
    elif isinstance(coll, Iterable):
        return iterator_seq(iter(coll)).seq()
    else:
        raise Exception("expected a seqable object")

def iterator_seq(it): # pylint: disable=invalid-name
    assert isinstance(it, Iterator), "iterator-seq expects an iterator"
    def _seq():
        value = next(it, _UNDEFINED)
        return cons(value, lazy_seq(_seq)) \
            if value is not _UNDEFINED else None
    return lazy_seq(_seq)

def is_seq(obj):
    return isinstance(obj, ISeq)

def is_seqable(obj):
    return obj is None or isinstance(obj, (ISeqable, Iterable))

def first(coll):
    coll = seq(coll)
    if coll is None:
        return None
    return coll.first()

def next_(coll):
    coll = seq(coll)
    if coll is None:
        return None
    return coll.next()

def rest(coll):
    coll = seq(coll)
    if coll is None:
        return _EMPTY_LIST
    return coll.rest()

def second(coll):
    return first(next_(coll))

def third(coll):
    return first(next_(next_(coll)))

def fourth(coll):
    return first(next_(next_(next_(coll))))

def get(obj, key, not_found=None):
    if isinstance(obj, IAssociative):
        return obj.lookup(key, not_found)
    else:
        return not_found

def assoc(obj, key, value):
    if obj is None:
        return hash_map(key, value)
    return obj.assoc(key, value)

def get_in(obj, path, not_found=_UNDEFINED):
    for key in path:
        obj = get(obj, key, not_found)
    return obj

def assoc_in(obj, path, value):
    path0 = first(path)
    path = next_(path)
    if path:
        child0 = get(obj, path0)
        return assoc(obj, path0, assoc_in(child0, path, value))
    else:
        return assoc(obj, path0, value)

def concat(*colls):
    num_colls = len(colls)
    if num_colls == 0:
        return _EMPTY_LIST
    elif num_colls == 1:
        return lazy_seq(lambda: colls[0])
    else:
        coll0 = colls[0]
        colls = colls[1:]
        def _seq():
            return cons(first(coll0), concat(rest(coll0), *colls)) \
                if seq(coll0) else concat(*colls)
        return lazy_seq(_seq)

def merge(*maps):
    def helper(m1, m2): # pylint: disable=invalid-name
        if m1 is None:
            return m2
        elif m2 is None:
            return m1
        else:
            return m1.merge(m2)
    return functools.reduce(helper, maps, None)
