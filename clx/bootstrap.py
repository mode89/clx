import ast
from bisect import bisect_left
from collections.abc import Iterable
import functools
import itertools
import re
import threading
import types
import weakref

from lepet_ext import \
    IMeta, ICounted, IAssociative, ISeqable, ISeq, \
    Symbol, symbol, is_symbol, is_simple_symbol, \
    Keyword, keyword, is_keyword, is_simple_keyword, \
    PersistentList, list_, is_list, \
    PersistentVector, vector, \
    PersistentHashMap, hash_map, hash_map_from, \
    cons, lazy_seq, seq, \
    atom, is_atom, \
    is_seq, is_record, \
    first, next_, rest, get, nth, conj, take, drop, count, \
    map_, filter_, reduce
from lepet_ext import is_hash_map as is_map
from lepet_ext import define_record as define_record0

Iterable.register(PersistentVector)
PersistentMap = PersistentHashMap

_DUMMY = type("Dummy", (), {})()

#************************************************************
# Utilities
#************************************************************

_MUNGE_TABLE = {
    "-": "_DASH_",
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

_SPECIAL_NAMES = {
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "False",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "None",
    "or",
    "return",
    "True",
    "try",
    "while",
    "with",
    "yield",
}

def munge(obj):
    if type(obj) is Symbol or type(obj) is Keyword:
        name = obj.namespace + "/" + obj.name \
            if obj.namespace else obj.name
    elif type(obj) is str:
        name = obj
    else:
        raise Exception("munge expects string, Symbol, or Keyword")

    if name in _SPECIAL_NAMES:
        return f"{name}_"
    if name[-1] == "_":
        if name[:-1] in _SPECIAL_NAMES:
            raise Exception(f"name '{name}' is reserved")

    return "".join(_MUNGE_TABLE.get(c, c) for c in name)

#************************************************************
# Types
#************************************************************

def is_vector(obj):
    return type(obj) is PersistentVector

def define_record(name, *fields):
    return define_record0(name, *[(f, munge(f)) for f in fields])

class DynamicVar:
    def __init__(self, value):
        self._lock = threading.Lock()
        self._root = value
        self._stack = threading.local()
        self._stack.value = None
        self._is_bound = False
    def deref(self):
        with self._lock:
            return self._root \
                if not self._is_bound \
                else self._stack.value.first()
    def push(self, value):
        with self._lock:
            self._stack.value = cons(value, self._stack.value)
            self._is_bound = True
    def pop(self):
        with self._lock:
            assert self._is_bound, "dynamic var is not bound"
            self._stack.value = self._stack.value.next()
            self._is_bound = self._stack.value is not None
    def set(self, value):
        with self._lock:
            assert self._stack.value is not None, \
                "dynamic var is not bound by this thread"
            self._stack.value = cons(value, self._stack.value.next())
    def is_bound(self):
        with self._lock:
            return self._is_bound

#************************************************************
# Constants
#************************************************************

_S_QUOTE = symbol("quote")
_S_UNQUOTE = symbol("unquote")
_S_SPLICE_UNQUOTE = symbol("splice-unquote")
_S_DEF = symbol("def")
_S_DO = symbol("do")
_S_LET = symbol("let")
_S_COND = symbol("cond")
_S_FN_STAR = symbol("fn*")
_S_LETFN = symbol("letfn")
_S_TRY = symbol("try")
_S_CATCH = symbol("catch")
_S_FINALLY = symbol("finally")
_S_LOOP = symbol("loop")
_S_RECUR = symbol("recur")
_S_DOT = symbol(".")
_S_IMPORT_STAR = symbol("import*")
_S_PYTHON = symbol("python*")
_S_PYTHON_WITH = symbol("python/with")
_S_LOCAL_CONTEXT = symbol("___local_context")
_S_AMPER = symbol("&")

def _core_symbol(name):
    return symbol("clx.core", name)

_S_WITH_META = _core_symbol("with-meta")
_S_VEC = _core_symbol("vec")
_S_VECTOR = _core_symbol("vector")
_S_LIST = _core_symbol("list")
_S_HASH_MAP = _core_symbol("hash-map")
_S_CONCAT = _core_symbol("concat")
_S_KEYWORD = _core_symbol("keyword")
_S_SYMBOL = _core_symbol("symbol")
_S_APPLY = _core_symbol("apply")
_S_RE_PATTERN = _core_symbol("re-pattern")
_S_DEREF = _core_symbol("deref")
_S_FIRST = _core_symbol("first")
_S_REST = _core_symbol("rest")

_K_LINE = keyword("line")
_K_COLUMN = keyword("column")
_K_CURRENT_NS = keyword("current-ns")
_K_ENV = keyword("env")
_K_PY_NAME = keyword("py-name")
_K_LOCAL = keyword("local")
_K_DYNAMIC = keyword("dynamic")
_K_IMPORTED_FROM = keyword("imported-from")
_K_ATTRIBUTE = keyword("attribute")
_K_BINDINGS = keyword("bindings")
_K_IMPORTS = keyword("imports")
_K_ALIASES = keyword("aliases")
_K_MACRO_QMARK = keyword("macro?")
_K_TOP_LEVEL_Q = keyword("top_level?")
_K_SHARED = keyword("shared")
_K_LOCAL = keyword("local")
_K_LOCK = keyword("lock")
_K_LOOP_BINDINGS = keyword("loop_bindings")
_K_TAIL_Q = keyword("tail?")
_K_AS = keyword("as")

_BINDING_META_LOCAL = hash_map(_K_LOCAL, True)

#************************************************************
# Reader
#************************************************************

Token = define_record("clx.bootstrap.Token",
    keyword("string"),
    keyword("line"),
    keyword("column")
)

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
        yield Token(token, line_id, column)

def read_form(tokens):
    token = first(tokens)
    try:
        assert isinstance(token, Token), "expected a token"
        rtokens = rest(tokens)
        tstring = token.string
        if tstring == "'":
            form, _rest = read_form(rtokens)
            return list_(_S_QUOTE, form), _rest
        elif tstring == "`":
            form, _rest = read_form(rtokens)
            return quasiquote(_gen_name("_"), form), _rest
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
        elif tstring == "@":
            form, _rest = read_form(rtokens)
            return list_(_S_DEREF, form), _rest
        elif tstring[0] == "#":
            return read_dispatch(tstring, rtokens)
        elif tstring == "(":
            l, rtokens = read_collection(token, rtokens, list_, ")")
            l0 = l.first()
            if type(l0) is Symbol and l0.name[0] == "." and l0.name != ".":
                member = symbol(l0.name[1:])
                assert l.count_() > 1, "expected (.member target ...)"
                target = l.rest().first()
                l = list_(_S_DOT, target, member, *l.rest().rest())
            if l.count_() > 0:
                l = l.with_meta(
                    hash_map(_K_LINE, token.line, _K_COLUMN, token.column))
            return l, rtokens
        elif tstring == "[":
            return read_collection(token, rtokens, vector, "]")
        elif tstring == "{":
            return read_collection(token, rtokens, hash_map, "}")
        else:
            return read_atom(tstring), rtokens
    except ReaderError:
        raise
    except Exception as ex:
        raise ReaderError(str(ex), token.line, token.column) from ex

def read_dispatch(t0, tokens):
    if t0 == "#":
        t1 = first(tokens).string
        if t1[0] == "\"":
            assert t1[-1] == "\"", "Unterminated regex"
            return re.compile(t1[1:-1]), rest(tokens)
        else:
            form, _rest = read_form(tokens)
            if type(form) is PersistentList:
                return _function_shorthand(form), _rest

    raise Exception("Unsupported reader macro")

def _function_shorthand(form):
    args = {}

    def genarg(x):
        return gensym(f"___arg_{x}_")

    def transform(form):
        if type(form) is Symbol:
            if match := re.fullmatch(r"%(&|\d*)", form.name):
                arg = match[1] if match[1] else "1"
                if arg not in args:
                    args[arg] = genarg(arg)
                return args[arg]
            else:
                return form
        else:
            return form
    body = postwalk(transform, form)

    varg = args.pop("&", None)
    argn = max(map(int, args.keys()), default=0)

    def make_arg(i):
        _i = str(i)
        return args[_i] if _i in args else genarg(_i)

    argv = [make_arg(i) for i in range(1, argn + 1)]
    if varg:
        argv.append(_S_AMPER)
        argv.append(varg)

    return list_(_S_FN_STAR, vec(argv), body)

def read_atom(token):
    assert isinstance(token, str), "expected a string"
    if re.match(INT_RE, token):
        return int(token)
    elif re.match(FLOAT_RE, token):
        return float(token)
    elif re.match(STRING_RE, token):
        return unescape(token[1:-1])
    elif token[0] == "\"":
        raise Exception("Unterminated string")
    elif token == "true":
        return True
    elif token == "false":
        return False
    elif token == "nil":
        return None
    elif token[0] == ":":
        return keyword(token[1:])
    else:
        return symbol(token)

def unescape(text):
    return text \
        .replace(r"\\", "\\") \
        .replace(r"\"", "\"") \
        .replace(r"\n", "\n")

def read_collection(
        token0,
        tokens,
        ctor,
        end):
    assert isinstance(token0, Token), "expected a token"
    elements = []
    while True:
        token = first(tokens)
        if token is None:
            raise Exception(f"Expected '{end}'")
        assert isinstance(token, Token), "expected a token"
        if token.string == end:
            return ctor(*elements), rest(tokens)
        element, tokens = read_form(tokens)
        elements.append(element)

def quasiquote(suffix, form):
    if is_list(form) and len(form) > 0:
        head = form.first()
        if head == _S_UNQUOTE:
            assert len(form) == 2, "unquote expects 1 argument"
            return second(form)
        elif head == _S_SPLICE_UNQUOTE:
            raise Exception("splice-unquote not in list")
        else:
            return list_(_S_APPLY, _S_LIST, _quasiquote_sequence(suffix, form))
    elif is_vector(form):
        return list_(_S_VEC, _quasiquote_sequence(suffix, form))
    elif is_map(form):
        elements = []
        for k, v in form.items():
            elements.append(k)
            elements.append(v)
        return list_(_S_APPLY, _S_HASH_MAP,
            _quasiquote_sequence(suffix, elements))
    elif is_symbol(form):
        if is_simple_symbol(form) and form.name[-1] == "#":
            form = symbol(None, form.name[:-1] + suffix)
        return list_(_S_QUOTE, form)
    else:
        return form

def _quasiquote_sequence(suffix, form):
    def entry(_f):
        if is_list(_f) and len(_f) > 0 and _f.first() == _S_UNQUOTE:
            assert len(_f) == 2, "unquote expects 1 argument"
            return list_(_S_LIST, second(_f))
        elif is_list(_f) and len(_f) > 0 and _f.first() == _S_SPLICE_UNQUOTE:
            assert len(_f) == 2, "splice-unquote expects 1 argument"
            return second(_f)
        else:
            return list_(_S_LIST, quasiquote(suffix, _f))
    return list_(_S_CONCAT, *map(entry, form))

class ReaderError(Exception):
    def __init__(self, message, line, column):
        super().__init__(message)
        self.line = line
        self.column = column

#************************************************************
# Printer
#************************************************************

def pr_str(obj, readably=False):
    if obj is None:
        return "nil"
    elif isinstance(obj, bool):
        return "true" if obj else "false"
    elif isinstance(obj, str):
        if readably:
            return "\"" + escape(obj) + "\""
        return obj
    elif type(obj) is Symbol:
        return f"{obj.namespace}/{obj.name}" \
            if obj.namespace else obj.name
    elif type(obj) is Keyword:
        return f":{obj.namespace}/{obj.name}" \
            if obj.namespace else f":{obj.name}"
    elif type(obj) is PersistentVector:
        return "[" + \
            " ".join(map(lambda x: pr_str(x, readably), obj)) + \
            "]"
    elif type(obj) is PersistentMap:
        kvs = functools.reduce(
            lambda acc, kv:
                acc.conj(pr_str(kv[1], readably))
                    .conj(pr_str(kv[0], readably)),
            obj.items(),
            list_())
        return "{" + " ".join(kvs) + "}"
    elif isinstance(obj, ISeq):
        return "(" + \
            " ".join(map(lambda x: pr_str(x, readably), obj)) + \
            ")"
    elif is_record(obj):
        def pr_field(k):
            v = obj.lookup(k, None)
            return pr_str(v, True)
        return "#" + obj.__module__ + "." + obj.__class__.__name__ + "{" + \
            ", ".join([f"{k} {pr_field(k)}" for k in obj.keys()]) + \
            "}"
    else:
        return repr(obj) if readably else str(obj)

def escape(text):
    return text \
        .replace("\\", r"\\") \
        .replace("\"", r"\"") \
        .replace("\n", r"\n")

#************************************************************
# Evaluation
#************************************************************

Context = define_record("clx.bootstrap.Context",
    keyword("namespaces"),
    keyword("py_globals")
)

# Local context defines state that is local to a single form
LocalContext = define_record("clx.bootstrap.LocalContext",
    _K_ENV,
    _K_LOOP_BINDINGS,
    _K_TAIL_Q,
    _K_TOP_LEVEL_Q,
    _K_LINE,
    _K_COLUMN,
)

Binding = define_record("clx.bootstrap.Binding",
    keyword("py_name"),
    keyword("meta")
)

def init_context(namespaces):
    namespaces = {
        "user": {},
        "clx.bootstrap": {
            "define-record": define_record,
        },
        "clx.core": {
            "IAssociative": IAssociative,
            "+": lambda *args: sum(args),
            "with-meta": with_meta,
            "meta": meta,
            "apply": apply,
            "keyword": keyword,
            "keyword?": is_keyword,
            "simple-keyword?": is_simple_keyword,
            "symbol": symbol,
            "symbol?": is_symbol,
            "simple-symbol?": is_simple_symbol,
            "list": list_,
            "list?": is_list,
            "vector": vector,
            "vector?": is_vector,
            "vec": vec,
            "hash-map": hash_map,
            "map?": is_map,
            "lazy-seq*": lazy_seq,
            "seq?": is_seq,
            "re-pattern": lambda pattern: re.compile(pattern, 0),
            "re-pattern?": lambda x: isinstance(x, re.Pattern),
            "atom": atom,
            "atom?": is_atom,
            "deref": lambda x: x.deref(),
            "record?": is_record,
            "first": first,
            "second": second,
            "rest": rest,
            "next": next_,
            "seq": seq,
            "concat*": _concat,
            "concat": concat,
            "cons": cons,
            "count": count,
            "assoc": assoc,
            "get": get,
            "nth": nth,
            "conj": conj,
            "take": take,
            "take-last": take_last,
            "drop": drop,
            "drop-last": drop_last,
            "last": last,
            "map": map_,
            "filter": filter_,
            "reduce": reduce,
            "pr-str": pr_str,
            "gensym": gensym,
        },
        **namespaces,
    }

    ctx = Context(atom(hash_map()), {})

    for ns_name, ns_bindings in namespaces.items():
        for name, value in ns_bindings.items():
            _intern(ctx, ns_name, name, value)

    weak_ctx = weakref.ref(ctx)

    # Don't make circular reference to context
    _intern(ctx, "clx.core", "*ns*", None, dynamic=True)
    _current_ns(ctx).push("user")
    _intern(ctx, "clx.core", "*file*", None, dynamic=True)
    _current_file(ctx).push("NO_SOURCE_PATH")
    _intern(ctx, "clx.core", "in-ns", lambda ns: _in_ns(weak_ctx(), ns))
    _intern(ctx, "clx.core", "eval",
        lambda form: _eval_form(weak_ctx(), _local_context(), form))
    _intern(ctx, "clx.core", "load-file",
        lambda path: load_file(weak_ctx(), path))
    _intern(ctx, "clx.core", "refer*",
        lambda from_ns, sym, _as=None:
            _refer(weak_ctx(), from_ns, sym, _as))
    _intern(ctx, "clx.core", "alias",
        lambda _as, ns: _alias(weak_ctx(), _as, ns))

    return ctx

def _intern(ctx, ns, name, value, dynamic=False):
    py_name = munge(f"{ns}/{name}")
    binding = Binding(py_name, None) \
        if not dynamic else Binding(py_name, hash_map(_K_DYNAMIC, True))
    ctx.namespaces.swap(assoc_in, list_(ns, _K_BINDINGS, name), binding)
    ctx.py_globals[py_name] = value if not dynamic else DynamicVar(value)

def _local_context( # pylint: disable=too-many-arguments
        env=hash_map(),
        loop_bindings=None,
        tail=False,
        top_level=True,
        line=1,
        column=1):
    return LocalContext(env, loop_bindings, tail, top_level, line, column)

def _eval_string(ctx, text):
    lctx = _local_context()
    tokens = seq(tokenize(text))
    result = None
    while seq(tokens):
        try:
            form, tokens = read_form(tokens)
        except ReaderError as ex:
            raise SyntaxError(str(ex), (
                _current_file(ctx).deref(),
                ex.line,
                ex.column,
                text.splitlines()[ex.line - 1],
            )) from ex
        result = _eval_form(ctx, lctx, form)
    return result

def _eval_form(ctx, lctx, form):
    result, body = _compile(ctx, lctx, form)
    body = ast.Module(body, type_ignores=[])
    result = ast.Expression(result, type_ignores=[])
    _transform_ast(ctx, body, result)
    file_name = _current_file(ctx).deref()
    exec(compile(body, file_name, "exec"), ctx.py_globals) # pylint: disable=exec-used
    return eval(compile(result, file_name, "eval"), ctx.py_globals) # pylint: disable=eval-used

def load_file(ctx, path):
    return _with_bindings1(ctx,
        lambda: _eval_string(ctx, slurp(path)),
        "*file*", path,
        "*ns*", _current_ns(ctx).deref())

def _with_bindings1(ctx, body, *bindings):
    _bindings = []
    for name, value in zip(bindings[::2], bindings[1::2]):
        binding = _resolve_symbol(ctx, None, symbol(name), None)
        assert binding is not None, f"Can't resolve symbol '{name}'"
        assert get(binding.meta, _K_DYNAMIC, False), \
            f"Can't re-bind non-dynamic symbol '{name}'"
        obj = _binding_object(ctx, binding)
        _bindings.append(obj)
        _bindings.append(value)
    return _with_bindings(body, *_bindings)

def _with_bindings(body, *bindings):
    for obj, value in zip(bindings[::2], bindings[1::2]):
        assert type(obj) is DynamicVar, "expected a dynamic var"
        obj.push(value)
    try:
        return body()
    finally:
        for obj, _ in zip(bindings[::2], bindings[1::2]):
            obj.pop()

def macroexpand(ctx, form):
    while True:
        _form = macroexpand1(ctx, form)
        if _form is form:
            return form
        form = _form

def macroexpand1(ctx, form):
    if is_list(form): # pylint: disable=too-many-nested-blocks
        head = form.first()
        if is_symbol(head):
            binding = _resolve_symbol(ctx, None, head, None)
            if binding is not None:
                if get(binding.meta, _K_MACRO_QMARK, False):
                    obj = _binding_object(ctx, binding)
                    return obj(*form.rest())
                else:
                    if get(binding, _K_MACRO_QMARK):
                        raise Exception(
                            f"macro '{head}' is defined, "
                            "but hasn't been evaluated yet")
    return form

def _compile(ctx, lctx, form):
    try:
        lctx = _update_source_location(lctx, form)
        form = macroexpand(ctx, form)
        if isinstance(form, PersistentList):
            head = form.first()
            compiler = _SPECIAL_FORM_COMPILERS[head] \
                if is_symbol(head) and head in _SPECIAL_FORM_COMPILERS \
                else _compile_call
            return compiler(ctx, lctx, form)
        elif isinstance(form, PersistentVector):
            return _compile_vector(ctx, lctx, form)
        elif isinstance(form, PersistentMap):
            return _compile_map(ctx, lctx, form)
        elif isinstance(form, Symbol):
            binding = _resolve_symbol(ctx, lctx, form)
            if get(binding.meta, _K_MACRO_QMARK, False):
                raise Exception(f"Can't take value of a macro: {form}")
            return _binding_node(lctx, ast.Load(), binding), []
        else:
            return \
                ast.Constant(
                    form,
                    lineno=lctx.line,
                    col_offset=lctx.column), \
                []
    except SyntaxError:
        raise
    except Exception as ex:
        fname = _current_file(ctx).deref()
        ftext = slurp(fname) if fname != "NO_SOURCE_PATH" else ""
        line = ftext.splitlines()[lctx.line - 1] if ftext else ""
        raise SyntaxError(str(ex),
            (fname, lctx.line, lctx.column, line,)) from ex

def _update_source_location(lctx, form):
    _meta = meta(form)
    line = get(_meta, _K_LINE)
    if line is not None:
        column = get(_meta, _K_COLUMN)
        return assoc(lctx, _K_LINE, line, _K_COLUMN, column)
    return lctx

def _compile_def(ctx, lctx, form):
    assert lctx.top_level_QMARK_, "def allowed only at top level"
    assert len(form) == 3, "def expects 2 arguments"
    name = second(form)
    assert is_simple_symbol(name), \
        "def expects a simple symbol as the first argument"
    current_ns = _current_ns(ctx).deref()
    py_name = munge(f"{current_ns}/{name.name}")
    # TODO disallow redefinition of bound dynamic vars
    ctx.namespaces.swap(assoc_in,
        list_(current_ns, _K_BINDINGS, name.name),
        Binding(py_name, meta(name)))
    value_expr, value_stmts = _compile(
        ctx, lctx.assoc(_K_TAIL_Q, False), third(form))
    return \
        _node(ast.Name, lctx, py_name, ast.Load()), \
        value_stmts + [
            _node(ast.Assign, lctx,
                [_node(ast.Name, lctx, py_name, ast.Store())],
                value_expr)
        ]

def _compile_do(ctx, lctx, form):
    forms = form.rest()
    stmts = []
    _lctx = lctx.assoc(_K_TAIL_Q, False)
    while forms:
        _f, forms = forms.first(), forms.rest()
        if forms:
            # Not last form
            _result, _stmts = _compile(ctx, _lctx, _f)
            stmts.extend(_stmts)
            stmts.append(_node(ast.Expr, lctx, _result))
        else:
            # Last form
            _result, _stmts = _compile(ctx, lctx, _f)
            stmts.extend(_stmts)
            return _result, stmts
    return _node(ast.Constant, lctx, None), []

def _compile_let(ctx, lctx, form):
    assert len(form) > 1, "let expects at least 1 argument"
    bindings = second(form)
    assert is_vector(bindings), \
        "let expects a vector as the first argument"
    assert len(bindings) % 2 == 0, \
        "bindings of let must have even number of elements"
    bindings = destructure(bindings)
    body = []
    for i in range(0, len(bindings), 2):
        _name, value = bindings[i], bindings[i + 1]
        assert is_simple_symbol(_name), \
            "first element of each binding pair must be a symbol"
        value_expr, value_stmts = _compile(
            ctx, lctx.assoc(_K_TAIL_Q, False), value)
        munged = munge(_name.name)
        py_name = _gen_name(f"{munged}_")
        body.extend(value_stmts)
        body.append(
            _node(ast.Assign, lctx,
                [_node(ast.Name, lctx, py_name, ast.Store())], value_expr))
        lctx = assoc_in(lctx, list_(_K_ENV, _name.name),
            Binding(py_name, _BINDING_META_LOCAL))
    body_expr, body_stmts = _compile(ctx, lctx,
        _wrap_with_do(form.rest().rest()))
    body.extend(body_stmts)
    return body_expr, body

def _compile_cond(ctx, lctx, form):
    lctx = assoc(lctx, _K_TOP_LEVEL_Q, False)
    result = _gen_name("___cond_")
    result_store = _node(ast.Name, lctx, result, ast.Store())
    result_load = _node(ast.Name, lctx, result, ast.Load())
    def compile_clauses(clauses):
        if clauses is None:
            return [
                _node(ast.Assign, lctx, [result_store],
                    _node(ast.Constant, lctx, None))
            ]
        else:
            clauses_next = clauses.next()
            assert clauses_next is not None, \
                "cond expects an even number of forms"
            # TODO detect true or keyword
            test_expr, test_stmts = _compile(
                ctx, lctx.assoc(_K_TAIL_Q, False), clauses.first())
            then_expr, then_stmts = _compile(ctx, lctx, clauses_next.first())
            return [
                *test_stmts,
                _node(ast.If, lctx,
                    test_expr, [
                        *then_stmts,
                        _node(ast.Assign, lctx, [result_store], then_expr),
                    ],
                    compile_clauses(clauses_next.next()),
                ),
            ]
    return result_load, compile_clauses(form.next())

def _compile_fn(ctx, lctx, form):
    arg1 = second(form)
    if is_symbol(arg1):
        fname = arg1
        params = third(form)
        body = fourth(form)
    else:
        fname = symbol(None, _gen_name("___fn_"))
        params = arg1
        body = third(form)

    fexpr, fstmt = _make_function_def(ctx, lctx, fname, params, body)
    return fexpr, [fstmt]

def _make_function_def(ctx, lctx, fname, params, body):
    assert is_symbol(fname), "function name must be a symbol"
    assert isinstance(params, PersistentVector), \
        "function parameters must be a vector"

    lctx = lctx.assoc(_K_LOOP_BINDINGS, None)

    pos_params = []
    rest_param = None
    env = lctx.env
    to_destructure = []
    while params:
        param = first(params)
        if is_symbol(param):
            assert is_simple_symbol(param), \
                "parameters of function must be simple symbols"
        else:
            to_destructure.append(param)
            param = gensym()
            to_destructure.append(param)
        if param == _S_AMPER:
            assert next_(next_(params)) is None, \
                "parameters should have a single entry after &"
            rest_param = second(params)
            if is_symbol(rest_param):
                assert is_simple_symbol(rest_param), \
                    "rest parameters of function must be a simple symbol"
            else:
                to_destructure.append(rest_param)
                rest_param = gensym()
                to_destructure.append(rest_param)
            env = assoc(env, rest_param.name,
                Binding(munge(rest_param.name), _BINDING_META_LOCAL))
            break
        pos_params.append(param)
        env = assoc(env, param.name,
            Binding(munge(param.name), _BINDING_META_LOCAL))
        params = rest(params)
    lctx = assoc(lctx, _K_ENV, env)

    if body is not None:
        if to_destructure:
            body = list_(_S_LET, vec(to_destructure), body)
        body_expr, body_stmts = _compile(
            ctx, assoc(lctx, _K_TOP_LEVEL_Q, False, _K_TAIL_Q, True), body)
        body_stmts.append(_node(ast.Return, lctx, body_expr))
    else:
        body_stmts = [_node(ast.Pass, lctx)]

    def arg(p):
        return _node(ast.arg, lctx, munge(p.name))

    py_name = munge(fname)
    return \
        _node(ast.Name, lctx, py_name, ast.Load()), \
        _node(ast.FunctionDef, lctx,
            py_name,
            ast.arguments(
                posonlyargs=[arg(p) for p in pos_params],
                args=[],
                vararg=arg(rest_param) if rest_param else None,
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body_stmts,
            [])

def _compile_letfn(ctx, lctx, form):
    fdefs = second(form)
    assert is_vector(fdefs), \
        "letfn expects a vector of function definitions"

    # First, populate local context with function names
    for fdef in fdefs:
        assert is_list(fdef), "letfn function definition must be a list"
        fname = first(fdef)
        assert is_simple_symbol(fname), \
            "function name must be a simple symbol"
        py_name = munge(fname.name)
        lctx = assoc_in(lctx, list_(_K_ENV, fname.name),
            Binding(py_name, _BINDING_META_LOCAL))

    stmts = []
    for fdef in fdefs:
        fname = first(fdef)
        params = second(fdef)
        assert is_vector(params), "function parameters must be a vector"
        fbody = _wrap_with_do(fdef.rest().rest())
        _lctx = lctx.assoc(_K_TOP_LEVEL_Q, False, _K_TAIL_Q, False)
        _fexpr, fstmt = _make_function_def(ctx, _lctx, fname, params, fbody)
        stmts.append(fstmt)

    body = _wrap_with_do(form.rest().rest())
    body_expr, body_stmts = _compile(ctx, lctx, body)
    stmts.extend(body_stmts)
    return body_expr, stmts

def _compile_try(ctx, lctx, form):
    lctx = assoc(lctx, _K_TOP_LEVEL_Q, False, _K_TAIL_Q, False)

    is_catch = lambda f: is_list(f) and f.first() == _S_CATCH
    is_finally = lambda f: is_list(f) and f.first() == _S_FINALLY
    is_prologue = lambda f: is_catch(f) or is_finally(f)
    id_body = lambda f: not is_prologue(f)

    body = itertools.takewhile(id_body, form.rest())
    body_expr, body_stmts = _compile(ctx, lctx, _wrap_with_do(body))
    prologue = list(itertools.dropwhile(id_body, form.rest()))

    if prologue:
        result = _gen_name("___try_")
        result_store = _node(ast.Name, lctx, result, ast.Store())
        result_load = _node(ast.Name, lctx, result, ast.Load())

        if is_finally(prologue[-1]):
            catch_clauses = prologue[:-1]
            finally_clause = prologue[-1]
        else:
            catch_clauses = prologue
            finally_clause = None

        try_body = [
            *body_stmts,
            _node(ast.Assign, lctx, [result_store], body_expr),
        ]

        handlers = []
        for clause in catch_clauses:
            if is_finally(clause):
                raise Exception("finally clause must be the last "
                    "in try expression")
            if not is_catch(clause):
                raise Exception("only catch and finally clauses can "
                    "follow catch in try expression")

            clazz = _resolve_symbol(ctx, lctx, second(clause))
            var = third(clause)
            assert is_simple_symbol(var), \
                "catch variable must be a simple symbol"

            py_var = munge(var.name)
            _lctx = assoc_in(lctx, list_(_K_ENV, var.name),
                Binding(py_var, _BINDING_META_LOCAL))
            catch_expr, catch_stmts = \
                _compile(ctx, _lctx,
                    _wrap_with_do(clause.rest().rest().rest()))

            handler = _node(ast.ExceptHandler, lctx,
                _binding_node(lctx, ast.Load(), clazz),
                py_var, [
                    *catch_stmts,
                    _node(ast.Assign, lctx, [result_store], catch_expr),
                ])
            handlers.append(handler)

        finalbody = []
        if finally_clause is not None:
            finally_expr, finally_stmts = \
                _compile(ctx, lctx, _wrap_with_do(finally_clause.rest()))
            finalbody.extend([
                *finally_stmts,
                _node(ast.Expr, lctx, finally_expr),
            ])

        return result_load, \
            [_node(ast.Try, lctx, try_body, handlers, [], finalbody)]
    else:
        return body_expr, body_stmts

def _compile_loop(ctx, lctx, form):
    assert len(form) > 1, "loop expects at least 1 argument"
    bindings = second(form)
    assert is_vector(bindings), \
        "loop expects a vector as the first argument"
    assert len(bindings) % 2 == 0, \
        "bindings of loop must have even number of elements"
    lctx = lctx.assoc(_K_TOP_LEVEL_Q, False, _K_TAIL_Q, False)
    stmts = []
    loop_bindings = []
    for i in range(0, len(bindings), 2):
        _name, value = bindings[i], bindings[i + 1]
        assert is_simple_symbol(_name), \
            "binding name must be a simple symbol"
        value_expr, value_stmts = _compile(ctx, lctx, value)
        munged = munge(_name.name)
        py_name = _gen_name(f"{munged}_")
        stmts.extend(value_stmts)
        stmts.append(
            _node(ast.Assign, lctx,
                [_node(ast.Name, lctx, py_name, ast.Store())], value_expr))
        binding = Binding(py_name, _BINDING_META_LOCAL)
        loop_bindings.append(binding)
        lctx = assoc_in(lctx, list_(_K_ENV, _name.name), binding)
    lctx = assoc(lctx, _K_LOOP_BINDINGS, loop_bindings)
    if len(form) > 2:
        result_name = _gen_name("___loop_")
        body_expr, body_stmts = _compile(
            ctx, lctx.assoc(_K_TAIL_Q, True),
            _wrap_with_do(form.rest().rest()))
        stmts.append(
            _node(ast.While, lctx, _node(ast.Constant, lctx, True), [
                *body_stmts,
                _node(ast.Assign, lctx,
                    [_node(ast.Name, lctx, result_name, ast.Store())],
                    body_expr),
                _node(ast.Break, lctx),
            ], []))
        expr = _node(ast.Name, lctx, result_name, ast.Load())
    else:
        expr = _node(ast.Constant, lctx, None)
    return expr, stmts

def _compile_recur(ctx, lctx, form):
    assert lctx.loop_bindings is not None, "recur allowed only in loop"
    assert len(form) == len(lctx.loop_bindings) + 1, \
        f"recur expects {len(lctx.loop_bindings)} arguments"
    assert lctx.tail_QMARK_, "recur allowed only in tail position"
    temps = []
    stmts = []
    for binding, value in zip(lctx.loop_bindings, form.rest()):
        value_expr, value_stmts = _compile(
            ctx, lctx.assoc(_K_LOOP_BINDINGS, None), value)
        stmts.extend(value_stmts)
        temp = _gen_name()
        temps.append(temp)
        stmts.append(
            _node(ast.Assign, lctx,
                [_node(ast.Name, lctx, temp, ast.Store())], value_expr))
    for binding, temp in zip(lctx.loop_bindings, temps):
        stmts.append(
            _node(ast.Assign, lctx,
                [_binding_node(lctx, ast.Store(), binding)],
                _node(ast.Name, lctx, temp, ast.Load())))
    stmts.append(_node(ast.Continue, lctx))
    return _node(ast.Constant, lctx, None), stmts

def _compile_quote(_ctx, lctx, form):
    assert len(form) == 2, "quote expects exactly 1 argument"
    return _node(ast.Constant, lctx, second(form)), []

def _compile_dot(ctx, lctx, form):
    assert len(form) >= 3, "dot expects at least 2 arguments"
    obj = second(form)
    attribute = third(form)
    assert is_simple_symbol(attribute), \
        "attribute name must be a simple symbol"
    lctx = assoc(lctx, _K_TAIL_Q, False)
    obj_expr, obj_stmts = _compile(ctx, lctx, obj)
    if attribute.name[0] == "-":
        assert len(form) == 3, "dot expects exactly 2 arguments"
        return \
            _node(ast.Attribute, lctx,
                obj_expr, attribute.name[1:], ast.Load()), \
            obj_stmts
    else:
        stmts = obj_stmts
        args = []
        for arg in form.rest().rest().rest():
            arg_expr, arg_body = _compile(ctx, lctx, arg)
            args.append(arg_expr)
            stmts.extend(arg_body)
        f = _node(ast.Attribute, lctx, obj_expr, attribute.name, ast.Load())
        return _node(ast.Call, lctx, f, args, []), stmts

def _compile_import(ctx, lctx, form):
    assert lctx.top_level_QMARK_, "import* allowed only at top level"
    assert len(form) <= 3, "import* expects at most 2 arguments"
    mod = second(form)
    assert is_simple_symbol(mod), \
        "name of module to import must be a simple symbol"
    _as = third(form)
    if _as is None:
        _as = mod
    assert is_simple_symbol(_as), "alias for module must be a simple symbol"
    alias_name = _gen_name(f"___import_{munge(mod.name)}_")
    current_ns = _current_ns(ctx).deref()
    def _update(namespaces):
        return assoc_in(namespaces,
            list_(current_ns, _K_IMPORTS, _as.name),
            alias_name)
    ctx.namespaces.swap(_update)
    return \
        _node(ast.Name, lctx, alias_name, ast.Load()), \
        [_node(ast.Import, lctx,
            [_node(ast.alias, lctx, mod.name, alias_name)])]

def _compile_python(ctx, lctx, form):
    def _eval_entry(entry):
        if isinstance(entry, Symbol):
            return _binding_string(_resolve_symbol(ctx, lctx, entry))
        elif isinstance(entry, str):
            return entry
        else:
            raise Exception("python* expects strings or symbols")
    source = "".join(map(_eval_entry, form.rest()))
    module = ast.parse(source)
    if module.body and isinstance(module.body[-1], ast.Expr):
        stmts = module.body[:-1]
        result = module.body[-1].value
    else:
        stmts = module.body
        result = _node(ast.Constant, lctx, None)
    return result, stmts

def _compile_python_with(ctx, lctx, form):
    assert len(form) >= 2, "python/with expects at least 1 argument"
    bindings = second(form)
    assert is_vector(bindings), "bindings of python/with must be a vector"
    assert len(bindings) % 2 == 0, \
        "bindings of python/with must have even number of elements"
    body = form.rest().rest()

    lctx1 = assoc(lctx, _K_TOP_LEVEL_Q, False, _K_TAIL_Q, False)
    lctx2 = lctx1

    stmts = []
    items = []
    for i in range(0, len(bindings), 2):
        _name, cm = bindings[i], bindings[i + 1]
        assert is_simple_symbol(_name), \
            "binding name must be a simple symbol"

        py_name = munge(_name.name)
        lctx2 = assoc_in(lctx2, list_(_K_ENV, _name.name),
            Binding(py_name, _BINDING_META_LOCAL))

        cm_expr, cm_stmts = _compile(ctx, lctx1, cm)
        stmts.extend(cm_stmts)

        items.append(
            _node(ast.withitem, lctx,
                cm_expr, _node(ast.Name, lctx, py_name, ast.Store())))

    body_expr, body_stmts = _compile(ctx, lctx2, _wrap_with_do(body))

    result = _gen_name("___with_")
    result_assign = lambda expr: \
        _node(ast.Assign, lctx,
            [_node(ast.Name, lctx, result, ast.Store())], expr)

    return \
        _node(ast.Name, lctx, result, ast.Load()), \
        [
            *stmts,
            result_assign(_node(ast.Constant, lctx, None)),
            _node(ast.With, lctx, items, [
                *body_stmts,
                result_assign(body_expr),
            ])
        ]

def _trace_local_context(_ctx, lctx, form):
    path = form.rest()
    return _node(ast.Constant, lctx, get_in(lctx, path)), []

_SPECIAL_FORM_COMPILERS = {
    _S_DEF: _compile_def,
    _S_DO: _compile_do,
    _S_LET: _compile_let,
    _S_COND: _compile_cond,
    _S_FN_STAR: _compile_fn,
    _S_LETFN: _compile_letfn,
    _S_TRY: _compile_try,
    _S_LOOP: _compile_loop,
    _S_RECUR: _compile_recur,
    _S_QUOTE: _compile_quote,
    _S_DOT: _compile_dot,
    _S_IMPORT_STAR: _compile_import,
    _S_PYTHON: _compile_python,
    _S_PYTHON_WITH: _compile_python_with,
    _S_LOCAL_CONTEXT: _trace_local_context,
}

def _compile_call(ctx, lctx, form):
    lctx = assoc(lctx, _K_TAIL_Q, False)
    args = []
    f, stmts = _compile(ctx, lctx, form.first())
    for arg in form.rest():
        arg_expr, arg_body = _compile(ctx, lctx, arg)
        args.append(arg_expr)
        stmts.extend(arg_body)
    return _node(ast.Call, lctx, f, args, []), stmts

def _compile_vector(ctx, lctx, form):
    lctx = assoc(lctx, _K_TAIL_Q, False)
    el_exprs = []
    stmts = []
    for elm in form:
        el_expr, el_stmts = _compile(ctx, lctx, elm)
        el_exprs.append(el_expr)
        stmts.extend(el_stmts)
    return \
        _node(ast.Call, lctx,
            _binding_node(lctx, ast.Load(), _core_binding(ctx, _S_VECTOR)),
            el_exprs,
            []), \
        stmts

def _compile_map(ctx, lctx, form):
    lctx = assoc(lctx, _K_TAIL_Q, False)
    args = []
    stmts = []
    for key, value in form.items():
        key_expr, key_stmts = _compile(ctx, lctx, key)
        args.append(key_expr)
        stmts.extend(key_stmts)
        value_expr, value_stmts = _compile(ctx, lctx, value)
        args.append(value_expr)
        stmts.extend(value_stmts)
    return \
        _node(ast.Call, lctx,
            _binding_node(lctx, ast.Load(), _core_binding(ctx, _S_HASH_MAP)),
            args,
            []), \
        stmts

def _resolve_symbol(ctx, lctx, sym, not_found=_DUMMY):
    assert is_symbol(sym), "expected a symbol"
    if is_simple_symbol(sym):
        if lctx is not None:
            binding = lctx.env.lookup(sym.name, None)
            if binding is not None:
                return binding

        current_ns = _current_ns(ctx).deref()
        assert type(current_ns) is str
        namespaces = ctx.namespaces.deref()
        binding = get_in(namespaces, list_(current_ns, _K_BINDINGS, sym.name))
        if binding is not None:
            return binding

        binding = get_in(namespaces, list_("clx.core", _K_BINDINGS, sym.name))
        if binding is not None:
            return binding
    else:
        namespaces = ctx.namespaces.deref()
        current_ns = _current_ns(ctx).deref()
        assert type(current_ns) is str

        _import = get_in(namespaces,
            list_(current_ns, _K_IMPORTS, sym.namespace))
        if _import is not None:
            return Binding(
                munge(sym.name),
                hash_map(_K_IMPORTED_FROM, _import))

        aliased_ns = get_in(namespaces,
            list_(current_ns, _K_ALIASES, sym.namespace))
        if aliased_ns is not None:
            binding = get_in(namespaces,
                list_(aliased_ns, _K_BINDINGS, sym.name))
            if binding is not None:
                return binding

        binding = get_in(namespaces,
            list_(sym.namespace, _K_BINDINGS, sym.name))
        if binding is not None:
            return binding

    if not_found is _DUMMY:
        raise Exception(f"Symbol '{pr_str(sym)}' not found")
    return not_found

def _resolve(ctx, _name):
    binding = _resolve_symbol(ctx, None, symbol(_name), None)
    return _binding_object(ctx, binding) if binding is not None else None

def _current_ns(ctx):
    return ctx.py_globals[munge("clx.core/*ns*")]

def _current_file(ctx):
    return ctx.py_globals[munge("clx.core/*file*")]

def _core_binding(ctx, sym):
    return get_in(ctx.namespaces.deref(),
        list_("clx.core", _K_BINDINGS, sym.name))

def _gen_name(prefix="___gen_"):
    if not hasattr(_gen_name, "counter"):
        _gen_name.counter = atom(10000)
    return f"{prefix}{_gen_name.counter.swap(lambda x: x + 1)}"

def _wrap_with_do(forms):
    forms = seq(forms)
    if forms is None:
        return None
    elif forms.next() is None:
        return forms.first()
    else:
        return list_(_S_DO, *forms)

def destructure(bindings):
    assert is_vector(bindings)
    assert len(bindings) % 2 == 0
    res = []
    for i in range(0, len(bindings), 2):
        bform, expr = bindings[i], bindings[i + 1]
        res += _destructure1(bform, expr)
    return vec(res)

def _destructure1(bform, expr):
    if is_simple_symbol(bform):
        return [bform, expr]
    elif is_vector(bform):
        return _destructure_sequence(bform, expr)
    elif is_map(bform):
        raise NotImplementedError()
    else:
        raise Exception("Invalid destructuring form")

def _destructure_sequence(bform, expr):
    bform_len = count(bform)
    if bform_len == 0:
        return [gensym(), expr]
    elif bform_len == 1:
        sfirst = gensym()
        return [sfirst, list_(_S_FIRST, expr)] + \
            _destructure1(first(bform), sfirst)
    elif first(take_last(2, bform)) == _K_AS:
        name = last(bform)
        assert is_simple_symbol(name), "Expected a simple symbol after :as"
        return [name, expr] + _destructure_sequence(drop_last(2, bform), name)
    elif first(bform) == _S_AMPER:
        assert bform_len == 2, "Expected exactly one entry after &"
        return _destructure1(second(bform), expr)
    else:
        sexpr = gensym()
        sfirst = gensym()
        srest = gensym()
        return [
            sexpr, expr, # Prevents evaluation of expr multiple times
            sfirst, list_(_S_FIRST, sexpr),
            srest, list_(_S_REST, sexpr),
        ] + _destructure1(first(bform), sfirst) + \
            _destructure_sequence(rest(bform), srest)

def _node(type_, lctx, *args):
    _n = type_(*args)
    _n.lineno = lctx.line
    _n.col_offset = lctx.column
    return _n

def _binding_node(lctx, load_or_store, binding):
    _meta = binding.meta
    if get(_meta, _K_LOCAL):
        return _node(ast.Name, lctx, binding.py_name, load_or_store)
    elif get(_meta, _K_IMPORTED_FROM):
        module = get(_meta, _K_IMPORTED_FROM)
        return \
            _node(ast.Attribute, lctx,
                _node(ast.Name, lctx, module, ast.Load()),
                binding.py_name,
                load_or_store)
    else:
        return _node(ast.Name, lctx, binding.py_name, load_or_store)

def _binding_object(ctx, binding, not_found=_DUMMY):
    _meta = binding.meta
    if get(_meta, _K_LOCAL): # pylint: disable=no-else-raise
        raise Exception("Can't take value of a local binding")
    elif get(_meta, _K_IMPORTED_FROM):
        module = get(_meta, _K_IMPORTED_FROM)
        return getattr(ctx.py_globals[module], binding.py_name, not_found)
    else:
        return ctx.py_globals.get(binding.py_name, not_found)

def _binding_string(binding):
    _meta = binding.meta
    if get(_meta, _K_LOCAL):
        return binding.py_name
    elif get(_meta, _K_IMPORTED_FROM):
        module = get(_meta, _K_IMPORTED_FROM)
        return f"{module}.{binding.py_name}"
    else:
        return binding.py_name

def _transform_ast(ctx, body, result):
    _fix_constants(ctx, body, result)

def _fix_constants(ctx, body, result):
    consts = {}

    lctx = _local_context()

    class Transformer(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, (
                    Keyword,
                    Symbol,
                    PersistentList,
                    PersistentVector,
                    PersistentMap,
                    re.Pattern)):
                name = _gen_name("___const_")
                consts[name] = _compile_constant(ctx, lctx, node.value)
                return ast.Name(name, ast.Load(), lineno=0, col_offset=0)
            else:
                self.generic_visit(node)
                return node

    Transformer().visit(body)
    Transformer().visit(result)

    body.body[:0] = [
        ast.Assign(
            [ast.Name(name, ast.Store(), lineno=0, col_offset=0)],
            value,
            lineno=0, col_offset=0)
        for name, value in consts.items()
    ]

def _compile_constant(ctx, lctx, value):
    if isinstance(value, (Keyword, Symbol)):
        _ns = ast.Constant(value.namespace, lineno=0, col_offset=0)
        _name = ast.Constant(value.name, lineno=0, col_offset=0)
        tree = ast.Call(
            _binding_node(lctx, ast.Load(),
                _core_binding(ctx,
                    _S_KEYWORD if type(value) is Keyword else _S_SYMBOL)),
            [_ns, _name], [], lineno=0, col_offset=0)
    elif isinstance(value, (PersistentList, PersistentVector)):
        tree = ast.Call(
            _binding_node(lctx, ast.Load(),
                _core_binding(ctx,
                    _S_LIST if type(value) is PersistentList else _S_VECTOR)),
            [_compile_constant(ctx, lctx, v) for v in value],
            [], lineno=0, col_offset=0)
    elif type(value) is PersistentMap:
        args = []
        for key, val in value.items():
            args.append(_compile_constant(ctx, lctx, key))
            args.append(_compile_constant(ctx, lctx, val))
        tree = ast.Call(
            _binding_node(lctx, ast.Load(),
                _core_binding(ctx, _S_HASH_MAP)),
            args, [], lineno=0, col_offset=0)
    elif type(value) is re.Pattern:
        tree = ast.Call(
            _binding_node(lctx, ast.Load(),
                _core_binding(ctx, _S_RE_PATTERN)),
            [ast.Constant(value.pattern, lineno=0, col_offset=0)],
            [], lineno=0, col_offset=0)
    else:
        tree = ast.Constant(value, lineno=0, col_offset=0)
    if meta(value) is not None:
        tree = ast.Call(
            _binding_node(lctx, ast.Load(),
                _core_binding(ctx, _S_WITH_META)),
            [tree, _compile_constant(ctx, lctx, meta(value))],
            [], lineno=0, col_offset=0)
    return tree

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
    if _meta is not None:
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

def vary_meta(obj, f, *args):
    return with_meta(obj, f(meta(obj), *args))

def vec(coll):
    if not coll:
        return vector()
    else:
        return vector(*coll)

def is_counted(x):
    return isinstance(x, ICounted)

def _count_seqable(x):
    num = 0
    while x is not None:
        num += 1
        x = x.next()
    return num

def is_seqable(obj):
    return obj is None or isinstance(obj, (ISeqable, Iterable))

def second(coll):
    return first(next_(coll))

def third(coll):
    return first(next_(next_(coll)))

def fourth(coll):
    return first(next_(next_(next_(coll))))

def assoc(obj, *kvs):
    assert len(kvs) % 2 == 0, "assoc expects even number of arguments"
    if obj is None:
        return hash_map(*kvs)
    return obj.assoc(*kvs)

def update(obj, key, f, *args):
    return assoc(obj, key, f(get(obj, key), *args))

def get_in(obj, path, not_found=None):
    if path is None:
        return obj
    for key in path:
        obj = get(obj, key, not_found)
    return obj

def assoc_in(obj, path, value):
    path0 = first(path)
    path = next_(path)
    if path is not None:
        child0 = get(obj, path0)
        return assoc(obj, path0, assoc_in(child0, path, value))
    else:
        return assoc(obj, path0, value)

def concat(*colls):
    return _concat(colls)

def _concat(colls):
    colls = seq(colls)
    if colls is None:
        return list_()
    elif colls.next() is None:
        return lazy_seq(colls.first)
    else:
        c0 = seq(colls.first())
        cs = colls.rest()
        def _seq():
            return cons(c0.first(), _concat(cons(c0.rest(), cs))) \
                if c0 is not None else _concat(cs)
        return lazy_seq(_seq)

def merge(*maps):
    def helper(m1, m2):
        if m1 is None:
            return m2
        elif m2 is None:
            return m1
        else:
            return m1.merge(m2)
    return functools.reduce(helper, maps, None)

def take_last(n, coll):
    size = count(coll)
    if n <= 0:
        return None
    elif n >= size:
        return seq(coll)
    else:
        return drop(size - n, coll)

def last(coll):
    return first(take_last(1, coll))

def drop_last(arg1, arg2=_DUMMY):
    if arg2 is _DUMMY:
        return drop_last(1, arg1)
    n = arg1
    coll = arg2
    if coll is None:
        return list_()
    elif n <= 0:
        return coll
    else:
        size = count(coll)
        return take(size - n, coll)

def _in_ns(ctx, ns):
    _current_ns(ctx).set(ns.name)

def _refer(ctx, from_ns, sym, _as):
    assert is_simple_symbol(from_ns), \
        "refer expects source namespace to be a simple symbol"
    assert is_simple_symbol(sym), \
        "refer expects referred symbol to be a simple symbol"
    _as = sym if _as is None else _as
    assert is_simple_symbol(_as), "refer expects alias to be a simple symbol"
    def _update(namespaces):
        cur_ns = _current_ns(ctx).deref()
        binding = get_in(namespaces,
            list_(from_ns.name, _K_BINDINGS, sym.name))
        assert binding is not None, \
            f"Symbol '{sym.name}' not found in namespace '{from_ns.name}'"
        return assoc_in(namespaces,
            list_(cur_ns, _K_BINDINGS, _as.name), binding)
    ctx.namespaces.swap(_update)

def _alias(ctx, _as, ns):
    assert is_simple_symbol(_as), "alias must be a simple symbol"
    assert is_simple_symbol(ns), \
        "alias expects namespace to be a simple symbol"
    def _update(namespaces):
        cur_ns = _current_ns(ctx).deref()
        return assoc_in(namespaces,
            list_(cur_ns, _K_ALIASES, _as.name), ns.name)
    ctx.namespaces.swap(_update)

def slurp(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

def gensym(prefix="___gen_"):
    return symbol(_gen_name(prefix))

def walk(inner, outer, form):
    if type(form) is PersistentList:
        return outer(with_meta(
            list_(*map(inner, form)),
            meta(form)))
    elif type(form) is PersistentVector:
        return outer(vec(map(inner, form)))
    elif type(form) is PersistentMap:
        return outer(hash_map_from(map(inner, form.items())))
    elif type(form) is tuple: # items of PersistentMap
        return outer(tuple(map(inner, form)))
    else:
        return outer(form)

def postwalk(f, form):
    return walk(lambda _form: postwalk(f, _form), f, form)
