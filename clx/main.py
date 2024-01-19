import ast
from bisect import bisect_left
from collections import namedtuple
import re
import sys
import threading

import pyrsistent as pr

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

class Symbol:
    def __init__(self, _namespace, _name, _meta):
        self.name = sys.intern(_name)
        self.namespace = sys.intern(_namespace) if _namespace else None
        self._hash = hash((self.namespace, self.name))
        self.__meta__ = _meta
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

Keyword = namedtuple("Keyword", ["namespace", "name", "munged"])

KEYWORD_TABLE = {}
KEYWORD_TABLE_LOCK = threading.Lock()

def keyword(arg1, arg2=None):
    if arg2 is None:
        if isinstance(arg1, str):
            if "/" in arg1:
                _ns, _name = arg1.split("/", 1)
                qname = arg1
            else:
                _ns = None
                _name = arg1
                qname = _name
        elif isinstance(arg1, Symbol):
            _ns = arg1.namespace
            _name = arg1.name
            qname = f"{_ns}/{_name}" if _ns else _name
        elif isinstance(arg1, Keyword):
            return arg1
        else:
            raise Exception(
                "keyword expects a string, a Keyword, or a Symbol as the "
                "first argument")
    else:
        _ns = arg1
        _name = arg2
        qname = f"{_ns}/{_name}"
    with KEYWORD_TABLE_LOCK:
        if qname in KEYWORD_TABLE:
            return KEYWORD_TABLE[qname]
        else:
            new_kw = Keyword(_ns, _name, munge(qname))
            KEYWORD_TABLE[qname] = new_kw
            return new_kw

def is_keyword(obj):
    return isinstance(obj, Keyword)

def is_simple_keyword(obj):
    return isinstance(obj, Keyword) and obj.namespace is None

class PersistentList:
    def __init__(self, impl, _meta):
        self._impl = impl
        self.__meta__ = _meta
    def __eq__(self, other):
        return isinstance(other, PersistentList) \
            and self._impl == other._impl
    def __hash__(self):
        return hash(self._impl)
    def __len__(self):
        return len(self._impl)
    def __getitem__(self, index):
        return self._impl[index]
    def with_meta(self, _meta):
        return PersistentList(self._impl, _meta)
    def cons(self, value):
        return PersistentList(self._impl.cons(value), _meta=None)

def list_(*elements):
    return PersistentList(pr.plist(elements), _meta=None)

def is_list(obj):
    return isinstance(obj, PersistentList)

class PersistentVector:
    def __init__(self, impl, _meta):
        assert isinstance(impl, pr.PVector), "Expected a PVector"
        self._impl = impl
        self.__meta__ = _meta
    def __eq__(self, other):
        return isinstance(other, PersistentVector) \
            and self._impl == other._impl
    def __len__(self):
        return len(self._impl)
    def __iter__(self):
        return iter(self._impl)
    def __getitem__(self, index):
        return self._impl[index]
    def with_meta(self, _meta):
        return PersistentVector(self._impl, _meta)
    def first(self):
        return self._impl[0]
    def rest(self):
        return PersistentVector(self._impl[1:], _meta=None)

def vec(coll):
    return PersistentVector(pr.pvector(coll), _meta=None)

def vector(*elements):
    return vec(elements)

def is_vector(obj):
    return isinstance(obj, PersistentVector)

class PersistentMap:
    def __init__(self, impl, _meta):
        assert isinstance(impl, pr.PMap), "Expected a PMap"
        self._impl = impl
        self.__meta__ = _meta
    def __eq__(self, other):
        return isinstance(other, PersistentMap) \
            and self._impl == other._impl
    def __len__(self):
        return len(self._impl)
    def __iter__(self):
        return iter(self._impl)
    def __getitem__(self, key):
        return self._impl[key]
    def items(self):
        return self._impl.items()
    def with_meta(self, _meta):
        return PersistentMap(self._impl, _meta)
    def get(self, key, default=None):
        return self._impl.get(key, default)
    def assoc(self, key, value):
        return PersistentMap(self._impl.set(key, value), _meta=None)

def hash_map(*elements):
    assert len(elements) % 2 == 0, "hash-map expects even number of elements"
    return PersistentMap(
        pr.pmap(dict(zip(elements[::2], elements[1::2]))),
        _meta=None)

def is_hash_map(obj):
    return isinstance(obj, PersistentMap)

class Record():
    def get(self, field):
        raise NotImplementedError()
    def assoc(self, field, value):
        raise NotImplementedError()

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
    _ns = {f"kw_{munge(f.name)}": keyword(f) for f in fields}
    exec( # pylint: disable=exec-used
        f"""
assoc_methods = {{
  {", ".join(assoc_methods)}
}}

class {name}:
  def __init__(self, {init_args}):
    {init_fields}
  def get(self, field):
    return getattr(self, field.munged)
  def assoc(self, field, value):
    return assoc_methods[field](self, value)
cls = {name}
        """,
        _ns)
    return _ns["cls"]

#************************************************************
# Constants
#************************************************************

_S_QUOTE = symbol("quote")
_S_UNQUOTE = symbol("unquote")
_S_SPLICE_UNQUOTE = symbol("splice-unquote")
_S_WITH_META = symbol("with-meta")
_S_VEC = symbol("vec")
_S_LIST = symbol("list")
_S_CONCAT = symbol("concat")
_K_LINE = keyword("line")
_K_COLUMN = keyword("column")

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
        line_id = bisect_left(line_starts, match.start(1) + 1)
        column = match.start(1) - line_starts[line_id - 1] + 1
        yield Token(string=token, line=line_id, column=column)

def read_form(tokens):
    token = tokens[0]
    tstring = token.string
    if tstring == "'":
        form, _rest = read_form(tokens[1:])
        return list_(_S_QUOTE, form), _rest
    elif tstring == "`":
        form, _rest = read_form(tokens[1:])
        return quasiquote(form), _rest
    elif tstring == "~":
        form, _rest = read_form(tokens[1:])
        return list_(_S_UNQUOTE, form), _rest
    elif tstring == "~@":
        form, _rest = read_form(tokens[1:])
        return list_(_S_SPLICE_UNQUOTE, form), _rest
    elif tstring == "^":
        _meta, rest1 = read_form(tokens[1:])
        form, rest2 = read_form(rest1)
        return list_(_S_WITH_META, form, _meta), rest2
    elif tstring == "(":
        return read_collection(tokens, list_, "(", ")")
    elif tstring == "[":
        return read_collection(tokens, vector, "[", "]")
    elif tstring == "{":
        return read_collection(tokens, hash_map, "{", "}")
    else:
        return read_atom(token), tokens[1:]

def read_atom(token):
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
    token0 = tokens[0]
    assert token0.string == start, f"Expected '{start}'"
    tokens = tokens[1:]
    elements = []
    while tokens[0].string != end:
        element, tokens = read_form(tokens)
        elements.append(element)
        if len(tokens) == 0:
            raise Exception(f"Expected '{end}'")
    return with_meta(
        ctor(*elements),
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
    def entry(_f):
        if is_list(_f) and len(_f) > 0 and _f[0] == _S_UNQUOTE:
            assert len(_f) == 2, "unquote expects 1 argument"
            return list_(_S_LIST, _f[1])
        elif is_list(_f) and len(_f) > 0 and _f[0] == _S_SPLICE_UNQUOTE:
            assert len(_f) == 2, "splice-unquote expects 1 argument"
            return _f[1]
        else:
            return list_(_S_LIST, quasiquote(_f))
    return cons(_S_CONCAT, list_(*map(entry, form)))

#************************************************************
# Evaluation
#************************************************************

Context = define_record("Context",
    keyword("namespaces"),
    keyword("current-ns"))

def eval_string(text):
    tokens = list(tokenize(text))
    ctx = Context(hash_map(), symbol("user"))
    _globals = {}
    while tokens:
        form, tokens = read_form(tokens)
        result, body, ctx = _compile(form, ctx)
        body_module = ast.Module(body, type_ignores=[])
        body_code = compile(body_module, "<none>", "exec")
        exec(body_code, _globals) # pylint: disable=exec-used
    result_expr = ast.Expression(result, type_ignores=[])
    result_code = compile(result_expr, "<none>", "eval")
    return eval(result_code, _globals) # pylint: disable=eval-used

def _compile(form, ctx):
    if isinstance(form, PersistentList):
        raise NotImplementedError()
    elif isinstance(form, PersistentVector):
        raise NotImplementedError()
    elif isinstance(form, PersistentMap):
        raise NotImplementedError()
    elif isinstance(form, Symbol):
        raise NotImplementedError()
    else:
        return ast.Constant(form, lineno=0, col_offset=0), None, ctx

#************************************************************
# Core
#************************************************************

def meta(obj):
    return obj.__meta__

def with_meta(obj, _meta):
    return obj.with_meta(_meta)

def cons(obj, coll):
    return coll.cons(obj)
