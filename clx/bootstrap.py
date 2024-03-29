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

_DUMMY = object()

#************************************************************
# Utilities
#************************************************************

_MUNGE_TABLE = {
    "-": "_",
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

class IPrintable(ABC):
    @abstractmethod
    def pr(self, readably):
        raise NotImplementedError()

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
    def assoc(self, *kvs):
        raise NotImplementedError()

class ICollection(ABC):
    @abstractmethod
    def conj(self, value):
        raise NotImplementedError()

class ISequential(ABC):
    pass

class Symbol(Hashable, IMeta, IPrintable):
    def __init__(self, _namespace, _name, _meta):
        self.name = sys.intern(_name)
        self.namespace = sys.intern(_namespace) if _namespace else None
        self._hash = hash((self.namespace, self.name))
        self.__meta__ = _meta
    def __repr__(self):
        return self.pr(True)
    def __str__(self):
        return self.pr(False)
    def __eq__(self, other):
        return isinstance(other, Symbol) \
            and self.name is other.name \
            and self.namespace is other.namespace
    def __hash__(self):
        return self._hash
    def with_meta(self, _meta):
        return Symbol(self.namespace, self.name, _meta)
    def pr(self, readably):
        return f"{self.namespace}/{self.name}" \
            if self.namespace else self.name

def symbol(arg1, arg2=None):
    if arg2 is None:
        if isinstance(arg1, str):
            if arg1[0] == "/":
                return Symbol(None, arg1, None)
            elif "/" in arg1:
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

class Keyword(Hashable, IPrintable):
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
    def __repr__(self):
        return self.pr(True)
    def __str__(self):
        return self.pr(False)
    def __call__(self, coll, not_found=None):
        return get(coll, self, not_found)
    def pr(self, readably):
        return f":{self.namespace}/{self.name}" \
            if self.namespace else f":{self.name}"

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

class PersistentList( # pylint: disable=too-many-ancestors
        Hashable,
        Sequence,
        IMeta,
        IPrintable,
        ICounted,
        ISeq,
        ISequential,
        ICollection):
    def __init__(self, _first, _rest, length, _hash, _meta):
        self._first = _first
        self._rest = _rest
        self._length = length
        self._hash = _hash
        self.__meta__ = _meta
    def __repr__(self):
        return self.pr(True)
    def __str__(self):
        return self.pr(False)
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
        if self._hash is None:
            self._hash = hash(tuple(self))
        return self._hash
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
        return PersistentList(
            self._first,
            self._rest,
            self._length,
            self._hash,
            _meta)
    def pr(self, readably):
        return "(" + \
            " ".join(map(lambda x: pr_str(x, readably), self)) + \
            ")"
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
        return PersistentList(value, self, self._length + 1, None, None)

_EMPTY_LIST = PersistentList(None, None, 0, None, None)
_EMPTY_LIST.rest = lambda: _EMPTY_LIST
_EMPTY_LIST.seq = lambda: None

def list_(*elements):
    return _list_from_iterable(elements)

def _list_from_iterable(iterable):
    _list = iterable if isinstance(iterable, list) else list(iterable)
    result = _EMPTY_LIST
    for elem in reversed(_list):
        result = PersistentList(
            elem, result, result._length + 1, None, None) # pylint: disable=protected-access
    return result

def is_list(obj):
    return type(obj) is PersistentList

class PersistentVector(
        Hashable,
        Sequence,
        IMeta,
        IPrintable,
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
        raise NotImplementedError()
    def __len__(self):
        return len(self._impl)
    def __iter__(self):
        return iter(self._impl)
    def __getitem__(self, index):
        return self._impl[index]
    def __call__(self, index):
        return self._impl[index]
    def with_meta(self, _meta):
        return PersistentVector(self._impl, _meta)
    def pr(self, readably):
        return "[" + \
            " ".join(map(lambda x: pr_str(x, readably), self)) + \
            "]"
    def count_(self):
        return len(self._impl)
    def seq(self):
        size = len(self._impl)
        if size == 0:
            return None
        elif size < 32:
            return IndexedSeq(self._impl, 0, None)
        return _list_from_iterable(self._impl)

_EMPTY_VECTOR = PersistentVector([], _meta=None)
_EMPTY_VECTOR.seq = lambda: None

def vec(coll):
    lst = list(coll)
    return PersistentVector(lst, None) if lst else _EMPTY_VECTOR

def vector(*elements):
    return vec(elements)

def is_vector(obj):
    return type(obj) is PersistentVector

class PersistentMap(
        Hashable,
        Mapping,
        IMeta,
        IPrintable,
        ICounted,
        ISeqable,
        IAssociative):
    def __init__(self, impl, _meta):
        self._impl = impl
        self.__meta__ = _meta
    def __repr__(self):
        return self.pr(True)
    def __str__(self):
        return self.pr(False)
    def __eq__(self, other):
        if type(other) is PersistentMap:
            return self._impl == other._impl
        else:
            raise NotImplementedError()
    def __hash__(self):
        raise NotImplementedError()
    def __len__(self):
        return len(self._impl)
    def __iter__(self):
        return iter(self._impl)
    def __getitem__(self, key):
        return self._impl[key]
    def __call__(self, key, not_found=None):
        return self._impl.get(key, not_found)
    def with_meta(self, _meta):
        return PersistentMap(self._impl, _meta)
    def pr(self, readably):
        ret = functools.reduce(
            lambda acc, kv:
                acc.conj(pr_str(kv[1], readably))
                    .conj(pr_str(kv[0], readably)),
            self.items(),
            list_())
        return "{" + " ".join(ret) + "}"
    def count_(self):
        return len(self._impl)
    def seq(self):
        if len(self._impl) == 0:
            return None
        raise NotImplementedError()
    def lookup(self, key, not_found):
        return self._impl.get(key, not_found)
    def assoc(self, *kvs):
        copy = self._impl.copy()
        for key, value in zip(kvs[::2], kvs[1::2]):
            copy[key] = value
        return PersistentMap(copy, None)
    def merge(self, other):
        copy = self._impl.copy()
        copy.update(other)
        return PersistentMap(copy, None)

_EMPTY_MAP = PersistentMap({}, None)

def hash_map(*elements):
    num = len(elements)
    assert num % 2 == 0, "hash-map expects even number of elements"
    if num == 0:
        return _EMPTY_MAP
    return PersistentMap(dict(zip(elements[::2], elements[1::2])), None)

def is_hash_map(obj):
    return type(obj) is PersistentMap

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
  def assoc(self, *kvs):
    obj = self
    for k, v in zip(kvs[::2], kvs[1::2]):
      obj = assoc_methods[k](obj, v)
    return obj
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

class LazySeq(Hashable, IMeta, ISeq, ISequential):
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
    def __hash__(self):
        raise NotImplementedError()
    def __iter__(self):
        s = self.seq()
        while s is not None:
            yield s.first()
            s = s.next()
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

class IndexedSeq(IMeta, ICounted, ISeq, ISequential):
    def __init__(self, coll, index, _meta):
        self._coll = coll
        self._index = index
        self.__meta__ = _meta
    def __eq__(self, other):
        return self is other \
            or _equiv_sequential(self, other)
    def with_meta(self, _meta):
        return IndexedSeq(self._coll, self._index, _meta)
    def count_(self):
        return len(self._coll) - self._index
    def first(self):
        return self._coll[self._index]
    def next(self):
        _index = self._index + 1
        if _index < len(self._coll):
            return IndexedSeq(self._coll, _index, None)
        return None
    def rest(self):
        _next = self.next()
        return _next if _next is not None else _EMPTY_LIST
    def seq(self):
        return self

def _equiv_sequential(x, y):
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

class Atom:
    def __init__(self, value):
        self._lock = threading.Lock()
        self._value = value
    def deref(self):
        with self._lock:
            return self._value
    def reset(self, new_value):
        with self._lock:
            self._value = new_value
            return new_value
    def swap(self, f, *args):
        with self._lock:
            self._value = f(self._value, *args)
            return self._value

def atom(value):
    return Atom(value)

class Box:
    def __init__(self, value):
        self._value = value
    def deref(self):
        return self._value
    def reset(self, new_value):
        self._value = new_value
        return new_value
    def swap(self, f, *args):
        self._value = f(self._value, *args)
        return self._value

#************************************************************
# Constants
#************************************************************

_S_QUOTE = symbol("quote")
_S_UNQUOTE = symbol("unquote")
_S_SPLICE_UNQUOTE = symbol("splice-unquote")
_S_DEF = symbol("def")
_S_DO = symbol("do")
_S_LET_STAR = symbol("let*")
_S_COND = symbol("cond")
_S_FN_STAR = symbol("fn*")
_S_LOOP_STAR = symbol("loop*")
_S_RECUR = symbol("recur")
_S_IN_NS = symbol("in-ns")
_S_DOT = symbol(".")
_S_IMPORT_STAR = symbol("import*")
_S_PYTHON = symbol("python*")
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

_K_LINE = keyword("line")
_K_COLUMN = keyword("column")
_K_CURRENT_NS = keyword("current-ns")
_K_NAMESPACES = keyword("namespaces")
_K_ENV = keyword("env")
_K_PY_GLOBALS = keyword("py-globals")
_K_PY_NAME = keyword("py-name")
_K_ATTRIBUTE = keyword("attribute")
_K_BINDINGS = keyword("bindings")
_K_IMPORTS = keyword("imports")
_K_MACRO_QMARK = keyword("macro?")
_K_TOP_LEVEL_Q = keyword("top-level?")
_K_SHARED = keyword("shared")
_K_LOCAL = keyword("local")
_K_LOCK = keyword("lock")
_K_LOOP_BINDINGS = keyword("loop-bindings")
_K_TAIL_Q = keyword("tail?")

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
    elif tstring == "(":
        l, rtokens = read_collection(token, rtokens, list_, ")")
        l0 = l.first()
        if type(l0) is Symbol and l0.name[0] == "." and l0.name != ".":
            member = symbol(l0.name[1:])
            assert l.count_() > 1, "expected (.member target ...)"
            target = l.rest().first()
            l = list_(_S_DOT, target, member, *l.rest().rest()).with_meta(
                hash_map(_K_LINE, token.line, _K_COLUMN, token.column))
        return l, rtokens
    elif tstring == "[":
        return read_collection(token, rtokens, vector, "]")
    elif tstring == "{":
        return read_collection(token, rtokens, hash_map, "}")
    else:
        return read_atom(tstring), rtokens

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
            return \
                with_meta(
                    ctor(*elements),
                    hash_map(
                        _K_LINE, token0.line,
                        _K_COLUMN, token0.column)), \
                rest(tokens)
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
    elif is_hash_map(form):
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
    if isinstance(obj, IPrintable):
        return obj.pr(readably)
    return repr(obj) if readably else str(obj)

def escape(text):
    return text \
        .replace("\\", r"\\") \
        .replace("\"", r"\"") \
        .replace("\n", r"\n")

#************************************************************
# Evaluation
#************************************************************

# Tracks state of the current evaluation thread
ThreadContext = define_record("Context",
    _K_SHARED,
    _K_CURRENT_NS,
)
# Tracks state shared across multiple evaluation threads
SharedContext = define_record("SharedContext",
    _K_NAMESPACES,
    _K_PY_GLOBALS,
)
# Local context defines state that is local to a single form
LocalContext = define_record("LocalContext",
    _K_ENV,
    _K_LOOP_BINDINGS,
    _K_TAIL_Q,
    _K_TOP_LEVEL_Q,
    _K_LINE,
    _K_COLUMN,
)
Namespace = define_record("Namespace", _K_BINDINGS, _K_IMPORTS)
Binding = define_record("Binding", _K_PY_NAME, _K_ATTRIBUTE)

def _known_binding(sym):
    return Binding(munge(sym), None)

_B_KEYWORD = _known_binding(_S_KEYWORD)
_B_SYMBOL = _known_binding(_S_SYMBOL)
_B_LIST = _known_binding(_S_LIST)
_B_VECTOR = _known_binding(_S_VECTOR)
_B_HASH_MAP = _known_binding(_S_HASH_MAP)
_B_WITH_META = _known_binding(_S_WITH_META)

def init_context(namespaces):
    _globals = {}

    namespaces = {
        "user": {},
        "clx.core": {
            "+": lambda *args: sum(args),
            "with-meta": with_meta,
            "meta": meta,
            "apply": apply,
            "keyword": keyword,
            "symbol": symbol,
            "symbol?": is_symbol,
            "list": list_,
            "vector": vector,
            "vector?": is_vector,
            "vec": vec,
            "hash-map": hash_map,
            "lazy-seq*": lazy_seq,
            "first": first,
            "second": second,
            "rest": rest,
            "next": next_,
            "seq": seq,
            "concat": concat,
            "cons": cons,
            "count": count,
            "pr-str": pr_str,
            "gensym": lambda prefix="___gen_": symbol(_gen_name(prefix)),
        },
        **namespaces,
    }

    _namespaces = hash_map()
    for ns_name, ns_bindings in namespaces.items():
        _bindings = hash_map()
        for name, value in ns_bindings.items():
            py_name = munge(f"{ns_name}/{name}")
            _bindings = assoc(_bindings, name, Binding(py_name, None))
            _globals[py_name] = value
        _namespaces = assoc(_namespaces, ns_name,
            Namespace(_bindings, hash_map()))

    return Box(
        ThreadContext(
            shared=SharedContext(
                namespaces=_namespaces,
                py_globals=_globals,
            ),
            current_ns="user",
        )
    )

def _load_string(ctx, file_name, text):
    lctx = LocalContext(
        env=hash_map(),
        loop_bindings=None,
        tail_QMARK_=False,
        top_level_QMARK_=True,
        line=1,
        column=1)
    tokens = list(tokenize(text))
    while tokens:
        form, tokens = read_form(tokens)
        result = _eval_form(ctx, lctx, file_name, form)
    return result

def _eval_form(ctx, lctx, file_name, form):
    result, body = _compile(ctx, lctx, form)
    body = ast.Module(body, type_ignores=[])
    result = ast.Expression(result, type_ignores=[])
    _transform_ast(body, result)
    print(ast.unparse(body))
    print(ast.unparse(result))
    _globals = ctx.deref().shared.py_globals
    exec(compile(body, file_name, "exec"), _globals) # pylint: disable=exec-used
    return eval(compile(result, file_name, "eval"), _globals) # pylint: disable=eval-used

def load_file(ctx, path):
    return _load_string(ctx, path, slurp(path))

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
                obj = _eval_binding(ctx, binding)
                if is_macro(obj):
                    return obj(*form.rest())
                else:
                    if get(binding, _K_MACRO_QMARK):
                        raise Exception(
                            f"macro '{head}' is defined, "
                            "but hasn't been evaluated yet")
    return form

def _compile(ctx, lctx, form):
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
        return _binding_node(lctx, ast.Load(), binding), []
    else:
        return \
            ast.Constant(
                form,
                lineno=lctx.line,
                col_offset=lctx.column), \
            []

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
    ns = ctx.deref().current_ns
    py_name = munge(f"{ns}/{name.name}")
    ctx.swap(assoc_in,
        list_(_K_SHARED, _K_NAMESPACES, ns, _K_BINDINGS, name.name),
        Binding(py_name, None))
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
    stmts = []
    forms = form.rest()
    result_name = None
    while forms:
        _f, forms = forms.first(), forms.rest()
        f_result, f_stmts = _compile(
            ctx, lctx.assoc(_K_TAIL_Q, False) if forms else lctx, _f)
        stmts.extend(f_stmts)
        result_name = _gen_name()
        stmts.append(
            _node(ast.Assign, lctx,
                [_node(ast.Name, lctx, result_name, ast.Store())],
                f_result))
    return \
        _node(ast.Name, lctx, result_name, ast.Load()) \
            if result_name is not None \
            else _node(ast.Constant, lctx, None), \
        stmts

def _compile_let(ctx, lctx, form):
    assert len(form) > 1, "let* expects at least 1 argument"
    assert len(form) < 4, "let* expects at most 2 arguments"
    bindings = second(form)
    assert is_vector(bindings), \
        "let* expects a vector as the first argument"
    assert len(bindings) % 2 == 0, \
        "bindings of let* must have even number of elements"
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
        lctx = assoc_in(lctx, list_(_K_ENV, _name.name), Binding(py_name, None))
    if len(form) == 3:
        body_expr, body_stmts = _compile(ctx, lctx, third(form))
        body.extend(body_stmts)
    else:
        body_expr = _node(ast.Constant, lctx, None)
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

    expr, fdef = _make_function_def(
        ctx, lctx.assoc(_K_LOOP_BINDINGS, None), fname, params, body)
    stmts = [fdef]

    _meta = meta(form)
    if _meta is not None:
        meta_expr, meta_stmts = _compile(
            ctx, lctx.assoc(_K_TAIL_Q, False), _meta)
        stmts.extend(meta_stmts)
        expr = _node(ast.Call, lctx,
            _binding_node(lctx, ast.Load(), _B_WITH_META),
            [expr, meta_expr], [])

    return expr, stmts

def _make_function_def(ctx, lctx, fname, params, body):
    assert is_symbol(fname), "function name must be a symbol"
    assert isinstance(params, PersistentVector), \
        "function parameters must be a vector"

    pos_params = []
    rest_param = None
    env = lctx.env
    while params:
        param = first(params)
        assert is_simple_symbol(param), \
            "parameters of function must be simple symbols"
        if param == _S_AMPER:
            assert next_(next_(params)) is None, \
                "parameters should have a single symbol after &"
            rest_param = second(params)
            assert is_simple_symbol(rest_param), \
                "rest parameters of function must be a simple symbol"
            env = assoc(env, rest_param.name,
                Binding(munge(rest_param.name), None))
            break
        pos_params.append(param)
        env = assoc(env, param.name, Binding(munge(param.name), None))
        params = rest(params)
    lctx = assoc(lctx, _K_ENV, env)

    if body is not None:
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

def _compile_loop(ctx, lctx, form):
    assert len(form) > 1, "loop* expects at least 1 argument"
    assert len(form) < 4, "loop* expects at most 2 arguments"
    bindings = second(form)
    assert is_vector(bindings), \
        "loop* expects a vector as the first argument"
    assert len(bindings) % 2 == 0, \
        "bindings of loop* must have even number of elements"
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
        binding = Binding(py_name, None)
        loop_bindings.append(binding)
        lctx = assoc_in(lctx, list_(_K_ENV, _name.name), binding)
    lctx = assoc(lctx, _K_LOOP_BINDINGS, loop_bindings)
    if len(form) == 3:
        result_name = _gen_name("___loop_")
        body_expr, body_stmts = _compile(
            ctx, lctx.assoc(_K_TAIL_Q, True), third(form))
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
    assert lctx.loop_bindings is not None, "recur allowed only in loop*"
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

def _compile_in_ns(ctx, lctx, form):
    assert lctx.top_level_QMARK_, "in-ns allowed only at top level"
    assert len(form) == 2, "in-ns expects exactly 1 argument"
    ns = second(form)
    assert is_symbol(ns), "in-ns expects a symbol"
    def _update_context(_ctx):
        _ctx = _ctx.assoc(_K_CURRENT_NS, ns.name)
        ns_obj = _ctx.shared.namespaces.lookup(ns.name, None)
        if ns_obj is None:
            _ctx = assoc_in(_ctx,
                list_(_K_SHARED, _K_NAMESPACES, ns.name),
                Namespace(hash_map(), hash_map()))
        return _ctx
    ctx.swap(_update_context)
    return _node(ast.Constant, lctx, None), []

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
    py_name = _gen_name(f"___import_{munge(mod.name)}_")
    def _update_context(_ctx):
        return assoc_in(_ctx,
            list_(
                _K_SHARED,
                _K_NAMESPACES,
                _ctx.current_ns,
                _K_IMPORTS,
                _as.name),
            py_name)
    ctx.swap(_update_context)
    return \
        _node(ast.Name, lctx, py_name, ast.Load()), \
        [_node(ast.Import, lctx,
            [_node(ast.alias, lctx, mod.name, py_name)])]

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

def _trace_local_context(_ctx, lctx, form):
    path = form.rest()
    return _node(ast.Constant, lctx, get_in(lctx, path)), []

_SPECIAL_FORM_COMPILERS = {
    _S_DEF: _compile_def,
    _S_DO: _compile_do,
    _S_LET_STAR: _compile_let,
    _S_COND: _compile_cond,
    _S_FN_STAR: _compile_fn,
    _S_LOOP_STAR: _compile_loop,
    _S_RECUR: _compile_recur,
    _S_QUOTE: _compile_quote,
    _S_IN_NS: _compile_in_ns,
    _S_DOT: _compile_dot,
    _S_IMPORT_STAR: _compile_import,
    _S_PYTHON: _compile_python,
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
            _binding_node(lctx, ast.Load(), _B_VECTOR),
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
            _binding_node(lctx, ast.Load(), _B_HASH_MAP),
            args,
            []), \
        stmts

def _resolve_symbol(ctx, lctx, sym, not_found=_DUMMY):
    _ctx = ctx.deref()
    if is_simple_symbol(sym):
        if lctx is not None:
            binding = lctx.env.lookup(sym.name, None)
            if binding is not None:
                return binding

        namespaces = _ctx.shared.namespaces
        binding = namespaces \
            .lookup(_ctx.current_ns, None) \
            .bindings.lookup(sym.name, None)
        if binding is not None:
            return binding

        binding = namespaces \
            .lookup("clx.core", None) \
            .bindings.lookup(sym.name, None)
        if binding is not None:
            return binding
    else:
        namespaces = _ctx.shared.namespaces
        import_alias = namespaces \
            .lookup(_ctx.current_ns, None) \
            .imports.lookup(sym.namespace, None)
        if import_alias is not None:
            return Binding(import_alias, munge(sym.name))

        namespace = namespaces.lookup(sym.namespace, None)
        if namespace is not None:
            binding = namespace.bindings.lookup(sym.name, None)
            if binding is not None:
                return binding

    if not_found is _DUMMY:
        raise Exception(f"Symbol '{pr_str(sym)}' not found")
    return not_found

def _gen_name(prefix="___gen_"):
    if not hasattr(_gen_name, "counter"):
        _gen_name.counter = Atom(10000)
    return f"{prefix}{_gen_name.counter.swap(lambda x: x + 1)}"

def _node(type_, lctx, *args):
    _n = type_(*args)
    _n.lineno = lctx.line
    _n.col_offset = lctx.column
    return _n

def _binding_node(lctx, load_or_store, binding):
    if binding.attribute is None:
        return _node(ast.Name, lctx, binding.py_name, load_or_store)
    else:
        return _node(ast.Attribute, lctx,
            _node(ast.Name, lctx, binding.py_name, ast.Load()),
            binding.attribute,
            load_or_store)

def _eval_binding(ctx, binding, not_found=_DUMMY):
    obj = ctx.deref().shared.py_globals.get(binding.py_name, not_found)
    if binding.attribute is not None:
        return getattr(obj, binding.attribute, not_found)
    return obj

def _binding_string(binding):
    if binding.attribute is None:
        return binding.py_name
    return f"{binding.py_name}.{binding.attribute}"

def _transform_ast(body, result):
    _fix_constants(body, result)

def _fix_constants(body, result):
    consts = {}

    lctx = LocalContext(
        env=hash_map(),
        loop_bindings=None,
        tail_QMARK_=False,
        top_level_QMARK_=True,
        line=1,
        column=1)

    class Transformer(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, (
                    Keyword,
                    Symbol,
                    PersistentList,
                    PersistentVector,
                    PersistentMap)):
                name = _gen_name("___const_")
                consts[name] = _compile_constant(lctx, node.value)
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

def _compile_constant(lctx, value):
    if isinstance(value, (Keyword, Symbol)):
        _ns = ast.Constant(value.namespace, lineno=0, col_offset=0)
        _name = ast.Constant(value.name, lineno=0, col_offset=0)
        tree = ast.Call(
            _binding_node(lctx, ast.Load(),
                _B_KEYWORD if type(value) is Keyword else _B_SYMBOL),
            [_ns, _name], [], lineno=0, col_offset=0)
    elif isinstance(value, (PersistentList, PersistentVector)):
        tree = ast.Call(
            _binding_node(lctx, ast.Load(),
                _B_LIST if type(value) is PersistentList else _B_VECTOR),
            [_compile_constant(lctx, v) for v in value],
            [], lineno=0, col_offset=0)
    elif type(value) is PersistentMap:
        args = []
        for key, val in value.items():
            args.append(_compile_constant(lctx, key))
            args.append(_compile_constant(lctx, val))
        tree = ast.Call(
            _binding_node(lctx, ast.Load(), _B_HASH_MAP),
            args, [], lineno=0, col_offset=0)
    else:
        tree = ast.Constant(value, lineno=0, col_offset=0)
    if meta(value) is not None:
        tree = ast.Call(
            _binding_node(lctx, ast.Load(), _B_WITH_META),
            [tree, _compile_constant(lctx, meta(value))],
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

def is_macro(obj):
    return callable(obj) and get(meta(obj), _K_MACRO_QMARK, False)

def is_counted(x):
    return isinstance(x, ICounted)

def count(x):
    if x is None:
        return 0
    elif isinstance(x, ICounted):
        return x.count_()
    elif isinstance(x, ISeqable):
        return _count_seqable(x)
    else:
        return len(x)

def _count_seqable(x):
    num = 0
    while x is not None:
        num += 1
        x = x.next()
    return num

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
    if coll is None:
        return None
    elif isinstance(coll, ISeqable):
        return coll.seq()
    elif type(coll) is tuple \
            or (type(coll) is list and len(coll) < 32) \
            or type(coll) is str:
        if len(coll) == 0:
            return None
        return IndexedSeq(coll, 0, None)
    elif isinstance(coll, Iterable):
        return iterator_seq(iter(coll)).seq()
    else:
        raise Exception("expected a seqable object")

def iterator_seq(it):
    assert isinstance(it, Iterator), "iterator-seq expects an iterator"
    def _seq():
        value = next(it, _DUMMY)
        return cons(value, lazy_seq(_seq)) \
            if value is not _DUMMY else None
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
    def helper(m1, m2):
        if m1 is None:
            return m2
        elif m2 is None:
            return m1
        else:
            return m1.merge(m2)
    return functools.reduce(helper, maps, None)

def slurp(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()
