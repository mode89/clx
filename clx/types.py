from abc import abstractmethod, ABC
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
import sys
import threading

class IMeta(ABC):
    @abstractmethod
    def with_meta(self, _meta):
        raise NotImplementedError()

class ICounted(ABC):
    @abstractmethod
    def count_(self):
        raise NotImplementedError()

class IIndexed(ICounted, ABC):
    @abstractmethod
    def nth(self, index, not_found):
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

class IRecord(IAssociative, ABC):
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

class Keyword(Hashable):
    def __init__(self, _namespace, _name):
        self.name = sys.intern(_name)
        self.namespace = sys.intern(_namespace) if _namespace else None
        self._hash = hash((_namespace, _name))
    def __eq__(self, other):
        return self is other
    def __hash__(self):
        return self._hash
    def __str__(self):
        return f"Keyword({self.namespace}, {self.name})"
    def __call__(self, obj, not_found=None):
        return obj.lookup(self, not_found)

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

class PersistentList( # pylint: disable=too-many-ancestors
        Hashable,
        Sequence,
        IMeta,
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

class PersistentVector(
        Hashable,
        Sequence,
        IMeta,
        IIndexed,
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
    def count_(self):
        return len(self._impl)
    def nth(self, index, not_found):
        return self._impl[index] \
            if 0 <= index < len(self._impl) \
            else not_found
    def seq(self):
        size = len(self._impl)
        return IndexedSeq(self._impl, 0, None) if size > 0 else None

_EMPTY_VECTOR = PersistentVector([], _meta=None)
_EMPTY_VECTOR.seq = lambda: None

def vec(coll):
    lst = list(coll)
    return PersistentVector(lst, None) if lst else _EMPTY_VECTOR

def vector(*elements):
    return vec(elements)

class PersistentMap(
        Hashable,
        Mapping,
        IMeta,
        ICounted,
        ISeqable,
        IAssociative):
    def __init__(self, impl, _meta):
        self._impl = impl
        self.__meta__ = _meta
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
        s = self.seq()
        return s.first() if s is not None else None
    def next(self):
        s = self.seq()
        return s.next() if s is not None else None
    def rest(self):
        s = self.seq()
        return s.rest() if s is not None else _EMPTY_LIST
    def seq(self):
        s = self._force1()
        while type(s) is LazySeq:
            s = s._force1() # pylint: disable=protected-access
        return seq(s)
    def _force1(self):
        with self._lock:
            if self._func is not None:
                self._seq = self._func()
                self._func = None
            return self._seq

def lazy_seq(func):
    return LazySeq(func, None, _meta=None)

class IndexedSeq(IMeta, IIndexed, ISeq, ISequential):
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
    def nth(self, index, not_found):
        return self._coll[index + self._index] \
            if 0 <= index < len(self._coll) - self._index \
            else not_found
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
        x, y = x.seq(), y.seq()
        print(x, y)
        while True:
            if x is None:
                return y is None
            elif y is None:
                return False
            elif x is y:
                return True
            elif x.first() != y.first():
                return False
            else:
                x, y = x.next(), y.next()
    else:
        return False

def cons(obj, coll):
    if coll is None:
        return PersistentList(obj, _EMPTY_LIST, 1, None, None)
    elif isinstance(coll, ISeq):
        return Cons(obj, coll, _meta=None)
    else:
        return Cons(obj, seq(coll), _meta=None)

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
    dummy = object()
    def _seq():
        value = next(it, dummy)
        return cons(value, lazy_seq(_seq)) \
            if value is not dummy else None
    return lazy_seq(_seq)

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
