from abc import ABC
from collections.abc import Hashable, Iterable, Iterator, Sequence
import threading

# pylint: disable=unused-import
from clx_rust import \
    IMeta, ICounted, ISeqable, ISeq, ICollection, ISequential, IIndexed, \
    IAssociative, \
    Symbol, symbol, is_symbol, is_simple_symbol, \
    Keyword, keyword, is_keyword, is_simple_keyword, \
    PersistentList, list_, is_list, \
    PersistentVector, vector, is_vector, \
    PersistentHashMap, hash_map, hash_map_from, is_hash_map

PersistentMap = PersistentHashMap

Iterable.register(PersistentVector)

class IRecord(IAssociative, ABC):
    pass

def _list_from_iterable(iterable):
    _list = iterable if isinstance(iterable, list) else list(iterable)
    result = list_()
    for elem in reversed(_list):
        result = result.conj(elem)
    return result

def vec(coll):
    if not coll:
        return vector()
    else:
        return vector(*coll)

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
        return s.rest() if s is not None else list_()
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
        return _next if _next is not None else list_()
    def seq(self):
        return self

def _equiv_sequential(x, y):
    assert isinstance(x, ISequential), "expected a sequential"
    if isinstance(y, ISequential):
        x, y = x.seq(), y.seq()
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
        return list_().conj(obj)
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
