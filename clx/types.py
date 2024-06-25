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
    PersistentHashMap, hash_map, hash_map_from, is_hash_map, \
    Cons, cons, \
    LazySeq, lazy_seq, seq, \
    Atom, atom, is_atom, \
    define_record, \
    first, next_, rest, get, nth

PersistentMap = PersistentHashMap

Iterable.register(PersistentVector)

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
