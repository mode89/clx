import pytest

from lepet_ext import \
    Symbol, symbol, \
    Keyword, keyword, \
    PersistentList, list_, \
    PersistentVector, vector, \
    PersistentHashMap, hash_map, hash_map_from, \
    CowSet, cow_set, cow_set_from, is_cow_set, \
    cons, lazy_seq, seq, \
    Atom, atom, \
    define_record
import clx.bootstrap as boot

K = keyword
S = symbol
L = list_
V = vector
M = hash_map
CS = cow_set

class NoHash:
    def __hash__(self):
        raise TypeError("unhashable type: 'NoHash'")

def _lazy_range(*args):
    assert len(args) <= 2
    if len(args) == 0:
        start = 0
        end = None
    elif len(args) == 1:
        start = 0
        end = args[0]
    elif len(args) == 2:
        start = args[0]
        end = args[1]
    def helper(i):
        return cons(i, lazy_seq(lambda: helper(i + 1))) \
            if end is None or i < end else None
    return lazy_seq(lambda: helper(start))

def test_keyword():
    hello = K("hello")
    assert isinstance(hello, Keyword)
    assert boot.is_keyword(hello)
    assert hello.name == "hello"
    assert hello.namespace is None
    assert hello is K("hello")
    hello_world = K("hello", "world")
    assert isinstance(hello_world, Keyword)
    assert boot.is_keyword(hello_world)
    assert hello_world.name == "world"
    assert hello_world.namespace == "hello"
    assert hello_world is K("hello", "world")
    foo_bar = K("foo/bar")
    assert isinstance(foo_bar, Keyword)
    assert boot.is_keyword(foo_bar)
    assert foo_bar.name == "bar"
    assert foo_bar.namespace == "foo"
    assert foo_bar is K("foo/bar")
    assert K(S("baz")) is K("baz")
    assert K(S("foo/bar")) is K("foo/bar")
    assert K(K("quux")) is K("quux")
    assert K(None, "foo") is K("foo")
    assert K("foo")(M(K("foo"), 42)) == 42
    assert K("foo")(M(K("bar"), 42)) is None
    assert K("foo")(M(K("bar"), 42), 43) == 43

def test_symbol():
    hello = S("hello")
    assert isinstance(hello, Symbol)
    assert boot.is_symbol(hello)
    assert hello.name == "hello"
    assert hello.namespace is None
    assert hello == S("hello")
    hello_world = S("hello", "world")
    assert isinstance(hello_world, Symbol)
    assert boot.is_symbol(hello_world)
    assert hello_world.name == "world"
    assert hello_world.namespace == "hello"
    assert hello_world == S("hello", "world")
    foo_bar = S("foo/bar")
    assert isinstance(foo_bar, Symbol)
    assert boot.is_symbol(foo_bar)
    assert foo_bar.name == "bar"
    assert foo_bar.namespace == "foo"
    assert foo_bar == S("foo/bar")
    assert S(S("quux")) == S("quux")
    assert boot.is_simple_symbol(S("foo"))
    assert not boot.is_simple_symbol(S("foo/bar"))
    assert S("/").name == "/"
    assert S("/").namespace is None
    assert S("".join(["a" for _ in range(100)])) == \
        S("".join(["a" for _ in range(100)]))
    assert S(None, "foo") == S("foo")

def test_list():
    assert isinstance(L(), PersistentList)
    assert L() is L()
    assert len(L()) == 0
    assert bool(L()) is False
    assert L().first() is None
    assert L().rest() is L()
    assert L().next() is None
    assert L().conj(1) == L(1)
    assert L().with_meta(M(1, 2)).__meta__ == M(1, 2)
    assert L().with_meta(M(1, 2)) is not L()
    assert L().with_meta(M(1, 2)) == L()
    assert L().with_meta(M(1, 2)) == L().with_meta(M(3, 4))
    # TODO: assert L().with_meta(M(1, 2)).rest() is L()
    # TODO: assert L().with_meta(M(1, 2)).seq() is None
    assert L(1) is not L()
    assert len(L(1)) == 1
    assert bool(L(1)) is True
    assert L(1).first() == 1
    assert L(1).rest() is L()
    assert L(1).next() is None
    assert L(1).conj(2) == L(2, 1)
    assert L(1, 2) == L(1, 2)
    assert len(L(1, 2)) == 2
    assert bool(L(1, 2)) is True
    assert L(1, 2).first() == 1
    assert L(1, 2).rest() == L(2)
    assert L(1, 2).next() == L(2)
    assert L(1, 2).conj(3) == L(3, 1, 2)
    assert isinstance(L(1, 2).conj(3), PersistentList)
    assert L() is not None
    assert L() != 42
    assert L() == V()
    assert L() != M()
    assert L() == _lazy_range(0)
    assert L() != L(1)
    assert L() != L(1, 2)
    assert L(1, 2, 3) != L(1, 2)
    assert L(1, 2, 3) == L(1, 2, 3)
    assert L(1, 2, 3) == V(1, 2, 3)
    assert L(1, 2, 3) == _lazy_range(1, 4)
    assert L(1, 2, 3) == [1, 2, 3]
    assert L(1, 2, 3) == (1, 2, 3)
    assert list(iter(L(1, 2, 3))) == [1, 2, 3]
    assert hash(L()) == hash(L())
    assert hash(L(1)) == hash(L(1))
    assert hash(L(1, 2)) == hash(L(1, 2))
    assert hash(L(1, 2)) != hash(L(2, 1))
    assert hash(L(1, 2, 3)) == hash(L(1, 2, 3))
    assert hash(L(1, 2, 3)) != hash(L(1, 2))
    assert hash(L(1, 2, 3).rest()) == hash(L(2, 3))
    with pytest.raises(TypeError):
        hash(L(NoHash(), 42))

def test_vector():
    v = V()
    assert isinstance(v, PersistentVector)
    assert v is V()
    assert len(v) == 0
    assert v.seq() is None
    with pytest.raises(IndexError):
        v[0] # pylint: disable=pointless-statement
    with pytest.raises(TypeError):
        v[None] # pylint: disable=pointless-statement
    v1 = V(1)
    assert len(v1) == 1
    assert v1.seq().first() == 1
    assert v1.seq().rest() is L()
    assert v1.seq().next() is None
    assert v1[0] == 1
    with pytest.raises(IndexError):
        v1[1] # pylint: disable=pointless-statement
    v23 = V(2, 3)
    assert v23 is not V()
    assert len(v23) == 2
    assert v23.seq().first() == 2
    assert v23.seq().rest() == L(3)
    assert v23.seq().next() == L(3)
    assert v23[0] == 2
    assert v23[1] == 3
    with pytest.raises(IndexError):
        v23[2] # pylint: disable=pointless-statement
    assert V() is not None
    assert V() != 42
    assert V() == L()
    assert V() != M()
    assert V() == _lazy_range(0)
    assert V() != V(1)
    assert V() != V(1, 2)
    assert V(1, 2, 3) != V(1, 2)
    assert V(1, 2, 3) == V(1, 2, 3)
    assert V(1, 2, 3) == L(1, 2, 3)
    assert V(1, 2, 3) == _lazy_range(1, 4)
    assert V(1, 2, 3) == [1, 2, 3]
    assert V(1, 2, 3) == (1, 2, 3)
    assert V(*range(1000)) == V(*range(1000))
    assert V(42, 9001)(0) == 42
    assert V(42, 9001)(1) == 9001
    with pytest.raises(IndexError):
        V(42, 9001)(2)
    with pytest.raises(TypeError):
        V(42, 9001)(None)
    vl = boot.vec(range(3))
    assert isinstance(vl, PersistentVector)
    assert vl == V(0, 1, 2)
    assert V(1, 2, 3).nth(0) == 1
    assert V(1, 2, 3).nth(1) == 2
    assert V(1, 2, 3).nth(2) == 3
    assert V(1, 2, 3).nth(3, 42) == 42
    with pytest.raises(IndexError):
        V(1, 2, 3).nth(3)
    with pytest.raises(IndexError):
        V(1, 2, 3).nth(-1)
    with pytest.raises(TypeError):
        V(1, 2, 3).nth(None)
    assert [*V(1, 2, 3)] == [1, 2, 3]
    assert V(1, 2).conj(3) == V(1, 2, 3)
    assert isinstance(V(1, 2).conj(3), PersistentVector)
    assert hash(V()) == hash(V())
    assert hash(V(1)) == hash(V(1))
    assert hash(V(1, 2)) == hash(V(1, 2))
    assert hash(V(1, 2)) != hash(V(2, 1))
    assert hash(V(1, 2, 3)) == hash(V(1, 2, 3))
    assert hash(V(1, 2, 3)) != hash(V(1, 2))

def test_hash_map():
    m0 = M()
    m1 = m0.assoc("a", 1)
    m2 = m1.assoc("b", 2)
    m3 = m2.merge(M("c", 3, "d", 4))
    m4 = m3.assoc("a", 5, "b", 6)
    assert type(m0) is PersistentHashMap
    assert m0 is M()
    assert type(m1) is PersistentHashMap
    assert m1.lookup("a", None) == 1
    assert m1.lookup("b", 42) == 42
    assert m1 == M("a", 1)
    assert type(m2) is PersistentHashMap
    assert m2 == M("a", 1, "b", 2)
    assert type(m3) is PersistentHashMap
    assert m3 == M("a", 1, "b", 2, "c", 3, "d", 4)
    assert type(m4) is PersistentHashMap
    assert m4 == M("a", 5, "b", 6, "c", 3, "d", 4)
    assert M(1, 2, 3, 4)(1) == 2
    assert M(1, 2, 3, 4)(2) is None
    assert M(1, 2, 3, 4)(2, 42) == 42
    assert M(1, 2, 3, 4)(3) == 4
    assert M(1, 2, 3, 4)(4) is None
    assert dict(M(1, 2, 3, 4)) == {1: 2, 3: 4}
    assert hash_map_from([]) is M()
    assert hash_map_from([("a", 42)]) == M("a", 42)
    assert hash_map_from([[9001, "foo"]]) == M(9001, "foo")
    assert len(M()) == 0
    assert len(M("a", 1)) == 1
    with pytest.raises(TypeError):
        hash(M(NoHash(), 42))
    assert M().seq() is None
    assert sorted(M(1, 2, 3, 4).seq()) == [(1, 2), (3, 4)]
    assert hash(M()) == hash(M())
    assert hash(M(1, 2)) == hash(M(1, 2))
    assert hash(M(*range(1000))) == hash(M(*range(1000)))

def test_cow_set():
    assert type(CS()) is CowSet
    assert CS() is CS()
    assert CS().count_() == 0
    assert CS().seq() is None
    assert CS().conj(1) == CS(1)
    assert CS().disj(1) is CS()
    assert CS().contains(1) is False
    assert CS(1, 2).contains(1) is True
    assert CS(1, 2).contains(2) is True
    assert CS(1, 2).contains(3) is False
    assert CS(1, 2).contains(None) is False
    assert CS(1, 2).conj(3).contains(1) is True
    assert CS(1, 2).conj(3).contains(2) is True
    assert CS(1, 2).conj(3).contains(3) is True
    assert CS(1, 2).conj(3).contains(None) is False
    assert CS(1, 2).disj(1).contains(1) is False
    assert CS(1, 2).disj(1).contains(2) is True
    assert CS(1, 2).disj(1).contains(3) is False
    assert CS(1, 2).disj(1).contains(None) is False
    assert CS(1, 2, 3) is not CS()
    assert CS(1, 2, 3) == CS(3, 1, 2)
    assert CS(1, 2, 3) != CS(1, 2)
    assert CS(1, 2, 3).count_() == 3
    assert sorted(CS(1, 2, 3).seq()) == [1, 2, 3]
    assert CS(1, 2, 3).conj(4) == CS(1, 2, 3, 4)
    assert CS(1, 2, 3).disj(2) == CS(1, 3)
    assert CS(1, 2, 3).disj(4) == CS(1, 2, 3)
    assert CS(1, 2, 3).disj(1, 2) == CS(3)
    assert CS(1, 2, 3).disj(3, 1, 2) is CS()
    assert is_cow_set(CS())
    assert is_cow_set(CS(1, 2, 3))
    assert not is_cow_set(L())
    assert cow_set_from([]) is CS()
    assert cow_set_from([1, 2, 3]) == CS(1, 2, 3)

def test_cons():
    assert cons(1, None).first() == 1
    assert cons(1, None).rest() is L()
    assert cons(1, None).next() is None
    assert cons(1, L(2, 3)).first() == 1
    assert cons(1, L(2, 3)).rest() == L(2, 3)
    assert cons(1, L(2, 3)).next() == L(2, 3)
    assert cons(1, V(2, 3)).first() == 1
    assert cons(1, V(2, 3)).rest() == L(2, 3)
    assert cons(1, V(2, 3)).next() == L(2, 3)
    assert cons(1, lazy_seq(lambda: None)).first() == 1
    assert cons(1, lazy_seq(lambda: None)).next() is None
    assert cons(1, lazy_seq(lambda: L(2, 3))).first() == 1
    assert cons(1, lazy_seq(lambda: L(2, 3))).next() == L(2, 3)
    assert cons(1, L(2, 3)) is not None
    assert cons(1, L(2, 3)) != 42
    assert cons(1, L(2, 3)) == L(1, 2, 3)
    assert cons(1, L(2, 3)) == V(1, 2, 3)
    assert cons(1, L(2, 3)) == _lazy_range(1, 4)
    inf = _lazy_range()
    assert cons(42, inf) == cons(42, inf)
    assert cons(42, inf) != cons(43, inf)
    assert cons(1, cons(2, cons(3, L()))) == L(1, 2, 3)
    assert cons(1, L()).next() is None
    assert [*cons(1, cons(2, cons(3, L())))] == [1, 2, 3]
    assert cons(1, L()).conj(2) == L(2, 1)
    assert cons(1, (cons(2, cons(3, None)))) == [1, 2, 3]
    assert cons(1, (cons(2, cons(3, None)))) == (1, 2, 3)

def test_lazy_seq():
    def nth(coll, n):
        for _ in range(n):
            coll = coll.rest()
        return coll.first()

    def _numbers(_realized, i):
        _realized.append(i)
        return cons(i, lazy_seq(lambda: _numbers(_realized, i + 1)))
    realized_numbers = []
    numbers = lazy_seq(lambda: _numbers(realized_numbers, 0))
    assert numbers.first() == 0
    assert realized_numbers == [0]
    assert numbers.rest().first() == 1
    assert realized_numbers == [0, 1]
    assert nth(numbers, 10) == 10
    assert realized_numbers == list(range(11))
    assert nth(numbers, 10000) == 10000
    assert realized_numbers == list(range(10001))

    def _range(_realized, end, i):
        if i < end:
            _realized.append(i)
            return cons(i, lazy_seq(
                lambda: _range(_realized, end, i + 1)))
        return None
    realized_range = []
    range_ = lazy_seq(lambda: _range(realized_range, 3, 0))
    assert range_.first() == 0
    assert realized_range == [0]
    assert range_.rest().first() == 1
    assert realized_range == [0, 1]
    assert nth(range_, 2) == 2
    assert realized_range == [0, 1, 2]
    assert nth(range_, 3) is None
    assert realized_range == [0, 1, 2]
    assert nth(range_, 4) is None
    assert realized_range == [0, 1, 2]
    assert range_.next().next().first() == 2
    assert range_.next().next().next() is None

    def recur_lazy(coll):
        return lazy_seq(
            lambda: lazy_seq(
                lambda: lazy_seq(
                    lambda: lazy_seq(
                        lambda: lazy_seq(
                            lambda: coll)))))
    assert recur_lazy(L()).first() is None
    assert recur_lazy(L()).next() is None
    assert recur_lazy(L()).rest() is L()
    assert bool(recur_lazy(L(42)))
    assert recur_lazy(L(42)).first() == 42
    assert recur_lazy(L(42)).next() is None
    assert recur_lazy(L(42)).rest() is L()

    inf = _lazy_range()
    assert lazy_seq(lambda: inf) == lazy_seq(lambda: inf)
    assert _lazy_range(0) == _lazy_range(0)
    assert _lazy_range(0) != _lazy_range(1)
    assert _lazy_range(1) != _lazy_range(0)
    assert _lazy_range(1) == _lazy_range(1)
    assert _lazy_range(1) == L(0)
    assert _lazy_range(1) == V(0)
    assert _lazy_range(3) == L(0, 1, 2)
    assert _lazy_range(3) == V(0, 1, 2)
    assert lazy_seq(L) == L()
    assert list(_lazy_range(3)) == [0, 1, 2]

    def raise_exception():
        raise Exception()

    s = lazy_seq(
        lambda: cons(42, lazy_seq(
            lambda: cons(9001, lazy_seq(raise_exception)))))
    assert s.first() == 42
    assert s.rest().first() == 9001
    with pytest.raises(Exception):
        s.rest().rest().first()
    with pytest.raises(Exception):
        # Check that the failed computation didn't affect the state
        s.rest().rest().first()
    assert s.next().first() == 9001
    with pytest.raises(Exception):
        s.next().next().first()
    assert [*_lazy_range(0)] == []
    assert [*_lazy_range(3)] == [0, 1, 2]

    assert lazy_seq(lambda: L(1, 2)).conj(3) == L(3, 1, 2)

    assert _lazy_range(3) == [0, 1, 2]
    assert _lazy_range(3) == (0, 1, 2)

def test_seq():
    assert seq(None) is None
    assert seq(L()) is None
    assert seq(L(1, 2)).first() == 1
    assert seq(L(1, 2)).rest().first() == 2
    assert seq(L(1, 2)).rest().next() is None
    assert seq(V()) is None
    assert seq(V(1, 2)).first() == 1
    assert seq(V(1, 2)).rest().first() == 2
    assert seq(V(1, 2)).rest().next() is None
    assert seq(M()) is None
    assert seq([]) is None
    assert seq([1, 2]).first() == 1
    assert seq([1, 2]).rest().first() == 2
    assert seq([1, 2]).rest().next() is None
    assert seq(tuple()) is None
    assert seq((1, 2)).first() == 1
    assert seq((1, 2)).rest().first() == 2
    assert seq((1, 2)).rest().next() is None
    assert seq(range(1, 1)) is None
    assert seq(range(1, 3)).first() == 1
    assert seq(range(1, 3)).rest().first() == 2
    assert seq(range(1, 3)).rest().next() is None
    assert seq("") is None
    assert seq("12").first() == "1"
    assert seq("12").rest().first() == "2"
    assert seq("12").rest().next() is None
    with pytest.raises(Exception, match="Don't know how to create ISeq"):
        seq(42)

def test_indexed_seq():
    s = seq("bar")
    assert s.first() == "b"
    assert s.rest().first() == "a"
    assert s.next().first() == "a"
    assert s.next().next().first() == "r"
    assert s.next().next().next() is None
    assert s.seq() is s
    assert s.count_() == 3
    assert s == L("b", "a", "r")
    s1 = s.rest().rest()
    assert s1.first() == "r"
    assert s1.count_() == 1
    assert s1 == L("r")
    assert seq("baz") == ["b", "a", "z"]
    assert seq("baz") == ("b", "a", "z")

def test_atom():
    a = atom(42)
    assert isinstance(a, Atom)
    assert a.deref() == 42
    assert a.reset(43) == 43
    assert a.deref() == 43
    assert a.swap(lambda x: x + 1) == 44
    assert a.deref() == 44
    assert a.swap(lambda x, y: x + y, 2) == 46
    assert a.deref() == 46

def test_record():
    with pytest.raises(Exception, match="expects.*two arguments"):
        define_record()
    with pytest.raises(Exception, match="expects.*two arguments"):
        define_record("TestRecord")
    with pytest.raises(Exception, match="must be.*tuple"):
        define_record("TestRecord", "a")
    with pytest.raises(Exception, match="expects keyword"):
        define_record("TestRecord", ("a", "b"))
    with pytest.raises(Exception, match="expects string"):
        define_record("TestRecord", (K("a"), K("b")))
    with pytest.raises(Exception, match="have two elements"):
        define_record("TestRecord", ("a", "b", "c"))
    def _defrec(n, *fs):
        return define_record(n, *[(K(f), f) for f in fs])
    with pytest.raises(Exception, match="name must be a string"):
        _defrec(["TestRecord"], "a", "b")
    with pytest.raises(Exception, match="name must not be empty"):
        _defrec("", "a", "b")
    TestRecord = _defrec("dummy.TestRecord", "a", "b", "c")
    assert isinstance(TestRecord, type)
    r1 = TestRecord("foo", K("bar"), 42)
    assert r1.a == "foo"
    assert r1.b is K("bar")
    assert r1.c == 42
    with pytest.raises(AttributeError):
        r1.d # pylint: disable=pointless-statement
    assert r1.lookup(K("a"), None) == "foo"
    assert r1.lookup(K("b"), None) is K("bar")
    assert r1.lookup(K("c"), None) == 42
    assert r1.lookup(K("d"), 9001) == 9001
    r2 = r1.assoc(K("a"), "bar")
    assert r2.a == "bar"
    assert r2.b is K("bar")
    assert r2.c == 42
    assert r2.lookup(K("a"), None) == "bar"
    assert r2.lookup(K("b"), None) is K("bar")
    assert r2.lookup(K("c"), None) == 42
    assert r1.a == "foo"
    assert r1.b is K("bar")
    assert r1.c == 42
    assert r1.lookup(K("a"), None) == "foo"
    assert r1.lookup(K("b"), None) is K("bar")
    assert r1.lookup(K("c"), None) == 42
    r3 = r2.assoc(K("b"), S("bar"), K("c"), 9001)
    assert r3.a == "bar"
    assert r3.b == S("bar")
    assert r3.c == 9001
    assert r3.lookup(K("a"), None) == "bar"
    assert r3.lookup(K("b"), None) == S("bar")
    assert r3.lookup(K("c"), None) == 9001
    assert r2.a == "bar"
    assert r2.b is K("bar")
    assert r2.c == 42
    assert r2.lookup(K("a"), None) == "bar"
    assert r2.lookup(K("b"), None) is K("bar")
    assert r2.lookup(K("c"), None) == 42
    assert r1.a == "foo"
    assert r1.b is K("bar")
    assert r1.c == 42
    assert r1.lookup(K("a"), None) == "foo"
    assert r1.lookup(K("b"), None) is K("bar")
    assert r1.lookup(K("c"), None) == 42
    assert TestRecord(1, 2, 3) == TestRecord(1, 2, 3)
    assert r1.keys() == (K("a"), K("b"), K("c"))
    assert r2.keys() == (K("a"), K("b"), K("c"))
    assert r3.keys() == (K("a"), K("b"), K("c"))
