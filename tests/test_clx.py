import pytest

import clx.main as clx
from clx.main import first, second, rest, next_, seq, lazy_seq, cons, \
    assoc, get, munge

K = clx.keyword
S = clx.symbol
L = clx.list_
V = clx.vector
M = clx.hash_map

def _eval(text):
    return clx.eval_string(text)[0]

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
    assert isinstance(hello, clx.Keyword)
    assert clx.is_keyword(hello)
    assert hello.name == "hello"
    assert hello.namespace is None
    assert hello is K("hello")
    hello_world = K("hello", "world")
    assert isinstance(hello_world, clx.Keyword)
    assert clx.is_keyword(hello_world)
    assert hello_world.name == "world"
    assert hello_world.namespace == "hello"
    assert hello_world is K("hello", "world")
    foo_bar = K("foo/bar")
    assert isinstance(foo_bar, clx.Keyword)
    assert clx.is_keyword(foo_bar)
    assert foo_bar.name == "bar"
    assert foo_bar.namespace == "foo"
    assert foo_bar is K("foo/bar")
    assert K(S("baz")) is K("baz")
    assert K(S("foo/bar")) is K("foo/bar")
    assert K(K("quux")) is K("quux")
    assert str(K("foo/bar")) == ":foo/bar"
    assert K(None, "foo") is K("foo")

def test_symbol():
    hello = S("hello")
    assert isinstance(hello, clx.Symbol)
    assert clx.is_symbol(hello)
    assert hello.name == "hello"
    assert hello.namespace is None
    assert hello == S("hello")
    hello_world = S("hello", "world")
    assert isinstance(hello_world, clx.Symbol)
    assert clx.is_symbol(hello_world)
    assert hello_world.name == "world"
    assert hello_world.namespace == "hello"
    assert hello_world == S("hello", "world")
    foo_bar = S("foo/bar")
    assert isinstance(foo_bar, clx.Symbol)
    assert clx.is_symbol(foo_bar)
    assert foo_bar.name == "bar"
    assert foo_bar.namespace == "foo"
    assert foo_bar == S("foo/bar")
    assert S(S("quux")) == S("quux")
    assert clx.is_simple_symbol(S("foo"))
    assert not clx.is_simple_symbol(S("foo/bar"))
    assert str(S("foo/bar")) == "foo/bar"

def test_list():
    assert isinstance(L(), clx.PersistentList)
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
    assert L(1, 2, 3) != [1, 2, 3]
    assert L(1, 2, 3) != (1, 2, 3)
    assert list(iter(L(1, 2, 3))) == [1, 2, 3]

def test_vector():
    v = V() # pylint: disable=invalid-name
    assert isinstance(v, clx.PersistentVector)
    assert v is V()
    assert len(v) == 0
    assert first(v) is None
    assert rest(v) is L()
    assert next_(v) is None
    with pytest.raises(IndexError):
        v[0] # pylint: disable=pointless-statement
    v1 = V(1) # pylint: disable=invalid-name
    assert len(v1) == 1
    assert first(v1) == 1
    assert rest(v1) is L()
    assert next_(v1) is None
    assert v1[0] == 1
    with pytest.raises(IndexError):
        v1[1] # pylint: disable=pointless-statement
    v23 = V(2, 3)
    assert v23 is not V()
    assert len(v23) == 2
    assert first(v23) == 2
    assert rest(v23) == L(3)
    assert next_(v23) == L(3)
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
    assert V(1, 2, 3) != [1, 2, 3]
    assert V(1, 2, 3) != (1, 2, 3)

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

def test_lazy_seq():
    def nth(coll, n): # pylint: disable=invalid-name
        for _ in range(n):
            coll = coll.rest()
        return coll.first()

    def _numbers(_realized, i):
        _realized.append(i)
        return clx.cons(i, lazy_seq(lambda: _numbers(_realized, i + 1)))
    realized_numbers = []
    numbers = lazy_seq(lambda: _numbers(realized_numbers, 0))
    assert first(numbers) == 0
    assert realized_numbers == [0]
    assert second(numbers) == 1
    assert realized_numbers == [0, 1]
    assert nth(numbers, 10) == 10
    assert realized_numbers == list(range(11))
    assert nth(numbers, 10000) == 10000
    assert realized_numbers == list(range(10001))

    def _range(_realized, end, i):
        if i < end:
            _realized.append(i)
            return clx.cons(i,
                lazy_seq(lambda: _range(_realized, end, i + 1)))
        return None
    realized_range = []
    range_ = lazy_seq(lambda: _range(realized_range, 3, 0))
    assert first(range_) == 0
    assert realized_range == [0]
    assert second(range_) == 1
    assert realized_range == [0, 1]
    assert nth(range_, 2) == 2
    assert realized_range == [0, 1, 2]
    assert nth(range_, 3) is None
    assert realized_range == [0, 1, 2]
    assert nth(range_, 4) is None
    assert realized_range == [0, 1, 2]
    assert first(next_(next_(range_))) == 2
    assert first(next_(next_(next_(range_)))) is None
    assert first(next_(next_(next_(next_(range_))))) is None

    def recur_lazy(coll):
        return lazy_seq(
            lambda: lazy_seq(
                lambda: lazy_seq(
                    lambda: lazy_seq(
                        lambda: lazy_seq(
                            lambda: coll)))))
    assert not bool(recur_lazy(L()))
    assert first(recur_lazy(L())) is None
    assert next_(recur_lazy(L())) is None
    assert rest(recur_lazy(L())) is L()
    assert bool(recur_lazy(L(42)))
    assert first(recur_lazy(L(42))) == 42
    assert next_(recur_lazy(L(42))) is None
    assert rest(recur_lazy(L(42))) is L()

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

def test_is_seq():
    assert clx.is_seq(None) is False
    assert clx.is_seq(L()) is True
    assert clx.is_seq(V()) is False
    assert clx.is_seq(M()) is False
    assert clx.is_seq([]) is False
    assert clx.is_seq(()) is False
    assert clx.is_seq({}) is False
    assert clx.is_seq(42) is False

def test_is_seqable():
    assert clx.is_seqable(None) is True
    assert clx.is_seqable(L()) is True
    assert clx.is_seqable(V()) is True
    assert clx.is_seqable(M()) is True
    assert clx.is_seqable([]) is True
    assert clx.is_seqable(()) is True
    assert clx.is_seqable({}) is True
    assert clx.is_seqable(42) is False

def test_hash_map():
    m0 = M() # pylint: disable=invalid-name
    m1 = assoc(m0, "a", 1) # pylint: disable=invalid-name
    m2 = assoc(m1, "b", 2) # pylint: disable=invalid-name
    m3 = m2.merge(M("c", 3, "d", 4)) # pylint: disable=invalid-name
    m4 = assoc(m3, "a", 5, "b", 6) # pylint: disable=invalid-name
    assert type(m0) is clx.PersistentMap
    assert m0 is M()
    assert type(m1) is clx.PersistentMap
    assert get(m1, "a") == 1
    assert m1 == M("a", 1)
    assert type(m2) is clx.PersistentMap
    assert m2 == M("a", 1, "b", 2)
    assert type(m3) is clx.PersistentMap
    assert m3 == M("a", 1, "b", 2, "c", 3, "d", 4)
    assert type(m4) is clx.PersistentMap
    assert m4 == M("a", 5, "b", 6, "c", 3, "d", 4)

def test_record():
    record = clx.define_record("TestRecord", K("a"), K("b"))
    _r1 = record(1, 2)
    assert isinstance(_r1, record)
    assert get(_r1, K("a")) == 1
    assert get(_r1, K("b")) == 2
    _r2 = assoc(_r1, K("a"), 3)
    assert isinstance(_r2, record)
    assert get(_r2, K("a")) == 3
    assert get(_r2, K("b")) == 2
    assert get(_r1, K("a")) == 1
    assert get(_r1, K("b")) == 2
    _r3 = assoc(_r1, K("b"), 4)
    assert isinstance(_r3, record)
    assert get(_r3, K("a")) == 1
    assert get(_r3, K("b")) == 4
    assert get(_r1, K("a")) == 1
    assert get(_r1, K("b")) == 2
    _r4 = assoc(_r3, K("a"), 5, K("b"), 6)
    assert isinstance(_r4, record)
    assert get(_r3, K("a")) == 1
    assert get(_r3, K("b")) == 4
    assert get(_r4, K("a")) == 5
    assert get(_r4, K("b")) == 6

def test_read_string():
    assert clx.read_string("1") == 1
    assert clx.read_string("1.23") == 1.23
    # assert clx.read_string("1.23e4") == 1.23e4 TODO
    assert clx.read_string("true") is True
    assert clx.read_string("false") is False
    assert clx.read_string("nil") is None
    assert clx.read_string("\"hello world\"") == "hello world"
    with pytest.raises(Exception, match=r"Unterminated string"):
        clx.read_string("\"hello world")
    assert clx.read_string(":hello") is K("hello")
    assert clx.read_string(":hello/world") is K("hello", "world")
    assert clx.read_string("hello") == S("hello")
    assert clx.read_string("hello/world") == S("hello", "world")
    assert clx.meta(clx.read_string("     \n  foo/bar")) == \
        M(K("line"), 2, K("column"), 3)
    assert clx.read_string("(1 2 3)") == L(1, 2, 3)
    with pytest.raises(Exception, match=r"Expected '\)'"):
        clx.read_string("(1 2 3")
    assert clx.read_string("[4 5 6]") == V(4, 5, 6)
    with pytest.raises(Exception, match=r"Expected '\]'"):
        clx.read_string("[4 5 6")
    assert clx.read_string("{:a 7 \"b\" eight}") == \
        M(K("a"), 7, "b", S("eight"))
    with pytest.raises(Exception, match=r"Expected '\}'"):
        clx.read_string("{:a 7 \"b\" eight")
    assert clx.read_string("'x") == L(S("quote"), S("x"))

def test_quasiquote():
    assert clx.read_string("`()") == L()
    assert clx.read_string("`a") == L(S("quote"), S("a"))
    assert clx.read_string("`~a") == S("a")
    assert clx.read_string("`(a)") == \
        L(S("concat"), L(S("list"), L(S("quote"), S("a"))))
    assert clx.read_string("`(~a)") == \
        L(S("concat"), L(S("list"), S("a")))
    assert clx.read_string("`(~@a)") == L(S("concat"), S("a"))
    assert clx.read_string("`(1 a ~b ~@c)") == \
        L(S("concat"),
            L(S("list"), 1),
            L(S("list"), L(S("quote"), S("a"))),
            L(S("list"), S("b")),
            S("c"))
    assert clx.read_string("`[]") == clx.vector()
    assert clx.read_string("`[1 a ~b ~@c]") == \
        L(S("vec"),
            L(S("concat"),
                L(S("list"), 1),
                L(S("list"), L(S("quote"), S("a"))),
                L(S("list"), S("b")),
                S("c")))
    with pytest.raises(Exception, match=r"splice-unquote not in list"):
        clx.read_string("`~@a")

def test_printer():
    assert clx.pr_str(None) == "nil"
    assert clx.pr_str(True) == "true"
    assert clx.pr_str(False) == "false"
    assert clx.pr_str("hello") == "hello"
    assert clx.pr_str("hello", True) == "\"hello\""
    assert clx.pr_str(42) == "42"
    assert clx.pr_str(K("hello")) == ":hello"
    assert clx.pr_str(K("hello", "world")) == ":hello/world"
    assert clx.pr_str(S("hello")) == "hello"
    assert clx.pr_str(S("hello", "world")) == "hello/world"
    assert clx.pr_str(L(1, True, "hello")) == "(1 true hello)"
    assert clx.pr_str(L(1, True, "hello"), True) == "(1 true \"hello\")"
    assert clx.pr_str(V(1, True, "hello")) == "[1 true hello]"
    assert clx.pr_str(V(1, True, "hello"), True) == "[1 true \"hello\"]"
    assert clx.pr_str(M("a", 1)) == "{a 1}"
    assert clx.pr_str(M("a", 1), True) == "{\"a\" 1}"

def test_munge():
    assert munge("foo") == "foo"
    assert munge("foo.bar/baz") == "foo_DOT_bar_SLASH_baz"
    assert munge("foo-bar.*baz*/+qux_fred!") == \
        "foo_bar_DOT__STAR_baz_STAR__SLASH__PLUS_qux_USCORE_fred_BANG_"

def test_eval_value():
    assert _eval("nil") is None
    assert _eval("true") is True
    assert _eval("false") is False
    assert _eval("42") == 42
    assert _eval("1.23") == 1.23
    assert _eval("\"hello world\"") == "hello world"
    assert _eval(":hello") is K("hello")
    assert _eval(":hello/world") is K("hello", "world")
    assert _eval("[]") == V()
    assert _eval("[1 2 3]") == V(1, 2, 3)
    assert _eval("[42 :foo \"bar\"]") == V(42, K("foo"), "bar")
    assert _eval("{}") == M()
    assert _eval("{1 2 3 4}") == M(1, 2, 3, 4)
    assert _eval("{:foo 42 \"bar\" :baz}") == M(K("foo"), 42, "bar", K("baz"))

def test_eval_quote():
    assert _eval("'42") == 42
    assert _eval("'hello") == S("hello")
    assert _eval("':world") == K("world")
    assert _eval("'()") == L()
    assert _eval("'(1 :two \"three\" four)") == \
        L(1, K("two"), "three", S("four"))
    assert _eval("'[]") == V()
    assert _eval("'[1 :two \"three\" four]") == \
        V(1, K("two"), "three", S("four"))
    assert _eval("'{}") == M()
    assert _eval("'{1 :two \"three\" four}") == \
        M(1, K("two"), "three", S("four"))
    assert _eval(
        """
        '(([1 :two] {\"three\" four})
          [(5 :six) {seven \"eight\"}]
          {:nine (ten 11) 12 [:thirteen \"fourteen\"]})
        """) == \
        L(L(V(1, K("two")), M("three", S("four"))),
          V(L(5, K("six")), M(S("seven"), "eight")),
          M(K("nine"), L(S("ten"), 11), 12, V(K("thirteen"), "fourteen")))

def test_eval_def():
    res, ctx, glob = clx.eval_string("(def foo 42)")
    assert res == 42
    assert clx.get_in(ctx,
        L(K("namespaces"),
          "user",
          K("bindings"),
          "foo",
          K("py-name"))) == munge("user/foo")
    assert glob[munge("user/foo")] == 42

def test_eval_do():
    assert _eval("(do)") is None
    assert _eval("(do 1 2 3)") == 3
    assert _eval(
        """
        (do (def foo 42)
            foo)
        """) == 42
    assert _eval(
        """
        (def foo
          (fn* []
            (def bar 42)))
        (do (foo)
            bar)
        """) == 42

def test_eval_call():
    assert _eval("(+ 1 2)") == 3
    assert _eval("(+ (def foo 3) (def bar 4))") == 7
    assert _eval(
        """
        (+ (do (def a 5)
               (def b 6)
               (def c (+ a b)))
           (do (def d 7)
               (def e 8)
               (def f (+ a b c d e))))
        """) == 48

def test_eval_let():
    assert _eval("(let* [])") is None
    assert _eval("(let* [a 42])") is None
    assert _eval("(let* [a 1] a)") == 1
    assert _eval("(let* [a 2 b 3] (+ a b))") == 5
    assert _eval("(let* [a 4 b (+ a 5)] b)") == 9
    with pytest.raises(Exception, match=r"Symbol 'b' not found"):
        _eval("(let* [a 1] b)")
    with pytest.raises(Exception, match=r"Symbol 'b' not found"):
        _eval(
            """
            (let* [a 1]
              (do (let* [b 2] b)
                  b))
            """)

def test_eval_if():
    assert _eval("(if true 1 2)") == 1
    assert _eval("(if false 1 2)") == 2
    assert _eval(
        """
        (if (let* [a 1
                   b 2]
              (odd? (+ a b)))
          (let* [c 3
                 d 4]
            (+ c d))
          (let* [e 5
                 f 6]
            (+ e f)))
        """) == 7
    assert _eval(
        """
        (if (let* [a 1
                   b 2]
              (even? (+ a b)))
          (let* [c 3
                 d 4]
            (+ c d))
          (let* [e 5
                 f 6]
            (+ e f)))
        """) == 11

def test_eval_fn():
    assert _eval("(fn* [])")() is None
    assert _eval("(fn* [] 1)")() == 1
    assert _eval("(fn* [x] 2)")(3) == 2
    assert _eval("(fn* [x y] (+ x y))")(4, 5) == 9
    assert _eval("(fn* [x y & z] (apply + x y z))")(1, 2, 3, 4, 5) == 15
    assert _eval("(fn* [& x] (apply + x))")(1, 2, 3, 4) == 10
    assert _eval(
        """
        (def foo (fn* [x y] (+ x y)))
        (foo 1 2)
        """) == 3
    with pytest.raises(Exception, match=r"Symbol 'x' not found"):
        _eval(
            """
            (def foo (fn* [x y] (+ x y)))
            x
            """)
    assert _eval(
        """
        (def x 1)
        (def y 2)
        (def foo
          (fn* [x]
            (+ x y)))
        (foo 3)
        """) == 5
    assert _eval(
        """
        (def foo
          (fn* []
            (def bar 42)))
        (foo)
        bar
        """) == 42

def test_eval_python():
    assert _eval("(___python)") is None
    assert _eval("(___python \"42\")") == 42
    assert _eval(
        """
        (def foo 42)
        (___python foo)
        """) == 42
    assert _eval(
        """
        (def a 1)
        (def b 2)
        (___python a " + " b)
        """) == 3
    assert _eval(
        """
        (let* [x 2
               y 3]
          (___python
            \"z = 4\\n\"
            x \" * \" y \" * z\"))
        """) == 24
    assert _eval(
        """
        (def foo
          (fn* []
            42))
        (___python foo "()")
        """) == 42
    assert _eval("(___python \"a = 42\")") is None

def test_meta():
    assert clx.meta(None) is None
    assert clx.meta(True) is None
    assert clx.meta(42) is None
    with pytest.raises(Exception, match=r"does not support metadata"):
        clx.with_meta(42, M())
    with pytest.raises(Exception, match=r"expects.*PersistentMap"):
        clx.with_meta(L(), 42)
    quux = clx.read_string("(def ^{:foo 42} ^{:bar 43} quux :fred)")
    assert quux.count_() == 3
    assert clx.meta(second(quux)).get(K("foo")) == 42
    assert clx.meta(second(quux)).get(K("bar")) == 43
    bar = lambda: 42 # pylint: disable=disallowed-name,unnecessary-lambda-assignment
    bar_with_meta = clx.with_meta(bar, M(K("quux"), 43))
    assert bar() == 42
    assert bar_with_meta() == 42
    assert clx.meta(bar) is None
    assert clx.meta(bar_with_meta) == M(K("quux"), 43)
    def fred():
        return 43
    fred_with_meta = clx.with_meta(fred, M("foo", 42))
    assert fred() == 43
    assert fred_with_meta() == 43
    assert clx.meta(fred) is None
    assert clx.meta(clx.vary_meta(fred_with_meta, assoc, K("bar"), 44)) == \
        M("foo", 42, K("bar"), 44)
    assert clx.meta(fred_with_meta) == M("foo", 42)

def test_resolve_symbol():
    resolve = clx._resolve_symbol # pylint: disable=protected-access
    ctx = clx.Context(
        current_ns="user",
        namespaces=M(
            "user",
                clx.Namespace(
                    bindings=M(
                        "foo", 1,
                        "bar", 2)),
            "baz",
                clx.Namespace(
                    bindings=M(
                        "quux", 3,
                        "fred", 4))),
        locals=M(
            "a", 5,
            "bar", 6),
        counter=0)
    assert resolve(ctx, S("a")) == 5
    assert resolve(ctx, S("bar")) == 6
    assert resolve(ctx, S("foo")) == 1
    assert resolve(ctx, S("baz/quux")) == 3
    assert resolve(ctx, S("baz/fred")) == 4
    assert resolve(ctx, S("user/foo")) == 1
    assert resolve(ctx, S("user/bar")) == 2
    with pytest.raises(Exception, match=r"Symbol 'user/baz' not found"):
        resolve(ctx, S("user/baz"))
    with pytest.raises(Exception, match=r"Namespace 'bar' not found"):
        resolve(ctx, S("bar/foo"))
    assert _eval(
        """
        (def forty-two 42)
        forty-two
        """) == 42

def test_apply():
    assert clx.apply(lambda: 42, []) == 42
    assert clx.apply(lambda x: x, [42]) == 42
    assert clx.apply(lambda x, y: x + y, [1, 2]) == 3
    assert clx.apply(lambda x, y, z: x + y + z, V(1, 2, 3)) == 6
    assert clx.apply(lambda x, y, z: x + y + z, L(1, 2, 3)) == 6
    with pytest.raises(Exception, match=r"expects a function"):
        clx.apply(42, [])
    with pytest.raises(Exception, match=r"expects at least 2 arguments"):
        clx.apply(lambda x: x)
    with pytest.raises(Exception, match=r"must be iterable"):
        clx.apply(lambda x: x, 42)

def test_get():
    _m = M("a", 1, "b", 2)
    assert get(_m, "a") == 1
    assert get(_m, "b") == 2
    assert get(_m, "c") is None
    assert get(_m, "c", 3) == 3
    assert get(41, "a") is None
    assert get(42, "b", 9001) == 9001

def test_assoc():
    assert assoc(None, "a", 1) == M("a", 1)
    assert assoc(M("a", 2), "a", 3) == M("a", 3)
    assert assoc(M("a", 4), "b", 5) == M("a", 4, "b", 5)
    assert assoc(M("a", 6), "b", 7, "a", 8) == M("a", 8, "b", 7)

def test_assoc_in():
    assert clx.assoc_in(M("a", M("b", 1)), L("a", "b"), 2) == M("a", M("b", 2))
    assert clx.assoc_in(M("a", 3), L("b"), 4) == M("a", 3, "b", 4)
    assert clx.assoc_in(M("a", 5), L("b", "c"), 6) == M("a", 5, "b", M("c", 6))

def test_first():
    assert clx.first(None) is None

def test_concat():
    concat = clx.concat
    assert concat() == L()
    assert concat(None) == L()
    assert concat(None, None) == L()
    assert concat(L()) == L()
    assert concat(L(), L()) == L()
    assert concat(L(1)) == L(1)
    assert concat(L(1, 2, 3)) == L(1, 2, 3)
    assert concat(L(1), L(2)) == L(1, 2)
    assert concat(L(1), L(2), L(3)) == L(1, 2, 3)
    assert concat(L(1, 2), L(3, 4)) == L(1, 2, 3, 4)
    assert concat(L(1, 2), L(3, 4), L(5, 6)) == L(1, 2, 3, 4, 5, 6)
    assert concat(L(1, 2), None, L(3, 4), None, L(5, 6)) == L(1, 2, 3, 4, 5, 6)
    assert concat(L(1, 2), None, V(3, 4), None, [5, 6]) == L(1, 2, 3, 4, 5, 6)
    assert concat(_lazy_range(1, 1500), range(1500, 3000)) == \
        seq(range(1, 3000))

def test_merge():
    assert clx.merge() is None
    assert clx.merge(None) is None
    assert clx.merge(None, None) is None
    assert clx.merge(M()) == M()
    assert clx.merge(M(), None) == M()
    assert clx.merge(None, M()) == M()
    assert clx.merge(M(), M()) == M()
    assert clx.merge(M("a", 1)) == M("a", 1)
    assert clx.merge(M("a", 1), None) == M("a", 1)
    assert clx.merge(None, M("a", 1)) == M("a", 1)
    assert clx.merge(M("a", 1), M("b", 2)) == M("a", 1, "b", 2)
    assert clx.merge(M("a", 1), M("a", 2)) == M("a", 2)
    assert clx.merge(M("a", 1), M("a", 2), M("a", 3)) == M("a", 3)
    assert clx.merge(M("a", 1), M("b", 2), M("a", 3)) == M("a", 3, "b", 2)
