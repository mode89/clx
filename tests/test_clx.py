import pytest

import clx.main as clx
from clx.main import assoc, get, munge

K = clx.keyword
S = clx.symbol
L = clx.list_
V = clx.vector
M = clx.hash_map

def _eval(text):
    return clx.eval_string(text)[0]

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
    assert str(K("foo/bar")) == "Keyword(foo, bar)"
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
    assert str(S("foo/bar")) == "Symbol(foo, bar)"

def test_hash_map():
    _m0 = M()
    assert isinstance(_m0, clx.PersistentMap)
    _m1 = assoc(_m0, "a", 1)
    assert _m0 is M()
    assert isinstance(_m1, clx.PersistentMap)
    assert get(_m1, "a") == 1

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
    assert clx.eval_string("(do)")[0] is None
    assert clx.eval_string("(do 1 2 3)")[0] == 3
    assert clx.eval_string(
        """
        (do (def foo 42)
            foo)
        """)[0] == 42

def test_eval_call():
    assert clx.eval_string("(+ 1 2)")[0] == 3
    assert clx.eval_string("(+ (def foo 3) (def bar 4))")[0] == 7
    assert clx.eval_string(
        """
        (+ (do (def a 5)
               (def b 6)
               (def c (+ a b)))
           (do (def d 7)
               (def e 8)
               (def f (+ a b c d e))))
        """)[0] == 48

def test_eval_let():
    assert clx.eval_string("(let* [])")[0] is None
    assert clx.eval_string("(let* [a 42])")[0] is None
    assert clx.eval_string("(let* [a 1] a)")[0] == 1
    assert clx.eval_string("(let* [a 2 b 3] (+ a b))")[0] == 5
    assert clx.eval_string("(let* [a 4 b (+ a 5)] b)")[0] == 9
    with pytest.raises(Exception, match=r"Symbol 'b' not found"):
        clx.eval_string("(let* [a 1] b)")
    with pytest.raises(Exception, match=r"Symbol 'b' not found"):
        clx.eval_string(
            """
            (let* [a 1]
              (do (let* [b 2] b)
                  b))
            """)

def test_eval_if():
    assert clx.eval_string("(if true 1 2)")[0] == 1
    assert clx.eval_string("(if false 1 2)")[0] == 2
    assert clx.eval_string(
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
        """)[0] == 7
    assert clx.eval_string(
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
        """)[0] == 11

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
    assert clx.eval_string(
        """
        (def forty-two 42)
        forty-two
        """)[0] == 42

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
    with pytest.raises(Exception, match=r"must be a sequence"):
        clx.apply(lambda x: x, 42)

def test_get():
    _m = M("a", 1, "b", 2)
    assert get(_m, "a") == 1
    assert get(_m, "b") == 2
    with pytest.raises(KeyError):
        get(_m, "c")
    assert get(_m, "c", 3) == 3

def test_assoc_in():
    assert clx.assoc_in(M("a", M("b", 1)), L("a", "b"), 2) == M("a", M("b", 2))
    assert clx.assoc_in(M("a", 1), L("b"), 2) == M("a", 1, "b", 2)
    with pytest.raises(KeyError):
        clx.assoc_in(M("a", 1), L("b", "c"), 2)
