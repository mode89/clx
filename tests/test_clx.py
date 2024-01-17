import pytest

import clx.main as clx

K = clx.keyword
S = clx.symbol
L = clx.list_
V = clx.vector
M = clx.hash_map

def test_keyword() -> None:
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

def test_symbol() -> None:
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

def test_record() -> None:
    Record = clx.define_record("Record", S("a"), S("b")) # pylint: disable=invalid-name
    _r1 = Record(1, 2)
    assert isinstance(_r1, Record)
    assert _r1.get(K("a")) == 1 # type: ignore
    assert _r1.get(K("b")) == 2 # type: ignore
    _r2 = _r1.assoc(K("a"), 3) # type: ignore
    assert isinstance(_r2, Record)
    assert _r2.get(K("a")) == 3 # type: ignore
    assert _r2.get(K("b")) == 2 # type: ignore
    assert _r1.get(K("a")) == 1 # type: ignore
    assert _r1.get(K("b")) == 2 # type: ignore
    _r3 = _r1.assoc(K("a"), 4, K("b"), 5) # type: ignore
    assert isinstance(_r3, Record)
    assert _r3.get(K("a")) == 4 # type: ignore
    assert _r3.get(K("b")) == 5 # type: ignore
    assert _r1.get(K("a")) == 1 # type: ignore
    assert _r1.get(K("b")) == 2 # type: ignore

def test_read_string() -> None:
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

def test_quasiquote() -> None:
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

def test_munge() -> None:
    assert clx.munge("foo") == "foo"
    assert clx.munge("foo.bar/baz") == "foo_DOT_bar_SLASH_baz"
    assert clx.munge("foo-bar.*baz*/+qux_fred!") == \
        "foo_bar_DOT__STAR_baz_STAR__SLASH__PLUS_qux_USCORE_fred_BANG_"
