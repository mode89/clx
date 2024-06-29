# pylint: disable=disallowed-name
# pylint: disable=protected-access
import re
from pathlib import Path

import pytest

import clx.bootstrap as clx
from clx.bootstrap import second, seq, lazy_seq, cons, \
    assoc, get, munge, nth, conj, map_, filter_, reduce, \
    _S_LIST, _S_VEC, _S_HASH_MAP, _S_CONCAT, _S_APPLY

K = clx.keyword
S = clx.symbol
L = clx.list_
V = clx.vector
M = clx.hash_map

TEST_DIR = Path(__file__).parent

def _make_test_context():
    box = [None]
    def set_box(value):
        box[0] = value
        return value
    def get_box():
        return box[0]

    return clx.init_context({
        "user": {
            "set-box": set_box,
            "get-box": get_box,
            "even?": lambda x: x % 2 == 0,
            "odd?": lambda x: x % 2 == 1,
        },
    })

def _eval(text):
    ctx = _make_test_context()
    return clx._eval_string(ctx, text)

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

def _throw():
    raise Exception()

def test_is_list():
    assert clx.is_list(None) is False
    assert clx.is_list(L()) is True
    assert clx.is_list(V()) is False
    assert clx.is_list(M()) is False
    assert clx.is_list([]) is False
    assert clx.is_list(()) is False
    assert clx.is_list({}) is False
    assert clx.is_list(set()) is False
    assert clx.is_list("hello") is False

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

def test_record():
    record = clx.define_record("dummy.TestRecord", K("a"), K("b"))
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
    assert clx.read_string("(1 2 3)") == L(1, 2, 3)
    with pytest.raises(Exception, match=r"Expected '\)'"):
        clx.read_string("(1 2 3")
    assert clx.read_string("[4 5 6]") == V(4, 5, 6)
    with pytest.raises(Exception, match=r"Expected '\]'"):
        clx.read_string("[4 5 6")
    assert clx.read_string("[eight]") == V(S("eight"))
    assert clx.read_string("{:a 7 \"b\" eight}") == \
        M(K("a"), 7, "b", S("eight"))
    with pytest.raises(Exception, match=r"Expected '\}'"):
        clx.read_string("{:a 7 \"b\" eight")
    assert clx.read_string("'x") == L(S("quote"), S("x"))
    assert clx.read_string("#\"a.*b+\")") == re.compile("a.*b+")

def test_quasiquote():
    assert clx.read_string("`()") == L()
    assert clx.read_string("`a") == L(S("quote"), S("a"))
    assert clx.read_string("`~a") == S("a")
    lconcat = lambda *args: \
        L(_S_APPLY, _S_LIST, L(_S_CONCAT, *args))
    assert clx.read_string("`(a)") == \
        lconcat(L(_S_LIST, L(S("quote"), S("a"))))
    assert clx.read_string("`(~a)") == lconcat(L(_S_LIST, S("a")))
    assert clx.read_string("`(~@a)") == lconcat(S("a"))
    assert clx.read_string("`(1 a ~b ~@c)") == \
        lconcat(
            L(_S_LIST, 1),
            L(_S_LIST, L(S("quote"), S("a"))),
            L(_S_LIST, S("b")),
            S("c"))
    assert clx.read_string("`[]") == L(_S_VEC, L(_S_CONCAT))
    assert clx.read_string("`[1 a ~b ~@c]") == \
        L(_S_VEC,
            L(_S_CONCAT,
                L(_S_LIST, 1),
                L(_S_LIST, L(S("quote"), S("a"))),
                L(_S_LIST, S("b")),
                S("c")))
    assert clx.read_string("`{}") == L(_S_APPLY, _S_HASH_MAP, L(_S_CONCAT))

    x = clx.read_string("`{1 a ~b ~@c}")
    assert \
        x == L(_S_APPLY, _S_HASH_MAP,
            L(_S_CONCAT,
                L(_S_LIST, 1),
                L(_S_LIST, L(S("quote"), S("a"))),
                L(_S_LIST, S("b")),
                S("c"))) or \
        x == L(_S_APPLY, _S_HASH_MAP,
            L(_S_CONCAT,
                L(_S_LIST, S("b")),
                S("c"),
                L(_S_LIST, 1),
                L(_S_LIST, L(S("quote"), S("a")))))

    assert re.fullmatch(r"\(quote x_\d+\)", clx.pr_str(clx.read_string("`x#")))
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
    assert clx.pr_str(_lazy_range(5, 10)) == "(5 6 7 8 9)"
    Record = clx.define_record("dummy.Record", K("a"), K("b"))
    assert clx.pr_str(Record("foo", 42)) == "#dummy.Record{:a \"foo\", :b 42}"

def test_munge():
    assert munge("foo") == "foo"
    assert munge("foo.bar/baz") == "foo_DOT_bar_SLASH_baz"
    assert munge("foo-bar.*baz*/+qux_fred!") == \
        "foo_DASH_bar_DOT__STAR_baz_STAR__SLASH__PLUS_qux_fred_BANG_"
    assert munge("if") == "if_"
    assert munge("def") == "def_"
    with pytest.raises(Exception, match=r"reserved"):
        munge("def_")

def test_trace_local_context():
    assert _eval("(___local_context :line)") == 1
    assert _eval("(___local_context :column)") == 1
    assert _eval("(___local_context :top_level?)") is True

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
    assert _eval("{:foo 42}") == M(K("foo"), 42)
    assert _eval("{:foo 42 \"bar\" :baz}") == M(K("foo"), 42, "bar", K("baz"))

def test_quote():
    assert _eval("'42") == 42
    assert _eval("'hello") == S("hello")
    assert _eval("':world") == K("world")
    assert _eval("'()") is L()
    assert _eval("'(1 :two \"three\" four)") == \
        L(1, K("two"), "three", S("four"))
    assert _eval("'[]") is V()
    assert _eval("'[1 :two \"three\" four]") == \
        V(1, K("two"), "three", S("four"))
    assert _eval("'{}") is M()
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

def test_def():
    ctx = _make_test_context()
    res = clx._eval_string(ctx, "(def foo 42)")
    assert res == 42
    assert clx.get_in(ctx.namespaces.deref(),
        L("user",
          K("bindings"),
          "foo")).py_name == munge("user/foo")
    assert ctx.py_globals[munge("user/foo")] == 42
    assert _eval("(def foo (___local_context :top_level?))") is True
    with pytest.raises(Exception, match=r"Symbol 'foo' not found"):
        _eval("foo")

def test_do():
    assert _eval("(do)") is None
    assert _eval("(do 1 2 3)") == 3
    assert _eval(
        """
        (do (def foo 42)
            foo)
        """) == 42
    assert _eval(
        """
        (do (set-box 9001)
            (get-box))
        """) == 9001
    assert _eval("(do (___local_context :top_level?))") is True

def test_call():
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
    with pytest.raises(Exception, match=r"'foo' not found"):
        _eval("(foo bar)")
    assert _eval("((fn* [] 42))") == 42

def test_let():
    assert _eval("(let [])") is None
    assert _eval("(let [a 42])") is None
    assert _eval("(let [a 1] a)") == 1
    assert _eval("(let [a 2 b 3] (+ a b))") == 5
    assert _eval("(let [a 4 b (+ a 5)] b)") == 9
    with pytest.raises(Exception, match=r"Symbol 'b' not found"):
        _eval("(let [a 1] b)")
    with pytest.raises(Exception, match=r"Symbol 'b' not found"):
        _eval(
            """
            (let [a 1]
              (do (let [b 2] b)
                  b))
            """)
    assert _eval("(let [a (___local_context :top_level?)] a)") is True

def test_cond():
    assert _eval("(cond)") is None
    assert _eval("(cond true 42)") == 42
    assert _eval("(cond false 42)") is None
    assert _eval("(cond true 1 true 2)") == 1
    assert _eval("(cond false 1 true 2)") == 2
    assert _eval("(cond false 1 false 2)") is None
    assert _eval(
        """
        (cond
          (let [a 1
                b 2]
            (even? (+ a b)))
          42

          (let [c 3
                d 4]
            (odd? (+ c d)))
          43)
        """) == 43
    assert _eval(
        """
        (cond
          (let [a 1
                b 2]
            (even? (+ a b)))
          42

          (let [c 3
                d 4]
            (even? (+ c d)))
          43)
        """) is None
    with pytest.raises(Exception, match=r"allowed only at top level"):
        _eval("(cond (def foo true) 42)")
    with pytest.raises(Exception, match=r"allowed only at top level"):
        _eval("(cond true (def foo 42))")
    with pytest.raises(Exception, match=r"allowed only at top level"):
        _eval("(cond true 42 (def foo true) 43)")
    with pytest.raises(Exception, match=r"allowed only at top level"):
        _eval("(cond true 42 true (def foo 43))")

def test_fn():
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
            (set-box 43)))
        (foo)
        (get-box)
        """) == 43
    assert _eval(
        """
        (def foo
          (fn* []
            (___local_context :top_level?)))
        (foo)
        """) is False
    with pytest.raises(Exception, match=r"allowed only at top level"):
        _eval("(fn* [] (def foo 1))")
    assert _eval("(fn* foo-bar/quux [] 42)").__name__ == munge("foo-bar/quux")
    assert _eval(
        """
        (def foo
          (fn* [x]
            (cond
              (python* x " > 10") x
              :else (foo (+ x x)))))
        (foo 1)
        """) == 16

def test_letfn():
    assert _eval("(letfn [])") is None
    assert _eval("(letfn [(foo [])])") is None
    assert _eval("(letfn [(foo [] 42)] (foo))") == 42
    assert _eval("(letfn [(foo [x] x)] (foo 43))") == 43
    assert _eval("(letfn [(foo [x y] (+ x y))] (foo 44 45))") == 89
    assert _eval(
        """
        (letfn [(foo [x y] (+ x y))
                (bar [x] (foo x x))]
          (bar 46))
        """) == 92
    assert _eval(
        """
        (letfn [(> [x y] (python* x " > " y))
                (foo [x]
                  (cond
                    (> x 10) x
                    :else (foo (+ x x))))]
          (foo 1))
        """) == 16
    assert _eval(
        """
        (letfn [(foo [x]
                  (cond
                    (< x 4) (bar (+ x 1))
                    :else 42))
                (bar [x]
                  (cond
                    (< x 4) (foo (+ x 1))
                    :else 43))
                (< [x y] (python* x " < " y))]
          [(foo 1) (bar 1)])
        """) == L(43, 42)

def test_try():
    assert _eval("(try nil)") is None
    assert _eval("(try 42)") == 42
    with pytest.raises(Exception, match=r"hello"):
        _eval("""
            (try
              (python* "raise Exception('hello')"))
        """)
    assert _eval(
        """
        (let [Exception (python* "Exception")
              foo (Exception "foo")]
          (try (python* "raise " foo)
            (catch Exception ex*
              (python* "str(" ex* ")"))))
        """) == "foo"
    assert _eval(
        """
        (def x nil)
        (try
          (python* x " = 42")
          (finally
            (python* x " += 1")))
        x
        """) == 43
    assert _eval(
        """
        (def Exception (python* "Exception"))
        (def RuntimeError (python* "RuntimeError"))
        (def x 5)
        (def y (try
                 :hello
                 (let [ex (Exception "world")]
                   (python* "raise " ex))
                 (python* x " *= 3")
                 x
                 (catch Exception ex
                   (python* x " += 1")
                   x)
                 (catch RuntimeError ex
                   (python* x " += 100")
                   42)
                 (finally
                   (python* x " *= 2")
                   x)))
        (list x y)
        """) == L(12, 6)

def test_loop():
    assert _eval(
        """
        (loop [])
        """) is None
    assert _eval(
        """
        (loop [x 1 y 2])
        """) is None
    assert _eval(
        """
        (loop [a 3 b 4]
          (+ a b))
        """) == 7
    assert _eval(
        """
        (loop [x 1]
          (cond
            (python* x "< 10") (recur (+ x 1))
            :else x))
        """) == 10
    assert _eval(
        """
        (loop [x 1
               xs (list)]
          (cond
            (python* x "< 5") (recur (+ x 1) (cons x xs))
            :else xs))
        """) == L(4, 3, 2, 1)
    assert _eval(
        """
        (loop [x 0
               y 0]
          (cond
            (python* x "< 4")
            (let [xy (loop [x 0
                            y (+ y 5)]
                       (cond (python* x "< 2") (recur (+ x 1) (+ y 1))
                             :else [x y]))]
              (recur (+ x (first xy)) (+ y (second xy))))
            :else [x y]))
        """) == L(4, 21)
    with pytest.raises(Exception, match=r"only in loop"):
        _eval("(recur 42)")
    with pytest.raises(Exception, match=r"only in loop"):
        _eval("(fn* [] (recur 42))")
    with pytest.raises(Exception, match=r"only in loop"):
        _eval("(loop [x 1] (fn* [] (recur 42)))")
    with pytest.raises(Exception, match=r"only in loop"):
        _eval("(loop [x 1] (recur (recur x)))")
    with pytest.raises(Exception, match=r"expects 3 arguments"):
        _eval("""(loop [x 1 y "hello" z :world] (recur 5 6))""")
    assert _eval(
        """
        (loop [x 1]
          (do 42
              (cond (python* x "< 10") (recur (+ x 1))
                    :else x)))
        """) == 10
    with pytest.raises(Exception, match=r"only in tail"):
        _eval("(loop [x 1] (do (recur x) 42))")
    with pytest.raises(Exception, match=r"only in tail"):
        _eval("(loop [x 1] (let [y (recur x)]))")
    with pytest.raises(Exception, match=r"only in tail"):
        _eval("(loop [x 1] (cond (recur x) 42))")
    with pytest.raises(Exception, match=r"only in tail"):
        _eval("(loop [x 1] ((recur x)))")
    with pytest.raises(Exception, match=r"only in tail"):
        _eval("(loop [x 1] (+ 42 (recur x)))")
    with pytest.raises(Exception, match=r"only in tail"):
        _eval("(loop [x 1] [42 (recur x)])")
    with pytest.raises(Exception, match=r"only in tail"):
        _eval("(loop [x 1] {42 (recur x)})")
    with pytest.raises(Exception, match=r"only at top level"):
        _eval("(loop [x 1] (def foo 42))")

def test_in_ns():
    ctx = _make_test_context()
    assert clx._eval_string(ctx,
        """
        (in-ns 'foo)
        @*ns*
        """) == "foo"

    ctx = _make_test_context()
    assert clx._eval_string(ctx,
        """
        (in-ns 'foo)
        (def bar 42)
        bar
        """) == 42

    ctx = _make_test_context()
    assert clx._eval_string(ctx,
        """
        (in-ns 'foo)
        (def bar 43)
        (in-ns 'baz)
        foo/bar
        """) == 43

def test_refer():
    assert _eval(
        """
        (refer* 'clx.core 'second 'my-second)
        (my-second '(1 42 3))
        """) == 42
    with pytest.raises(Exception,
            match=r"Symbol 'hundredth' not found in namespace 'clx.core'"):
        _eval("(refer* 'clx.core 'hundredth)")

def test_alias():
    assert _eval(
        """
        (alias 'core* 'clx.core)
        (core*/second '(1 42 3))
        """) == 42

def test_macros():
    assert _eval(
        """
        (def ^{:macro? true} foo
          (fn* []
            (list '+ 1 2)))
        (foo)
        """) == 3
    assert _eval(
        """
        (def ^{:macro? true} foo
          (fn* []
            '(+ 3 4 5)))
        (foo)
        """) == 12
    assert _eval(
        """
        (def ^{:macro? true} foo
          (fn* [x]
            (list '+ x x)))
        (foo 42)
        """) == 84
    assert _eval(
        """
        (def ^{:macro? true} foo
          (fn* [x]
            `(+ ~x 1 ~x ~x 2)))
        (foo 7)
        """) == 24
    assert _eval(
        """
        (def ^{:macro? true} foo
          (fn* [x y]
            (let [x2 [x x]
                  y3 [y y y]]
              `(+ ~@x2 4 ~@y3))))
        (foo 1 2)
        """) == 12
    assert _eval(
        """
        (def ^{:macro? true} bar
          (fn* [x]
            `(let [foo# ~x]
              (+ foo# foo#))))
        (bar 5)
        """) == 10
    with pytest.raises(Exception, match=r"'quux_\d+' not found"):
        _eval(
            """
            (def ^{:macro? true} foo
              (fn* []
                `(+ 1 quux#)))
            (def quux# 2)
            (foo)
            """)
    assert _eval(
        """
        (def ^{:macro? true} foo
          (fn* []
            `{42 (fn* [] 9001)}))
        (foo)
        """).lookup(42, None)() == 9001
    with pytest.raises(Exception, match=r"Can't take value of a macro"):
        _eval(
            """
            (def ^{:macro? true} foo
              (fn* []
                42))
            foo
            """)

def test_dot():
    assert _eval("(. 'hello/world -name)") == "world"
    assert callable(_eval("(. '(1 2 3) -first)"))
    assert _eval("(. '(1 2 3) first)") == 1
    assert _eval("(. {1 2 3 4} lookup 3 nil)") == 4
    assert _eval("(.-namespace 'hello/world)") == "hello"
    assert _eval("(.lookup {1 2 3 4} 1 nil)") == 2
    with pytest.raises(Exception, match=r"expected .*member.*target"):
        _eval("(.lookup)")

def test_import():
    mod = _eval("(import* collections)")
    assert mod is __import__("collections")
    assert _eval(
        """
        (import* builtins blts)
        (blts/str 42)
        """) == "42"

def test_python():
    assert _eval("(python*)") is None
    assert _eval("(python* \"42\")") == 42
    assert _eval(
        """
        (def foo 42)
        (python* foo)
        """) == 42
    assert _eval(
        """
        (def a 1)
        (def b 2)
        (python* a " + " b)
        """) == 3
    assert _eval(
        """
        (let [x 2
              y 3]
          (python*
            \"z = 4\\n\"
            x \" * \" y \" * z\"))
        """) == 24
    assert _eval(
        """
        (def foo
          (fn* []
            42))
        (python* foo "()")
        """) == 42
    assert _eval("(python* \"a = 42\")") is None

def test_python_with():
    assert _eval(
        """
        (def contextmanager
          (python* "__import__('contextlib').contextmanager"))
        (def foo (python* "[]"))
        (def cm
          (contextmanager
            (fn* []
              (do (.append foo 1)
                  (python* "yield " foo)
                  (.append foo 2)))))
        (.append foo
          (python/with [cm* (cm)]
            (.append cm* 3)
            (python* "sum(" foo ")")))
        foo
        """) == [1, 3, 2, 4]
    assert _eval(
        """
        (def Exception (python* "Exception"))
        (def contextmanager
          (python* "__import__('contextlib').contextmanager"))
        (def foo (python* "[]"))
        (def cm
          (contextmanager
            (fn* []
              (try (python* "yield " foo)
                (catch Exception ex
                  nil)))))
        (.append foo
          (python/with [cm* (cm)]
            (.append cm* 42)
            (let [ex (Exception "hello")]
              (python* "raise " ex))
            (python* "sum(" foo ")")))
        foo
        """) == [42, None]

def test_regex():
    with open(TEST_DIR / "examples" / "regex.clj", encoding="utf-8") as f:
        examples = f.readlines()
    assert _eval(examples[0]) == re.compile("[a-z]+")
    assert _eval(examples[1]) == re.compile(r"[\s]?")
    assert _eval(examples[2]) == re.compile(r"\\")
    assert _eval(examples[3]) == re.compile(r"\n")
    assert _eval(examples[4]) == re.compile(r"\d+")
    assert _eval(examples[5]) == re.compile(r"\w\"\\")
    assert _eval(examples[6]) == re.compile(r"[^\\\"]")
    assert re.compile(_eval(examples[7])).search(examples[8]) is not None

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
    bar = lambda: 42
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
    assert clx.meta(clx.with_meta(fred_with_meta, None)) is None
    assert clx.meta(clx.read_string("()")) is None
    assert clx.meta(clx.read_string("(1 2)")) is not None
    assert clx.meta(clx.read_string("[]")) is None
    assert clx.meta(clx.read_string("[1 2]")) is None
    assert clx.meta(clx.read_string("{}")) is None
    assert clx.meta(clx.read_string("{1 2}")) is None

def test_resolve_symbol():
    ctx = clx.Context(
        clx.atom(M(
            "clx.core",
                M(K("bindings"), M("list", clx.list_),
                  K("imports"),  M()),
            "user",
                M(K("bindings"), M("foo", 1,
                                   "bar", 2),
                  K("imports"),  M(),
                  K("aliases"),  M("bz", "baz")),
            "baz",
                M(K("bindings"), M("quux", 3,
                                   "fred", 4),
                  K("imports"),  M()),
                )),
        {
            munge("clx.core/*ns*"): clx.DynamicVar("user"),
        },
    )
    lctx = clx.LocalContext(M("a", 5, "bar", 6), None, False, True, 1, 1)
    resolve = lambda x: clx._resolve_symbol(ctx, lctx, x)
    assert resolve(S("a")) == 5
    assert resolve(S("bar")) == 6
    assert resolve(S("foo")) == 1
    assert resolve(S("baz/quux")) == 3
    assert resolve(S("baz/fred")) == 4
    assert resolve(S("user/foo")) == 1
    assert resolve(S("user/bar")) == 2
    assert resolve(S("bz/quux")) == 3
    with pytest.raises(Exception, match=r"Symbol 'user/baz' not found"):
        resolve(S("user/baz"))
    with pytest.raises(Exception, match=r"Symbol 'bar/foo' not found"):
        resolve(S("bar/foo"))
    assert resolve(S("list")) is clx.list_
    with pytest.raises(Exception, match=r"expected a symbol"):
        resolve(42)

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

def test_get_in():
    get_in = clx.get_in
    assert get_in(None, None) is None
    assert get_in(None, L("a")) is None
    assert get_in(None, L("a", "b")) is None
    assert get_in(M(), None) is M()
    assert get_in(M(), L("a")) is None
    assert get_in(M(), L("a", "b")) is None
    assert get_in(M("a", 1), None) == M("a", 1)
    assert get_in(M("a", 1), L("a")) == 1
    assert get_in(M("a", 1), L("b")) is None
    assert get_in(M("a", 1), L("b", "c")) is None
    assert get_in(M("a", M("b", 2)), L("a")) == M("b", 2)
    assert get_in(M("a", M("b", 2)), L("a", "b")) == 2
    assert get_in(M("a", M("b", 2)), L("a", "c")) is None
    assert get_in(M("a", M("b", 2)), L("b", "c"), 3) == 3
    assert get_in(M("a", M("b", 2)), L("a", "b", "c")) is None

def test_nth():
    assert nth(None, 0) is None
    assert nth(None, 1) is None
    assert nth(None, 2) is None
    with pytest.raises(IndexError):
        nth(L(), 0)
    with pytest.raises(IndexError):
        nth(L(), 1)
    with pytest.raises(IndexError):
        nth(L(), 2)
    assert nth(L(1), 0) == 1
    with pytest.raises(IndexError):
        nth(L(1), 1)
    with pytest.raises(IndexError):
        nth(L(1), 2)
    assert nth(L(1, 2), 0) == 1
    assert nth(L(1, 2), 1) == 2
    with pytest.raises(IndexError):
        nth(L(1, 2), 2)
    assert nth(L(1, 2), 2, 42) == 42
    assert nth(L(1, 2), -1, 42) == 42
    with pytest.raises(IndexError):
        nth(L(1, 2), -1)
    assert nth(V(1, 2, 3), 0) == 1
    assert nth(V(1, 2, 3), 1) == 2
    assert nth(V(1, 2, 3), 2) == 3
    with pytest.raises(IndexError):
        nth(V(1, 2, 3), 3)
    assert nth(V(1, 2, 3), 3, 42) == 42
    assert nth(V(1, 2, 3), -1, 42) == 42
    with pytest.raises(IndexError):
        nth(V(1, 2, 3), -1)
    assert nth(tuple(), 0, 42) == 42
    with pytest.raises(IndexError):
        nth(tuple(), 0)
    assert nth(tuple([1]), 0) == 1
    assert nth(tuple([1]), 0, 42) == 1
    assert nth(tuple([1, 2]), 1) == 2
    assert nth(tuple([1, 2]), 1, 42) == 2
    assert nth(tuple([1, 2]), 2, 42) == 42
    with pytest.raises(IndexError):
        nth(tuple([1, 2]), 2)
    assert nth(_lazy_range(0), 0, 42) == 42
    with pytest.raises(IndexError):
        nth(_lazy_range(0), 0)
    assert nth(_lazy_range(1), 0) == 0
    assert nth(_lazy_range(1), 0, 42) == 0
    assert nth(_lazy_range(2), 1) == 1
    assert nth(_lazy_range(2), 1, 42) == 1
    assert nth(_lazy_range(2), 2, 42) == 42
    with pytest.raises(IndexError):
        nth(_lazy_range(2), 2)
    assert nth(_lazy_range(10000000), -1, 42) == 42
    with pytest.raises(IndexError):
        nth(_lazy_range(10000000), -1)

def test_conj():
    assert conj(None, 42) == L(42)
    assert conj(L(), 42) == L(42)
    assert conj(L(1), 2) == L(2, 1)
    assert conj(L(1, 2), 3) == L(3, 1, 2)
    assert clx.is_list(conj(L(1, 2), 3))
    assert conj(V(), 42) == V(42)
    assert conj(V(1), 2) == V(1, 2)
    assert conj(V(1, 2), 3) == V(1, 2, 3)
    assert clx.is_vector(conj(V(1, 2), 3))
    assert conj(cons(1, None), 2) == L(2, 1)
    assert conj(cons(1, cons(2, None)), 3) == L(3, 1, 2)
    assert conj(lazy_seq(lambda: L(1, 2)), 3) == L(3, 1, 2)
    assert conj(seq(V(1, 2)), 3) == L(3, 1, 2)

def test_assoc():
    assert assoc(None, "a", 1) == M("a", 1)
    assert assoc(M("a", 2), "a", 3) == M("a", 3)
    assert assoc(M("a", 4), "b", 5) == M("a", 4, "b", 5)
    assert assoc(M("a", 6), "b", 7, "a", 8) == M("a", 8, "b", 7)

def test_assoc_in():
    assert clx.assoc_in(M("a", M("b", 1)), L("a", "b"), 2) == M("a", M("b", 2))
    assert clx.assoc_in(M("a", 3), L("b"), 4) == M("a", 3, "b", 4)
    assert clx.assoc_in(M("a", 5), L("b", "c"), 6) == M("a", 5, "b", M("c", 6))

def test_update():
    inc = lambda x: x + 1 if x is not None else 1
    assert clx.update(M("a", 1), "a", inc) == M("a", 2)
    assert clx.update(M("a", 2), "b", inc) == M("a", 2, "b", 1)
    assert clx.update(M("a", 3), "a", lambda x, y: x + y, 4) == M("a", 7)

def test_first():
    assert clx.first(None) is None
    assert clx.first(L()) is None
    assert clx.first(L(1)) == 1
    assert clx.first(L(2, 3)) == 2
    assert clx.first(V()) is None
    assert clx.first(V(1)) == 1
    assert clx.first(V(2, 3)) == 2
    assert clx.first(tuple()) is None

def test_map():
    inc = lambda x: x + 1

    assert map_(inc, None) == L()
    assert map_(inc, L(1, 2, 3)) == L(2, 3, 4)
    assert map_(inc, V(1, 2, 3)) == L(2, 3, 4)
    s = map_(lambda x: x if x < 3 else _throw(), V(1, 2, 3))
    assert s.first() == 1
    assert s.next().first() == 2
    with pytest.raises(Exception):
        s.next().next()
    assert s.first() == 1
    assert s.next().first() == 2
    s = map_(lambda x: x * 3, _lazy_range(10000000))
    assert s.first() == 0
    assert s.next().first() == 3

def test_filter():
    odd = lambda x: x % 2 == 1
    assert filter_(None, None) == L()
    assert filter_(odd, None) == L()
    assert filter_(odd, L(1, 2, 3)) == L(1, 3)
    assert filter_(odd, V(5, 6, 7, 8)) == L(5, 7)
    s = filter_(lambda x: odd(x) if x < 3 else _throw(), V(1, 2, 3))
    assert s.first() == 1
    with pytest.raises(Exception):
        s.next()
    assert s.first() == 1
    s = filter_(odd, _lazy_range(10000000))
    assert s.first() == 1
    assert s.next().first() == 3
    assert s.next().next().first() == 5

def test_reduce():
    add = lambda x, y: x + y
    assert reduce(None, None, None) is None
    assert reduce(add, L(42)) == 42
    assert reduce(add, L(1, 2, 3)) == 6
    assert reduce(add, V(4, 5, 6)) == 15
    assert reduce(add, 7, L(8, 9, 10)) == 34
    assert reduce(add, 42, _lazy_range(10)) == 42 + sum(range(10))

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

def test_eval_string():
    assert _eval("") is None

def test_slurp():
    with open("tests/examples/hello-world.clj", encoding="utf-8") as file:
        assert clx.slurp("tests/examples/hello-world.clj") == file.read()

def test_load_file():
    ctx = _make_test_context()
    clx._current_file(ctx).set("<no-file>")
    hello_world = clx.load_file(ctx, "tests/examples/hello-world.clj")
    assert hello_world() is K("hello-world")
    assert clx._eval_string(ctx, "@*file*") == "<no-file>"

    ctx = _make_test_context()
    clx._eval_string(ctx, "(load-file \"tests/examples/load-file.clj\")")
    assert clx._eval_string(ctx, "@*file*") == "NO_SOURCE_PATH"
    assert clx._eval_string(ctx, "@*ns*") == "user"
    assert clx._eval_string(ctx, "(example/message)") == K("hello", "world")
    assert clx._eval_string(ctx, "example/ns") == "example"
    assert clx._eval_string(ctx, "example/file") == "tests/examples/load-file.clj"

def test_current_file():
    ctx = _make_test_context()
    clx._current_file(ctx).set("some/file.clj")
    assert clx._eval_string(ctx, "@*file*") == "some/file.clj"

def test_count():
    assert clx.count(None) == 0
    assert clx.count(L()) == 0
    assert clx.count(L(1, 2, 3)) == 3
    assert clx.count(V()) == 0
    assert clx.count(V(1, 2, 3, 4, 5)) == 5
    assert clx.count(M()) == 0
    assert clx.count(M(1, 2, 3, 4)) == 2
    assert clx.count([1, 2, 3 ,4]) == 4
    assert clx.count((1, 2, 3, 4, 5, 6)) == 6
    assert clx.count(_lazy_range(0, 2000)) == 2000

def test_deref():
    assert _eval("(deref (atom 42))") == 42
    assert _eval("@(atom 43)") == 43

def test_walk():
    identity = lambda x: x
    inc = lambda x: x + 1
    assert clx.walk(None, identity, None) is None
    assert clx.walk(identity, identity, L()) == L()
    assert type(clx.walk(inc, identity, L(1, 2, 3))) is clx.PersistentList
    assert clx.walk(inc, identity, L(1, 2, 3)) == L(2, 3, 4)
    assert clx.meta(clx.walk(identity, identity,
        clx.with_meta(L(1, 2, 3), M("foo", 42)))) == M("foo", 42)
    assert type(clx.walk(inc, identity, V(4, 5, 6))) is clx.PersistentVector
    assert clx.walk(inc, identity, V(4, 5, 6)) == V(5, 6, 7)
    assert clx.walk(inc, sum, V(1, 2, 3)) == 9
    assert type(clx.walk(identity, identity, M("a", 1, K("b"), 2))) is clx.PersistentMap
    assert clx.walk(identity, identity, M("a", 1, K("b"), 2)) == M("a", 1, K("b"), 2)
    inner_map = lambda kv: (kv[0] + kv[0], inc(kv[1]))
    assert clx.walk(inner_map, identity, M("c", 7, "d", 8)) == M("cc", 8, "dd", 9)

def test_postwalk():
    identity = lambda x: x
    assert clx.postwalk(identity, None) is None
    assert clx.postwalk(identity, L()) == L()
    assert clx.postwalk(
        lambda x: x + 1 if isinstance(x, int) else x,
        L(1, 2, 3)) == L(2, 3, 4)
    assert clx.postwalk(
        lambda x: x + 1 if isinstance(x, int) else x,
        M("e", 9)) == M("e", 10)

def test_function_shorthand():
    # TODO (#()) == '()
    assert _eval("#(+ 1 2)")() == 3
    assert _eval("#(+ % %)")(4) == 8
    assert _eval("#(+ %1 %)")(5) == 10
    assert _eval("#(+ %1 %2)")(6, 42) == 48
    assert _eval("#(vector %5 %2)")(1, 2, 3, 4, 5) == V(5, 2)
    assert _eval("#(count %&)")(4, 5, 6) == 3
    assert _eval("#(+ %1 %2 (apply + %&))")(3, 4, 5, 6) == 18
    assert _eval("#(let [x % y %2] (+ x y %1))")(7, 8) == 22
    assert _eval("#(get {%1 %2 %3 %4} %5)")("a", 1, K("b"), 2, K("b")) == 2
    # TODO nesting
