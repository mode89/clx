# pylint: disable=disallowed-name
# pylint: disable=protected-access
import pytest

import clx
import clx.bootstrap as bs

def clone_context():
    ctx = clx.bootstrap_context.deref()
    sctx = bs.update(ctx.shared, bs._K_PY_GLOBALS, lambda x: x.copy())
    return bs.Box(
        bs.assoc(ctx,
            bs._K_SHARED, sctx,
            bs._K_CURRENT_NS, "user",
        )
    )

@pytest.fixture
def _eval():
    def impl(text):
        return bs._load_string(clone_context(), "<string>", text)
    return impl

def test_throw(_eval):
    with pytest.raises(Exception, match="foo"):
        _eval("(throw (Exception \"foo\"))")

def test_fn(_eval):
    assert _eval("((fn [] 42))") == 42
    assert _eval(
        """
        (def f (fn inc [x] (+ x 1)))
        (f 9001)
        """) == 9002
    assert _eval(
        """
        (def foo
          (fn [x y]
            (+ x 3)
            (+ y 7)))
        (foo 1 2)
        """) == 9
    foo = _eval(
        """
        (fn
          ([] 42)
          ([x] (+ x 1))
          ([x y] (+ x y))
          ([x y & zs] (apply list x y zs)))
        """)
    assert foo() == 42
    assert foo(1) == 2
    assert foo(3, 4) == 7
    assert foo(5, 6, 7, 8) == bs.list_(5, 6, 7, 8)

def test_defn(_eval):
    assert _eval(
        """
        (defn add [x y] (+ x y))
        (add 1 2)
        """) == 3
    assert _eval(
        """
        (defn bar [x y]
          (+ x 10)
          (+ y 20))
        (bar 3 5)
        """) == 25
    foo = _eval(
        """
        (defn foo
          ([] 42)
          ([x] (+ x 1))
          ([x y] (+ x y))
          ([x y & zs] (apply list x y zs)))
        foo
        """)
    assert foo() == 42
    assert foo(1) == 2
    assert foo(3, 4) == 7
    assert foo(5, 6, 7, 8) == bs.list_(5, 6, 7, 8)

def test_defmacro(_eval):
    assert _eval(
        """
        (defmacro if* [pred then else]
          `(cond ~pred ~then true ~else))
        (if* true
          42
          (throw (Exception)))
        """) == 42
    assert _eval(
        """
        (defmacro +
          ([] 0)
          ([x] x)
          ([x y] `(let [x# ~x y# ~y] (python* x# " + " y#)))
          ([x y & more] `(+ (+ ~x ~y) ~@more)))
        [(+) (+ 1) (+ 2 3) (+ 4 5 6) (+ 7 8 9 10 11)]
        """) == bs.list_(0, 1, 5, 15, 45)

def test_let(_eval):
    assert _eval(
        """
        (let [x 42]
          x)
        """) == 42
    assert _eval(
        """
        (let [x 42
              y 9001]
          (+ x y)
          (+ y 10))
        """) == 9011
    with pytest.raises(Exception, match="even number"):
        _eval("(let [x 42 y] x)")
    with pytest.raises(Exception, match="vector"):
        _eval("(let (x 42) x)")

def test_set_bang(_eval):
    assert _eval(
        """
        (def foo
          (python*
            "from types import SimpleNamespace\n"
            "SimpleNamespace()"))
        (set! foo bar 42)
        foo
        """).bar == 42

def test_if(_eval):
    assert _eval(
        """
        (if true
          42
          (throw (Exception)))
        """) == 42
    assert _eval(
        """
        (if false
          (throw (Exception))
          43)
        """) == 43

def test_if_not(_eval):
    assert _eval(
        """
        (if-not false
          42
          (throw (Exception)))
        """) == 42
    assert _eval(
        """
        (if-not true
          (throw (Exception))
          43)
        """) == 43

def test_when(_eval):
    assert _eval(
        """
        (when true
          42)
        """) == 42
    assert _eval(
        """
        (when false
          (throw (Exception)))
        """) is None

def test_when_not(_eval):
    assert _eval(
        """
        (when-not false
          42)
        """) == 42
    assert _eval(
        """
        (when-not true
          (throw (Exception)))
        """) is None

def test_when_let(_eval):
    assert _eval(
        """
        (when-let [x 42]
          x)
        """) == 42
    assert _eval(
        """
        (when-let [x nil]
          (throw (Exception)))
        """) is None

def test_assert(_eval):
    assert _eval(
        """
        (assert true)
        42
        """) == 42
    with pytest.raises(Exception):
        _eval("(assert false)")
    with pytest.raises(Exception, match="Hello, World!"):
        _eval("(assert false \"Hello, World!\")")

def test_lazy_seq(_eval):
    assert _eval("(lazy-seq nil)") == bs.list_()
    assert _eval("(lazy-seq '(1 2 3))") == bs.list_(1, 2, 3)
    assert _eval(
        """
        (defn foo [x]
          (lazy-seq (cons x (foo (+ x 1)))))
        (first (next (next (next (foo 1)))))
        """) == 4
    s = _eval(
        """
        (lazy-seq
          (cons 42
            (lazy-seq
              (cons 9001
                (lazy-seq
                  (throw (Exception)))))))
        """)
    assert s.first() == 42
    assert s.rest().first() == 9001
    s.rest().rest()
    with pytest.raises(Exception):
        s.rest().rest().first()
    assert s.next().first() == 9001
    with pytest.raises(Exception):
        s.next().next()

def test_even(_eval):
    assert _eval("(even? 0)") is True
    assert _eval("(even? 1)") is False
    assert _eval("(even? 2)") is True
    assert _eval("(even? 3)") is False

def test_odd(_eval):
    assert _eval("(odd? 0)") is False
    assert _eval("(odd? 1)") is True
    assert _eval("(odd? 2)") is False
    assert _eval("(odd? 3)") is True

def test_inc(_eval):
    assert _eval("(inc 0)") == 1
    assert _eval("(inc 1)") == 2
    assert _eval("(inc 2)") == 3
    assert _eval("(inc -2)") == -1

def test_dec(_eval):
    assert _eval("(dec 0)") == -1
    assert _eval("(dec 1)") == 0
    assert _eval("(dec 2)") == 1
    assert _eval("(dec -2)") == -3

def test_operators(_eval):
    assert _eval("(+ 1 2)") == 3
    assert _eval("(- 3 4)") == -1
    assert _eval("(* 5 6)") == 30
    assert _eval("(/ 7 8)") == 7 / 8
    assert _eval("(= 9 10)") is False
    assert _eval("(= 11 11)") is True
    assert _eval("(< 12 13)") is True
    assert _eval("(< 14 14)") is False
    assert _eval("(< 16 15)") is False
    assert _eval("(<= 17 18)") is True
    assert _eval("(<= 19 19)") is True
    assert _eval("(<= 21 20)") is False
    assert _eval("(> 22 23)") is False
    assert _eval("(> 24 24)") is False
    assert _eval("(> 26 25)") is True
    assert _eval("(>= 27 28)") is False
    assert _eval("(>= 29 29)") is True
    assert _eval("(>= 31 30)") is True

def test_map(_eval):
    assert _eval("(map inc nil)") == bs.list_()
    assert _eval("(map inc '(1 2 3))") == bs.list_(2, 3, 4)
    assert _eval("(map inc [1 2 3])") == bs.list_(2, 3, 4)
    s = _eval(
        """
        (defn foo [x]
          (if (< x 3)
            x
            (throw (Exception))))
        (map foo [1 2 3])
        """)
    assert s.first() == 1
    assert s.next().first() == 2
    with pytest.raises(Exception):
        s.next().next()
