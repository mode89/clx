# pylint: disable=protected-access
import threading

import pytest

import clx
import clx.bootstrap as bs

def clone_context():
    ctx = clx.bootstrap_context
    return bs.Context(
        shared=bs.SharedContext(
            lock=threading.Lock(),
            namespaces=bs.Box(ctx.shared.namespaces.deref()),
            py_globals=ctx.shared.py_globals.copy(),
            counter=bs.Box(ctx.shared.counter.deref()),
        ),
        local=bs.LocalContext(
            env=bs.hash_map(),
            top_level_QMARK_=True,
            line=1,
            column=1,
        ),
        current_ns=bs.Box("user"),
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

def test_defmacro(_eval):
    assert _eval(
        """
        (defmacro if* [pred then else]
          `(cond ~pred ~then true ~else))
        (if* true
          42
          (throw (Exception)))
        """) == 42

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
