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
        (defmacro if* [pred a b]
          `(if ~pred ~a ~b))
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
