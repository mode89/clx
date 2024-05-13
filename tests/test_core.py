# pylint: disable=disallowed-name
# pylint: disable=protected-access
import pytest

import clx
import clx.compiler as comp

DEFN_RANGE = """
  (defn range*
    ([end] (range* 0 end))
    ([start end]
      (lazy-seq
        (when (< start end)
          (cons start (range* (+ start 1) end))))))
"""

TEST_CONTEXT = clx.init_context()

@pytest.fixture
def _eval():
    namespaces = TEST_CONTEXT.namespaces.deref()
    _globals = TEST_CONTEXT.py_globals.copy()

    def impl(text):
        return comp._eval_string(TEST_CONTEXT, text)

    yield impl

    TEST_CONTEXT.namespaces.reset(namespaces)
    TEST_CONTEXT.py_globals.clear()
    TEST_CONTEXT.py_globals.update(_globals)

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
    assert foo(5, 6, 7, 8) == comp.list_(5, 6, 7, 8)

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
    assert foo(5, 6, 7, 8) == comp.list_(5, 6, 7, 8)

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
        """) == comp.list_(0, 1, 5, 15, 45)
