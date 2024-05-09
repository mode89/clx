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
    assert _eval("(lazy-seq nil)") == comp.list_()
    assert _eval("(lazy-seq '(1 2 3))") == comp.list_(1, 2, 3)
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

def test_is_nil(_eval):
    assert _eval("(nil? nil)") is True
    assert _eval("(nil? false)") is False
    assert _eval("(nil? true)") is False
    assert _eval("(nil? 42)") is False
    assert _eval("(nil? :hello)") is False
    assert _eval("(nil? '())") is False
    assert _eval("(nil? [])") is False
    assert _eval("(nil? (python/list))") is False

def test_is_string(_eval):
    assert _eval("(string? nil)") is False
    assert _eval("(string? false)") is False
    assert _eval("(string? true)") is False
    assert _eval("(string? 42)") is False
    assert _eval("(string? :hello)") is False
    assert _eval("(string? '())") is False
    assert _eval("(string? [])") is False
    assert _eval("(string? (python/list))") is False
    assert _eval("(string? \"hello\")") is True

def test_some(_eval):
    assert _eval("(some? nil)") is False
    assert _eval("(some? false)") is True
    assert _eval("(some? true)") is True
    assert _eval("(some? 42)") is True
    assert _eval("(some? :hello)") is True
    assert _eval("(some? '())") is True
    assert _eval("(some? [])") is True
    assert _eval("(some? (python/list))") is True

def test_not(_eval):
    assert _eval("(not nil)") is True
    assert _eval("(not false)") is True
    assert _eval("(not true)") is False
    assert _eval("(not 42)") is False
    assert _eval("(not :hello)") is False
    # TODO assert _eval("(not '())") is False
    # TODO assert _eval("(not [])") is False
    # TODO assert _eval("(not (python/list))") is False

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

def test_and(_eval):
    assert _eval("(and)") is True
    assert _eval("(and true)") is True
    assert _eval("(and false)") is False
    assert _eval("(and true true)") is True
    assert _eval("(and true false)") is False
    assert _eval("(and false true)") is False
    assert _eval("(and false false)") is False
    assert _eval("(and true true true)") is True
    assert _eval("(and true true false)") is False
    assert _eval("(and true false true)") is False
    assert _eval("(and true false false)") is False
    assert _eval("(and false true true)") is False
    assert _eval("(and false true false)") is False
    assert _eval("(and false false true)") is False
    assert _eval("(and false false false)") is False
    assert _eval("(and nil)") is None
    assert _eval("(and true nil)") is None
    assert _eval("(and nil false)") is None
    assert _eval("(and false nil)") is False
    assert _eval("(and nil nil)") is None
    assert _eval("(and true true nil)") is None

def test_or(_eval):
    assert _eval("(or)") is None
    assert _eval("(or true)") is True
    assert _eval("(or false)") is False
    assert _eval("(or true true)") is True
    assert _eval("(or true false)") is True
    assert _eval("(or false true)") is True
    assert _eval("(or false false)") is False
    assert _eval("(or true true true)") is True
    assert _eval("(or true true false)") is True
    assert _eval("(or true false true)") is True
    assert _eval("(or true false false)") is True
    assert _eval("(or false true true)") is True
    assert _eval("(or false true false)") is True
    assert _eval("(or false false true)") is True
    assert _eval("(or false false false)") is False
    assert _eval("(or nil)") is None
    assert _eval("(or true nil)") is True
    assert _eval("(or nil false)") is False
    assert _eval("(or false nil)") is None
    assert _eval("(or nil nil)") is None
    assert _eval("(or false false nil)") is None
    assert _eval("(or false nil false)") is False

def test_map(_eval):
    assert _eval("(map inc nil)") == comp.list_()
    assert _eval("(map inc '(1 2 3))") == comp.list_(2, 3, 4)
    assert _eval("(map inc [1 2 3])") == comp.list_(2, 3, 4)
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

    ls = _eval(
        f"""
        {DEFN_RANGE}
        (map (fn [x] (* x 3))
             (range* 10000000))
        """)
    assert ls.first() == 0
    assert ls.next().first() == 3

def test_filter(_eval):
    assert _eval("(filter nil nil)") == comp.list_()
    assert _eval("(filter odd? nil)") == comp.list_()
    assert _eval("(filter odd? '(1 2 3))") == comp.list_(1, 3)
    assert _eval("(filter odd? [1 2 3])") == comp.list_(1, 3)
    s = _eval(
        """
        (defn foo [x]
          (if (< x 3)
            (odd? x)
            (throw (Exception))))
        (filter foo [1 2 3])
        """)
    assert s.first() == 1
    with pytest.raises(Exception):
        s.next()

    ls = _eval(
        f"""
        {DEFN_RANGE}
        (filter odd? (range* 10000000))
        """)
    assert ls.first() == 1
    assert ls.next().first() == 3
    assert ls.next().next().first() == 5

def test_reduce(_eval):
    assert _eval("(reduce nil nil nil)") is None
    assert _eval("(reduce + '(42))") == 42
    assert _eval("(reduce + '(1 2 3))") == 6
    assert _eval("(reduce + [4 5 6])") == 15
    assert _eval("(reduce + 7 '(8 9 10))") == 34
    assert _eval(
        f"""
        {DEFN_RANGE}
        (reduce
          (fn [x y]
            (+ x y))
          42
          (range* 10))
        """) == 42 + sum(range(10))

def test_eval(_eval):
    assert _eval("(eval '(+ 1 2))") == 3
    assert _eval(
        """
        (def x 42)
        (eval '(+ x 5))
        """) == 47

def test_load_file(_eval):
    assert _eval(
        """
        (load-file "tests/examples/hello-world.clj")
        (hello-world)
        """) == comp.keyword("hello-world")

    _eval("(load-file \"tests/examples/load-file.clj\")")
    assert _eval("*file*") == "NO_SOURCE_PATH"
    assert _eval("*ns*") == "user"
    assert _eval("(example/message)") == comp.keyword("hello", "world")
    assert _eval("example/ns") == "example"
    assert _eval("example/file") == "tests/examples/load-file.clj"

def test_python_builtins(_eval):
    assert _eval("(python/str 42)") == "42"
    assert _eval("(python/abs -1234)") == 1234
    assert _eval("(python/max 1 5 3 2 4)") == 5
    assert _eval("(python/pow 2 10)") == 1024

def test_str(_eval):
    assert _eval("(str)") == ""
    assert _eval("(str 42)") == "42"
    assert _eval("(str 42 :hello)") == "42:hello"
    assert _eval("(str 42 :hello \"world\")") == "42:helloworld"
    assert _eval("(str \"hello\" nil :world)") == "hello:world"
    assert _eval("(str nil)") == ""
    assert _eval("(str nil nil)") == ""
    assert _eval("(str nil 'foo  nil)") == "foo"
    assert _eval("(str [1 2 3])") == "[1 2 3]"
    assert _eval("(str '(1 2 3))") == "(1 2 3)"
    assert _eval("(str {:a 1})") == "{:a 1}"

def test_instance(_eval):
    assert _eval("(instance? python/int 42)") is True
    assert _eval("(instance? python/int 42.0)") is False
    assert _eval("(instance? python/float 42.0)") is True
    assert _eval("(instance? python/float 42)") is False
    assert _eval("(instance? python/str \"hello\")") is True
    assert _eval("(instance? python/str 42)") is False

def test_re_find(_eval):
    assert _eval("(re-find #\"a\" \"hello\")") is None
    assert _eval("(re-find #\"l\" \"hello\")") == "l"
    assert _eval("(re-find #\"(e.)(l.)\" \"hello\")") == \
        comp.list_("ello", "el", "lo")
