# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from clx import list_, read_string1, symbol, vector # pylint: disable=import-error
import pytest

_s = symbol

def test_symbol():
    assert _s("a") is _s("a")
    assert _s("a") != _s("b")

def test_quasiquote():
    assert read_string1("`()") == list_()
    assert read_string1("`a") == list_(_s("quote"), _s("a"))
    assert read_string1("`~a") is _s("a")
    assert read_string1("`(a)") == \
        list_(_s("concat"), list_(_s("list"), list_(_s("quote"), _s("a"))))
    assert read_string1("`(~a)") == \
        list_(_s("concat"), list_(_s("list"), _s("a")))
    assert read_string1("`(~@a)") == list_(_s("concat"), _s("a"))
    assert read_string1("`(1 a ~b ~@c)") == \
        list_(_s("concat"),
            list_(_s("list"), 1),
            list_(_s("list"), list_(_s("quote"), _s("a"))),
            list_(_s("list"), _s("b")),
            _s("c"))
    assert read_string1("`[]") == vector()
    assert read_string1("`[1 a ~b ~@c]") == \
        list_(_s("vec"),
            list_(_s("concat"),
                list_(_s("list"), 1),
                list_(_s("list"), list_(_s("quote"), _s("a"))),
                list_(_s("list"), _s("b")),
                _s("c")))
    with pytest.raises(Exception, match=r"splice-unquote not in list"):
        read_string1("`~@a")
