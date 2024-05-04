from pathlib import Path
import sys
import types

import clx.compiler as comp

PKG_DIR = Path(__file__).parent

def init_context():
    ctx = comp.init_context({})
    comp.load_file(ctx, PKG_DIR / "core.clj")
    return ctx

DEFAULT_CONTEXT = init_context()

_globals = DEFAULT_CONTEXT.py_globals
core_ns = DEFAULT_CONTEXT.namespaces.deref().lookup("clx.core", None)
core_mod = types.ModuleType("clx.core")
for name, binding in core_ns.lookup(comp.keyword("bindings"), None).items():
    munged_name = comp.munge(name)
    core_mod.__dict__[munged_name] = _globals[binding.py_name]
sys.modules["clx.core"] = core_mod
