from pathlib import Path
import sys
import types

import clx.bootstrap as bs

PKG_DIR = Path(__file__).parent

bootstrap_context = bs.init_context({})
bs.load_file(bootstrap_context, PKG_DIR / "core.clj")

_globals = bootstrap_context.deref().shared.py_globals
core_ns = bootstrap_context.deref().shared.namespaces.lookup("clx.core", None)
core_mod = types.ModuleType("clx.core")
for name, binding in core_ns.bindings.items():
    munged_name = bs.munge(name)
    core_mod.__dict__[munged_name] = _globals[binding.py_name]
sys.modules["clx.core"] = core_mod
