from pathlib import Path
import sys
import types

import clx.bootstrap as boot

PKG_DIR = Path(__file__).parent

def init_context():
    ctx = boot.init_context({})
    boot.load_file(ctx, PKG_DIR / "core.clj")
    return ctx

DEFAULT_CONTEXT = init_context()

def generate_module(ns_name):
    _globals = DEFAULT_CONTEXT.py_globals
    namespaces = DEFAULT_CONTEXT.namespaces.deref()
    assert ns_name in namespaces, f"Namespace {ns_name} not found"
    ns = namespaces.lookup(ns_name, None)
    mod = types.ModuleType(ns_name)
    for name, binding in ns.lookup(boot.keyword("bindings"), None).items():
        munged_name = boot.munge(name)
        mod.__dict__[munged_name] = _globals[binding.py_name]
    sys.modules[ns_name.replace("-", "_")] = mod
