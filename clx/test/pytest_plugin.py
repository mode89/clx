import pytest

import clx
import clx.compiler as comp

def pytest_collect_file(file_path, parent):
    if file_path.name.endswith("-test.clj"):
        return TestFile.from_parent(parent, path=file_path)

class TestFile(pytest.File):
    def collect(self):
        ctx = clx.DEFAULT_CONTEXT
        namespaces0 = ctx.namespaces.deref()
        comp.load_file(ctx, self.path)
        namespaces1 = ctx.namespaces.deref()
        items = []
        for ns_name, ns in namespaces1.items():
            if ns_name not in namespaces0 and ns_name.endswith("-test"):
                bindings = ns.lookup(comp.keyword("bindings"), None)
                for name, binding in bindings.items():
                    if name.startswith("test-"):
                        func = ctx.py_globals[binding.py_name]
                        items.append(
                            pytest.Function.from_parent(
                                self, name=name[5:], callobj=func))
        return items
