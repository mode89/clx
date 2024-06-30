import clx_rust as rust
from timeit import timeit

def nums():
    def do(x):
        return rust.cons(x, rust.lazy_seq(lambda: do(x + 1)))
    return do(0)

def drop(n, coll):
    for _ in range(n):
        coll = coll.rest()
    return coll

ka = rust.keyword("a")
kb = rust.keyword("b")
MyRecord = rust.define_record("dummy.MyRecord", (ka, "a"), (kb, "b"))
r = MyRecord(*[str(i) for i in range(2)])
print(r.a, r.b)
r2 = r.assoc(ka, 42)
print(r2.a, r2.b)

l = rust.list_(*range(1000000))
del l

print(rust.symbol("hello", "world"))
print("symbol hello, world:", timeit(
    lambda: rust.symbol("hello", "world"),
    number=1000000))

print("symbol hello:", timeit(
    lambda: rust.symbol("hello"),
    number=1000000))

print("symbol hello/world:", timeit(
    lambda: rust.symbol("hello/world"),
    number=1000000))

print("keyword hello, world:", timeit(
    lambda: rust.keyword("hello", "world"),
    number=1000000))

print("keyword hello:", timeit(
    lambda: rust.keyword("hello"),
    number=1000000))

print("keyword hello/world:", timeit(
    lambda: rust.keyword("hello/world"),
    number=1000000))

print("lazy_seq + cons:", timeit(
    lambda: rust.lazy_seq(
        lambda: rust.cons(42, None)).first(),
    number=1000000))

print("lazy_seq drop:", timeit(
    lambda: drop(1000, nums()),
    number=1000))
