#![allow(dead_code)]
// #![allow(unused_imports)]
// #![allow(unused_variables)]

use lazy_static::lazy_static;
use regex::Regex;
use pyo3::prelude::*;
use pyo3::types::*;
use std::collections::{
    HashMap,
    hash_map::DefaultHasher,
};
use std::hash::{Hash, Hasher};
use std::sync::RwLock;

/// A Python module implemented in Rust.
#[pymodule]
fn clx(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(symbol, m)?)?;
    m.add_function(wrap_pyfunction!(keyword, m)?)?;
    m.add_function(wrap_pyfunction!(list, m)?)?;
    m.add_function(wrap_pyfunction!(foo, m)?)?;
    Ok(())
}

#[pyfunction]
fn foo() -> i64 {
    42
}

struct Token {
    text: String,
}

lazy_static! {
    static ref RE_TOKEN: Regex = Regex::new(
        concat!(
            r"[\s,]*",
            "(",
                "~@", "|",
                r"[\[\]{}()'`~^@]", "|",
                r#""(?:[\\].|[^\\"])*"?"#, "|",
                ";.*", "|",
                r#"[^\s\[\]{}()'"`@,;]+"#,
            ")"))
        .unwrap();
    static ref RE_INT: Regex = Regex::new(r"^-?[0-9]+$").unwrap();
    static ref RE_FLOAT: Regex = Regex::new(r"^-?[0-9]+\.[0-9]+$").unwrap();
    static ref RE_STRING: Regex =
        Regex::new(r#"^"(?:[\\].|[^\\"])*"$"#).unwrap();
}

fn tokenize(text: &str) -> Vec<Token> {
    RE_TOKEN
        .captures_iter(text)
        .map(|cap| Token{ text: cap[1].to_string() })
        .collect()
}

fn read_atom(py: Python<'_>, token: &str) -> PyObject {
    if RE_INT.is_match(token) {
        token.parse::<i64>().unwrap().to_object(py)
    } else if RE_FLOAT.is_match(token) {
        token.parse::<f64>().unwrap().to_object(py)
    } else if RE_STRING.is_match(token) {
        let end = token.len() - 1;
        unescape(&token[1..end]).to_object(py)
    } else if token.starts_with("\"") {
        panic!("Unterminated string")
    } else if token == "true" {
        true.to_object(py)
    } else if token == "false" {
        false.to_object(py)
    } else if token == "nil" {
        ().to_object(py)
    } else if token.starts_with(":") {
        keyword(Some(PyString::new(py, &token[1..])), None)
    } else {
        symbol(Some(PyString::new(py, token)), None)
    }
}

fn unescape(text: &str) -> String {
    text.replace("\\\\", "\\")
        .replace("\\\"", "\"")
        .replace("\\n", "\n")
}

#[pyclass(frozen)]
struct Symbol {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    namespace: Option<String>,
    hash: i64,
    meta: Option<PyObject>,
}

impl Symbol {
    fn new(namespace: Option<&str>, name: &str) -> Symbol {
        let mut hasher = DefaultHasher::new();
        name.hash(&mut hasher);
        namespace.hash(&mut hasher);
        Symbol {
            name: name.to_string(),
            namespace: namespace.map(|s| s.to_string()),
            hash: hasher.finish() as i64,
            meta: None,
        }
    }
}

#[pymethods]
impl Symbol {
    pub fn __str__(&self) -> String {
        self.print()
    }

    pub fn __repr__(&self) -> String {
        self.print()
    }

    pub fn __eq__(&self, other: &PyAny) -> bool {
        if other.is_exact_instance_of::<Symbol>() {
            let _other = other.extract::<PyRef<Symbol>>().unwrap();
            self.name == _other.name && self.namespace == _other.namespace
        } else {
            false
        }
    }

    pub fn __hash__(&self) -> i64 {
        self.hash
    }

    pub fn print(&self) -> String {
        match &self.namespace {
            Some(ns) => format!("{}/{}", ns, self.name),
            None => self.name.clone(),
        }
    }
}

#[pyfunction]
fn symbol(arg1: Option<&PyAny>, arg2: Option<&PyAny>) -> PyObject {
    match arg2 {
        Some(arg2) => {
            let py = arg2.py();
            match arg1 {
                Some(arg1) => {
                    let ns = arg1.extract::<&str>().unwrap();
                    let name = arg2.extract::<&str>().unwrap();
                    Symbol::new(Some(ns), name).into_py(py)
                }
                None => {
                    let name = arg2.extract::<&str>().unwrap();
                    Symbol::new(None, name).into_py(py)
                }
            }
        }
        None => {
            match arg1 {
                Some(arg1) => {
                    let py = arg1.py();
                    if arg1.is_exact_instance_of::<PyString>() {
                        let s = arg1.extract::<&str>().unwrap();
                        match s.chars().position(|c| c == '/') {
                            Some(index) => {
                                let (ns, name) = s.split_at(index);
                                Symbol::new(Some(&ns), &name[1..])
                            }
                            None => Symbol::new(None, s)
                        }.into_py(py)
                    } else if is_symbol(arg1) {
                        arg1.into_py(py)
                    } else {
                        panic!("symbol expects string or Symbol")
                    }
                }
                None => panic!("symbol expects string or Symbol")
            }
        }
    }
}

#[pyfunction]
fn is_symbol(obj: &PyAny) -> bool {
    obj.is_exact_instance_of::<Symbol>()
}

#[pyfunction]
fn is_simple_symbol(obj: &PyAny) -> bool {
    obj.is_exact_instance_of::<Symbol>() &&
        obj.extract::<PyRef<Symbol>>().unwrap().namespace.is_none()
}

#[pyclass(frozen)]
struct Keyword {
    name: String,
    namespace: Option<String>,
    hash: i64,
}

impl Keyword {
    fn new(namespace: Option<&str>, name: &str) -> Keyword {
        let mut hasher = DefaultHasher::new();
        name.hash(&mut hasher);
        namespace.hash(&mut hasher);
        Keyword {
            name: name.to_string(),
            namespace: namespace.map(|s| s.to_string()),
            hash: hasher.finish() as i64,
        }
    }
}

#[pymethods]
impl Keyword {
    pub fn __str__(&self) -> String {
        self.print()
    }

    pub fn __repr__(&self) -> String {
        self.print()
    }

    pub fn __eq__(&self, other: &PyAny) -> bool {
        match other.extract::<PyRef<Keyword>>() {
            Ok(_other) => {
                std::ptr::eq(&self.name, &_other.name) &&
                    std::ptr::eq(&self.namespace, &_other.namespace)
            }
            Err(_) => false,
        }
    }

    pub fn __hash__(&self) -> i64 {
        self.hash
    }

    pub fn print(&self) -> String {
        match &self.namespace {
            Some(ns) => format!(":{}/{}", ns, self.name),
            None => format!(":{}", self.name),
        }
    }
}

#[pyfunction]
fn keyword(arg1: Option<&PyAny>, arg2: Option<&PyAny>) -> PyObject {
    match arg2 {
        Some(arg2) => {
            match arg1 {
                Some(arg1) => {
                    let ns = arg1.extract::<&str>().unwrap();
                    let name = arg2.extract::<&str>().unwrap();
                    intern_keyword(arg1.py(), Some(ns), name)
                }
                None => {
                    let name = arg2.extract::<&str>().unwrap();
                    intern_keyword(arg2.py(), None, name)
                }
            }
        }
        None => {
            match arg1 {
                Some(arg1) => {
                    if arg1.is_exact_instance_of::<PyString>() {
                        let s = arg1.extract::<&str>().unwrap();
                        match s.chars().position(|c| c == '/') {
                            Some(index) => {
                                let (ns, name) = s.split_at(index);
                                intern_keyword(arg1.py(), Some(ns), &name[1..])
                            }
                            None => intern_keyword(arg1.py(), None, s)
                        }
                    } else if is_keyword(arg1) {
                        arg1.into_py(arg1.py())
                    } else if is_symbol(arg1) {
                        let sym = arg1.extract::<PyRef<Symbol>>().unwrap();
                        let ns = &sym.namespace;
                        let name = &sym.name;
                        intern_keyword(arg1.py(), ns.as_deref(), name)
                    } else {
                        panic!("keyword expects string, Symbol, or Keyword")
                    }
                }
                None => panic!("keyword expects string, Symbol, or Keyword")
            }
        }
    }
}

lazy_static! {
    static ref KEYWORD_TABLE:
        RwLock<HashMap<(Option<String>, String), PyObject>> =
            RwLock::new(HashMap::new());
}

fn intern_keyword(py: Python<'_>, ns: Option<&str>, name: &str) -> PyObject {
    let mut table = KEYWORD_TABLE.write().unwrap();
    table.entry((ns.map(|s| s.to_string()), name.to_string()))
        .or_insert_with(|| {
            Keyword::new(ns, name).into_py(py)
        }).clone_ref(py)
}

#[pyfunction]
fn is_keyword(obj: &PyAny) -> bool {
    obj.is_exact_instance_of::<Keyword>()
}

#[pyclass(frozen)]
struct PersistentList {
    first: PyObject,
    tail: Option<PyObject>,
    length: i64,
    meta: Option<PyObject>,
}

#[pymethods]
impl PersistentList {
//     def __repr__(self):
//         return self.pr(True)
//     def __str__(self):
//         return self.pr(False)
//     def __eq__(self, other):
//         if self is other:
//             return True
//         elif type(other) is PersistentList:
//             if self._length != other._length:
//                 return False
//             else:
//                 lst1, lst2 = self, other
//                 while lst1._length > 0:
//                     if lst1._first != lst2._first:
//                         return False
//                     lst1, lst2 = lst1._rest, lst2._rest
//                 return True
//         else:
//             return _equiv_sequential(self, other)
    fn __eq__(self_: &PyCell<Self>, other: &PyAny) -> bool {
        let py = self_.py();
        match other.extract::<&PyCell<PersistentList>>() {
            Ok(other) => {
                let mut lst1 = self_.get();
                let mut lst2 = other.get();
                if lst1.length == lst2.length {
                    while lst1.length > 0 {
                        if !PyAny::eq(
                                lst1.first.as_ref(py),
                                lst2.first.as_ref(py)).unwrap() {
                            return false;
                        }
                        lst1 = lst1.tail
                            .as_ref()
                            .unwrap()
                            .extract::<&PyCell<PersistentList>>(py)
                            .unwrap()
                            .get();
                        lst2 = lst2.tail
                            .as_ref()
                            .unwrap()
                            .extract::<&PyCell<PersistentList>>(py)
                            .unwrap()
                            .get();
                    }
                    true
                } else {
                    false
                }
            }
            Err(_) => panic!("Not implemented")
        }
    }
//     def __hash__(self):
//         if self._hash is None:
//             self._hash = hash(tuple(self))
//         return self._hash
    fn __hash__(self_: &PyCell<Self>) -> i64 {
        let py = self_.py();
        let mut lst = self_.get();
        let mut hash = 42;
        while lst.length > 0 {
            println!("{}", lst.length);
            // hash = hash ^ lst.length ^ HASH.call1(py,
            //     (lst.first.as_ref(py),)
            //         .to_object(py)
            //         .extract::<&PyTuple>(py)
            //         .unwrap())
            //     .unwrap()
            //     .extract::<i64>(py)
            //     .unwrap();
            lst = lst.tail
                .as_ref()
                .unwrap()
                .extract::<&PyCell<PersistentList>>(py)
                .unwrap()
                .get();
        }
        hash
    }

//     def __len__(self):
//         return self._length
    fn count(&self) -> i64 {
        self.length
    }

//     def __iter__(self):
//         lst = self
//         while lst._length > 0:
//             yield lst._first
//             lst = lst._rest
//     def __getitem__(self, index):
//         raise NotImplementedError()
//     def with_meta(self, _meta):
//         return PersistentList(
//             self._first,
//             self._rest,
//             self._length,
//             self._hash,
//             _meta)
//     def pr(self, readably):
//         return "(" + \
//             " ".join(map(lambda x: pr_str(x, readably), self)) + \
//             ")"

    fn first(&self) -> PyObject {
        self.first.clone()
    }

    fn next(&self) -> Option<PyObject> {
        if self.length < 2 {
            None
        } else {
            self.tail.clone()
        }
    }

    fn rest(self_: PyRef<'_, Self>) -> PyObject {
        if self_.length == 0 {
            let py = self_.py();
            self_.into_py(py)
        } else {
            self_.tail.as_ref().unwrap().clone()
        }
    }

    fn seq(self_: PyRef<'_, Self>) -> PyObject {
        let py = self_.py();
        self_.into_py(py)
    }

    fn conj(self_: PyRef<'_, Self>, value: &PyAny) -> PyObject {
        let py = self_.py();
        let length = self_.length + 1;
        PersistentList {
            first: value.into(),
            tail: Some(self_.into_py(py)),
            length,
            meta: None,
        }.into_py(py)
    }
}

lazy_static! {
    static ref EMPTY_LIST: PyObject = {
        Python::with_gil(|py| {
            PersistentList {
                first: py.None(),
                tail: None,
                length: 0,
                meta: None,
            }.into_py(py)
        })
    };

    // static ref HASH: PyObject = {
    //     Python::with_gil(|py| {
    //         let builtins = py.import("builtins").unwrap();
    //         let hash = builtins.getattr("hash").unwrap().into_py(py)
    //         Box::new(|_py: Python<'_>, obj: &PyAny| {
    //             hash.call1(_py,
    //                 (obj.as_ref(_py),)
    //                     .to_object(_py)
    //                     .extract::<&PyTuple>(_py)
    //                     .unwrap())
    //         })
    //     })
    // };
}

#[pyfunction]
#[pyo3(signature = (*args))]
fn list(args: &PyTuple) -> PyObject {
    if args.is_empty() {
        EMPTY_LIST.clone()
    } else {
        let py = args.py();
        let mut tail = EMPTY_LIST.clone();
        for i in (0..args.len()).rev() {
            tail = PersistentList::conj(
                tail.extract::<PyRef<PersistentList>>(py).unwrap(),
                args.get_item(i).unwrap());
        }
        tail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer() {
        let _tokenize = |text: &str| tokenize(text)
            .iter()
            .map(|t| t.text.clone())
            .collect::<Vec<_>>();
        assert_eq!(_tokenize("    "), Vec::<String>::new());
        assert_eq!(_tokenize(" ~@ "), vec!["~@"]);
        assert_eq!(_tokenize("  \"hello\n world!\"   "),
            vec!["\"hello\n world!\""]);
        assert_eq!(_tokenize("(foo :bar [1 \"hello\"] {:baz 'quux})"),
            vec!["(", "foo", ":bar",
                "[", "1", "\"hello\"", "]",
                "{", ":baz", "'", "quux", "}", ")"]);
    }

    #[test]
    fn symbols() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let _s = |s: &str| symbol(Some(PyString::new(py, s)), None);
            let _s2 = |ns: &str, name: &str|
                symbol(
                    Some(PyString::new(py, ns)),
                    Some(PyString::new(py, name)));

            let hello = _s("hello");
            assert!(is_symbol(hello.as_ref(py)));
            assert!(is_simple_symbol(hello.as_ref(py)));
            let _hello = hello.extract::<PyRef<Symbol>>(py).unwrap();
            assert_eq!(_hello.name, "hello");
            assert!(_hello.namespace.is_none());
            assert!(PyAny::eq(hello.as_ref(py), _s("hello")).unwrap());

            let hello_world = _s2("hello", "world");
            let _hello_world =
                hello_world.extract::<PyRef<Symbol>>(py).unwrap();
            assert!(is_symbol(hello_world.as_ref(py)));
            assert!(!is_simple_symbol(hello_world.as_ref(py)));
            assert_eq!(_hello_world.name, "world");
            assert_eq!(_hello_world.namespace, Some("hello".to_string()));
            assert!(PyAny::eq(
                hello_world.as_ref(py),
                _s2("hello", "world")).unwrap());
            assert_eq!(_hello_world.print(), "hello/world");

            let foo_bar = _s("foo/bar");
            assert!(is_symbol(foo_bar.as_ref(py)));
            let _foo_bar = foo_bar.extract::<PyRef<Symbol>>(py).unwrap();
            assert_eq!(_foo_bar.name, "bar");
            assert_eq!(_foo_bar.namespace, Some("foo".to_string()));
            assert!(PyAny::eq(
                foo_bar.as_ref(py),
                _s2("foo", "bar")).unwrap());

            assert!(PyAny::eq(
                symbol(Some(_s("quux").as_ref(py)), None).as_ref(py),
                _s("quux")).unwrap());

            assert!(PyAny::eq(
                symbol(None, Some(PyString::new(py, "foo"))).as_ref(py),
                symbol(Some(PyString::new(py, "foo")), None)).unwrap());
        });
    }

    #[test]
    fn keywords() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let _s = |s: &str| PyString::new(py, s);
            let _k2 = |ns: &str, name: &str|
                keyword(Some(_s(ns)), Some(_s(name)));
            let _k = |name: &str| keyword(Some(_s(name)), None);

            let hello = _k("hello");
            assert!(is_keyword(_k("hello").as_ref(py)));
            let _hello = hello.extract::<PyRef<Keyword>>(py).unwrap();
            assert_eq!(_hello.name, "hello");
            assert!(_hello.namespace.is_none());
            assert!(hello.is(&_k("hello")));
            assert!(PyAny::eq(hello.as_ref(py), _k("hello")).unwrap());

            let hello_world = _k2("hello", "world");
            let _hello_world =
                hello_world.extract::<PyRef<Keyword>>(py).unwrap();
            assert!(is_keyword(hello_world.as_ref(py)));
            assert_eq!(_hello_world.name, "world");
            assert_eq!(_hello_world.namespace, Some("hello".to_string()));
            assert!(hello_world.is(&_k2("hello", "world")));
            assert!(PyAny::eq(
                hello_world.as_ref(py),
                _k2("hello", "world")).unwrap());

            let foo_bar = _k("foo/bar");
            assert!(is_keyword(foo_bar.as_ref(py)));
            let _foo_bar = foo_bar.extract::<PyRef<Keyword>>(py).unwrap();
            assert_eq!(_foo_bar.name, "bar");
            assert_eq!(_foo_bar.namespace, Some("foo".to_string()));
            assert!(foo_bar.is(&_k2("foo", "bar")));
            assert!(PyAny::eq(
                foo_bar.as_ref(py),
                _k2("foo", "bar")).unwrap());

            assert!(
                keyword(
                    Some(symbol(Some(_s("baz")), None).as_ref(py)),
                    None)
                .is(&_k("baz")));
            assert!(
                keyword(
                    Some(symbol(Some(_s("foo/bar")), None).as_ref(py)),
                    None)
                .is(&_k("foo/bar")));
            assert!(keyword(Some(_k("quux").as_ref(py)), None)
                .is(&_k("quux")));
            assert_eq!(
                _k("foo/bar")
                    .extract::<PyRef<Keyword>>(py)
                    .unwrap()
                    .print(),
                ":foo/bar");
            assert!(keyword(None, Some(_s("foo"))).is(&_k("foo")));

            // TODO
            // assert K("foo")(M(K("foo"), 42)) == 42
            // assert K("foo")(M(K("bar"), 42)) is None
            // assert K("foo")(M(K("bar"), 42), 43) == 43
        })
    }

    #[test]
    fn lists() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            macro_rules! L {
                () => {
                    list(PyTuple::empty(py))
                };
                ($x:expr) => {
                    list(( $x, )
                        .to_object(py)
                        .extract::<&PyTuple>(py)
                        .unwrap())
                };
                ($($x:expr),*) => {
                    list(( $($x),* )
                        .to_object(py)
                        .extract::<&PyTuple>(py)
                        .unwrap())
                };
            }
            macro_rules! _L {
                ($($x:expr),*) => {
                    L!($($x),*)
                        .extract::<PyRef<PersistentList>>(py)
                        .unwrap()
                };
            }
            macro_rules! is {
                ($x:expr, $y:expr) => {
                    $x.is($y.as_ref(py))
                }
            }
            macro_rules! eq {
                ($x:expr, $y:expr) => {
                    PyAny::eq($x.as_ref(py), $y).unwrap()
                }
            }
            // list((1, "hello", 3).to_object(py).extract::<&PyTuple>(py).unwrap());
            // assert isinstance(L(), clx.PersistentList)
            // assert L() is L()
            assert!(is!(L!(), L!()));
            // assert len(L()) == 0
            assert!(_L!().count() == 0);
            // TODO assert bool(L()) is False
            // assert L().first() is None
            assert!(_L!().first().is_none(py));
            // assert L().rest() is L()
            assert!(is!(PersistentList::rest(_L!()), L!()));
            // assert L().next() is None
            assert!(_L!().next().is_none());
            // assert L().conj(1) == L(1)
            assert!(eq!(
                PersistentList::conj(
                    _L!(), 1_i64.to_object(py).as_ref(py)),
                L!(1)));
            // assert L().with_meta(M(1, 2)).__meta__ == M(1, 2)
            // assert L().with_meta(M(1, 2)) is not L()
            // assert L().with_meta(M(1, 2)) == L()
            // assert L().with_meta(M(1, 2)) == L().with_meta(M(3, 4))
            // assert L(1) is not L()
            // assert len(L(1)) == 1
            // assert bool(L(1)) is True
            // assert L(1).first() == 1
            // assert L(1).rest() is L()
            // assert L(1).next() is None
            // assert L(1).conj(2) == L(2, 1)
            // assert L(1, 2) == L(1, 2)
            // assert len(L(1, 2)) == 2
            // assert bool(L(1, 2)) is True
            // assert L(1, 2).first() == 1
            // assert L(1, 2).rest() == L(2)
            // assert L(1, 2).next() == L(2)
            // assert L(1, 2).conj(3) == L(3, 1, 2)
            // assert L() is not None
            // assert L() != 42
            // assert L() == V()
            // assert L() != M()
            // assert L() == _lazy_range(0)
            // assert L() != L(1)
            // assert L() != L(1, 2)
            // assert L(1, 2, 3) != L(1, 2)
            assert!(!eq!(L!(1, 2, 3), L!(1, 2)));
            // assert L(1, 2, 3) == L(1, 2, 3)
            assert!(eq!(L!(1, 2, 3), L!(1, 2, 3)));
            // assert L(1, 2, 3) == V(1, 2, 3)
            // assert L(1, 2, 3) == _lazy_range(1, 4)
            // assert L(1, 2, 3) != [1, 2, 3]
            // assert L(1, 2, 3) != (1, 2, 3)
            // assert list(iter(L(1, 2, 3))) == [1, 2, 3]
            // assert hash(L()) == hash(L())
            // assert hash(L(1)) == hash(L(1))
            // assert hash(L(1, 2)) == hash(L(1, 2))
            // assert hash(L(1, 2)) != hash(L(2, 1))
            // assert hash(L(1, 2, 3)) == hash(L(1, 2, 3))
            // assert hash(L(1, 2, 3)) != hash(L(1, 2))
            // assert hash(L(1, 2, 3).rest()) == hash(L(2, 3))
        })
    }

    #[test]
    fn read_atoms() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            assert_eq!(read_atom(py, &"42".to_string())
                .extract::<i64>(py).unwrap(), 42);
            assert_eq!(read_atom(py, &"-42".to_string())
                .extract::<i64>(py).unwrap(), -42);
            assert_eq!(read_atom(py, &"1.23".to_string())
                .extract::<f64>(py).unwrap(), 1.23);
            assert_eq!(read_atom(py, &"-45.678".to_string())
                .extract::<f64>(py).unwrap(), -45.678);
            assert_eq!(read_atom(py, &"\"hello, world!\"".to_string())
                .extract::<&str>(py).unwrap(), "hello, world!");
            assert_eq!(read_atom(py, &"\"foo\\nbar\\baz\\\"quux\"".to_string())
                .extract::<&str>(py).unwrap(), "foo\nbar\\baz\"quux");
            assert_eq!(read_atom(py, &"true".to_string())
                .extract::<bool>(py).unwrap(), true);
            assert_eq!(read_atom(py, &"false".to_string())
                .extract::<bool>(py).unwrap(), false);
            assert!(read_atom(py, &"nil".to_string()).is_none(py));
            assert!(PyAny::eq(
                read_atom(py, &":foo/bar".to_string()).as_ref(py),
                keyword(
                    Some(PyString::new(py, "foo")),
                    Some(PyString::new(py, "bar")))).unwrap());
            assert!(PyAny::eq(
                read_atom(py, &"foo/bar".to_string()).as_ref(py),
                symbol(
                    Some(PyString::new(py, "foo")),
                    Some(PyString::new(py, "bar")))).unwrap());
        })
    }
}
