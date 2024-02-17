use regex::Regex;
use pyo3::prelude::*;
use pyo3::types::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// A Python module implemented in Rust.
#[pymodule]
fn clx(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(symbol, m)?)?;
    Ok(())
}

struct Token {
    text: String,
}

fn tokenize(text: &str) -> Vec<Token> {
    Regex::new(
        concat!(
            "[\\s,]*",
            "(",
                "~@", "|",
                "[\\[\\]{}()'`~^@]", "|",
                "\"(?:[\\\\].", "|", "[^\\\\\"])*\"?", "|",
                ";.*", "|",
                "[^\\s\\[\\]{}()'\"`@,;]+",
            ")"))
        .unwrap()
        .captures_iter(text)
        .map(|cap| Token{ text: cap[1].to_string() })
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_string() {
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
}
