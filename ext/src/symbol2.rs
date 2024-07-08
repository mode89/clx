use pyo3::exceptions::*;
use pyo3::prelude::*;
use pyo3::types::*;
use std::hash::DefaultHasher;
use std::hash::{Hash, Hasher};

pub fn init_module(module: *mut pyo3_ffi::PyObject) {
    Python::with_gil(|py| {
        let module: Py<PyModule> = unsafe {
            Py::from_borrowed_ptr(py, module)
        };
        let module = module.into_bound(py);
        module.add_function(
            wrap_pyfunction!(symbol2, &module).unwrap()).unwrap();
    });
}

#[pyclass]
struct Symbol {
    #[pyo3(get)]
    namespace: PyObject,
    #[pyo3(get)]
    name: PyObject,
    #[pyo3(get)]
    __meta__: PyObject,
    hash: Option<isize>,
}

#[pymethods]
impl Symbol {
    fn __hash__(&mut self) -> isize {
        match self.hash {
            Some(hash) => hash,
            None => {
                Python::with_gil(|py| {
                    let mut hasher = DefaultHasher::new();
                    self.namespace.bind(py).hash().unwrap().hash(&mut hasher);
                    self.name.bind(py).hash().unwrap().hash(&mut hasher);
                    let hash = hasher.finish() as isize;
                    self.hash = Some(hash);
                    hash
                })
            }
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.namespace.is(&other.namespace) && self.name.is(&other.name)
    }

    fn __repr__(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            if self.namespace.is_none(py) {
                Ok(self.name.clone_ref(py))
            } else {
                Ok(PyString::new_bound(py,
                    format!("{}/{}", self.namespace, self.name).as_str())
                   .into())
            }
        })
    }
}

#[pyfunction]
#[pyo3(signature = (arg1, arg2 = None))]
fn symbol2(
    arg1: &Bound<'_, PyAny>,
    arg2: Option<&Bound<'_, PyAny>>
) -> PyResult<PyObject> {
    if let Some(arg2) = arg2 {
        let py = arg1.py();
        Py::new(py, Symbol {
            namespace: PyString::intern_bound(
                py, arg1.extract::<&str>()?).into(),
            name: PyString::intern_bound(
                py, arg2.extract::<&str>()?).into(),
            __meta__: arg1.py().None(),
            hash: None,
        }).map(|x| x.into_any())
    } else {
        if let Ok(s) = arg1.extract::<&str>() {
            let py = arg1.py();
            match s.chars().position(|c| c == '/') {
                Some(index) => {
                    if s == "/" {
                        Py::new(py, Symbol {
                            namespace: py.None(),
                            name: PyString::intern_bound(py, "/").into(),
                            __meta__: py.None(),
                            hash: None,
                        }).map(|x| x.into_any())
                    } else {
                        let (ns, name) = s.split_at(index);
                        Py::new(py, Symbol {
                            namespace: PyString::intern_bound(py, ns).into(),
                            name: PyString::intern_bound(py, &name[1..]).into(),
                            __meta__: py.None(),
                            hash: None,
                        }).map(|x| x.into_any())
                    }
                }
                None => {
                    Py::new(py, Symbol {
                        namespace: py.None(),
                        name: PyString::intern_bound(py, s).into(),
                        __meta__: py.None(),
                        hash: None,
                    }).map(|x| x.into_any())
                }
            }
        } else if arg1.is_exact_instance_of::<Symbol>() {
            Ok(arg1.clone().unbind())
        } else {
            return Err(PyTypeError::new_err("argument must be a string"));
        }
    }
}
