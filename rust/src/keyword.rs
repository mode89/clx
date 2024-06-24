use crate::object::PyObj;
use crate::type_object as tpo;
use crate::symbol::*;
use crate::utils;
use std::ffi::CString;
use std::sync::Mutex;
use std::collections::{
    HashMap,
    hash_map::DefaultHasher,
};
use std::hash::{Hash, Hasher};
use pyo3_ffi::*;

pub fn init_module(module: *mut PyObject) {
    utils::module_add_method!(module, keyword, py_keyword);
    utils::module_add_method!(module, is_keyword, py_is_keyword);
    utils::module_add_method!(module, is_simple_keyword, py_is_simple_keyword);
    utils::module_add_type!(module, Keyword, keyword_type());
}

#[repr(C)]
pub struct Keyword {
    ob_base: PyObject,
    name: PyObj,
    namespace: PyObj,
    hash: Option<isize>,
}

pub fn keyword_type() -> &'static PyObj {
    tpo::static_type!(
        tpo::TypeSpec {
            name: "clx_rust.Keyword",
            flags: Py_TPFLAGS_DEFAULT,
            size: std::mem::size_of::<Keyword>(),
            new: Some(utils::disallowed_new!(keyword_type)),
            dealloc: Some(utils::generic_dealloc::<Keyword>),
            repr: Some(py_keyword_repr),
            hash: Some(py_keyword_hash),
            compare: Some(py_keyword_compare),
            call: Some(py_keyword_call),
            members: vec![
                tpo::member!("name"),
                tpo::member!("namespace"),
            ],
            ..Default::default()
        }
    )
}

extern "C" fn py_keyword(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        match nargs {
            1 => {
                let argo = PyObj::borrow(unsafe { *args });
                if argo.is_string() {
                    let arg = argo.as_cstr().unwrap().to_str().unwrap();
                    Ok(match arg.chars().position(|c| c == '/') {
                        Some(index) => {
                            if arg == "/" {
                                intern_keyword(
                                    None, CString::new("/").unwrap())
                            } else {
                                let (ns, name) = arg.split_at(index);
                                intern_keyword(
                                    Some(CString::new(ns).unwrap()),
                                    CString::new(&name[1..]).unwrap())
                            }
                        }
                        None => intern_keyword(
                            None, CString::new(arg).unwrap())
                    })
                } else if argo.type_is(keyword_type()) {
                    Ok(argo)
                } else if argo.type_is(symbol_type()) {
                    let sym = unsafe { argo.as_ref::<Symbol>() };
                    let ns = &sym.namespace;
                    let name = &sym.name;
                    Ok(intern_keyword(
                        if ns.is_none() {
                            None
                        } else {
                            Some(ns.as_cstr().unwrap().to_owned())
                        },
                        name.as_cstr().unwrap().to_owned()
                    ))
                } else {
                    utils::raise_exception(
                        "keyword() argument must be string, keyword or symbol")
                }
            },
            2 => {
                let ns = PyObj::borrow(unsafe { *args });
                if !ns.is_none() && !ns.is_string() {
                    return utils::raise_exception(
                        "keyword namespace must be a string");
                }

                let name = PyObj::borrow(unsafe { *args.add(1) });
                if !name.is_string() {
                    return utils::raise_exception(
                        "keyword name must be a string");
                }

                Ok(intern_keyword(
                    if ns.is_none() {
                        None
                    } else {
                        Some(ns.as_cstr().unwrap().to_owned())
                    },
                    name.as_cstr().unwrap().to_owned()
                ))
            },
            _ => {
                return utils::raise_exception(
                    "keyword() takes 1 or 2 positional arguments");
            }
        }
    })
}

fn intern_keyword(ns: Option<CString>, name: CString) -> PyObj {
    type Key = (Option<CString>, CString);
    let mut table = utils::lazy_static!(
        Mutex<HashMap<Key, PyObj>>, {
            Mutex::new(HashMap::new())
        }).lock().unwrap();
    table.entry((ns, name))
        .or_insert_with_key(|key| unsafe {
            let obj = PyObj::alloc(keyword_type());
            let keyword = obj.as_ref::<Keyword>();
            let ns = &key.0;
            std::ptr::write(&mut keyword.namespace, match ns {
                Some(ns) => utils::intern_string(ns),
                None => PyObj::none()
            });
            let name = &key.1;
            std::ptr::write(&mut keyword.name, utils::intern_string(name));
            keyword.hash = None;
            obj
        }).clone()
}

unsafe extern "C" fn py_keyword_repr(
    self_: *mut PyObject,
) -> *mut PyObject {
    let self_ = PyObj::borrow(self_);
    let keyword = self_.as_ref::<Keyword>();
    if keyword.namespace.is_none() {
        PyUnicode_FromFormat(":%s\0".as_ptr().cast(),
            keyword.name.as_cstr().unwrap().as_ptr())
    } else {
        PyUnicode_FromFormat(":%s/%s\0".as_ptr().cast(),
            keyword.namespace.as_cstr().unwrap().as_ptr(),
            keyword.name.as_cstr().unwrap().as_ptr())
    }
}

extern "C" fn py_keyword_hash(
    self_: *mut PyObject,
) -> isize {
    let self_ = PyObj::borrow(self_);
    let keyword = unsafe { self_.as_ref::<Keyword>() };
    match keyword.hash {
        Some(hash) => hash,
        None => {
            let mut hasher = DefaultHasher::new();

            if !keyword.namespace.is_none() {
                keyword.namespace.as_cstr().unwrap().hash(&mut hasher);
            }
            keyword.name.as_cstr().unwrap().hash(&mut hasher);

            let hash = hasher.finish() as isize;
            (*keyword).hash = Some(hash);
            hash
        }
    }
}

extern "C" fn py_keyword_compare(
    self_: *mut PyObject,
    other: *mut PyObject,
    op: i32,
) -> *mut PyObject {
    utils::wrap_body!({
        let other = PyObj::borrow(other);
        if other.type_is(keyword_type()) {
            let self_ = PyObj::borrow(self_);
            let self_ = unsafe { self_.as_ref::<Keyword>() };
            let other = unsafe { other.as_ref::<Keyword>() };
            match op {
                pyo3_ffi::Py_EQ => Ok(PyObj::from(
                    self_.namespace.is(&other.namespace) &&
                    self_.name.is(&other.name))),
                pyo3_ffi::Py_NE => Ok(PyObj::from(
                    !self_.namespace.is(&other.namespace) ||
                    !self_.name.is(&other.name))),
                _ => utils::raise_exception(
                    "keyword comparison not supported")
            }
        } else {
            Ok(PyObj::from(false))
        }
    })
}

extern "C" fn py_keyword_call(
    self_: *mut PyObject,
    args: *mut PyObject,
    _kw: *mut PyObject,
) -> *mut PyObject {
    utils::wrap_body!({
        let coll: *mut PyObject = std::ptr::null_mut();
        let default: *mut PyObject = std::ptr::null_mut();
        if unsafe { PyArg_UnpackTuple(args,
                "Keyword.__call__()\0".as_ptr().cast(),
                1, 2, &coll, &default) } != 0 {
            let coll = PyObj::borrow(coll);
            coll.call_method2(
                &utils::static_pystring!("lookup"),
                PyObj::borrow(self_),
                if !default.is_null() {
                    PyObj::borrow(default)
                } else {
                    PyObj::none()
                })
        } else {
            Err(())
        }
    })
}

extern "C" fn py_is_keyword(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    _nargs: isize,
) -> *mut PyObject {
    utils::handle_gil!({
        PyObj::from(unsafe {
            Py_TYPE(*args) == keyword_type().as_ptr()
        }).into_ptr()
    })
}

extern "C" fn py_is_simple_keyword(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    _nargs: isize,
) -> *mut PyObject {
    utils::handle_gil!({
        let argo = PyObj::borrow(unsafe { *args });
        if argo.type_is(keyword_type()) {
            let arg = unsafe { argo.as_ref::<Keyword>() };
            PyObj::from(arg.namespace.is_none()).into_ptr()
        } else {
            utils::ref_false()
        }
    })
}
