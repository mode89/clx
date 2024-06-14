use crate::object::PyObj;
use crate::utils;
use crate::protocols::*;
use std::ffi::{CStr, CString};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use pyo3_ffi::*;

pub fn init_module(module: *mut PyObject) {
    utils::module_add_method!(module, symbol);
    utils::module_add_method!(module, is_symbol);
    utils::module_add_method!(module, is_simple_symbol);
    utils::module_add_type!(module, Symbol, symbol_type());
}

#[repr(C)]
pub struct Symbol {
    ob_base: PyObject,
    pub name: PyObj,
    pub namespace: PyObj,
    meta: PyObj,
    hash: Option<isize>,
}

pub fn symbol_type() -> &'static PyObj {
    utils::static_type!(
        utils::TypeSpec {
            name: "clx_rust.Symbol",
            bases: vec![
                imeta_type(),
            ],
            flags: Py_TPFLAGS_DEFAULT |
                   Py_TPFLAGS_DISALLOW_INSTANTIATION,
            size: std::mem::size_of::<Symbol>(),
            dealloc: Some(utils::generic_dealloc::<Symbol>),
            repr: Some(symbol_repr),
            hash: Some(symbol_hash),
            compare: Some(symbol_compare),
            members: vec![
                "name",
                "namespace",
                "__meta__"
            ],
            methods: vec![
                ("with_meta", symbol_with_meta),
            ],
            ..Default::default()
        }
    )
}

unsafe extern "C" fn symbol(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::handle_gil_and_panic!({
        if nargs < 1 || nargs > 2 {
            return utils::raise_exception(
                "symbol() takes 1 or 2 positional arguments");
        }

        let obj = if nargs == 1 {
            let arg = PyObj::borrow(*args);
            if arg.type_is(symbol_type()) {
                return utils::incref(*args);
            }

            let arg_ptr = PyUnicode_AsUTF8AndSize(*args, std::ptr::null_mut());
            if arg_ptr.is_null() {
                return utils::raise_exception(
                    "symbol() argument must be a string");
            }

            let obj = PyObj::alloc(symbol_type());
            let sym = obj.as_ref::<Symbol>();
            let arg = CStr::from_ptr(arg_ptr).to_str().unwrap();
            match arg.chars().position(|c| c == '/') {
                Some(index) => {
                    if arg == "/" {
                        std::ptr::write(&mut sym.namespace, PyObj::none());
                        std::ptr::write(&mut sym.name,
                            utils::intern_string(
                                utils::static_cstring!("/")));
                    } else {
                        let (ns, name) = arg.split_at(index);
                        let ns = CString::new(ns).unwrap();
                        std::ptr::write(&mut sym.namespace,
                            utils::intern_string(ns.as_c_str()));
                        std::ptr::write(&mut sym.name,
                            utils::intern_string(
                                CStr::from_ptr(name[1..].as_ptr().cast())));
                    }
                }
                None => {
                    std::ptr::write(&mut sym.namespace, PyObj::none());
                    std::ptr::write(&mut sym.name,
                        utils::intern_string_in_place(PyObj::borrow(*args)));
                }
            }
            obj
        } else {
            let ns = PyObj::borrow(*args);
            if !ns.is_none() && !ns.is_string() {
                return utils::raise_exception(
                    "symbol namespace must be a string");
            }

            let name = PyObj::borrow(*args.add(1));
            if !name.is_string() {
                return utils::raise_exception(
                    "symbol name must be a string");
            }

            let obj = PyObj::alloc(symbol_type());
            let sym = obj.as_ref::<Symbol>();
            std::ptr::write(&mut sym.namespace,
                utils::intern_string_in_place(ns));
            std::ptr::write(&mut sym.name, name);
            obj
        };
        let sym = obj.as_ref::<Symbol>();
        std::ptr::write(&mut sym.meta, PyObj::none());
        sym.hash = None;
        obj.into_ptr()
    })
}

unsafe extern "C" fn symbol_repr(
    self_: *mut PyObject,
) -> *mut PyObject {
    let self_ = PyObj::borrow(self_);
    let self_ = self_.as_ref::<Symbol>();
    if self_.namespace.is_none() {
        let name = PyUnicode_AsUTF8AndSize(*self_.name,
            std::ptr::null_mut());
        PyUnicode_FromFormat("%s\0".as_ptr().cast(), name)
    } else {
        let namespace = PyUnicode_AsUTF8AndSize(*self_.namespace,
            std::ptr::null_mut());
        let name = PyUnicode_AsUTF8AndSize(*self_.name,
            std::ptr::null_mut());
        PyUnicode_FromFormat("%s/%s\0".as_ptr().cast(), namespace, name)
    }
}

unsafe extern "C" fn symbol_hash(
    self_: *mut PyObject,
) -> isize {
    let self_ = PyObj::borrow(self_);
    let self_ = self_.as_ref::<Symbol>();
    match self_.hash {
        Some(hash) => hash,
        None => {
            let mut hasher = DefaultHasher::new();

            if !self_.namespace.is_none() {
                let namespace = PyUnicode_AsUTF8AndSize(
                    *self_.namespace, std::ptr::null_mut());
                namespace.hash(&mut hasher);
            }
            let name = PyUnicode_AsUTF8AndSize(
                *self_.name, std::ptr::null_mut());
            name.hash(&mut hasher);

            let hash = hasher.finish() as isize;
            self_.hash = Some(hash);
            hash
        }
    }
}

unsafe extern "C" fn symbol_compare(
    self_: *mut PyObject,
    other: *mut PyObject,
    op: i32,
) -> *mut PyObject {
    utils::handle_gil_and_panic!({
        let other = PyObj::borrow(other);
        if other.type_is(symbol_type()) {
            let self_ = PyObj::borrow(self_);
            let self_ = self_.as_ref::<Symbol>();
            let other = other.as_ref::<Symbol>();
            PyObj::from(match op {
                pyo3_ffi::Py_EQ => {
                    *self_.namespace == *other.namespace
                        && *self_.name == *other.name
                }
                pyo3_ffi::Py_NE => {
                    *self_.namespace != *other.namespace
                        || *self_.name != *other.name
                }
                _ => {
                    return utils::raise_exception(
                        "symbol comparison not supported");
                }
            }).into_ptr()
        } else {
            utils::ref_false()
        }
    })
}

unsafe extern "C" fn symbol_with_meta(
    self_: *mut PyObject,
    args: *mut *mut PyObject,
    _nargs: isize,
) -> *mut PyObject {
    utils::handle_gil_and_panic!({
        let self_ = PyObj::borrow(self_);
        let self_ = self_.as_ref::<Symbol>();
        let new_obj = PyObj::alloc(symbol_type());
        let new_sym = new_obj.as_ref::<Symbol>();
        std::ptr::write(&mut new_sym.name, self_.name.clone());
        std::ptr::write(&mut new_sym.namespace, self_.namespace.clone());
        std::ptr::write(&mut new_sym.meta, PyObj::borrow(*args));
        new_sym.hash = self_.hash;
        new_obj.into_ptr()
    })
}

unsafe extern "C" fn is_symbol(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    _nargs: isize,
) -> *mut PyObject {
    utils::handle_gil!({
        PyObj::from(Py_TYPE(*args) == symbol_type().as_ptr()).into_ptr()
    })
}

unsafe extern "C" fn is_simple_symbol(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    _nargs: isize,
) -> *mut PyObject {
    utils::handle_gil!({
        let obj = PyObj::borrow(*args);
        if obj.type_ptr() == symbol_type().as_ptr() {
            let sym = obj.as_ref::<Symbol>();
            PyObj::from(sym.namespace.is_none()).into_ptr()
        } else {
            utils::ref_false()
        }
    })
}
