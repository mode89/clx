use crate::object::PyObj;
use crate::type_object as tpo;
use crate::utils;
use crate::protocols::*;
use std::ffi::{CStr, CString};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use pyo3_ffi::*;

pub fn init_module(module: *mut PyObject) {
    utils::module_add_method!(module, symbol, py_symbol);
    utils::module_add_method!(module, is_symbol, py_is_symbol);
    utils::module_add_method!(module, is_simple_symbol, py_is_simple_symbol);
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
    tpo::static_type!(
        tpo::TypeSpec {
            name: "clx_rust.Symbol",
            bases: vec![
                imeta_type(),
            ],
            flags: Py_TPFLAGS_DEFAULT,
            size: std::mem::size_of::<Symbol>(),
            new: Some(utils::disallowed_new!(symbol_type)),
            dealloc: Some(tpo::generic_dealloc::<Symbol>),
            repr: Some(py_symbol_repr),
            hash: Some(py_symbol_hash),
            compare: Some(py_symbol_compare),
            members: vec![
                tpo::member!("name"),
                tpo::member!("namespace"),
                tpo::member!("__meta__"),
            ],
            methods: vec![
                tpo::method!("with_meta", py_symbol_with_meta),
            ],
            ..Default::default()
        }
    )
}

extern "C" fn py_symbol(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs == 1 {
            let argo = PyObj::borrow(unsafe { *args });
            if argo.type_is(symbol_type()) {
                Ok(argo)
            } else if let Ok(arg) = argo.as_cstr() {
                let arg = arg.to_str().unwrap();
                match arg.chars().position(|c| c == '/') {
                    Some(index) => {
                        if arg == "/" {
                            Ok(_symbol(
                                PyObj::none(),
                                utils::intern_string(
                                    utils::static_cstring!("/")),
                                PyObj::none(),
                                None))
                        } else {
                            let (ns, name) = arg.split_at(index);
                            let ns = CString::new(ns).unwrap();
                            Ok(_symbol(
                                utils::intern_string(ns.as_c_str()),
                                utils::intern_string(unsafe {
                                    CStr::from_ptr(name[1..].as_ptr().cast())
                                }),
                                PyObj::none(),
                                None))
                        }
                    }
                    None => {
                        Ok(_symbol(
                            PyObj::none(),
                            argo.intern_string_in_place(),
                            PyObj::none(),
                            None))
                    }
                }
            } else {
                utils::raise_exception("symbol() argument must be a string")
            }
        } else if nargs == 2 {
            let ns = PyObj::borrow(unsafe { *args });
            if !ns.is_none() && !ns.is_string() {
                return utils::raise_exception(
                    "symbol namespace must be a string");
            }

            let name = PyObj::borrow(unsafe { *args.add(1) });
            if !name.is_string() {
                return utils::raise_exception(
                    "symbol name must be a string");
            }

            Ok(_symbol(
                ns.intern_string_in_place(),
                name,
                PyObj::none(),
                None))
        } else {
            utils::raise_exception("symbol() takes 1 or 2 arguments")
        }
    })
}

#[inline]
fn _symbol(
    namespace: PyObj,
    name: PyObj,
    meta: PyObj,
    hash: Option<isize>
) -> PyObj {
    let obj = PyObj::alloc(symbol_type());
    unsafe {
        let sym = obj.as_ref::<Symbol>();
        std::ptr::write(&mut sym.namespace, namespace);
        std::ptr::write(&mut sym.name, name);
        std::ptr::write(&mut sym.meta, meta);
        sym.hash = hash;
    }
    obj
}

unsafe extern "C" fn py_symbol_repr(
    self_: *mut PyObject,
) -> *mut PyObject {
    let self_ = PyObj::borrow(self_);
    let self_ = self_.as_ref::<Symbol>();
    if self_.namespace.is_none() {
        PyUnicode_FromFormat("%s\0".as_ptr().cast(),
            self_.name.as_cstr().unwrap().as_ptr())
    } else {
        PyUnicode_FromFormat("%s/%s\0".as_ptr().cast(),
            self_.namespace.as_cstr().unwrap().as_ptr(),
            self_.name.as_cstr().unwrap().as_ptr())
    }
}

extern "C" fn py_symbol_hash(
    self_: *mut PyObject,
) -> isize {
    let self_ = PyObj::borrow(self_);
    let self_ = unsafe { self_.as_ref::<Symbol>() };
    match self_.hash {
        Some(hash) => hash,
        None => {
            let mut hasher = DefaultHasher::new();

            if !self_.namespace.is_none() {
                self_.namespace.as_cstr().unwrap().hash(&mut hasher);
            }
            self_.name.as_cstr().unwrap().hash(&mut hasher);

            let hash = hasher.finish() as isize;
            self_.hash = Some(hash);
            hash
        }
    }
}

extern "C" fn py_symbol_compare(
    self_: *mut PyObject,
    other: *mut PyObject,
    op: i32,
) -> *mut PyObject {
    utils::wrap_body!({
        let other = PyObj::borrow(other);
        if other.type_is(symbol_type()) {
            let self_ = PyObj::borrow(self_);
            let self_ = unsafe { self_.as_ref::<Symbol>() };
            let other = unsafe { other.as_ref::<Symbol>() };
            match op {
                pyo3_ffi::Py_EQ => Ok(PyObj::from(
                    self_.namespace.is(&other.namespace) &&
                    self_.name.is(&other.name))),
                pyo3_ffi::Py_NE => Ok(PyObj::from(
                    !self_.namespace.is(&other.namespace) ||
                    !self_.name.is(&other.name))),
                _ => utils::raise_exception(
                    "symbol comparison not supported")
            }
        } else {
            Ok(PyObj::from(false))
        }
    })
}

extern "C" fn py_symbol_with_meta(
    self_: *mut PyObject,
    args: *mut *mut PyObject,
    _nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        let self_ = PyObj::borrow(self_);
        let self_ = unsafe { self_.as_ref::<Symbol>() };
        Ok(_symbol(
            self_.namespace.clone(),
            self_.name.clone(),
            PyObj::borrow(unsafe { *args }),
            self_.hash))
    })
}

extern "C" fn py_is_symbol(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    _nargs: isize,
) -> *mut PyObject {
    utils::handle_gil!({
        PyObj::from(unsafe {
            Py_TYPE(*args) == symbol_type().as_ptr()
        }).into_ptr()
    })
}

extern "C" fn py_is_simple_symbol(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    _nargs: isize,
) -> *mut PyObject {
    utils::handle_gil!({
        let obj = PyObj::borrow(unsafe { *args });
        if obj.type_ptr() == unsafe { symbol_type().as_ptr() } {
            let sym = unsafe { obj.as_ref::<Symbol>() };
            PyObj::from(sym.namespace.is_none()).into_ptr()
        } else {
            utils::ref_false()
        }
    })
}
