use crate::object::PyObj;
use pyo3_ffi::*;

#[inline]
pub fn intern_string(s: &std::ffi::CStr) -> PyObj {
    unsafe {
        PyObj::own(PyUnicode_InternFromString(s.as_ptr()))
    }
}

#[inline]
pub fn incref(obj: *mut PyObject) -> *mut PyObject {
    unsafe { Py_XINCREF(obj) };
    obj
}

#[inline]
pub fn ref_false() -> *mut PyObject {
    incref(unsafe { Py_False() })
}

macro_rules! lazy_static {
    ($type:ty, $code:block) => {
        {
            use std::sync::OnceLock;
            static mut CELL: OnceLock<$type> = OnceLock::new();
            static CODE: fn() -> $type = || $code;
            unsafe { CELL.get_or_init(CODE) }
        }
    };
}

pub(crate) use lazy_static;

macro_rules! static_cstring {
    ($val:expr) => {
        {
            use std::ffi::CString;
            crate::utils::lazy_static!(CString, {
                CString::new($val).unwrap()
            })
        }
    };
}

pub(crate) use static_cstring;

macro_rules! static_pystring {
    ($val:expr) => {
        {
            #[allow(unused_unsafe)]
            crate::utils::lazy_static!(PyObj, {
                let cstring = std::ffi::CString::new($val).unwrap();
                unsafe {
                    PyObj::own(PyUnicode_InternFromString(cstring.as_ptr()))
                }
            }).clone()
        }
    };
}

pub(crate) use static_pystring;

macro_rules! static_pynumber {
    ($val:expr) => {
        crate::utils::lazy_static!(PyObj, { PyObj::from($val) })
    };
}

pub(crate) use static_pynumber;

macro_rules! module_add_method {
    ($module:expr, $name:ident, $func:ident) => {
        {
            let method = utils::lazy_static!(PyMethodDef, {
                PyMethodDef {
                    ml_name: utils::static_cstring!(stringify!($name))
                        .as_ptr().cast(),
                    ml_meth: PyMethodDefPointer { _PyCFunctionFast: $func },
                    ml_flags: METH_FASTCALL,
                    ml_doc: std::ptr::null(),
                }
            });

            let name = utils::static_cstring!(stringify!($name));
            unsafe {
                PyModule_AddObject(
                    $module,
                    name.as_ptr().cast(),
                    PyCFunction_New(
                        method as *const _ as *mut _,
                        std::ptr::null_mut())
                );
            }
        }
    };
}

pub(crate) use module_add_method;

macro_rules! module_add_type {
    ($module:expr, $type:ty, $type_obj:expr) => {
        {
            let name = utils::static_cstring!(stringify!($type));
            unsafe {
                PyModule_AddObject(
                    $module,
                    name.as_ptr().cast(),
                    $type_obj.clone().into_ptr()
                );
            }
        }
    };
}

pub(crate) use module_add_type;

macro_rules! wrap_body {
    ($code:block) => {{
        let gil = unsafe { PyGILState_Ensure() };
        let result: Result<PyObj, ()> = std::panic::catch_unwind(|| {
            $code
        }).unwrap_or_else(|_| {
            crate::utils::raise_exception("panic occurred in Rust code")
        });
        unsafe { PyGILState_Release(gil) };
        match result {
            Ok(obj) => obj.into_ptr(),
            Err(_) => std::ptr::null_mut(),
        }
    }}
}

pub(crate) use wrap_body;

macro_rules! handle_gil {
    ($code:block) => {
        {
            let gil = unsafe { PyGILState_Ensure() };
            let result = { $code };
            unsafe { PyGILState_Release(gil) };
            result
        }
    }
}

pub(crate) use handle_gil;

macro_rules! disallowed_new {
    ($type:expr) => {
        {
            extern "C" fn new(
                cls: *mut PyTypeObject,
                args: *mut PyObject,
                kws: *mut PyObject,
            ) -> *mut PyObject {
                if cls == unsafe { $type().as_ptr() } {
                    utils::wrap_body!({
                        use crate::object::PyObj;
                        use crate::utils;
                        let cls = PyObj::borrow(cls as *mut PyObject);
                        let name = cls.get_attr(
                            &utils::static_pystring!("__name__"))?;
                        let name = name.as_cstr()?.to_str().unwrap();
                        let module = cls.get_attr(
                            &utils::static_pystring!("__module__"))?;
                        let module = module.as_cstr()?.to_str().unwrap();
                        utils::raise_exception(
                            &format!("{}.{} cannot be instantiated directly",
                                module, name))
                    })
                } else {
                    unsafe { PyType_GenericNew(cls, args, kws) }
                }
            }
            new
        }
    }
}

pub(crate) use disallowed_new;

pub extern "C" fn generic_dealloc<T>(obj: *mut PyObject) {
    unsafe {
        std::ptr::drop_in_place::<T>(obj.cast());
        let obj_type = &*Py_TYPE(obj);
        let free = obj_type.tp_free.unwrap();
        free(obj.cast());
    }
}

pub fn raise_exception(msg: &str) -> Result<PyObj, ()> {
    set_exception(msg);
    Err(())
}

pub fn py_assert(cond: bool, msg: &str) -> Result<PyObj, ()> {
    if cond {
        Ok(PyObj::none())
    } else {
        raise_exception(msg)
    }
}

#[inline]
pub fn set_exception(msg: &str) {
    let msg = std::ffi::CString::new(msg).unwrap();
    unsafe {
        PyErr_SetString(PyExc_Exception, msg.as_ptr().cast());
    }
}
