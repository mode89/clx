use crate::object::PyObj;
use crate::type_object::*;
use pyo3_ffi::*;

#[inline]
pub fn intern_string(s: &std::ffi::CStr) -> PyObj {
    unsafe {
        PyObj::own(PyUnicode_InternFromString(s.as_ptr()))
    }
}

#[inline]
pub fn intern_string_in_place(obj: PyObj) -> PyObj {
    unsafe {
        PyUnicode_InternInPlace(&*obj as *const _ as *mut *mut PyObject);
    }
    obj
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

macro_rules! module_add_method {
    ($module:expr, $func:expr) => {
        {
            let method = utils::lazy_static!(PyMethodDef, {
                PyMethodDef {
                    ml_name: utils::static_cstring!(stringify!($func))
                        .as_ptr().cast(),
                    ml_meth: PyMethodDefPointer { _PyCFunctionFast: $func },
                    ml_flags: METH_FASTCALL,
                    ml_doc: std::ptr::null(),
                }
            });

            let name = utils::static_cstring!(stringify!($func));
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

macro_rules! handle_gil_and_panic {
    ($code:block) => {{
        let gil = unsafe { PyGILState_Ensure() };
        let result = std::panic::catch_unwind(|| {
            $code
        }).unwrap_or_else(|_| {
            crate::utils::raise_exception("panic occurred in Rust code")
        });
        unsafe { PyGILState_Release(gil) };
        result
    }}
}

pub(crate) use handle_gil_and_panic;

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

#[allow(unused_macros)]
macro_rules! member_def {
    ($index:expr, $name:expr) => {
        PyMemberDef {
            name: utils::static_cstring!(stringify!($name)).as_ptr().cast(),
            type_code: Py_T_OBJECT_EX,
            offset: (std::mem::size_of::<PyObject>() +
                     std::mem::size_of::<PyObj>() * $index) as isize,
            flags: Py_READONLY,
            doc: std::ptr::null(),
        }
    };
}

#[allow(unused_imports)]
pub(crate) use member_def;

#[allow(unused_macros)]
macro_rules! method_def {
    ($name:expr, $func:expr) => {
        PyMethodDef {
            ml_name: utils::static_cstring!($name).as_ptr().cast(),
            ml_meth: PyMethodDefPointer { _PyCFunctionFast: $func },
            ml_flags: METH_FASTCALL,
            ml_doc: std::ptr::null(),
        }
    };
}

#[allow(unused_imports)]
pub(crate) use method_def;

pub extern "C" fn generic_dealloc<T>(obj: *mut PyObject) {
    unsafe {
        std::ptr::drop_in_place::<T>(obj.cast());
        let obj_type = &*Py_TYPE(obj);
        let free = obj_type.tp_free.unwrap();
        free(obj.cast());
    }
}

pub fn raise_exception(msg: &str) -> *mut PyObject {
    let msg = std::ffi::CString::new(msg).unwrap();
    unsafe {
        PyErr_SetString(PyExc_Exception, msg.as_ptr().cast());
    }
    std::ptr::null_mut()
}

pub fn seq(obj: &PyObj) -> PyObj {
    if obj.is_none() {
        return obj.clone();
    } else if obj.type_is(crate::list::list_type()) {
        crate::list::list_seq(obj)
    } else {
        obj.call_method0(&static_pystring!("seq"))
    }
}

pub fn first(obj: &PyObj) -> PyObj {
    if obj.is_none() {
        return obj.clone();
    } else if obj.type_is(crate::list::list_type()) {
        crate::list::list_first(obj)
    } else {
        obj.call_method0(&static_pystring!("first"))
    }
}

pub fn next(obj: &PyObj) -> PyObj {
    if obj.is_none() {
        return obj.clone();
    } else if obj.type_is(crate::list::list_type()) {
        crate::list::list_next(obj)
    } else {
        obj.call_method0(&static_pystring!("next"))
    }
}

pub fn sequential_eq(self_: &PyObj, other: &PyObj) -> bool {
    let mut x = seq(self_);
    let mut y = seq(other);
    loop {
        if x.is_none() {
            return y.is_none();
        } else if y.is_none() {
            return false;
        } else if x.is(&y) {
            return true;
        } else if first(&x) != first(&y) {
            return false;
        } else {
            x = next(&x);
            y = next(&y);
        }
    }
}

#[repr(C)]
pub struct SeqIterator {
    ob_base: PyObject,
    seq: PyObj,
}

pub fn seq_iterator_type() -> &'static PyObj {
    static_type!(TypeSpec {
        name: "clx_rust.SeqIterator",
        flags: Py_TPFLAGS_DEFAULT | Py_TPFLAGS_DISALLOW_INSTANTIATION,
        size: std::mem::size_of::<SeqIterator>(),
        dealloc: Some(generic_dealloc::<SeqIterator>),
        iter: Some(seq_iterator_iter),
        next: Some(seq_iterator_next),
        ..Default::default()
    })
}

pub fn seq_iterator(coll: PyObj) -> PyObj {
    let obj = PyObj::alloc(seq_iterator_type());
    unsafe {
        let iter = obj.as_ref::<SeqIterator>();
        std::ptr::write(&mut iter.seq, seq(&coll));
    }
    obj
}

unsafe extern "C" fn seq_iterator_iter(
    self_: *mut PyObject,
) -> *mut PyObject {
    PyObj::borrow(self_).into_ptr()
}

unsafe extern "C" fn seq_iterator_next(
    self_: *mut PyObject,
) -> *mut PyObject {
    let self_ = PyObj::borrow(self_);
    let iter = self_.as_ref::<SeqIterator>();
    let s = &iter.seq;
    if s.is_none() {
        PyErr_SetNone(PyExc_StopIteration);
        std::ptr::null_mut()
    } else {
        let item = first(s);
        (*iter).seq = next(s);
        item.into_ptr()
    }
}
