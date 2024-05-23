use pyo3_ffi::*;

pub const PY_TYPE_SLOT_DUMMY: PyType_Slot = PyType_Slot {
    slot: 0,
    pfunc: std::ptr::null_mut(),
};

pub const PY_MEMBER_DEF_DUMMY: PyMemberDef = PyMemberDef {
    name: std::ptr::null(),
    type_code: 0,
    offset: 0,
    flags: 0,
    doc: std::ptr::null(),
};

pub const PY_METHOD_DEF_DUMMY: PyMethodDef = PyMethodDef {
    ml_name: std::ptr::null(),
    ml_meth: unsafe { std::mem::transmute(0usize) },
    ml_flags: 0,
    ml_doc: std::ptr::null(),
};

pub const fn py_member_offset(offset: usize) -> isize {
    (std::mem::size_of::<PyObject>() + offset) as isize
}

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

// The pointer must be at the beginning of the structure, otherwise
// the offset calculation will be incorrect and member definitions will
// be broken
pub struct PyObj(*mut PyObject);

impl Drop for PyObj {
    #[inline]
    fn drop(&mut self) {
        unsafe { Py_XDECREF(self.0) };
    }
}

impl Clone for PyObj {
    #[inline]
    fn clone(&self) -> Self {
        unsafe { Py_XINCREF(self.0) };
        PyObj(self.0)
    }
}

impl std::ops::Deref for PyObj {
    type Target = *mut PyObject;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl PyObj {
    #[inline]
    pub fn from_borrowed_ptr(ptr: *mut PyObject) -> PyObj {
        unsafe { Py_XINCREF(ptr) };
        PyObj(ptr)
    }

    #[inline]
    pub fn from_owned_ptr(ptr: *mut PyObject) -> PyObj {
        PyObj(ptr)
    }

    #[inline]
    pub fn borrow(ptr: *mut PyObject) -> PyObj {
        PyObj::from_borrowed_ptr(ptr)
    }

    #[inline]
    pub fn own(ptr: *mut PyObject) -> PyObj {
        PyObj::from_owned_ptr(ptr)
    }

    #[inline]
    pub fn alloc(tp: &PyObj) -> PyObj {
        PyObj::own(unsafe { PyType_GenericAlloc(tp.as_ptr(), 0) })
    }

    #[inline]
    pub unsafe fn as_ref<T>(&self) -> &mut T {
        &mut *self.0.cast()
    }

    #[inline]
    pub unsafe fn as_ptr<T>(&self) -> *mut T {
        self.0.cast()
    }

    #[inline]
    pub fn into_ptr(self) -> *mut PyObject {
        let ptr = self.0;
        std::mem::forget(self);
        ptr
    }

    #[inline]
    pub fn is_none(&self) -> bool {
        unsafe { self.0 == Py_None() }
    }

    #[inline]
    pub fn is_string(&self) -> bool {
        unsafe { PyUnicode_Check(self.0) != 0 }
    }

    #[inline]
    pub fn type_ptr(&self) -> *mut PyTypeObject {
        unsafe { Py_TYPE(self.0) }
    }

    #[inline]
    pub fn type_is(&self, tp: &PyObj) -> bool {
        self.type_ptr() == unsafe { tp.as_ptr() }
    }

    #[inline]
    pub fn none() -> PyObj {
        PyObj::from_borrowed_ptr(unsafe { Py_None() })
    }
}

impl From<bool> for PyObj {
    #[inline]
    fn from(val: bool) -> Self {
        PyObj::from_borrowed_ptr(
            unsafe { if val { Py_True() } else { Py_False() }})
    }
}

impl From<i64> for PyObj {
    #[inline]
    fn from(val: i64) -> Self {
        PyObj::own(unsafe { PyLong_FromLong(val) })
    }
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
        #[allow(unused_unsafe)]
        utils::lazy_static!(*mut PyObject, {
            let cstring = std::ffi::CString::new($val).unwrap();
            unsafe { PyUnicode_InternFromString(cstring.as_ptr()) }
        }).clone()
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

pub(crate) use member_def;

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

pub(crate) use method_def;

pub extern "C" fn generic_dealloc<T>(obj: *mut PyObject) {
    unsafe {
        std::ptr::drop_in_place::<T>(obj.cast());
        let obj_type = &*Py_TYPE(obj);
        let free = obj_type.tp_free.unwrap();
        free(obj.cast());
    }
}

pub fn generic_dealloc_slot<T>() -> PyType_Slot {
    PyType_Slot {
        slot: Py_tp_dealloc,
        pfunc: generic_dealloc::<T> as *mut _,
    }
}

pub fn raise_exception(msg: &str) -> *mut PyObject {
    let msg = std::ffi::CString::new(msg).unwrap();
    unsafe {
        PyErr_SetString(PyExc_Exception, msg.as_ptr().cast());
    }
    std::ptr::null_mut()
}
