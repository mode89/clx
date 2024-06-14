use pyo3_ffi::*;
use std::hash::{Hash, Hasher};

// The pointer must be at the beginning of the structure, otherwise
// the offset calculation will be incorrect and member definitions will
// be broken
pub struct PyObj(*mut PyObject);

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
    pub fn as_int(&self) -> Option<isize> {
        let val = unsafe { PyLong_AsLong(self.0) };
        if val == -1 && unsafe { !PyErr_Occurred().is_null() } {
            None
        } else {
            Some(val as isize)
        }
    }

    #[inline]
    pub fn tuple2(obj1: PyObj, obj2: PyObj) -> PyObj {
        PyObj::from_owned_ptr(unsafe {
            PyTuple_Pack(2, obj1.into_ptr(), obj2.into_ptr())
        })
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
    pub fn is_instance(&self, tp: &PyObj) -> bool {
        unsafe { PyObject_IsInstance(self.0, tp.0) == 1 }
    }

    #[inline]
    pub fn none() -> PyObj {
        PyObj::from_borrowed_ptr(unsafe { Py_None() })
    }

    #[inline]
    pub fn is(&self, tp: &PyObj) -> bool {
        self.0 == tp.0
    }

    #[inline]
    pub fn py_hash(&self) -> isize {
        unsafe { PyObject_Hash(self.0) }
    }

    #[inline]
    pub fn get_item(&self, key: &PyObj) -> Option<PyObj> {
        let ptr = unsafe { PyObject_GetItem(self.0, key.0) };
        if ptr.is_null() {
            None
        } else {
            Some(PyObj::from_owned_ptr(ptr))
        }
    }

    #[inline]
    pub fn get_attr(&self, name: &PyObj) -> Option<PyObj> {
        let ptr = unsafe { PyObject_GetAttr(self.0, name.0) };
        if ptr.is_null() {
            None
        } else {
            Some(PyObj::from_owned_ptr(ptr))
        }
    }

    #[inline]
    pub fn get_iter(&self) -> Option<PyObj> {
        let ptr = unsafe { PyObject_GetIter(self.0) };
        if ptr.is_null() {
            None
        } else {
            Some(PyObj::from_owned_ptr(ptr))
        }
    }

    #[inline]
    pub fn next(&self) -> Option<PyObj> {
        let ptr = unsafe { PyIter_Next(self.0) };
        if ptr.is_null() {
            None
        } else {
            Some(PyObj::from_owned_ptr(ptr))
        }
    }

    #[inline]
    pub fn call_method0(&self, name: &PyObj) -> PyObj {
        PyObj::from_owned_ptr(unsafe {
            PyObject_CallMethodObjArgs(self.0, name.0,
                std::ptr::null::<PyObject>())
        })
    }

    #[inline]
    pub fn import(name: &str) -> PyObj {
        PyObj::from_owned_ptr(unsafe {
            let name = std::ffi::CString::new(name).unwrap();
            PyImport_ImportModule(name.as_ptr())
        })
    }
}

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

impl Hash for PyObj {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_isize(self.py_hash());
    }
}

impl PartialEq for PyObj {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        unsafe { PyObject_RichCompareBool(self.0, other.0, Py_EQ) == 1 }
    }
}

impl Eq for PyObj {}
