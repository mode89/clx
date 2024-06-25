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

    #[allow(dead_code)]
    #[inline]
    pub fn ref_count(&self) -> isize {
        unsafe { (*self.0).ob_refcnt }
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
    pub fn as_int(&self) -> Result<isize, ()> {
        let val = unsafe { PyLong_AsLong(self.0) };
        if val == -1 && unsafe { !PyErr_Occurred().is_null() } {
            Err(())
        } else {
            Ok(val as isize)
        }
    }

    #[inline]
    pub fn as_cstr(&self) -> Result<&std::ffi::CStr, ()> {
        let ptr = unsafe {
            PyUnicode_AsUTF8AndSize(self.0, std::ptr::null_mut())
        };
        if !ptr.is_null() {
            Ok(unsafe { std::ffi::CStr::from_ptr(ptr) })
        } else {
            Err(())
        }
    }

    #[allow(dead_code)]
    #[inline]
    pub fn to_pystr(&self) -> Option<PyObj> {
        let ptr = unsafe { PyObject_Str(self.0) };
        if !ptr.is_null() {
            Some(PyObj::from_owned_ptr(ptr))
        } else {
            None
        }
    }

    #[inline]
    pub fn tuple(len: isize) -> Result<PyObj, ()> {
        result_from_owned_ptr(unsafe { PyTuple_New(len) })
    }

    #[inline]
    pub fn tuple2(obj1: PyObj, obj2: PyObj) -> PyObj {
        PyObj::from_owned_ptr(unsafe {
            PyTuple_Pack(2, obj1.into_ptr(), obj2.into_ptr())
        })
    }

    #[inline]
    pub fn is_tuple(&self) -> bool {
        unsafe { PyTuple_Check(self.0) != 0 }
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
    pub fn get_type(&self) -> PyObj {
        PyObj::from_borrowed_ptr(unsafe { Py_TYPE(self.0) }.cast())
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
    pub fn py_hash(&self) -> Result<isize, ()> {
        let hash = unsafe { PyObject_Hash(self.0) };
        if hash == -1 && unsafe { !PyErr_Occurred().is_null() } {
            Err(())
        } else {
            Ok(hash)
        }
    }

    #[inline]
    pub fn into_hashable(self) -> Result<PyObjHashable, ()> {
        let hash = self.py_hash()?;
        Ok(PyObjHashable {
            obj: self.clone(),
            hash,
        })
    }

    #[inline]
    pub fn len(&self) -> Result<isize, ()> {
        let len = unsafe { PyObject_Size(self.0) };
        if len == -1 && unsafe { !PyErr_Occurred().is_null() } {
            Err(())
        } else {
            Ok(len)
        }
    }

    #[inline]
    pub fn get_item(&self, key: &PyObj) -> Result<PyObj, ()> {
        result_from_owned_ptr(unsafe { PyObject_GetItem(self.0, key.0) })
    }

    #[inline]
    pub fn get_attr(&self, name: &PyObj) -> Result<PyObj, ()> {
        result_from_owned_ptr(unsafe { PyObject_GetAttr(self.0, name.0) })
    }

    #[inline]
    pub fn get_iter(&self) -> Result<PyObj, ()> {
        result_from_owned_ptr(unsafe { PyObject_GetIter(self.0) })
    }

    #[inline]
    pub fn get_tuple_item(&self, index: isize) -> Result<PyObj, ()> {
        let ptr = unsafe { PyTuple_GetItem(self.0, index) };
        if !ptr.is_null() {
            Ok(PyObj::from_borrowed_ptr(ptr))
        } else {
            Err(())
        }
    }

    #[inline]
    pub fn set_tuple_item(
        &self,
        index: isize,
        value: PyObj
    ) -> Result<(), ()> {
        if unsafe { PyTuple_SetItem(self.0, index, value.into_ptr()) } == 0 {
            Ok(())
        } else {
            Err(())
        }
    }

    #[inline]
    pub fn next(&self) -> Option<PyObj> {
        let ptr = unsafe { PyIter_Next(self.0) };
        if !ptr.is_null() {
            Some(PyObj::from_owned_ptr(ptr))
        } else {
            None
        }
    }

    #[inline]
    pub fn is_callable(&self) -> bool {
        unsafe { PyCallable_Check(self.0) == 1 }
    }

    #[inline]
    pub fn call(&self, args: PyObj) -> Result<PyObj, ()> {
        result_from_owned_ptr(unsafe { PyObject_CallObject(self.0, args.0) })
    }

    #[inline]
    pub fn call0(&self) -> Result<PyObj, ()> {
        result_from_owned_ptr(unsafe { PyObject_CallNoArgs(self.0) })
    }

    #[inline]
    pub fn call_method0(&self, name: &PyObj) -> Result<PyObj, ()> {
        result_from_owned_ptr(unsafe {
            PyObject_CallMethodObjArgs(
                self.0, name.0, std::ptr::null::<PyObject>())
        })
    }

    #[inline]
    pub fn call_method1(
        &self,
        name: &PyObj,
        arg: PyObj
    ) -> Result<PyObj, ()> {
        result_from_owned_ptr(unsafe {
            PyObject_CallMethodObjArgs(self.0, name.0, arg.0,
                std::ptr::null::<PyObject>())
        })
    }

    #[inline]
    pub fn call_method2(
        &self,
        name: &PyObj,
        arg1: PyObj,
        arg2: PyObj
    ) -> Result<PyObj, ()> {
        result_from_owned_ptr(unsafe {
            PyObject_CallMethodObjArgs(self.0, name.0,
                arg1.0, arg2.0, std::ptr::null::<PyObject>())
        })
    }

    #[inline]
    pub fn import(name: &str) -> Result<PyObj, ()> {
        let name = std::ffi::CString::new(name).unwrap();
        result_from_owned_ptr(unsafe { PyImport_ImportModule(name.as_ptr()) })
    }

    #[inline]
    pub fn intern_string_in_place(self) -> PyObj {
        unsafe {
            PyUnicode_InternInPlace(
                &self.0 as *const _ as *mut *mut PyObject);
        }
        self
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

impl PartialEq for PyObj {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        unsafe { PyObject_RichCompareBool(self.0, other.0, Py_EQ) == 1 }
    }
}

impl Eq for PyObj {}

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

#[inline]
fn result_from_owned_ptr(ptr: *mut PyObject) -> Result<PyObj, ()> {
    if !ptr.is_null() {
        Ok(PyObj::from_owned_ptr(ptr))
    } else {
        Err(())
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct PyObjHashable {
    obj: PyObj,
    hash: isize,
}

impl PyObjHashable {
    #[inline]
    pub fn as_pyobj(&self) -> &PyObj {
        &self.obj
    }
}

impl Hash for PyObjHashable {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash.hash(state);
    }
}
