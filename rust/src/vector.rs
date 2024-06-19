use crate::object::PyObj;
use crate::type_object::*;
use crate::list;
use crate::utils;
use crate::protocols::*;
use pyo3_ffi::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub fn init_module(module: *mut PyObject) {
    utils::module_add_method!(module, vector);
    utils::module_add_method!(module, is_vector);
    utils::module_add_type!(module, PersistentVector, vector_type());
}

#[repr(C)]
pub struct Vector {
    ob_base: PyObject,
    meta: PyObj,
    impl_: Vec<PyObj>,
    hash: Option<isize>,
}

pub fn vector_type() -> &'static PyObj {
    static_type!(
        TypeSpec {
            name: "clx_rust.PersistentVector".to_string(),
            bases: vec![
                imeta_type(),
                iindexed_type(),
                iseqable_type(),
                isequential_type(),
                icollection_type(),
            ],
            flags: Py_TPFLAGS_DEFAULT |
                   Py_TPFLAGS_DISALLOW_INSTANTIATION,
            size: std::mem::size_of::<Vector>(),
            dealloc: Some(utils::generic_dealloc::<Vector>),
            sequence_length: Some(py_vector_len),
            compare: Some(py_vector_compare),
            // TODO hash: Some(py_vector_hash),
            sequence_item: Some(py_vector_item),
            call: Some(py_vector_call),
            members: vec![ "__meta__".to_string() ],
            methods: vec![
                // TODO ("with_meta", py_vector_with_meta),
                ("count_", py_vector_count),
                ("nth", py_vector_nth),
                ("first", py_vector_first),
                ("seq", py_vector_seq),
                // TODO ("conj", py_vector_conj),
            ],
            ..TypeSpec::default()
        }
    )
}

fn _vector(impl_: Vec<PyObj>, meta: PyObj, hash: Option<isize>) -> PyObj {
    unsafe {
        let obj = PyObj::alloc(vector_type());
        let v = obj.as_ref::<Vector>();
        std::ptr::write(&mut v.impl_, impl_);
        std::ptr::write(&mut v.meta, meta);
        v.hash = hash;
        obj
    }
}

fn empty_vector() -> PyObj {
    utils::lazy_static!(PyObj, {
        let mut hasher = DefaultHasher::new();
        "empty_vector".hash(&mut hasher);
        _vector(vec![], PyObj::none(), Some(hasher.finish() as isize))
    }).clone()
}

extern "C" fn vector(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs == 0 {
            Ok(empty_vector())
        } else {
            let mut impl_ = Vec::with_capacity(nargs as usize);
            for i in 0..nargs {
                impl_.push(PyObj::borrow(unsafe { *args.offset(i) }));
            }
            Ok(_vector(impl_, PyObj::none(), None))
        }
    })
}

unsafe extern "C" fn is_vector(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    _nargs: isize,
) -> *mut PyObject {
    utils::handle_gil!({
        PyObj::from(Py_TYPE(*args) == vector_type().as_ptr()).into_ptr()
    })
}

extern "C" fn py_vector_compare(
    self_: *mut PyObject,
    other: *mut PyObject,
    op: i32,
) -> *mut PyObject {
    utils::wrap_body!({
        let self_ = PyObj::borrow(self_);
        let other = PyObj::borrow(other);
        match op {
            pyo3_ffi::Py_EQ => Ok(PyObj::from(vector_eq(&self_, &other)?)),
            pyo3_ffi::Py_NE => Ok(PyObj::from(!vector_eq(&self_, &other)?)),
            _ => utils::raise_exception("vector comparison not supported")
        }
    })
}

fn vector_eq(self_: &PyObj, other: &PyObj) -> Result<bool, ()> {
    if self_.is(other) {
        Ok(true)
    } else {
        let vself = unsafe { self_.as_ref::<Vector>() };
        if other.type_is(vector_type()) {
            let vother = unsafe { other.as_ref::<Vector>() };
            if vself.impl_.len() == vother.impl_.len() {
                for i in 0..vself.impl_.len() {
                    if vself.impl_[i] != vother.impl_[i] {
                        return Ok(false);
                    }
                }
                Ok(true)
            } else {
                Ok(false)
            }
        } else if other.is_instance(isequential_type()) {
            utils::sequential_eq(&vector_seq(self_), other)
        } else {
            Ok(false)
        }
    }
}

extern "C" fn py_vector_count(
    self_: *mut PyObject,
    _args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs != 0 {
            utils::raise_exception(
                "PersistentVector.count_() takes no arguments")
        } else {
            let self_ = PyObj::borrow(self_);
            let self_ = unsafe { self_.as_ref::<Vector>() };
            Ok(PyObj::from(self_.impl_.len() as i64))
        }
    })
}

extern "C" fn py_vector_first(
    self_: *mut PyObject,
    _args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs != 0 {
            utils::raise_exception(
                "PersistentVector.first() takes no arguments")
        } else {
            Ok(vector_first(&PyObj::borrow(self_)))
        }
    })
}

#[inline]
pub fn vector_first(self_: &PyObj) -> PyObj {
    let v = unsafe { self_.as_ref::<Vector>() };
    if v.impl_.is_empty() {
        PyObj::none()
    } else {
        v.impl_[0].clone()
    }
}

unsafe extern "C" fn py_vector_len(
    self_: *mut PyObject,
) -> isize {
    let self_ = self_ as *mut Vector;
    (*self_).impl_.len() as isize
}

extern "C" fn py_vector_seq(
    self_: *mut PyObject,
    _args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs != 0 {
            utils::raise_exception(
                "PersistentVector.seq() takes no arguments")
        } else {
            Ok(vector_seq(&PyObj::borrow(self_)))
        }
    })
}

pub fn vector_seq(self_: &PyObj) -> PyObj {
    let v = unsafe { self_.as_ref::<Vector>() };
    if v.impl_.is_empty() {
        PyObj::none()
    } else {
        let mut lst = list::empty_list();
        for i in (0..v.impl_.len()).rev() {
            let item = v.impl_[i].clone();
            lst = list::list_conj(lst, item);
        }
        lst
    }
}

extern "C" fn py_vector_item(
    self_: *mut PyObject,
    index: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        let self_ = PyObj::borrow(self_);
        vector_nth(&self_, index, None)
    })
}

extern "C" fn py_vector_nth(
    self_: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        let not_found = match nargs {
            1 => None,
            2 => Some(PyObj::borrow(unsafe { *args.offset(1) })),
            _ => {
                return utils::raise_exception(
                    "PersistentVector.nth() takes one or two arguments")
            }
        };

        let index = unsafe { PyObj::borrow(*args) }.as_int()?;
        let self_ = PyObj::borrow(self_);
        vector_nth(&self_, index, not_found)
    })
}

unsafe extern "C" fn py_vector_call(
    self_: *mut PyObject,
    args: *mut PyObject,
    _kw: *mut PyObject,
) -> *mut PyObject {
    utils::wrap_body!({
        let index: *mut PyObject = std::ptr::null_mut();
        if PyArg_UnpackTuple(args,
                "PersistentVector.__call__()\0".as_ptr().cast(),
                1, 1, &index) != 0 {
            let index = PyObj::borrow(index).as_int()?;
            let self_ = PyObj::borrow(self_);
            vector_nth(&self_, index, None)
        } else {
            Err(())
        }
    })
}

fn vector_nth(
    self_: &PyObj,
    index: isize,
    not_found: Option<PyObj>
) -> Result<PyObj, ()> {
    let v = unsafe { self_.as_ref::<Vector>() };
    if index < 0 || index >= v.impl_.len() as isize {
        match not_found {
            Some(not_found) => Ok(not_found),
            None => raise_index_error()
        }
    } else {
        Ok(v.impl_[index as usize].clone())
    }
}

fn raise_index_error() -> Result<PyObj, ()> {
    unsafe {
        PyErr_SetString(
            PyExc_IndexError,
            "PersistentVector index out of range\0".as_ptr().cast(),
        );
    }
    Err(())
}
