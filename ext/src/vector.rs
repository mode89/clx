use crate::object::PyObj;
use crate::type_object as tpo;
use crate::utils;
use crate::protocols::*;
use crate::indexed_seq;
use crate::common;
use pyo3_ffi::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub fn init_module(module: *mut PyObject) {
    utils::module_add_method!(module, vector, py_vector);
    utils::module_add_method!(module, is_vector, py_is_vector);
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
    tpo::static_type!(
        tpo::TypeSpec {
            name: "lepet_ext.PersistentVector",
            bases: vec![
                imeta_type(),
                iindexed_type(),
                iseqable_type(),
                isequential_type(),
                icollection_type(),
            ],
            flags: Py_TPFLAGS_DEFAULT,
            size: std::mem::size_of::<Vector>(),
            new: Some(utils::disallowed_new!(vector_type)),
            dealloc: Some(tpo::generic_dealloc::<Vector>),
            sequence_length: Some(py_vector_len),
            compare: Some(py_vector_compare),
            // TODO hash: Some(py_vector_hash),
            sequence_item: Some(py_vector_item),
            call: Some(py_vector_call),
            members: vec![ tpo::member!("__meta__") ],
            methods: vec![
                // TODO ("with_meta", py_vector_with_meta),
                tpo::method!("count_", py_vector_count),
                tpo::method!("nth", py_vector_nth),
                tpo::method!("seq", py_vector_seq),
                tpo::method!("conj", py_conj),
            ],
            ..Default::default()
        }
    )
}

extern "C" fn py_vector(
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
            Ok(vector(impl_, PyObj::none(), None))
        }
    })
}

fn vector(impl_: Vec<PyObj>, meta: PyObj, hash: Option<isize>) -> PyObj {
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
        vector(vec![], PyObj::none(), Some(hasher.finish() as isize))
    }).clone()
}

unsafe extern "C" fn py_is_vector(
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
            common::sequential_eq(self_, other)
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
            Ok(seq(&PyObj::borrow(self_)))
        }
    })
}

pub fn seq(self_: &PyObj) -> PyObj {
    let v = unsafe { self_.as_ref::<Vector>() };
    if !v.impl_.is_empty() {
        indexed_seq::new(self_.clone(), v.impl_.len(), 0)
    } else {
        PyObj::none()
    }
}

extern "C" fn py_conj(
    self_: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs == 1 {
            Ok(conj(PyObj::borrow(self_), PyObj::borrow(unsafe { *args })))
        } else {
            utils::raise_exception(
                "PersistentVector.conj() expects one argument")
        }
    })
}

#[inline]
pub fn conj(self_: PyObj, item: PyObj) -> PyObj {
    let v = unsafe { self_.as_ref::<Vector>() };
    let mut impl_ = v.impl_.clone();
    impl_.push(item.clone());
    vector(impl_, PyObj::none(), None)
}

extern "C" fn py_vector_item(
    self_: *mut PyObject,
    index: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        let self_ = PyObj::borrow(self_);
        nth(&self_, index, None)
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
        nth(&self_, index, not_found)
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
            nth(&self_, index, None)
        } else {
            Err(())
        }
    })
}

pub fn nth(
    self_: &PyObj,
    index: isize,
    not_found: Option<PyObj>
) -> Result<PyObj, ()> {
    let v = unsafe { self_.as_ref::<Vector>() };
    if index < 0 || index >= v.impl_.len() as isize {
        match not_found {
            Some(not_found) => Ok(not_found),
            None => utils::raise_index_error()
        }
    } else {
        Ok(v.impl_[index as usize].clone())
    }
}
