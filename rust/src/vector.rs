use crate::list;
use crate::utils;
use crate::protocols::*;
use utils::PyObj;
use pyo3_ffi::*;

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
    utils::static_type!(
        utils::TypeSpec {
            name: "clx_rust.PersistentVector",
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
            members: vec![ "__meta__" ],
            methods: vec![
                // TODO ("with_meta", py_vector_with_meta),
                ("count_", py_vector_count),
                ("nth", py_vector_nth),
                ("first", py_vector_first),
                ("seq", py_vector_seq),
                // TODO ("conj", py_vector_conj),
            ],
            ..utils::TypeSpec::default()
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
        _vector(vec![], PyObj::none(), None)
    }).clone()
}

extern "C" fn vector(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::handle_gil_and_panic!({
        if nargs == 0 {
            empty_vector().into_ptr()
        } else {
            let mut impl_ = Vec::with_capacity(nargs as usize);
            for i in 0..nargs {
                impl_.push(PyObj::borrow(unsafe { *args.offset(i) }));
            }
            _vector(impl_, PyObj::none(), None).into_ptr()
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
    utils::handle_gil_and_panic!({
        let self_ = PyObj::borrow(self_);
        let other = PyObj::borrow(other);
        PyObj::from(match op {
            pyo3_ffi::Py_EQ => vector_eq(&self_, &other),
            pyo3_ffi::Py_NE => !vector_eq(&self_, &other),
            _ => {
                return utils::raise_exception(
                    "vector comparison not supported");
            }
        }).into_ptr()
    })
}

fn vector_eq(self_: &PyObj, other: &PyObj) -> bool {
    if self_.is(other) {
        true
    } else {
        let vself = unsafe { self_.as_ref::<Vector>() };
        if other.type_is(vector_type()) {
            let vother = unsafe { other.as_ref::<Vector>() };
            if vself.impl_.len() == vother.impl_.len() {
                for i in 0..vself.impl_.len() {
                    if vself.impl_[i] != vother.impl_[i] {
                        return false;
                    }
                }
                true
            } else {
                false
            }
        } else if other.is_instance(isequential_type()) {
            utils::sequential_eq(&vector_seq(self_), other)
        } else {
            false
        }
    }
}

extern "C" fn py_vector_count(
    self_: *mut PyObject,
    _args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::handle_gil_and_panic!({
        if nargs != 0 {
            utils::raise_exception(
                "PersistentVector.count_() takes no arguments")
        } else {
            let self_ = PyObj::borrow(self_);
            let self_ = unsafe { self_.as_ref::<Vector>() };
            PyObj::from(self_.impl_.len() as i64).into_ptr()
        }
    })
}

extern "C" fn py_vector_first(
    self_: *mut PyObject,
    _args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::handle_gil_and_panic!({
        if nargs != 0 {
            utils::raise_exception(
                "PersistentVector.first() takes no arguments")
        } else {
            vector_first(&PyObj::borrow(self_)).into_ptr()
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
    utils::handle_gil_and_panic!({
        if nargs != 0 {
            utils::raise_exception(
                "PersistentVector.seq() takes no arguments")
        } else {
            vector_seq(&PyObj::borrow(self_)).into_ptr()
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
    utils::handle_gil_and_panic!({
        let self_ = PyObj::borrow(self_);
        match vector_nth(&self_, index, None) {
            Some(item) => item.into_ptr(),
            None => std::ptr::null_mut(),
        }
    })
}

extern "C" fn py_vector_nth(
    self_: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::handle_gil_and_panic!({
        let not_found = match nargs {
            1 => None,
            2 => Some(PyObj::borrow(unsafe { *args.offset(1) })),
            _ => {
                return utils::raise_exception(
                    "PersistentVector.nth() takes one or two arguments")
            }
        };

        let index = unsafe { PyObj::borrow(*args) };
        match index.as_int() {
            Some(index) => {
                let v = PyObj::borrow(self_);
                match vector_nth(&v, index, not_found) {
                    Some(item) => item.into_ptr(),
                    None => std::ptr::null_mut(),
                }
            },
            None => std::ptr::null_mut(),
        }
    })
}

unsafe extern "C" fn py_vector_call(
    self_: *mut PyObject,
    args: *mut PyObject,
    _kw: *mut PyObject,
) -> *mut PyObject {
    utils::handle_gil_and_panic!({
        let index: *mut PyObject = std::ptr::null_mut();
        if PyArg_UnpackTuple(args,
                "PersistentVector.__call__()\0".as_ptr().cast(),
                1, 1, &index) != 0 {
            let index = PyObj::borrow(index);
            let self_ = PyObj::borrow(self_);
            match index.as_int() {
                Some(index) => {
                    match vector_nth(&self_, index, None) {
                        Some(item) => item.into_ptr(),
                        None => std::ptr::null_mut(),
                    }
                },
                None => std::ptr::null_mut(),
            }
        } else {
            std::ptr::null_mut()
        }
    })
}

fn vector_nth(
    self_: &PyObj,
    index: isize,
    not_found: Option<PyObj>
) -> Option<PyObj> {
    let v = unsafe { self_.as_ref::<Vector>() };
    if index < 0 || index >= v.impl_.len() as isize {
        match not_found {
            Some(not_found) => Some(not_found),
            None => {
                raise_index_error();
                return None;
            }
        }
    } else {
        Some(v.impl_[index as usize].clone())
    }
}

fn raise_index_error() {
    unsafe {
        PyErr_SetString(
            PyExc_IndexError,
            "PersistentVector index out of range\0".as_ptr().cast(),
        );
    }
}
