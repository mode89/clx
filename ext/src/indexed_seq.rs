use crate::object::PyObj;
use crate::type_object as tpo;
use crate::utils;
use crate::protocols::*;
use crate::list;
use crate::common;
use crate::seq_iterator;
use pyo3_ffi::*;

pub fn init_module(module: *mut PyObject) {
    utils::module_add_type!(module, IndexedSeq, class());
}

#[repr(C)]
pub struct IndexedSeq
{
    ob_base: PyObject,
    meta: PyObj,
    coll: PyObj,
    len: usize,
    offset: usize,
}

pub fn class() -> &'static PyObj {
    tpo::static_type!(tpo::TypeSpec {
        name: "lepet_ext.IndexedSeq",
        bases: vec![
            imeta_type(),
            iseq_type(),
            isequential_type(),
            icounted_type(),
        ],
        flags: Py_TPFLAGS_DEFAULT,
        size: std::mem::size_of::<IndexedSeq>(),
        new: Some(utils::disallowed_new!(class)),
        dealloc: Some(tpo::generic_dealloc::<IndexedSeq>),
        compare: Some(py_compare),
        iter: Some(py_iter),
        members: vec![ tpo::member!("__meta__") ],
        methods: vec![
            tpo::method!("first", py_first),
            tpo::method!("rest", py_rest),
            tpo::method!("next", py_next),
            tpo::method!("seq", py_seq),
            tpo::method!("count_", py_count),
        ],
        ..Default::default()
    })
}

#[inline]
pub fn new(coll: PyObj, len: usize, offset: usize) -> PyObj {
    let obj = PyObj::alloc(class());
    unsafe {
        let iseq = obj.as_ref::<IndexedSeq>();
        std::ptr::write(&mut iseq.meta, PyObj::none());
        std::ptr::write(&mut iseq.coll, coll);
        iseq.len = len;
        iseq.offset = offset;
    }
    obj
}

extern "C" fn py_compare(
    self_: *mut PyObject,
    other: *mut PyObject,
    op: i32,
) -> *mut PyObject {
    utils::wrap_body!({
        let self_ = PyObj::borrow(self_);
        let other = PyObj::borrow(other);
        Ok(PyObj::from(match op {
            pyo3_ffi::Py_EQ => common::sequential_eq(&self_, &other)?,
            pyo3_ffi::Py_NE => !common::sequential_eq(&self_, &other)?,
            _ => {
                return utils::raise_exception(
                    "IndexedSeq comparison not supported");
            }
        }))
    })
}

extern "C" fn py_first(
    self_: *mut PyObject,
    _args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs != 0 {
            utils::raise_exception("IndexedSeq.first() takes no arguments")
        } else {
            first(&PyObj::borrow(self_))
        }
    })
}

#[inline]
pub fn first(self_: &PyObj) -> Result<PyObj, ()> {
    let self_ = unsafe { self_.as_ref::<IndexedSeq>() };
    common::nth(&self_.coll, self_.offset as isize, Some(PyObj::none()))
}

extern "C" fn py_rest(
    self_: *mut PyObject,
    _args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs != 0 {
            utils::raise_exception("IndexedSeq.rest() takes no arguments")
        } else {
            rest(&PyObj::borrow(self_))
        }
    })
}

#[inline]
pub fn rest(self_: &PyObj) -> Result<PyObj, ()> {
    let self_ = unsafe { self_.as_ref::<IndexedSeq>() };
    if self_.offset < (self_.len - 1) {
        Ok(new(self_.coll.clone(), self_.len, self_.offset + 1))
    } else {
        Ok(list::empty_list())
    }
}

extern "C" fn py_next(
    self_: *mut PyObject,
    _args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs != 0 {
            utils::raise_exception("IndexedSeq.next() takes no arguments")
        } else {
            next(&PyObj::borrow(self_))
        }
    })
}

#[inline]
pub fn next(self_: &PyObj) -> Result<PyObj, ()> {
    let self_ = unsafe { self_.as_ref::<IndexedSeq>() };
    if self_.offset < (self_.len - 1) {
        Ok(new(self_.coll.clone(), self_.len, self_.offset + 1))
    } else {
        Ok(PyObj::none())
    }
}

extern "C" fn py_seq(
    self_: *mut PyObject,
    _args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs != 0 {
            utils::raise_exception("IndexedSeq.seq() takes no arguments")
        } else {
            seq(&PyObj::borrow(self_))
        }
    })
}

#[inline]
pub fn seq(self_: &PyObj) -> Result<PyObj, ()> {
    Ok(self_.clone())
}

extern "C" fn py_count(
    self_: *mut PyObject,
    _args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs != 0 {
            utils::raise_exception("IndexedSeq.count_() takes no arguments")
        } else {
            let self_ = PyObj::borrow(self_);
            Ok(PyObj::from(count(&self_)))
        }
    })
}

#[inline]
pub fn count(self_: &PyObj) -> i64 {
    let self_ = unsafe { self_.as_ref::<IndexedSeq>() };
    (self_.len - self_.offset) as i64
}

extern "C" fn py_iter(
    self_: *mut PyObject,
) -> *mut PyObject {
    utils::wrap_body!({
        let self_ = PyObj::borrow(self_);
        seq_iterator::from(self_)
    })
}

#[inline]
pub fn drop(self_: PyObj, n: usize) -> PyObj {
    let self_ = unsafe { self_.as_ref::<IndexedSeq>() };
    let offset = self_.offset + n;
    if offset < self_.len {
        new(self_.coll.clone(), self_.len, offset)
    } else {
        list::empty_list()
    }
}
