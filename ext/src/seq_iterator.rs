use crate::object::PyObj;
use crate::common;
use crate::utils;
use crate::type_object as tpo;
use pyo3_ffi::*;

#[repr(C)]
pub struct SeqIterator {
    ob_base: PyObject,
    seq: PyObj,
}

pub fn from(coll: PyObj) -> Result<PyObj, ()> {
    let coll = common::seq(&coll)?;
    let obj = PyObj::alloc(seq_iterator_type());
    unsafe {
        let iter = obj.as_ref::<SeqIterator>();
        std::ptr::write(&mut iter.seq, coll);
    }
    Ok(obj)
}

pub fn seq_iterator_type() -> &'static PyObj {
    tpo::static_type!(tpo::TypeSpec {
        name: "lepet_ext.SeqIterator",
        flags: Py_TPFLAGS_DEFAULT,
        size: std::mem::size_of::<SeqIterator>(),
        new: Some(utils::disallowed_new!(seq_iterator_type)),
        dealloc: Some(tpo::generic_dealloc::<SeqIterator>),
        iter: Some(py_iter),
        next: Some(py_next),
        ..Default::default()
    })
}

extern "C" fn py_iter(
    self_: *mut PyObject,
) -> *mut PyObject {
    PyObj::borrow(self_).into_ptr()
}

unsafe extern "C" fn py_next(
    self_: *mut PyObject,
) -> *mut PyObject {
    utils::wrap_body!({
        let self_ = PyObj::borrow(self_);
        let iter = self_.as_ref::<SeqIterator>();
        let s = &iter.seq;
        if s.is_none() {
            PyErr_SetNone(PyExc_StopIteration);
            Err(())
        } else {
            let item = common::first(s)?;
            (*iter).seq = common::next(s)?;
            Ok(item)
        }
    })
}
