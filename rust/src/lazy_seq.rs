use crate::object::PyObj;
use crate::type_object as tpo;
use crate::utils;
use crate::protocols::*;
use crate::common;
use crate::seq_iterator;
use pyo3_ffi::*;
use std::sync::Mutex;

pub fn init_module(module: *mut PyObject) {
    utils::module_add_method!(module, lazy_seq, py_lazy_seq);
    utils::module_add_type!(module, LazySeq, lazyseq_type());
}

#[repr(C)]
pub struct LazySeq
{
    ob_base: PyObject,
    meta: PyObj,
    func: Option<Box<Func>>,
    pub seq: Mutex<PyObj>,
}

type Func = dyn Fn() -> Result<PyObj, ()>;

pub fn lazyseq_type() -> &'static PyObj {
    tpo::static_type!(tpo::TypeSpec {
        name: "clx_rust.LazySeq",
        bases: vec![
            imeta_type(),
            iseq_type(),
            isequential_type(),
        ],
        flags: Py_TPFLAGS_DEFAULT,
        size: std::mem::size_of::<LazySeq>(),
        new: Some(utils::disallowed_new!(lazyseq_type)),
        dealloc: Some(utils::generic_dealloc::<LazySeq>),
        compare: Some(py_lazyseq_compare),
        iter: Some(py_lazyseq_iter),
        // TODO hash: Some(py_vector_hash),
        members: vec![ tpo::member!("__meta__") ],
        methods: vec![
            // TODO ("with_meta", py_vector_with_meta),
            tpo::method!("first", py_lazyseq_first),
            tpo::method!("rest", py_lazyseq_rest),
            tpo::method!("next", py_lazyseq_next),
            tpo::method!("seq", py_lazyseq_seq),
            // TODO ("conj", py_vector_conj),
        ],
        ..Default::default()
    })
}

extern "C" fn py_lazy_seq(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        let func = PyObj::borrow(unsafe { *args });
        if nargs != 1 || !func.is_callable() {
            utils::raise_exception(
                "lazy_seq() requires a callable as its argument")
        } else {
            Ok(lazy_seq(Box::new(move || func.call0())))
        }
    })
}

#[inline]
pub fn lazy_seq(func: Box<Func>) -> PyObj {
    let obj = PyObj::alloc(lazyseq_type());
    unsafe {
        let lseq = obj.as_ref::<LazySeq>();
        std::ptr::write(&mut lseq.meta, PyObj::none());
        std::ptr::write(&mut lseq.func, Some(func));
        std::ptr::write(&mut lseq.seq, Mutex::new(PyObj::none()));
    }
    obj
}

extern "C" fn py_lazyseq_compare(
    self_: *mut PyObject,
    other: *mut PyObject,
    op: i32,
) -> *mut PyObject {
    utils::wrap_body!({
        let self_ = PyObj::borrow(self_);
        let other = PyObj::borrow(other);
        Ok(PyObj::from(match op {
            pyo3_ffi::Py_EQ => equal(&self_, &other)?,
            pyo3_ffi::Py_NE => !equal(&self_, &other)?,
            _ => {
                return utils::raise_exception(
                    "LazySeq comparison not supported");
            }
        }))
    })
}

fn equal(self_: &PyObj, other: &PyObj) -> Result<bool, ()> {
    if self_.is(other) {
        Ok(true)
    } else {
        if other.is_instance(isequential_type()) {
            common::sequential_eq(self_, other)
        } else {
            Ok(false)
        }
    }
}

#[inline]
fn force1(self_: &PyObj) -> Result<PyObj, ()> {
    unsafe {
        let self_ = self_.as_ref::<LazySeq>();
        let mut seq = self_.seq.lock().unwrap();
        if let Some(func) = &self_.func {
            *seq = func()?;
            self_.func = None;
        }
        Ok((*seq).clone())
    }
}

extern "C" fn py_lazyseq_first(
    self_: *mut PyObject,
    _args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs != 0 {
            utils::raise_exception("LazySeq.first() takes no arguments")
        } else {
            first(&PyObj::borrow(self_))
        }
    })
}

#[inline]
pub fn first(self_: &PyObj) -> Result<PyObj, ()> {
    let s = force1(self_)?;
    common::first(&s)
}

extern "C" fn py_lazyseq_rest(
    self_: *mut PyObject,
    _args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs != 0 {
            utils::raise_exception("LazySeq.rest() takes no arguments")
        } else {
            rest(&PyObj::borrow(self_))
        }
    })
}

#[inline]
pub fn rest(self_: &PyObj) -> Result<PyObj, ()> {
    let s = force1(self_)?;
    common::rest(&s)
}

extern "C" fn py_lazyseq_next(
    self_: *mut PyObject,
    _args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs != 0 {
            utils::raise_exception("LazySeq.next() takes no arguments")
        } else {
            next(&PyObj::borrow(self_))
        }
    })
}

#[inline]
pub fn next(self_: &PyObj) -> Result<PyObj, ()> {
    let s = force1(self_)?;
    common::next(&s)
}

extern "C" fn py_lazyseq_seq(
    self_: *mut PyObject,
    _args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs != 0 {
            utils::raise_exception("LazySeq.seq() takes no arguments")
        } else {
            seq(&PyObj::borrow(self_))
        }
    })
}

#[inline]
pub fn seq(self_: &PyObj) -> Result<PyObj, ()> {
    let s = force1(self_)?;
    common::seq(&s)
}

extern "C" fn py_lazyseq_iter(
    self_: *mut PyObject,
) -> *mut PyObject {
    utils::wrap_body!({
        let self_ = PyObj::borrow(self_);
        seq_iterator::from(self_)
    })
}
