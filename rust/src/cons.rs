use crate::object::PyObj;
use crate::type_object as tpo;
use crate::utils;
use crate::protocols::*;
use crate::common;
use pyo3_ffi::*;

pub fn init_module(module: *mut PyObject) {
    utils::module_add_type!(module, Cons, cons_type());
}

#[repr(C)]
pub struct Cons {
    ob_base: PyObject,
    meta: PyObj,
    first: PyObj,
    rest: PyObj,
}

pub fn cons_type() -> &'static PyObj {
    tpo::static_type!(tpo::TypeSpec {
        name: "clx_rust.Cons",
        bases: vec![
            imeta_type(),
            iseq_type(),
            isequential_type(),
        ],
        flags: Py_TPFLAGS_DEFAULT,
        size: std::mem::size_of::<Cons>(),
        new: Some(utils::disallowed_new!(cons_type)),
        dealloc: Some(utils::generic_dealloc::<Cons>),
        compare: Some(py_cons_compare),
        members: vec![ tpo::member!("__meta__") ],
        methods: vec![
            tpo::method!("seq", py_cons_seq),
            tpo::method!("first", py_cons_first),
            tpo::method!("rest", py_cons_rest),
            tpo::method!("next", py_cons_next),
        ],
        ..Default::default()
    })
}

pub fn cons(x: PyObj, coll: PyObj) -> PyObj {
    unsafe {
        let obj = PyObj::alloc(cons_type());
        let cons = obj.as_ref::<Cons>();
        std::ptr::write(&mut cons.meta, PyObj::none());
        std::ptr::write(&mut cons.first, x.clone());
        std::ptr::write(&mut cons.rest, coll.clone());
        obj
    }
}

extern "C" fn py_cons_seq(
    self_: *mut PyObject,
    _args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    if nargs != 0 {
        utils::set_exception("Cons.seq() takes no arguments");
        std::ptr::null_mut()
    } else {
        utils::incref(self_)
    }
}

extern "C" fn py_cons_first(
    self_: *mut PyObject,
    _args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs != 0 {
            utils::raise_exception("Cons.first() takes no arguments")
        } else {
            Ok(first(&PyObj::borrow(self_)))
        }
    })
}

#[inline]
pub fn first(self_: &PyObj) -> PyObj {
    unsafe { self_.as_ref::<Cons>() }.first.clone()
}

extern "C" fn py_cons_rest(
    self_: *mut PyObject,
    _args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs != 0 {
            utils::raise_exception("Cons.rest() takes no arguments")
        } else {
            Ok(rest(&PyObj::borrow(self_)))
        }
    })
}

#[inline]
pub fn rest(self_: &PyObj) -> PyObj {
    unsafe { self_.as_ref::<Cons>() }.rest.clone()
}

extern "C" fn py_cons_next(
    self_: *mut PyObject,
    _args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs != 0 {
            utils::raise_exception("Cons.next() takes no arguments")
        } else {
            next(&PyObj::borrow(self_))
        }
    })
}

#[inline]
pub fn next(self_: &PyObj) -> Result<PyObj, ()> {
    let cons = unsafe { self_.as_ref::<Cons>() };
    common::seq(&cons.rest)
}

extern "C" fn py_cons_compare(
    self_: *mut PyObject,
    other: *mut PyObject,
    op: i32,
) -> *mut PyObject {
    utils::wrap_body!({
        let self_ = PyObj::borrow(self_);
        let other = PyObj::borrow(other);
        Ok(PyObj::from(match op {
            pyo3_ffi::Py_EQ => cons_eq(&self_, &other)?,
            pyo3_ffi::Py_NE => !cons_eq(&self_, &other)?,
            _ => {
                return utils::raise_exception(
                    "Cons comparison not supported");
            }
        }))
    })
}

fn cons_eq(self_: &PyObj, other: &PyObj) -> Result<bool, ()> {
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
