use crate::cons;
use crate::lazy_seq;
use crate::list;
use crate::vector;
use crate::object::PyObj;
use crate::utils;
use crate::protocols::*;
use pyo3_ffi::*;

pub fn init_module(module: *mut PyObject) {
    utils::module_add_method!(module, cons, py_cons);
    utils::module_add_method!(module, seq, py_seq);
    utils::module_add_method!(module, first, py_first);
    utils::module_add_method!(module, next_, py_next);
    utils::module_add_method!(module, rest, py_rest);
}

extern "C" fn py_cons(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs != 2 {
            utils::raise_exception("cons() requires exactly two arguments")
        } else {
            let obj = PyObj::borrow(unsafe { *args });
            let coll = PyObj::borrow(unsafe { *args.add(1) });
            Ok(cons::cons(obj,
                if coll.is_none() {
                    list::empty_list()
                } else if coll.is_instance(iseq_type()) {
                    coll
                } else {
                    seq(&coll)?
                }
            ))
        }
    })
}

extern "C" fn py_first(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs != 1 {
            utils::raise_exception("first() takes exactly 1 argument")
        } else {
            first(&PyObj::borrow(unsafe { *args }))
        }
    })
}

pub fn first(coll: &PyObj) -> Result<PyObj, ()> {
    if coll.is_none() {
        Ok(coll.clone())
    } else if coll.type_is(cons::cons_type()) {
        Ok(cons::first(coll))
    } else if coll.type_is(lazy_seq::lazyseq_type()) {
        lazy_seq::first(coll)
    } else if coll.type_is(list::list_type()) {
        Ok(list::first(coll))
    } else if coll.type_is(vector::vector_type()) {
        Ok(vector::nth(coll, 0, Some(PyObj::none()))?)
    } else if coll.is_tuple() {
        coll.get_tuple_item(0).or_else(|_| {
            utils::clear_exception();
            Ok(PyObj::none())
        })
    } else if coll.is_instance(iseq_type()) {
        coll.call_method0(&utils::static_pystring!("first"))
    } else if coll.is_instance(iindexed_type()) {
        nth(coll, 0, Some(PyObj::none()))
    } else {
        first(&seq(coll)?)
    }
}

extern "C" fn py_next(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs == 1 {
            next(&PyObj::borrow(unsafe { *args }))
        } else {
            utils::raise_exception("next() takes exactly 1 argument")
        }
    })
}

pub fn next(coll: &PyObj) -> Result<PyObj, ()> {
    if coll.is_none() {
        Ok(coll.clone())
    } else if coll.type_is(cons::cons_type()) {
        cons::next(coll)
    } else if coll.type_is(lazy_seq::lazyseq_type()) {
        lazy_seq::next(coll)
    } else if coll.type_is(list::list_type()) {
        Ok(list::next(coll))
    } else if coll.is_instance(iseq_type()) {
        coll.call_method0(&utils::static_pystring!("next"))
    } else {
        next(&seq(coll)?)
    }
}

extern "C" fn py_rest(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs == 1 {
            rest(&PyObj::borrow(unsafe { *args }))
        } else {
            utils::raise_exception("rest() takes exactly 1 argument")
        }
    })
}

pub fn rest(coll: &PyObj) -> Result<PyObj, ()> {
    if coll.is_none() {
        Ok(list::empty_list())
    } else if coll.type_is(cons::cons_type()) {
        Ok(cons::rest(coll))
    } else if coll.type_is(lazy_seq::lazyseq_type()) {
        lazy_seq::rest(coll)
    } else if coll.type_is(list::list_type()) {
        Ok(list::rest(coll))
    } else if coll.is_instance(iseq_type()) {
        coll.call_method0(&utils::static_pystring!("rest"))
    } else {
        rest(&seq(coll)?)
    }
}

extern "C" fn py_seq(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs != 1 {
            utils::raise_exception("seq() takes exactly 1 argument")
        } else {
            seq(&PyObj::borrow(unsafe { *args }))
        }
    })
}

pub fn seq(coll: &PyObj) -> Result<PyObj, ()> {
    if coll.is_none() {
        Ok(PyObj::none())
    } else if coll.type_is(cons::cons_type()) {
        Ok(coll.clone())
    } else if coll.type_is(lazy_seq::lazyseq_type()) {
        lazy_seq::seq(coll)
    } else if coll.type_is(list::list_type()) {
        Ok(list::seq(coll))
    } else if coll.type_is(vector::vector_type()) {
        Ok(vector::seq(coll))
    } else if coll.is_instance(iterable_type()) {
        seq(&iterator_seq(&coll.get_iter()?)?)
    } else if coll.is_instance(iseqable_type()) {
        Ok(coll.call_method0(&utils::static_pystring!("seq"))?)
    } else {
        let type_name = coll.get_type()
            .get_attr(&utils::static_pystring!("__name__"))?;
        let msg = format!("Don't know how to create ISeq from '{}'",
            type_name.as_cstr()?.to_str().unwrap());
        utils::raise_exception(&msg)
    }
}

pub fn iterator_seq(it: &PyObj) -> Result<PyObj, ()> {
    fn _seq(it: PyObj) -> Result<PyObj, ()> {
        Ok(match it.next() {
            Some(value) =>
                cons::cons(value,
                    lazy_seq::lazy_seq(Box::new(move || _seq(it.clone())))),
            None => PyObj::none(),
        })
    }
    let it = it.clone();
    Ok(lazy_seq::lazy_seq(Box::new(move || _seq(it.clone()))))
}

pub fn sequential_eq(self_: &PyObj, other: &PyObj) -> Result<bool, ()> {
    let mut x = seq(self_)?;
    let mut y = seq(other)?;
    loop {
        if x.is_none() {
            return Ok(y.is_none());
        } else if y.is_none() {
            return Ok(false);
        } else if x.is(&y) {
            return Ok(true);
        } else if first(&x)? != first(&y)? {
            return Ok(false);
        } else {
            x = next(&x)?;
            y = next(&y)?;
        }
    }
}
