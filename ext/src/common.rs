use crate::cons;
use crate::lazy_seq;
use crate::indexed_seq;
use crate::list;
use crate::vector;
use crate::hash_map;
use crate::record;
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
    utils::module_add_method!(module, get, py_get);
    utils::module_add_method!(module, nth, py_nth);
    utils::module_add_method!(module, conj, py_conj);
    utils::module_add_method!(module, drop, py_drop);
    utils::module_add_method!(module, count, py_count);
    utils::module_add_method!(module, map_, py_map);
    utils::module_add_method!(module, filter_, py_filter);
    utils::module_add_method!(module, reduce, py_reduce);
    utils::module_add_method!(module, is_seq, py_is_seq);
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
    } else if coll.type_is(indexed_seq::class()) {
        indexed_seq::first(coll)
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
    } else if coll.type_is(indexed_seq::class()) {
        indexed_seq::next(coll)
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
    } else if coll.type_is(indexed_seq::class()) {
        indexed_seq::rest(coll)
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
    } else if coll.type_is(indexed_seq::class()) {
        Ok(coll.clone())
    } else if coll.is_tuple() || coll.is_string() {
        let len = coll.len()?;
        Ok(if len > 0 {
            indexed_seq::new(coll.clone(), len as usize, 0)
        } else {
            PyObj::none()
        })
    } else if coll.is_instance(iseqable_type()) {
        Ok(coll.call_method0(&utils::static_pystring!("seq"))?)
    } else if coll.is_instance(iterable_type()) {
        seq(&iterator_seq(&coll.get_iter()?)?)
    } else {
        let msg = format!("Don't know how to create ISeq from '{}'",
            coll.class().qual_name_string()?);
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
    if self_.is(other) {
        Ok(true)
    } else if (other.is_instance(isequential_type())
            || other.is_instance(sequence_type()))
            && !other.is_string() {
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
    } else {
        Ok(false)
    }
}

extern "C" fn py_get(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs == 2 {
            get(&PyObj::borrow(unsafe { *args }),
                PyObj::borrow(unsafe { *args.add(1) }),
                PyObj::none())
        } else if nargs == 3 {
            get(&PyObj::borrow(unsafe { *args }),
                PyObj::borrow(unsafe { *args.add(1) }),
                PyObj::borrow(unsafe { *args.add(2) }))
        } else {
            utils::raise_exception("get() expects 2 or 3 arguments")
        }
    })
}

pub fn get(coll: &PyObj, key: PyObj, not_found: PyObj) -> Result<PyObj, ()> {
    if coll.is_none() {
        Ok(PyObj::none())
    } else if coll.type_is(hash_map::hash_map_type()) {
        hash_map::lookup(coll, key, not_found)
    } else if coll.is_instance(irecord_type()) {
        Ok(record::lookup(coll, key, not_found))
    } else if coll.is_instance(iassociative_type()) {
        coll.call_method2(&utils::static_pystring!("lookup"), key, not_found)
    } else {
        Ok(not_found)
    }
}

extern "C" fn py_nth(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs == 2 {
            nth(&PyObj::borrow(unsafe { *args }),
                PyObj::borrow(unsafe { *args.add(1) }).as_int()?,
                None)
        } else if nargs == 3 {
            nth(&PyObj::borrow(unsafe { *args }),
                PyObj::borrow(unsafe { *args.add(1) }).as_int()?,
                Some(PyObj::borrow(unsafe { *args.add(2) })))
        } else {
            utils::raise_exception("get() expects 2 or 3 arguments")
        }
    })
}

pub fn nth(
    coll: &PyObj,
    index: isize,
    not_found: Option<PyObj>
) -> Result<PyObj, ()> {
    if coll.is_none() {
        Ok(PyObj::none())
    } else if coll.type_is(vector::vector_type()) {
        vector::nth(coll, index, not_found)
    } else if coll.is_tuple() {
        coll.get_tuple_item(index).or_else(|_| {
            match not_found {
                Some(not_found) => {
                    utils::clear_exception();
                    Ok(not_found)
                },
                None => Err(())
            }
        })
    } else if coll.is_string() {
        coll.get_sequence_item(index).or_else(|_| {
            match not_found {
                Some(not_found) => {
                    utils::clear_exception();
                    Ok(not_found)
                },
                None => Err(())
            }
        })
    } else if coll.is_instance(iindexed_type()) {
        match not_found {
            None => coll.call_method1(
                &utils::static_pystring!("nth"),
                PyObj::from(index as i64)),
            Some(not_found) => coll.call_method2(
                &utils::static_pystring!("nth"),
                PyObj::from(index as i64),
                not_found),
        }
    } else if coll.is_instance(iseqable_type()) {
        if index >= 0 {
            let mut coll = seq(coll)?;
            let mut index = index;
            loop {
                if coll.is_none() {
                    return match not_found {
                        Some(not_found) => Ok(not_found),
                        None => utils::raise_index_error(),
                    }
                }
                if index == 0 {
                    return Ok(first(&coll)?);
                }
                coll = next(&coll)?;
                index -= 1;
            }
        } else {
            match not_found {
                Some(not_found) => Ok(not_found),
                None => utils::raise_index_error(),
            }
        }
    } else {
        let msg = format!("nth() not supported for '{}'",
            coll.class().qual_name_string()?);
        utils::raise_exception(&msg)
    }
}

extern "C" fn py_conj(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs == 2 {
            let coll = PyObj::borrow(unsafe { *args });
            let item = PyObj::borrow(unsafe { *args.add(1) });
            if coll.is_none() {
                Ok(cons::cons(item, list::empty_list()))
            } else if coll.type_is(vector::vector_type()) {
                Ok(vector::conj(coll, item))
            } else if coll.type_is(list::list_type()) {
                Ok(list::conj(coll, item))
            } else if coll.is_instance(iseq_type()) {
                Ok(cons::cons(item, coll))
            } else if coll.is_instance(icollection_type()) {
                coll.call_method1(&utils::static_pystring!("conj"), item)
            } else {
                utils::raise_exception(
                    &format!("conj() not supported for '{}'",
                        coll.class().qual_name_string()?))
            }
        } else {
            utils::raise_exception("conj() expects exactly 2 arguments")
        }
    })
}

extern "C" fn py_drop(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs == 2 {
            let n_obj = PyObj::borrow(unsafe { *args });
            let coll = PyObj::borrow(unsafe { *args.add(1) });
            let n = n_obj.as_int()?;
            if coll.is_none() {
                Ok(list::empty_list())
            } else if n <= 0 {
                let res = seq(&coll)?;
                if res.is_none() {
                    Ok(list::empty_list())
                } else {
                    Ok(res)
                }
            } else if coll.type_is(vector::vector_type()) {
                Ok(vector::drop(coll, n as usize))
            } else if coll.type_is(indexed_seq::class()) {
                Ok(indexed_seq::drop(coll, n as usize))
            } else if coll.is_tuple() || coll.is_string() {
                Ok(indexed_seq::drop(seq(&coll)?, n as usize))
            } else {
                Ok(lazy_seq::lazy_seq(Box::new(move || {
                    let mut coll = seq(&coll)?;
                    for _ in 0..n {
                        if coll.is_none() {
                            break;
                        }
                        coll = next(&coll)?;
                    }
                    Ok(coll)
                })))
            }
        } else {
            utils::raise_exception("drop() expects exactly 2 arguments")
        }
    })
}

extern "C" fn py_count(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs == 1 {
            let coll = PyObj::borrow(unsafe { *args });
            Ok(PyObj::from(
                if coll.is_none() {
                    0
                } else if coll.type_is(vector::vector_type()) {
                    vector::count(&coll)
                } else if coll.type_is(indexed_seq::class()) {
                    indexed_seq::count(&coll)
                } else if coll.type_is(list::list_type()) {
                    list::count(&coll)
                } else if coll.type_is(hash_map::hash_map_type()) {
                    hash_map::count(&coll)
                } else if coll.is_instance(icounted_type()) {
                    coll.call_method0(
                        &utils::static_pystring!("count"))?.as_int()? as i64
                } else if coll.is_instance(iseqable_type()) {
                    let mut count = 0;
                    let mut coll = seq(&coll)?;
                    loop {
                        if coll.is_none() {
                            break;
                        }
                        count += 1;
                        coll = next(&coll)?;
                    }
                    count
                } else {
                    coll.len()? as i64
                }
            ))
        } else {
            utils::raise_exception("count() expects exactly 1 argument")
        }
    })
}

extern "C" fn py_map(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs == 2 {
            fn step(f: PyObj, coll: PyObj) -> Result<PyObj, ()> {
                Ok(if coll.is_none() {
                    PyObj::none()
                } else {
                    cons::cons(f.call1(first(&coll)?)?,
                        lazy_seq::lazy_seq(
                            Box::new(move ||
                                step(f.clone(), next(&coll)?))))
                })
            }

            let f = PyObj::borrow(unsafe { *args });
            let coll = seq(&PyObj::borrow(unsafe { *args.add(1) }))?;
            Ok(lazy_seq::lazy_seq(
                Box::new(move || step(f.clone(), coll.clone()))))
        } else {
            utils::raise_exception("wrong number of arguments")
        }
    })
}

extern "C" fn py_filter(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs == 2 {
            fn step(pred: PyObj, coll: PyObj) -> Result<PyObj, ()> {
                Ok(if coll.is_none() {
                    PyObj::none()
                } else {
                    let item = first(&coll)?;
                    if pred.call1(item.clone())?.is_truthy() {
                        cons::cons(item,
                            lazy_seq::lazy_seq(
                                Box::new(move ||
                                    step(pred.clone(), next(&coll)?))))
                    } else {
                        lazy_seq::lazy_seq(
                            Box::new(move ||
                                step(pred.clone(), next(&coll)?)))
                    }
                })
            }

            let pred = PyObj::borrow(unsafe { *args });
            let coll = seq(&PyObj::borrow(unsafe { *args.add(1) }))?;
            Ok(lazy_seq::lazy_seq(
                Box::new(move || step(pred.clone(), coll.clone()))))
        } else {
            utils::raise_exception("wrong number of arguments")
        }
    })
}

extern "C" fn py_reduce(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs == 2 {
            let f = PyObj::borrow(unsafe { *args });
            let coll = PyObj::borrow(unsafe { *args.add(1) });
            reduce(f, first(&coll)?, next(&coll)?)
        } else if nargs == 3 {
            reduce(PyObj::borrow(unsafe { *args }),
                PyObj::borrow(unsafe { *args.add(1) }),
                seq(&PyObj::borrow(unsafe { *args.add(2) }))?)
        } else {
            utils::raise_exception("wrong number of arguments")
        }
    })
}

fn reduce(f: PyObj, init: PyObj, coll: PyObj) -> Result<PyObj, ()> {
    let mut acc = init;
    let mut coll = coll;
    loop {
        if coll.is_none() {
            return Ok(acc);
        }
        acc = f.call2(acc, first(&coll)?)?;
        coll = next(&coll)?;
    }
}

extern "C" fn py_is_seq(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs == 1 {
            let obj = PyObj::borrow(unsafe { *args });
            Ok(PyObj::from(obj.is_instance(iseq_type())))
        } else {
            utils::raise_exception("seq? takes exactly one argument")
        }
    })
}
