use crate::object::PyObj;
use crate::type_object as tpo;
use crate::utils;
use crate::protocols::*;
use crate::common;
use crate::lazy_seq;
use crate::seq_iterator;
use pyo3_ffi::*;
use std::collections::LinkedList;

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
        name: "lepet_ext.Cons",
        bases: vec![
            imeta_type(),
            iseq_type(),
            isequential_type(),
            icollection_type(),
        ],
        flags: Py_TPFLAGS_DEFAULT,
        size: std::mem::size_of::<Cons>(),
        new: Some(utils::disallowed_new!(cons_type)),
        dealloc: Some(py_cons_dealloc),
        compare: Some(py_cons_compare),
        iter: Some(py_iter),
        members: vec![ tpo::member!("__meta__") ],
        methods: vec![
            tpo::method!("seq", py_cons_seq),
            tpo::method!("first", py_cons_first),
            tpo::method!("rest", py_cons_rest),
            tpo::method!("next", py_cons_next),
            tpo::method!("conj", py_conj),
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

extern "C" fn py_cons_dealloc(obj: *mut PyObject) {
    unsafe {
        let cobj = obj.cast::<Cons>();

        // Dereferencing a deeply nested list can cause stack overflow.
        // To avoid this, we first unlink the list starting from the end.
        if !(*cobj).rest.is_none() {
            let cons_type = cons_type().as_ptr();
            let lseq_type = lazy_seq::lazyseq_type().as_ptr();
            let mut nodes: LinkedList<*mut PyObject> = LinkedList::new();
            // Collect nodes that are about to be deallocated
            let mut node = obj;
            loop {
                if (*node).ob_refcnt <= 1 {
                    let node_type = (*node).ob_type;
                    if node_type == cons_type {
                        nodes.push_back(node);
                        node = (*node.cast::<Cons>()).rest.as_ptr();
                    } else if node_type == lseq_type {
                        nodes.push_back(node);
                        let lseq = node.cast::<lazy_seq::LazySeq>();
                        node = (*lseq).seq.lock().unwrap().as_ptr();
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            // Unlink nodes in reverse order
            while let Some(node) = nodes.pop_back() {
                let node_type = (*node).ob_type;
                if node_type == cons_type {
                    (*node.cast::<Cons>()).rest = PyObj::none();
                } else if node_type == lseq_type {
                    let lseq = node.cast::<lazy_seq::LazySeq>();
                    *(*lseq).seq.lock().unwrap() = PyObj::none();
                }
            }
        }

        std::ptr::drop_in_place(cobj);
        tpo::free(obj);
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

extern "C" fn py_conj(
    self_: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs == 1 {
            Ok(cons(PyObj::borrow(unsafe { *args }), PyObj::borrow(self_)))
        } else {
            utils::raise_exception("Cons.conj() takes exactly one argument")
        }
    })
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

extern "C" fn py_iter(self_: *mut PyObject,) -> *mut PyObject {
    utils::wrap_body!({
        let self_ = PyObj::borrow(self_);
        seq_iterator::from(self_)
    })
}
