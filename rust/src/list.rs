use crate::object::PyObj;
use crate::utils;
use crate::protocols::*;
use pyo3_ffi::*;
use std::collections::LinkedList;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub fn init_module(module: *mut PyObject) {
    utils::module_add_method!(module, list_);
    utils::module_add_method!(module, is_list);
    utils::module_add_type!(module, PersistentList, list_type());
}

#[repr(C)]
pub struct List {
    ob_base: PyObject,
    meta: PyObj,
    first: PyObj,
    rest: PyObj,
    length: i64,
    hash: Option<isize>,
}

pub fn list_type() -> &'static PyObj {
    utils::static_type!(
        utils::TypeSpec {
            name: "clx_rust.PersistentList",
            bases: vec![
                imeta_type(),
                icounted_type(),
                iseq_type(),
                isequential_type(),
                icollection_type(),
            ],
            flags: Py_TPFLAGS_DEFAULT |
                   Py_TPFLAGS_DISALLOW_INSTANTIATION,
            size: std::mem::size_of::<List>(),
            dealloc: Some(list_dealloc),
            sequence_length: Some(py_list_len),
            compare: Some(py_list_compare),
            hash: Some(py_list_hash),
            iter: Some(py_list_iter),
            members: vec![ "__meta__" ],
            methods: vec![
                ("with_meta", py_list_with_meta),
                ("count_", py_list_count),
                ("first", py_list_first),
                ("next", py_list_next),
                ("rest", py_list_rest),
                ("seq", py_list_seq),
                ("conj", py_list_conj),
            ],
            ..utils::TypeSpec::default()
        }
    )
}

pub fn empty_list() -> PyObj {
    utils::lazy_static!(PyObj, {
        let mut hasher = DefaultHasher::new();
        "empty_list".hash(&mut hasher);
        _list(PyObj::none(),
            PyObj::none(),
            PyObj::none(),
            0,
            Some(hasher.finish() as isize))
    }).clone()
}

extern "C" fn list_(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::handle_gil_and_panic!({
        let mut obj = empty_list();
        let mut i = nargs - 1;
        while i >= 0 {
            unsafe {
                let item = *args.offset(i);
                obj = _list(
                    PyObj::borrow(item),
                    obj.clone(),
                    PyObj::none(),
                    (nargs - i) as i64,
                    None,
                );
                i -= 1;
            }
        }
        obj.into_ptr()
    })
}

fn _list(first: PyObj,
         rest: PyObj,
         meta: PyObj,
         length: i64,
         hash: Option<isize>) -> PyObj {
    unsafe {
        let obj = PyObj::alloc(list_type());
        let l = obj.as_ref::<List>();
        std::ptr::write(&mut l.first, first);
        std::ptr::write(&mut l.rest, rest);
        std::ptr::write(&mut l.meta, meta);
        l.length = length;
        l.hash = hash;
        obj
    }
}

extern "C" fn list_dealloc(obj: *mut PyObject) {
    unsafe {
        let l = obj as *mut List;

        // Dereferencing a deeply nested list can cause stack overflow.
        // To avoid this, we first unlink the list starting from the end.
        if !(*l).rest.is_none() {
            let mut ls: LinkedList<*mut List> = LinkedList::new();
            // Collect nodes that are about to be deallocated
            let mut l = l;
            while (*l).ob_base.ob_refcnt <= 1 {
                ls.push_back(l);
                l = (*l).rest.as_ptr();
            }
            // Unlink nodes in reverse order
            while let Some(l) = ls.pop_back() {
                (*l).rest = PyObj::none();
            }
        }

        std::ptr::drop_in_place(l);
        let obj_type = &*Py_TYPE(obj);
        let free = obj_type.tp_free.unwrap();
        free(obj.cast());
    }
}

unsafe extern "C" fn is_list(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    _nargs: isize,
) -> *mut PyObject {
    utils::handle_gil_and_panic!({
        PyObj::from(Py_TYPE(*args) == list_type().as_ptr()).into_ptr()
    })
}

extern "C" fn py_list_compare(
    self_: *mut PyObject,
    other: *mut PyObject,
    op: i32,
) -> *mut PyObject {
    utils::handle_gil_and_panic!({
        let self_ = PyObj::borrow(self_);
        let other = PyObj::borrow(other);
        PyObj::from(match op {
            pyo3_ffi::Py_EQ => list_eq(&self_, &other),
            pyo3_ffi::Py_NE => !list_eq(&self_, &other),
            _ => {
                return utils::raise_exception(
                    "list comparison not supported");
            }
        }).into_ptr()
    })
}

fn list_eq(self_: &PyObj, other: &PyObj) -> bool {
    if self_.is(other) {
        true
    } else {
        let lself = unsafe { self_.as_ref::<List>() };
        if other.type_is(list_type()) {
            let lother = unsafe { other.as_ref::<List>() };
            if lself.length == lother.length {
                let mut l1 = lself;
                let mut l2 = lother;
                while l1.length > 0 {
                    if l1.first != l2.first {
                        return false
                    }
                    l1 = unsafe { l1.rest.as_ref::<List>() };
                    l2 = unsafe { l2.rest.as_ref::<List>() };
                }
                true
            } else {
                false
            }
        } else if other.is_instance(isequential_type()) {
            utils::sequential_eq(self_, other)
        } else {
            false
        }
    }
}

extern "C" fn py_list_count(
    self_: *mut PyObject,
    _args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::handle_gil_and_panic!({
        if nargs != 0 {
            utils::raise_exception("PersistentList.count_() takes no arguments")
        } else {
            let self_ = PyObj::borrow(self_);
            let self_ = unsafe { self_.as_ref::<List>() };
            PyObj::from(self_.length).into_ptr()
        }
    })
}

extern "C" fn py_list_first(
    self_: *mut PyObject,
    _args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::handle_gil_and_panic!({
        if nargs != 0 {
            utils::raise_exception("PersistentList.first() takes no arguments")
        } else {
            list_first(&PyObj::borrow(self_)).into_ptr()
        }
    })
}

#[inline]
pub fn list_first(self_: &PyObj) -> PyObj {
    unsafe { self_.as_ref::<List>() }.first.clone()
}

extern "C" fn py_list_next(
    self_: *mut PyObject,
    _args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::handle_gil_and_panic!({
        if nargs != 0 {
            utils::raise_exception("PersistentList.next() takes no arguments")
        } else {
            list_next(&PyObj::borrow(self_)).into_ptr()
        }
    })
}

#[inline]
pub fn list_next(self_: &PyObj) -> PyObj {
    let self_ = unsafe { self_.as_ref::<List>() };
    if self_.length <= 1 {
        PyObj::none()
    } else {
        self_.rest.clone()
    }
}

extern "C" fn py_list_rest(
    self_: *mut PyObject,
    _args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::handle_gil_and_panic!({
        if nargs != 0 {
            utils::raise_exception("PersistentList.rest() takes no arguments")
        } else {
            list_rest(&PyObj::borrow(self_)).into_ptr()
        }
    })
}

#[inline]
pub fn list_rest(self_: &PyObj) -> PyObj {
    let self_ = unsafe { self_.as_ref::<List>() };
    if self_.length == 0 {
        empty_list()
    } else {
        self_.rest.clone()
    }
}

unsafe extern "C" fn py_list_len(
    self_: *mut PyObject,
) -> isize {
    let self_ = self_ as *mut List;
    (*self_).length as isize
}

unsafe extern "C" fn py_list_with_meta(
    self_: *mut PyObject,
    args: *mut *mut PyObject,
    _nargs: isize,
) -> *mut PyObject {
    utils::handle_gil_and_panic!({
        let self_ = PyObj::borrow(self_);
        let self_ = self_.as_ref::<List>();
        let meta = PyObj::borrow(*args);
        _list(self_.first.clone(),
            self_.rest.clone(),
            meta,
            self_.length,
            self_.hash).into_ptr()
    })
}

extern "C" fn py_list_seq(
    self_: *mut PyObject,
    _args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    if nargs != 0 {
        utils::raise_exception("PersistentList.seq() takes no arguments")
    } else {
        list_seq(&PyObj::borrow(self_)).into_ptr()
    }
}

pub fn list_seq(self_: &PyObj) -> PyObj {
    let l = unsafe { self_.as_ref::<List>() };
    if l.length == 0 {
        PyObj::none()
    } else {
        self_.clone()
    }
}

unsafe extern "C" fn py_list_conj(
    self_: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::handle_gil_and_panic!({
        if nargs != 1 {
            utils::raise_exception("PersistentList.conj() takes one argument")
        } else {
            let oself = PyObj::borrow(self_);
            let item = PyObj::borrow(*args);
            list_conj(oself, item).into_ptr()
        }
    })
}

pub fn list_conj(oself: PyObj, item: PyObj) -> PyObj {
    let self_ = unsafe { oself.as_ref::<List>() };
    let length = self_.length + 1;
    _list(item.clone(), oself.clone(), PyObj::none(), length, None)
}

extern "C" fn py_list_iter(
    self_: *mut PyObject,
) -> *mut PyObject {
    utils::handle_gil_and_panic!({
        let self_ = PyObj::borrow(self_);
        utils::seq_iterator(self_).into_ptr()
    })
}

extern "C" fn py_list_hash(
    self_: *mut PyObject,
) -> isize {
    utils::handle_gil!({
        let self_ = PyObj::borrow(self_);
        let self_ = unsafe { self_.as_ref::<List>() };
        if let Some(hash) = self_.hash {
            hash
        } else {
            let mut nodes: LinkedList<*mut List> = LinkedList::new();
            let mut l = self_ as *mut List;
            unsafe {
                while (*l).hash.is_none() {
                    nodes.push_back(l);
                    l = (*l).rest.as_ptr();
                }
            }
            while let Some(l) = nodes.pop_back() {
                let mut hasher = DefaultHasher::new();
                let l = unsafe { &mut *l };
                l.rest.hash(&mut hasher);
                l.first.hash(&mut hasher);
                l.hash = Some(hasher.finish() as isize);
            }
            self_.hash.unwrap()
        }
    })
}
