use crate::object::{PyObj, PyObjHashable};
use crate::type_object as tpo;
use crate::utils;
use crate::protocols::*;
use crate::common;
use pyo3_ffi::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub fn init_module(module: *mut PyObject) {
    utils::module_add_method!(module, cow_set, py_new);
    utils::module_add_method!(module, cow_set_from, py_from);
    utils::module_add_method!(module, is_cow_set, py_is_cow_set);
    utils::module_add_type!(module, CowSet, class());
}

#[repr(C)]
pub struct CowSet {
    ob_base: PyObject,
    meta: PyObj,
    impl_: SetImpl,
    hash: Option<isize>,
}

type SetImpl = std::collections::HashSet<PyObjHashable>;
type SetIter<'a> = std::collections::hash_set::Iter<'a, PyObjHashable>;

pub fn class() -> &'static PyObj {
    tpo::static_type!(tpo::TypeSpec {
        name: "lepet_ext.CowSet",
        bases: vec![
            imeta_type(),
            icounted_type(),
            iseqable_type(),
            icollection_type(),
            set_type(),
        ],
        flags: Py_TPFLAGS_DEFAULT,
        size: std::mem::size_of::<CowSet>(),
        dealloc: Some(tpo::generic_dealloc::<CowSet>),
        new: Some(utils::disallowed_new!(class)),
        compare: Some(py_compare),
        hash: Some(py_hash),
        call: Some(py_call),
        iter: Some(py_iter),
        members: vec![ tpo::member!("__meta__") ],
        methods: vec![
            tpo::method!("count_", py_count),
            tpo::method!("conj", py_conj),
            tpo::method!("disj", py_disj),
            tpo::method!("contains", py_contains),
            tpo::method!("seq", py_seq),
        ],
        ..Default::default()
    })
}

extern "C" fn py_new(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs == 0 {
            Ok(empty())
        } else {
            let mut impl_ = SetImpl::new();
            for i in 0..nargs {
                let obj = PyObj::borrow(unsafe { *args.offset(i) });
                impl_.insert(obj.into_hashable()?);
            }
            Ok(new(impl_, PyObj::none(), None))
        }
    })
}

fn new(
    impl_: SetImpl,
    meta: PyObj,
    hash: Option<isize>
) -> PyObj {
    unsafe {
        let obj = PyObj::alloc(class());
        let v = obj.as_ref::<CowSet>();
        std::ptr::write(&mut v.impl_, impl_);
        std::ptr::write(&mut v.meta, meta);
        v.hash = hash;
        obj
    }
}

fn empty() -> PyObj {
    utils::lazy_static!(PyObj, {
        let mut hasher = DefaultHasher::new();
        "empty_cow_set".hash(&mut hasher);
        new(SetImpl::new(), PyObj::none(), Some(hasher.finish() as isize))
    }).clone()
}

extern "C" fn py_is_cow_set(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    _nargs: isize,
) -> *mut PyObject {
    utils::handle_gil!({
        PyObj::from(unsafe {
            Py_TYPE(*args) == class().as_ptr()
        }).into_ptr()
    })
}

extern "C" fn py_from(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs == 1 {
            let coll = PyObj::borrow(unsafe { *args });
            let iter = coll.get_iter()?;
            let mut impl_ = SetImpl::new();
            while let Some(item) = iter.next() {
                impl_.insert(item.into_hashable()?);
            }
            Ok(if impl_.is_empty() {
                empty()
            } else {
                new(impl_, PyObj::none(), None)
            })
        } else {
            utils::raise_exception("CowSet.from() requires one argument")
        }
    })
}

extern "C" fn py_conj(
    self_: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs > 0 {
            let self_ = PyObj::borrow(self_);
            let self_ = unsafe { self_.as_ref::<CowSet>() };
            let mut impl_ = self_.impl_.clone();
            for i in 0..nargs {
                let obj = PyObj::borrow(unsafe { *args.offset(i) });
                impl_.insert(obj.into_hashable()?);
            }
            Ok(new(impl_, PyObj::none(), None))
        } else {
            utils::raise_exception(
                "CowSet.conj() requires at least one argument")
        }
    })
}

extern "C" fn py_disj(
    self_: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs > 0 {
            let self_ = PyObj::borrow(self_);
            let self_ = unsafe { self_.as_ref::<CowSet>() };
            let mut impl_ = self_.impl_.clone();
            for i in 0..nargs {
                let obj = PyObj::borrow(unsafe { *args.offset(i) });
                impl_.remove(&obj.into_hashable()?);
            }

            Ok(if !impl_.is_empty() {
                new(impl_, PyObj::none(), None)
            } else {
                empty()
            })
        } else {
            utils::raise_exception(
                "CowSet.disj() requires at least one argument")
        }
    })
}

extern "C" fn py_count(
    self_: *mut PyObject,
    _args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs == 0 {
            let self_ = PyObj::borrow(self_);
            Ok(PyObj::from(count(&self_)))
        } else {
            utils::raise_exception("CowSet.count_() takes no arguments")
        }
    })
}

#[inline]
pub fn count(self_: &PyObj) -> i64 {
    let self_ = unsafe { self_.as_ref::<CowSet>() };
    self_.impl_.len() as i64
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
            pyo3_ffi::Py_EQ => equal(&self_, &other),
            pyo3_ffi::Py_NE => !equal(&self_, &other),
            _ => {
                return utils::raise_exception(
                    "CowSet comparison not supported");
            }
        }))
    })
}

fn equal(self_: &PyObj, other: &PyObj) -> bool {
    if self_.is(other) {
        true
    } else {
        let self_ = unsafe { self_.as_ref::<CowSet>() };
        if other.type_is(class()) {
            let other = unsafe { other.as_ref::<CowSet>() };
            self_.impl_ == other.impl_
        } else {
            false
        }
    }
}
extern "C" fn py_hash(
    self_: *mut PyObject,
) -> isize {
    utils::handle_gil!({
        let self_ = PyObj::borrow(self_);
        match hash(&self_) {
            Ok(hash) => hash,
            Err(()) => -1
        }
    })
}

fn hash(self_: &PyObj) -> Result<isize, ()> {
    let set = unsafe { self_.as_ref::<CowSet>() };
    match set.hash {
        Some(hash) => Ok(hash),
        None => {
            let mut hasher = DefaultHasher::new();
            let mut items = set.impl_.iter().collect::<Vec<_>>();
            // TODO won't work if collisions are present
            items.sort_by_key(|x| x.hash);
            for item in items {
                item.hash.hash(&mut hasher);
            }
            let hash = hasher.finish() as isize;
            set.hash = Some(hash);
            Ok(hash)
        }
    }
}

extern "C" fn py_contains(
    self_: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs == 1 {
            Ok(PyObj::from(contains(
                &PyObj::borrow(self_),
                PyObj::borrow(unsafe { *args }))?))
        } else {
            utils::raise_exception(
                "CowSet.contains() requires one arguments")
        }
    })
}

#[inline]
pub fn contains(self_: &PyObj, obj: PyObj) -> Result<bool, ()> {
    let self_ = unsafe { self_.as_ref::<CowSet>() };
    let obj = obj.into_hashable()?;
    Ok(self_.impl_.contains(&obj))
}

extern "C" fn py_seq(
    self_: *mut PyObject,
    _args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs == 0 {
            seq(PyObj::borrow(self_))
        } else {
            utils::raise_exception("CowSet.seq() takes no arguments")
        }
    })
}

pub fn seq(oself: PyObj) -> Result<PyObj, ()> {
    let rself = unsafe { oself.as_ref::<CowSet>() };
    Ok(if !rself.impl_.is_empty() {
        common::iterator_seq(&oself.get_iter()?)
    } else {
        PyObj::none()
    })
}

extern "C" fn py_call(
    self_: *mut PyObject,
    args: *mut PyObject,
    _kwargs: *mut PyObject,
) -> *mut PyObject {
    utils::wrap_body!({
        let obj: *mut PyObject = std::ptr::null_mut();
        if unsafe { PyArg_UnpackTuple(args,
                "CowSet.__call__()\0".as_ptr().cast(),
                1, 1, &obj) } != 0 {
            let obj = PyObj::borrow(obj);
            let self_ = PyObj::borrow(self_);
            let self_ = unsafe { self_.as_ref::<CowSet>() };
            Ok(match self_.impl_.get(&obj.into_hashable()?) {
                Some(found) => found.as_pyobj().clone(),
                None => PyObj::none(),
            })
        } else {
            Err(())
        }
    })
}

extern "C" fn py_iter(
    self_: *mut PyObject,
) -> *mut PyObject {
    utils::wrap_body!({
        let oself = PyObj::borrow(self_);
        let self_ = unsafe { oself.as_ref::<CowSet>() };
        let oiter = PyObj::alloc(set_iterator_type());
        unsafe {
            let iter = oiter.as_ref::<SetIterator>();
            std::ptr::write(&mut iter.coll, oself.clone());
            std::ptr::write(&mut iter.iterator, self_.impl_.iter());
        }
        Ok(oiter)
    })
}

#[repr(C)]
pub struct SetIterator<'a> {
    ob_base: PyObject,
    coll: PyObj,
    iterator: SetIter<'a>,
}

pub fn set_iterator_type() -> &'static PyObj {
    tpo::static_type!(tpo::TypeSpec {
        name: "lepet_ext.CowSetIterator",
        flags: Py_TPFLAGS_DEFAULT,
        size: std::mem::size_of::<SetIterator>(),
        new: Some(utils::disallowed_new!(set_iterator_type)),
        dealloc: Some(tpo::generic_dealloc::<SetIterator>),
        iter: Some(set_iterator_iter),
        next: Some(set_iterator_next),
        ..Default::default()
    })
}

unsafe extern "C" fn set_iterator_iter(
    self_: *mut PyObject,
) -> *mut PyObject {
    self_
}

unsafe extern "C" fn set_iterator_next(
    self_: *mut PyObject,
) -> *mut PyObject {
    utils::wrap_body!({
        let self_ = PyObj::borrow(self_);
        let iter = self_.as_ref::<SetIterator>();
        match iter.iterator.next() {
            Some(obj) => Ok(obj.as_pyobj().clone()),
            None => {
                PyErr_SetNone(PyExc_StopIteration);
                Err(())
            }
        }
    })
}
