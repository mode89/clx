use crate::object::{PyObj, PyObjHashable};
use crate::type_object as tpo;
use crate::list;
use crate::utils;
use crate::protocols::*;
use pyo3_ffi::*;

pub fn init_module(module: *mut PyObject) {
    utils::module_add_method!(module, hash_map, py_hash_map);
    utils::module_add_method!(module, is_hash_map, py_is_hash_map);
    utils::module_add_method!(module, hash_map_from, py_hash_map_from);
    utils::module_add_type!(module, PersistentHashMap, hash_map_type());
}

#[repr(C)]
pub struct HashMap {
    ob_base: PyObject,
    meta: PyObj,
    impl_: HashMapImpl,
    hash: Option<isize>,
}

type HashMapImpl = std::collections::HashMap<PyObjHashable, PyObj>;

pub fn hash_map_type() -> &'static PyObj {
    tpo::static_type!(
        tpo::TypeSpec {
            name: "clx_rust.PersistentHashMap",
            bases: vec![
                imeta_type(),
                icounted_type(),
                iseqable_type(),
                iassociative_type(),
                mapping_type(),
            ],
            flags: Py_TPFLAGS_DEFAULT,
            size: std::mem::size_of::<HashMap>(),
            dealloc: Some(utils::generic_dealloc::<HashMap>),
            new: Some(utils::disallowed_new!(hash_map_type)),
            compare: Some(py_hash_map_compare),
            // TODO hash: Some(py_vector_hash),
            call: Some(py_hash_map_call),
            iter: Some(py_hash_map_iter),
            mapping_length: Some(py_hash_map_len),
            mapping_subscript: Some(py_hash_map_subscript),
            members: vec![ tpo::member!("__meta__") ],
            methods: vec![
                // TODO ("with_meta", py_vector_with_meta),
                tpo::method!("__getitem__", py_hash_map_getitem),
                tpo::method!("assoc", py_hash_map_assoc),
                tpo::method!("lookup", py_hash_map_lookup),
                tpo::method!("merge", py_hash_map_merge),
                tpo::method!("count_", py_hash_map_count),
                tpo::method!("seq", py_hash_map_seq),
                // TODO ("conj", py_vector_conj),
            ],
            ..Default::default()
        }
    )
}

extern "C" fn py_hash_map(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs == 0 {
            Ok(empty_hash_map())
        } else if nargs % 2 != 0 {
            utils::raise_exception(
                "PersistentHashMap() requires an even number of arguments")
        } else {
            let mut impl_ = HashMapImpl::new();
            let kv_num = nargs / 2;
            for i in 0..kv_num {
                let key = PyObj::borrow(unsafe { *args.offset(i * 2) });
                let value = PyObj::borrow(unsafe { *args.offset(i * 2 + 1) });
                impl_.insert(key.into_hashable()?, value);
            }
            Ok(hash_map(impl_, PyObj::none(), None))
        }
    })
}

fn hash_map(
    impl_: HashMapImpl,
    meta: PyObj,
    hash: Option<isize>
) -> PyObj {
    unsafe {
        let obj = PyObj::alloc(hash_map_type());
        let v = obj.as_ref::<HashMap>();
        std::ptr::write(&mut v.impl_, impl_);
        std::ptr::write(&mut v.meta, meta);
        v.hash = hash;
        obj
    }
}

fn empty_hash_map() -> PyObj {
    utils::lazy_static!(PyObj, {
        hash_map(HashMapImpl::new(), PyObj::none(), None)
    }).clone()
}

extern "C" fn py_is_hash_map(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    _nargs: isize,
) -> *mut PyObject {
    utils::handle_gil!({
        PyObj::from(unsafe {
            Py_TYPE(*args) == hash_map_type().as_ptr()
        }).into_ptr()
    })
}

extern "C" fn py_hash_map_from(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs != 1 {
            utils::raise_exception(
                "PersistentHashMap.from_iter() requires one argument")
        } else {
            let coll = PyObj::borrow(unsafe { *args });
            let iter = coll.get_iter()?;
            let mut impl_ = HashMapImpl::new();
            let _0 = utils::static_pynumber!(0);
            let _1 = utils::static_pynumber!(1);
            while let Some(item) = iter.next() {
                let key = item.get_item(_0)?;
                let value = item.get_item(_1)?;
                impl_.insert(key.into_hashable()?, value);
            }
            Ok(if impl_.is_empty() {
                empty_hash_map()
            } else {
                hash_map(impl_, PyObj::none(), None)
            })
        }
    })
}

extern "C" fn py_hash_map_assoc(
    self_: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs == 0 {
            utils::raise_exception(
                "PersistentHashMap.assoc() requires at least one argument")
        } else if nargs % 2 != 0 {
            utils::raise_exception(
                "PersistentHashMap.assoc() requires an even number of arguments")
        } else {
            let mut impl_ = unsafe {
                PyObj::borrow(self_).as_ref::<HashMap>()
            }.impl_.clone();
            let kv_num = nargs / 2;
            for i in 0..kv_num {
                let key = PyObj::borrow(unsafe { *args.offset(i * 2) });
                let value = PyObj::borrow(unsafe { *args.offset(i * 2 + 1) });
                impl_.insert(key.into_hashable()?, value);
            }
            Ok(hash_map(impl_, PyObj::none(), None))
        }
    })
}

extern "C" fn py_hash_map_lookup(
    self_: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs != 2 {
            utils::raise_exception(
                "PersistentHashMap.lookup() requires two arguments")
        } else {
            let self_ = PyObj::borrow(self_);
            let self_ = unsafe { self_.as_ref::<HashMap>() };
            let key = PyObj::borrow(unsafe { *args });
            let not_found = PyObj::borrow(unsafe { *args.offset(1) });
            Ok(match self_.impl_.get(&key.into_hashable()?) {
                Some(value) => value.clone(),
                None => not_found,
            })
        }
    })
}

extern "C" fn py_hash_map_count(
    self_: *mut PyObject,
    _args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs != 0 {
            utils::raise_exception(
                "PersistentHashMap.count_() takes no arguments")
        } else {
            let self_ = PyObj::borrow(self_);
            let self_ = unsafe { self_.as_ref::<HashMap>() };
            Ok(PyObj::from(self_.impl_.len() as i64))
        }
    })
}

extern "C" fn py_hash_map_merge(
    self_: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs != 1 {
            utils::raise_exception(
                "PersistentHashMap.merge() requires one argument")
        } else {
            let other = PyObj::borrow(unsafe { *args });
            if !other.type_is(hash_map_type()) {
                utils::raise_exception(
                    "PersistentHashMap.merge() requires a PersistentHashMap argument")
            } else {
                let self_ = PyObj::borrow(self_);
                let self_ = unsafe { self_.as_ref::<HashMap>() };
                let other = unsafe { other.as_ref::<HashMap>() };
                let mut impl_ = self_.impl_.clone();
                for (key, value) in other.impl_.iter() {
                    impl_.insert(key.clone(), value.clone());
                }
                Ok(hash_map(impl_, PyObj::none(), None))
            }
        }
    })
}

extern "C" fn py_hash_map_compare(
    self_: *mut PyObject,
    other: *mut PyObject,
    op: i32,
) -> *mut PyObject {
    utils::wrap_body!({
        let self_ = PyObj::borrow(self_);
        let other = PyObj::borrow(other);
        Ok(PyObj::from(match op {
            pyo3_ffi::Py_EQ => hash_map_eq(&self_, &other),
            pyo3_ffi::Py_NE => !hash_map_eq(&self_, &other),
            _ => {
                return utils::raise_exception(
                    "hash-map comparison not supported");
            }
        }))
    })
}

fn hash_map_eq(self_: &PyObj, other: &PyObj) -> bool {
    if self_.is(other) {
        true
    } else {
        let self_ = unsafe { self_.as_ref::<HashMap>() };
        if other.type_is(hash_map_type()) {
            let other = unsafe { other.as_ref::<HashMap>() };
            self_.impl_ == other.impl_
        } else {
            false
        }
    }
}

extern "C" fn py_hash_map_len(self_: *mut PyObject) -> isize {
    utils::handle_gil!({
        let self_ = self_ as *mut HashMap;
        unsafe { (*self_).impl_.len() as isize }
    })
}

extern "C" fn py_hash_map_subscript(
    self_: *mut PyObject,
    key: *mut PyObject,
) -> *mut PyObject {
    utils::wrap_body!({
        let self_ = PyObj::borrow(self_);
        let key = PyObj::borrow(key);
        match hash_map_getitem(&self_, key.into_hashable()?) {
            Some(value) => Ok(value),
            None => raise_key_error()
        }
    })
}

extern "C" fn py_hash_map_getitem(
    self_: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs != 1 {
            utils::raise_exception(
                "PersistentHashMap.__getitem__() takes one argument")
        } else {
            let self_ = PyObj::borrow(self_);
            let key = PyObj::borrow(unsafe { *args });
            match hash_map_getitem(&self_, key.into_hashable()?) {
                Some(value) => Ok(value),
                None => raise_key_error(),
            }
        }
    })
}

fn hash_map_getitem(self_: &PyObj, key: PyObjHashable) -> Option<PyObj> {
    let self_ = unsafe { self_.as_ref::<HashMap>() };
    self_.impl_.get(&key).cloned()
}

extern "C" fn py_hash_map_seq(
    self_: *mut PyObject,
    _args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs != 0 {
            utils::raise_exception(
                "PersistentVector.seq() takes no arguments")
        } else {
            Ok(seq(&PyObj::borrow(self_)))
        }
    })
}

pub fn seq(self_: &PyObj) -> PyObj {
    let self_ = unsafe { self_.as_ref::<HashMap>() };
    if self_.impl_.is_empty() {
        PyObj::none()
    } else {
        let mut lst = list::empty_list();
        for (key, value) in self_.impl_.iter() {
            let kv = PyObj::tuple2(key.as_pyobj().clone(), value.clone());
            lst = list::list_conj(lst, kv);
        }
        lst
    }
}

extern "C" fn py_hash_map_call(
    self_: *mut PyObject,
    args: *mut PyObject,
    _kwargs: *mut PyObject,
) -> *mut PyObject {
    utils::wrap_body!({
        let key: *mut PyObject = std::ptr::null_mut();
        let not_found: *mut PyObject = std::ptr::null_mut();
        if unsafe { PyArg_UnpackTuple(args,
                "PersistentHashMap.__call__()\0".as_ptr().cast(),
                1, 2, &key, &not_found) } != 0 {
            let key = PyObj::borrow(key);
            let self_ = PyObj::borrow(self_);
            let self_ = unsafe { self_.as_ref::<HashMap>() };
            Ok(match self_.impl_.get(&key.into_hashable()?) {
                Some(value) => value.clone(),
                None => {
                    if !not_found.is_null() {
                        PyObj::borrow(not_found)
                    } else {
                        PyObj::none()
                    }
                }
            })
        } else {
            Err(())
        }
    })
}

extern "C" fn py_hash_map_iter(
    self_: *mut PyObject,
) -> *mut PyObject {
    utils::wrap_body!({
        let oself = PyObj::borrow(self_);
        let self_ = unsafe { oself.as_ref::<HashMap>() };
        let oiter = PyObj::alloc(hash_map_iterator_type());
        unsafe {
            let iter = oiter.as_ref::<HashMapIterator>();
            std::ptr::write(&mut iter.map, oself.clone());
            std::ptr::write(&mut iter.iterator, self_.impl_.iter());
        }
        Ok(oiter)
    })
}

#[repr(C)]
pub struct HashMapIterator<'a> {
    ob_base: PyObject,
    map: PyObj,
    iterator: std::collections::hash_map::Iter<'a, PyObjHashable, PyObj>,
}

pub fn hash_map_iterator_type() -> &'static PyObj {
    tpo::static_type!(tpo::TypeSpec {
        name: "clx_rust.PersistentHashMapIterator",
        flags: Py_TPFLAGS_DEFAULT,
        size: std::mem::size_of::<HashMapIterator>(),
        new: Some(utils::disallowed_new!(hash_map_iterator_type)),
        dealloc: Some(utils::generic_dealloc::<HashMapIterator>),
        iter: Some(hash_map_iterator_iter),
        next: Some(hash_map_iterator_next),
        ..Default::default()
    })
}

unsafe extern "C" fn hash_map_iterator_iter(
    self_: *mut PyObject,
) -> *mut PyObject {
    self_
}

unsafe extern "C" fn hash_map_iterator_next(
    self_: *mut PyObject,
) -> *mut PyObject {
    utils::wrap_body!({
        let self_ = PyObj::borrow(self_);
        let iter = self_.as_ref::<HashMapIterator>();
        match iter.iterator.next() {
            Some((key, _value)) => Ok(key.as_pyobj().clone()),
            None => {
                PyErr_SetNone(PyExc_StopIteration);
                Err(())
            }
        }
    })
}

fn raise_key_error() -> Result<PyObj, ()> {
    unsafe {
        PyErr_SetString(PyExc_KeyError, "Key not found\0".as_ptr().cast());
    }
    Err(())
}
