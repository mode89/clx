use crate::object::PyObj;
use crate::list;
use crate::utils;
use crate::protocols::*;
use pyo3_ffi::*;

pub fn init_module(module: *mut PyObject) {
    utils::module_add_method!(module, hash_map);
    utils::module_add_method!(module, is_hash_map);
    utils::module_add_method!(module, hash_map_from);
    utils::module_add_type!(module, PersistentHashMap, hash_map_type());
}

#[repr(C)]
pub struct HashMap {
    ob_base: PyObject,
    meta: PyObj,
    impl_: std::collections::HashMap<PyObj, PyObj>,
    hash: Option<isize>,
}

pub fn hash_map_type() -> &'static PyObj {
    utils::static_type!(
        utils::TypeSpec {
            name: "clx_rust.PersistentHashMap",
            bases: vec![
                imeta_type(),
                icounted_type(),
                iseqable_type(),
                iassociative_type(),
                mapping_type(),
            ],
            flags: Py_TPFLAGS_DEFAULT |
                   Py_TPFLAGS_DISALLOW_INSTANTIATION,
            size: std::mem::size_of::<HashMap>(),
            dealloc: Some(utils::generic_dealloc::<HashMap>),
            compare: Some(py_hash_map_compare),
            // TODO hash: Some(py_vector_hash),
            call: Some(py_hash_map_call),
            iter: Some(py_hash_map_iter),
            mapping_length: Some(py_hash_map_len),
            mapping_subscript: Some(py_hash_map_subscript),
            members: vec![ "__meta__" ],
            methods: vec![
                // TODO ("with_meta", py_vector_with_meta),
                ("__getitem__", py_hash_map_getitem),
                ("assoc", py_hash_map_assoc),
                ("lookup", py_hash_map_lookup),
                ("merge", py_hash_map_merge),
                ("count_", py_hash_map_count),
                ("seq", py_hash_map_seq),
                // TODO ("conj", py_vector_conj),
            ],
            ..utils::TypeSpec::default()
        }
    )
}

fn _hash_map(
    impl_: std::collections::HashMap<PyObj, PyObj>,
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
        _hash_map(std::collections::HashMap::new(), PyObj::none(), None)
    }).clone()
}

extern "C" fn hash_map(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::handle_gil_and_panic!({
        if nargs == 0 {
            empty_hash_map().into_ptr()
        } else if nargs % 2 != 0 {
            utils::raise_exception(
                "PersistentHashMap() requires an even number of arguments")
        } else {
            let mut impl_ = std::collections::HashMap::new();
            let kv_num = nargs / 2;
            for i in 0..kv_num {
                let key = PyObj::borrow(unsafe { *args.offset(i * 2) });
                let value = PyObj::borrow(unsafe { *args.offset(i * 2 + 1) });
                impl_.insert(key, value);
            }
            _hash_map(impl_, PyObj::none(), None).into_ptr()
        }
    })
}

extern "C" fn is_hash_map(
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

extern "C" fn hash_map_from(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::handle_gil_and_panic!({
        if nargs != 1 {
            utils::raise_exception(
                "PersistentHashMap.from_iter() requires one argument")
        } else {
            let coll = PyObj::borrow(unsafe { *args });
            if let Some(iter) = coll.get_iter() {
                let mut impl_ = std::collections::HashMap::new();
                while let Some(item) = iter.next() {
                    let _0 = utils::lazy_static!(PyObj, { PyObj::from(0) });
                    let _1 = utils::lazy_static!(PyObj, { PyObj::from(1) });
                    if let Some(key) = item.get_item(_0) {
                        if let Some(value) = item.get_item(_1) {
                            impl_.insert(key, value);
                        } else {
                            return std::ptr::null_mut()
                        }
                    } else {
                        return std::ptr::null_mut()
                    }
                }
                if impl_.is_empty() {
                    empty_hash_map().into_ptr()
                } else {
                    _hash_map(impl_, PyObj::none(), None).into_ptr()
                }
            } else {
                std::ptr::null_mut()
            }
        }
    })
}

extern "C" fn py_hash_map_assoc(
    self_: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::handle_gil!({
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
                impl_.insert(key, value);
            }
            _hash_map(impl_, PyObj::none(), None).into_ptr()
        }
    })
}

extern "C" fn py_hash_map_lookup(
    self_: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::handle_gil!({
        if nargs != 2 {
            utils::raise_exception(
                "PersistentHashMap.lookup() requires two arguments")
        } else {
            let self_ = PyObj::borrow(self_);
            let self_ = unsafe { self_.as_ref::<HashMap>() };
            let key = PyObj::borrow(unsafe { *args });
            let not_found = PyObj::borrow(unsafe { *args.offset(1) });
            match self_.impl_.get(&key) {
                Some(value) => value.clone().into_ptr(),
                None => not_found.into_ptr(),
            }
        }
    })
}

extern "C" fn py_hash_map_count(
    self_: *mut PyObject,
    _args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::handle_gil!({
        if nargs != 0 {
            utils::raise_exception(
                "PersistentHashMap.count_() takes no arguments")
        } else {
            let self_ = PyObj::borrow(self_);
            let self_ = unsafe { self_.as_ref::<HashMap>() };
            PyObj::from(self_.impl_.len() as i64).into_ptr()
        }
    })
}

extern "C" fn py_hash_map_merge(
    self_: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::handle_gil!({
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
                _hash_map(impl_, PyObj::none(), None).into_ptr()
            }
        }
    })
}

extern "C" fn py_hash_map_compare(
    self_: *mut PyObject,
    other: *mut PyObject,
    op: i32,
) -> *mut PyObject {
    utils::handle_gil_and_panic!({
        let self_ = PyObj::borrow(self_);
        let other = PyObj::borrow(other);
        PyObj::from(match op {
            pyo3_ffi::Py_EQ => hash_map_eq(&self_, &other),
            pyo3_ffi::Py_NE => !hash_map_eq(&self_, &other),
            _ => {
                return utils::raise_exception(
                    "hash-map comparison not supported");
            }
        }).into_ptr()
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
    utils::handle_gil_and_panic!({
        let self_ = PyObj::borrow(self_);
        let key = PyObj::borrow(key);
        match hash_map_getitem(&self_, key) {
            Some(value) => value.into_ptr(),
            None => {
                raise_key_error();
                std::ptr::null_mut()
            }
        }
    })
}

extern "C" fn py_hash_map_getitem(
    self_: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::handle_gil_and_panic!({
        if nargs != 1 {
            utils::raise_exception(
                "PersistentHashMap.__getitem__() takes one argument")
        } else {
            let self_ = PyObj::borrow(self_);
            let key = PyObj::borrow(unsafe { *args });
            match hash_map_getitem(&self_, key) {
                Some(value) => value.into_ptr(),
                None => {
                    raise_key_error();
                    std::ptr::null_mut()
                }
            }
        }
    })
}

fn hash_map_getitem(self_: &PyObj, key: PyObj) -> Option<PyObj> {
    let self_ = unsafe { self_.as_ref::<HashMap>() };
    self_.impl_.get(&key).cloned()
}

extern "C" fn py_hash_map_seq(
    self_: *mut PyObject,
    _args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::handle_gil_and_panic!({
        if nargs != 0 {
            utils::raise_exception(
                "PersistentVector.seq() takes no arguments")
        } else {
            hash_map_seq(&PyObj::borrow(self_)).into_ptr()
        }
    })
}

pub fn hash_map_seq(self_: &PyObj) -> PyObj {
    let self_ = unsafe { self_.as_ref::<HashMap>() };
    if self_.impl_.is_empty() {
        PyObj::none()
    } else {
        let mut lst = list::empty_list();
        for (key, value) in self_.impl_.iter() {
            let kv = PyObj::tuple2(key.clone(), value.clone());
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
    utils::handle_gil_and_panic!({
        let key: *mut PyObject = std::ptr::null_mut();
        let not_found: *mut PyObject = std::ptr::null_mut();
        if unsafe { PyArg_UnpackTuple(args,
                "PersistentHashMap.__call__()\0".as_ptr().cast(),
                1, 2, &key, &not_found) } != 0 {
            let key = PyObj::borrow(key);
            let self_ = PyObj::borrow(self_);
            let self_ = unsafe { self_.as_ref::<HashMap>() };
            match self_.impl_.get(&key) {
                Some(value) => value.clone().into_ptr(),
                None => {
                    if !not_found.is_null() {
                        PyObj::borrow(not_found).into_ptr()
                    } else {
                        PyObj::none().into_ptr()
                    }
                }
            }
        } else {
            std::ptr::null_mut()
        }
    })
}

extern "C" fn py_hash_map_iter(
    self_: *mut PyObject,
) -> *mut PyObject {
    utils::handle_gil_and_panic!({
        let oself = PyObj::borrow(self_);
        let self_ = unsafe { oself.as_ref::<HashMap>() };
        let oiter = PyObj::alloc(hash_map_iterator_type());
        unsafe {
            let iter = oiter.as_ref::<HashMapIterator>();
            std::ptr::write(&mut iter.map, oself.clone());
            std::ptr::write(&mut iter.iterator, self_.impl_.iter());
        }
        oiter.into_ptr()
    })
}

#[repr(C)]
pub struct HashMapIterator<'a> {
    ob_base: PyObject,
    map: PyObj,
    iterator: std::collections::hash_map::Iter<'a, PyObj, PyObj>,
}

pub fn hash_map_iterator_type() -> &'static PyObj {
    utils::static_type!(utils::TypeSpec {
        name: "clx_rust.PersistentHashMapIterator",
        flags: Py_TPFLAGS_DEFAULT | Py_TPFLAGS_DISALLOW_INSTANTIATION,
        size: std::mem::size_of::<HashMapIterator>(),
        dealloc: Some(utils::generic_dealloc::<HashMapIterator>),
        iter: Some(hash_map_iterator_iter),
        next: Some(hash_map_iterator_next),
        ..Default::default()
    })
}

unsafe extern "C" fn hash_map_iterator_iter(
    self_: *mut PyObject,
) -> *mut PyObject {
    utils::handle_gil_and_panic!({
        self_
    })
}

unsafe extern "C" fn hash_map_iterator_next(
    self_: *mut PyObject,
) -> *mut PyObject {
    utils::handle_gil_and_panic!({
        let self_ = PyObj::borrow(self_);
        let iter = self_.as_ref::<HashMapIterator>();
        match iter.iterator.next() {
            Some((key, _value)) => key.clone().into_ptr(),
            None => {
                PyErr_SetNone(PyExc_StopIteration);
                std::ptr::null_mut()
            }
        }
    })
}

fn raise_key_error() {
    unsafe {
        PyErr_SetString(PyExc_KeyError, "Key not found\0".as_ptr().cast());
    }
}
