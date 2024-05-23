use crate::symbol::*;
use crate::utils;
use utils::PyObj;
use std::ffi::{CStr, CString};
use std::sync::Mutex;
use std::collections::{
    HashMap,
    hash_map::DefaultHasher,
};
use std::hash::{Hash, Hasher};
use pyo3_ffi::*;

pub fn init_module(module: *mut PyObject) {
    utils::module_add_method!(module, keyword);
    utils::module_add_method!(module, is_keyword);
    utils::module_add_method!(module, is_simple_keyword);
    utils::module_add_type!(module, Keyword, keyword_type());
}

#[repr(C)]
pub struct Keyword {
    ob_base: PyObject,
    name: PyObj,
    namespace: PyObj,
    hash: Option<i64>,
}

fn keyword_type() -> &'static PyObj {
    utils::lazy_static!(PyObj, {
        let members = vec![
            utils::member_def!(0, name),
            utils::member_def!(1, namespace),
            utils::PY_MEMBER_DEF_DUMMY
        ];

        let slots = vec![
            utils::generic_dealloc_slot::<Keyword>(),
            PyType_Slot {
                slot: Py_tp_members,
                pfunc: members.as_ptr() as *mut _,
            },
            PyType_Slot {
                slot: Py_tp_repr,
                pfunc: keyword_repr as *mut _,
            },
            PyType_Slot {
                slot: Py_tp_hash,
                pfunc: keyword_hash as *mut _,
            },
            PyType_Slot {
                slot: Py_tp_richcompare,
                pfunc: keyword_compare as *mut _,
            },
            PyType_Slot {
                slot: Py_tp_call,
                pfunc: keyword_call as *mut _,
            },
            utils::PY_TYPE_SLOT_DUMMY
        ];

        let spec = PyType_Spec {
            name: utils::static_cstring!("clx_rust.Keyword").as_ptr().cast(),
            basicsize: std::mem::size_of::<Keyword>() as i32,
            itemsize: 0,
            flags: (Py_TPFLAGS_DEFAULT |
                    Py_TPFLAGS_DISALLOW_INSTANTIATION) as u32,
            slots: slots.as_ptr() as *mut PyType_Slot,
        };

        unsafe {
            PyObj::own(PyType_FromSpec(&spec as *const _ as *mut _))
        }
    })
}

unsafe extern "C" fn keyword(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::handle_gil_and_panic!({
        match nargs {
            1 => {
                let argo = PyObj::borrow(*args);
                if argo.is_string() {
                    let arg_ptr = PyUnicode_AsUTF8AndSize(
                        *argo, std::ptr::null_mut());
                    if arg_ptr.is_null() {
                        return utils::raise_exception(
                            "keyword() argument must be a string");
                    }

                    let arg = CStr::from_ptr(arg_ptr).to_str().unwrap();
                    match arg.chars().position(|c| c == '/') {
                        Some(index) => {
                            if arg == "/" {
                                intern_keyword(
                                    None, CString::new("/").unwrap())
                                    .into_ptr()
                            } else {
                                let (ns, name) = arg.split_at(index);
                                intern_keyword(
                                    Some(CString::new(ns).unwrap()),
                                    CString::new(&name[1..]).unwrap())
                                    .into_ptr()
                            }
                        }
                        None => intern_keyword(
                            None, CString::new(arg).unwrap()).into_ptr()
                    }
                } else if argo.type_is(keyword_type()) {
                    argo.into_ptr()
                } else if argo.type_is(symbol_type()) {
                    let sym = argo.as_ref::<Symbol>();
                    let ns = &sym.namespace;
                    let name = &sym.name;
                    intern_keyword(
                        if ns.is_none() {
                            None
                        } else {
                            Some(CStr::from_ptr(
                                PyUnicode_AsUTF8AndSize(
                                    **ns, std::ptr::null_mut()))
                                .to_owned())
                        },
                        CStr::from_ptr(
                            PyUnicode_AsUTF8AndSize(
                                **name, std::ptr::null_mut()))
                            .to_owned()
                    ).into_ptr()
                } else {
                    return utils::raise_exception(
                        "keyword() argument must be string, keyword or symbol");
                }
            },
            2 => {
                let ns = PyObj::borrow(*args);
                if !ns.is_none() && !ns.is_string() {
                    return utils::raise_exception(
                        "keyword namespace must be a string");
                }

                let name = PyObj::borrow(*args.add(1));
                if !name.is_string() {
                    return utils::raise_exception(
                        "keyword name must be a string");
                }

                intern_keyword(
                    if ns.is_none() {
                        None
                    } else {
                        Some(CStr::from_ptr(
                            PyUnicode_AsUTF8AndSize(
                                *ns, std::ptr::null_mut())).to_owned())
                    },
                    CStr::from_ptr(
                        PyUnicode_AsUTF8AndSize(*name, std::ptr::null_mut()))
                        .to_owned()
                ).into_ptr()
            },
            _ => {
                return utils::raise_exception(
                    "keyword() takes 1 or 2 positional arguments");
            }
        }
    })
}

unsafe fn intern_keyword(ns: Option<CString>, name: CString) -> PyObj {
    type Key = (Option<CString>, CString);
    let mut table = utils::lazy_static!(
        Mutex<HashMap<Key, PyObj>>, {
            Mutex::new(HashMap::new())
        }).lock().unwrap();
    table.entry((ns, name))
        .or_insert_with_key(|key| {
            let obj = PyObj::alloc(keyword_type());
            let keyword = obj.as_ref::<Keyword>();
            let ns = &key.0;
            std::ptr::write(&mut keyword.namespace, match ns {
                Some(ns) => utils::intern_string(ns),
                None => PyObj::none()
            });
            let name = &key.1;
            std::ptr::write(&mut keyword.name, utils::intern_string(name));
            keyword.hash = None;
            obj
        }).clone()
}

unsafe extern "C" fn keyword_repr(
    self_: *mut PyObject,
) -> *mut PyObject {
    let self_ = PyObj::borrow(self_);
    let keyword = self_.as_ref::<Keyword>();
    if keyword.namespace.is_none() {
        let name = PyUnicode_AsUTF8AndSize(*keyword.name,
            std::ptr::null_mut());
        PyUnicode_FromFormat(":%s\0".as_ptr().cast(), name)
    } else {
        let namespace = PyUnicode_AsUTF8AndSize(*keyword.namespace,
            std::ptr::null_mut());
        let name = PyUnicode_AsUTF8AndSize(*keyword.name,
            std::ptr::null_mut());
        PyUnicode_FromFormat(":%s/%s\0".as_ptr().cast(), namespace, name)
    }
}

unsafe extern "C" fn keyword_hash(
    self_: *mut PyObject,
) -> i64 {
    let self_ = PyObj::borrow(self_);
    let keyword = self_.as_ref::<Keyword>();
    match (*keyword).hash {
        Some(hash) => hash,
        None => {
            let mut hasher = DefaultHasher::new();

            if !keyword.namespace.is_none() {
                let namespace = PyUnicode_AsUTF8AndSize(
                    *keyword.namespace, std::ptr::null_mut());
                namespace.hash(&mut hasher);
            }
            let name = PyUnicode_AsUTF8AndSize(
                *keyword.name, std::ptr::null_mut());
            name.hash(&mut hasher);

            let hash = hasher.finish() as i64;
            (*keyword).hash = Some(hash);
            hash
        }
    }
}

unsafe extern "C" fn keyword_compare(
    self_: *mut PyObject,
    other: *mut PyObject,
    op: i32,
) -> *mut PyObject {
    utils::handle_gil_and_panic!({
        let other = PyObj::borrow(other);
        if other.type_is(keyword_type()) {
            let self_ = PyObj::borrow(self_);
            let self_ = self_.as_ref::<Keyword>();
            let other = other.as_ref::<Keyword>();
            PyObj::from(match op {
                pyo3_ffi::Py_EQ => {
                    *self_.namespace == *other.namespace
                        && *self_.name == *other.name
                }
                pyo3_ffi::Py_NE => {
                    *self_.namespace != *other.namespace
                        || *self_.name != *other.name
                }
                _ => {
                    return utils::raise_exception(
                        "keyword comparison not supported");
                }
            }).into_ptr()
        } else {
            utils::ref_false()
        }
    })
}

unsafe extern "C" fn keyword_call(
    self_: *mut PyObject,
    args: *mut PyObject,
    _kw: *mut PyObject,
) -> *mut PyObject {
    utils::handle_gil_and_panic!({
        let mut result = std::ptr::null_mut();
        let coll: *mut PyObject = std::ptr::null_mut();
        let default: *mut PyObject = std::ptr::null_mut();
        if PyArg_UnpackTuple(args, "Keyword.__call__()\0".as_ptr().cast(),
                1, 2, &coll, &default) != 0 {
            result = PyObject_CallMethodObjArgs(coll,
                utils::static_pystring!("lookup"),
                self_,
                if !default.is_null() {
                    default
                } else {
                    Py_None()
                },
                std::ptr::null_mut() as *mut PyObject)
        }
        result
    })
}

unsafe extern "C" fn is_keyword(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    _nargs: isize,
) -> *mut PyObject {
    utils::handle_gil_and_panic!({
        PyObj::from(Py_TYPE(*args) == keyword_type().as_ptr()).into_ptr()
    })
}

unsafe extern "C" fn is_simple_keyword(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    _nargs: isize,
) -> *mut PyObject {
    utils::handle_gil_and_panic!({
        let argo = PyObj::borrow(*args);
        if argo.type_is(keyword_type()) {
            let arg = argo.as_ref::<Keyword>();
            PyObj::from(arg.namespace.is_none()).into_ptr()
        } else {
            utils::ref_false()
        }
    })
}
