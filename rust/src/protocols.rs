use crate::object::PyObj;
use crate::type_object::*;
use crate::utils;
use pyo3_ffi::*;

pub fn init_module(module: *mut PyObject) {
    utils::module_add_type!(module, IMeta, imeta_type());
    utils::module_add_type!(module, ICounted, icounted_type());
    utils::module_add_type!(module, ISeqable, iseqable_type());
    utils::module_add_type!(module, ISeq, iseq_type());
    utils::module_add_type!(module, ISequential, isequential_type());
    utils::module_add_type!(module, ICollection, icollection_type());
    utils::module_add_type!(module, IIndexed, iindexed_type());
    utils::module_add_type!(module, IAssociative, iassociative_type());
}

extern "C" fn dummy_method(
    _self: *mut PyObject,
    _args: *mut *mut PyObject,
    _nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        utils::raise_exception("Not implemented")
    })
}

pub fn imeta_type() -> &'static PyObj {
    static_type!(TypeSpec {
        name: "clx_rust.IMeta".to_string(),
        flags: Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        methods: vec![
            utils::method!("with_meta", dummy_method),
        ],
        ..Default::default()
    })
}

pub fn icounted_type() -> &'static PyObj {
    static_type!(TypeSpec {
        name: "clx_rust.ICounted".to_string(),
        flags: Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        methods: vec![
            utils::method!("count_", dummy_method),
        ],
        ..Default::default()
    })
}

pub fn iseqable_type() -> &'static PyObj {
    static_type!(TypeSpec {
        name: "clx_rust.ISeqable".to_string(),
        flags: Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        methods: vec![
            utils::method!("seq", dummy_method),
        ],
        ..Default::default()
    })
}

pub fn iseq_type() -> &'static PyObj {
    static_type!(TypeSpec {
        name: "clx_rust.ISeq".to_string(),
        bases: vec![iseqable_type()],
        flags: Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        methods: vec![
            utils::method!("first", dummy_method),
            utils::method!("next", dummy_method),
            utils::method!("rest", dummy_method),
        ],
        ..Default::default()
    })
}

pub fn isequential_type() -> &'static PyObj {
    static_type!(TypeSpec {
        name: "clx_rust.ISequential".to_string(),
        flags: Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        ..Default::default()
    })
}

pub fn icollection_type() -> &'static PyObj {
    static_type!(TypeSpec {
        name: "clx_rust.ICollection".to_string(),
        flags: Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        methods: vec![
            utils::method!("conj", dummy_method),
        ],
        ..Default::default()
    })
}

pub fn iindexed_type() -> &'static PyObj {
    static_type!(TypeSpec {
        name: "clx_rust.IIndexed".to_string(),
        bases: vec![icounted_type()],
        flags: Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        methods: vec![
            utils::method!("nth", dummy_method),
        ],
        ..Default::default()
    })
}

pub fn iassociative_type() -> &'static PyObj {
    static_type!(TypeSpec {
        name: "clx_rust.IAssociative".to_string(),
        flags: Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        methods: vec![
            utils::method!("lookup", dummy_method),
            utils::method!("assoc", dummy_method),
        ],
        ..Default::default()
    })
}

pub fn mapping_type() -> &'static PyObj {
    utils::lazy_static!(PyObj, {
        let module = PyObj::import("collections.abc").unwrap();
        module.get_attr(&utils::static_pystring!("Mapping")).unwrap()
    })
}
