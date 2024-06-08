use crate::utils;
use utils::PyObj;
use pyo3_ffi::*;

pub fn init_module(module: *mut PyObject) {
    utils::module_add_type!(module, IMeta, imeta_type());
    utils::module_add_type!(module, ICounted, icounted_type());
    utils::module_add_type!(module, ISeqable, iseqable_type());
    utils::module_add_type!(module, ISeq, iseq_type());
    utils::module_add_type!(module, ISequential, isequential_type());
    utils::module_add_type!(module, ICollection, icollection_type());
    utils::module_add_type!(module, IIndexed, iindexed_type());
}

extern "C" fn dummy_method(
    _self: *mut PyObject,
    _args: *mut *mut PyObject,
    _nargs: isize,
) -> *mut PyObject {
    utils::handle_gil_and_panic!({
        utils::raise_exception("Not implemented")
    })
}

pub fn imeta_type() -> &'static PyObj {
    utils::static_type!(utils::TypeSpec {
        name: "clx_rust.IMeta",
        flags: Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        methods: vec![
            ("with_meta", dummy_method),
        ],
        ..Default::default()
    })
}

pub fn icounted_type() -> &'static PyObj {
    utils::static_type!(utils::TypeSpec {
        name: "clx_rust.ICounted",
        flags: Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        methods: vec![
            ("count_", dummy_method),
        ],
        ..Default::default()
    })
}

pub fn iseqable_type() -> &'static PyObj {
    utils::static_type!(utils::TypeSpec {
        name: "clx_rust.ISeqable",
        flags: Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        methods: vec![
            ("seq", dummy_method),
        ],
        ..Default::default()
    })
}

pub fn iseq_type() -> &'static PyObj {
    utils::static_type!(utils::TypeSpec {
        name: "clx_rust.ISeq",
        bases: vec![iseqable_type()],
        flags: Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        methods: vec![
            ("first", dummy_method),
            ("next", dummy_method),
            ("rest", dummy_method),
        ],
        ..Default::default()
    })
}

pub fn isequential_type() -> &'static PyObj {
    utils::static_type!(utils::TypeSpec {
        name: "clx_rust.ISequential",
        flags: Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        ..Default::default()
    })
}

pub fn icollection_type() -> &'static PyObj {
    utils::static_type!(utils::TypeSpec {
        name: "clx_rust.ICollection",
        flags: Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        methods: vec![
            ("conj", dummy_method),
        ],
        ..Default::default()
    })
}

pub fn iindexed_type() -> &'static PyObj {
    utils::static_type!(utils::TypeSpec {
        name: "clx_rust.IIndexed",
        bases: vec![icounted_type()],
        flags: Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        methods: vec![
            ("nth", dummy_method),
        ],
        ..Default::default()
    })
}
