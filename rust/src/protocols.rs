use crate::object::PyObj;
use crate::type_object as tpo;
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
    utils::module_add_type!(module, IRecord, irecord_type());
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
    tpo::static_type!(tpo::TypeSpec {
        name: "clx_rust.IMeta",
        flags: Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        methods: vec![
            tpo::method!("with_meta", dummy_method),
        ],
        ..Default::default()
    })
}

pub fn icounted_type() -> &'static PyObj {
    tpo::static_type!(tpo::TypeSpec {
        name: "clx_rust.ICounted",
        flags: Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        methods: vec![
            tpo::method!("count_", dummy_method),
        ],
        ..Default::default()
    })
}

pub fn iseqable_type() -> &'static PyObj {
    tpo::static_type!(tpo::TypeSpec {
        name: "clx_rust.ISeqable",
        flags: Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        methods: vec![
            tpo::method!("seq", dummy_method),
        ],
        ..Default::default()
    })
}

pub fn iseq_type() -> &'static PyObj {
    tpo::static_type!(tpo::TypeSpec {
        name: "clx_rust.ISeq",
        bases: vec![iseqable_type()],
        flags: Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        methods: vec![
            tpo::method!("first", dummy_method),
            tpo::method!("next", dummy_method),
            tpo::method!("rest", dummy_method),
        ],
        ..Default::default()
    })
}

pub fn isequential_type() -> &'static PyObj {
    tpo::static_type!(tpo::TypeSpec {
        name: "clx_rust.ISequential",
        flags: Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        ..Default::default()
    })
}

pub fn icollection_type() -> &'static PyObj {
    tpo::static_type!(tpo::TypeSpec {
        name: "clx_rust.ICollection",
        flags: Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        methods: vec![
            tpo::method!("conj", dummy_method),
        ],
        ..Default::default()
    })
}

pub fn iindexed_type() -> &'static PyObj {
    tpo::static_type!(tpo::TypeSpec {
        name: "clx_rust.IIndexed",
        bases: vec![icounted_type()],
        flags: Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        methods: vec![
            tpo::method!("nth", dummy_method),
        ],
        ..Default::default()
    })
}

pub fn iassociative_type() -> &'static PyObj {
    tpo::static_type!(tpo::TypeSpec {
        name: "clx_rust.IAssociative",
        flags: Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        methods: vec![
            tpo::method!("lookup", dummy_method),
            tpo::method!("assoc", dummy_method),
        ],
        ..Default::default()
    })
}

pub fn irecord_type() -> &'static PyObj {
    tpo::static_type!(tpo::TypeSpec {
        name: "clx_rust.IRecord",
        bases: vec![iassociative_type()],
        flags: Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        ..Default::default()
    })
}

pub fn mapping_type() -> &'static PyObj {
    utils::lazy_static!(PyObj, {
        let module = PyObj::import("collections.abc").unwrap();
        module.get_attr(&utils::static_pystring!("Mapping")).unwrap()
    })
}
