use crate::object::PyObj;
use pyo3_ffi::*;
use std::ffi::CString;

// Define a type with the given specification and return a reference to
// a static instance of the type
macro_rules! static_type {
    ($spec:expr) => {
        {
            use crate::utils;
            utils::lazy_static!(crate::object::PyObj, {
                crate::type_object::_make_type(
                    utils::lazy_static!(crate::type_object::_TypeBuffer, {
                        crate::type_object::_make_type_buffer($spec)
                    }))
            })
        }
    };
}

pub(crate) use static_type;

#[derive(Default)]
pub struct TypeSpec {
    pub name: String,
    pub bases: Vec<&'static PyObj>,
    pub flags: u64,
    pub size: usize,
    pub dealloc: Option<destructor>,
    pub repr: Option<reprfunc>,
    pub hash: Option<hashfunc>,
    pub call: Option<ternaryfunc>,
    pub compare: Option<richcmpfunc>,
    pub iter: Option<getiterfunc>,
    pub next: Option<iternextfunc>,
    pub sequence_length: Option<lenfunc>,
    pub sequence_item: Option<ssizeargfunc>,
    pub mapping_length: Option<lenfunc>,
    pub mapping_subscript: Option<binaryfunc>,
    pub members: Vec<String>,
    pub methods: Vec<(&'static str, _PyCFunctionFast)>,
}

pub fn _make_type(tbuf: &'static _TypeBuffer) -> PyObj {
    let spec = PyType_Spec {
        name: tbuf.name.as_ptr().cast(),
        basicsize: tbuf.spec.size as i32,
        itemsize: 0,
        flags: tbuf.spec.flags as u32,
        slots: tbuf.slots.as_ptr() as *mut PyType_Slot,
    };

    let bases: Vec<PyObj> = tbuf.spec.bases.iter()
        .map(|x| (*x).clone())
        .collect();

    let otype = type_from_spec(&spec, bases);
    let _type = unsafe { otype.as_ref::<PyTypeObject>() };
    _type.tp_as_sequence = &tbuf.sequence_methods as *const _ as *mut _;
    _type.tp_as_mapping = &tbuf.mapping_methods as *const _ as *mut _;
    otype
}

// Some of the data that is used to create a type is required to be
// staticaly allocated, so we use this structure to hold that data
pub struct _TypeBuffer {
    spec: TypeSpec,
    name: CString,
    slots: Vec<PyType_Slot>,
    sequence_methods: PySequenceMethods,
    mapping_methods: PyMappingMethods,
    _member_names: Vec<CString>,
    _members: Vec<PyMemberDef>,
    _method_names: Vec<CString>,
    _methods: Vec<PyMethodDef>,
}

pub fn _make_type_buffer(spec: TypeSpec) -> _TypeBuffer {
    let mut slots: Vec<PyType_Slot> = vec![];

    if let Some(dealloc) = spec.dealloc {
        slots.push(PyType_Slot {
            slot: Py_tp_dealloc,
            pfunc: dealloc as *mut _,
        });
    }

    if let Some(repr) = spec.repr {
        slots.push(PyType_Slot {
            slot: Py_tp_repr,
            pfunc: repr as *mut _,
        });
    }

    if let Some(hash) = spec.hash {
        slots.push(PyType_Slot {
            slot: Py_tp_hash,
            pfunc: hash as *mut _,
        });
    }

    if let Some(call) = spec.call {
        slots.push(PyType_Slot {
            slot: Py_tp_call,
            pfunc: call as *mut _,
        });
    }

    if let Some(compare) = spec.compare {
        slots.push(PyType_Slot {
            slot: Py_tp_richcompare,
            pfunc: compare as *mut _,
        });
    }

    if let Some(iter) = spec.iter {
        slots.push(PyType_Slot {
            slot: Py_tp_iter,
            pfunc: iter as *mut _,
        });
    }

    if let Some(next) = spec.next {
        slots.push(PyType_Slot {
            slot: Py_tp_iternext,
            pfunc: next as *mut _,
        });
    }

    let _member_names = spec.members.iter()
        .map(|x| CString::new(x.clone()).unwrap())
        .collect::<Vec<_>>();

    let mut _members = _member_names.iter()
        .enumerate()
        .map(|(i, x)| PyMemberDef {
            name: x.as_ptr().cast(),
            type_code: Py_T_OBJECT_EX,
            offset: (std::mem::size_of::<PyObject>() +
                     std::mem::size_of::<PyObj>() * i) as isize,
            flags: Py_READONLY,
            doc: std::ptr::null(),
        })
        .collect::<Vec<_>>();
    _members.push(PY_MEMBER_DEF_DUMMY);

    if spec.members.len() > 0 {
        slots.push(PyType_Slot {
            slot: Py_tp_members,
            pfunc: _members.as_ptr() as *mut _,
        });
    }

    let _method_names = spec.methods.iter()
        .map(|x| CString::new(x.0).unwrap())
        .collect::<Vec<_>>();

    let mut _methods = spec.methods.iter()
        .zip(_method_names.iter())
        .map(|(mdef, cname)| PyMethodDef {
            ml_name: cname.as_ptr().cast(),
            ml_meth: PyMethodDefPointer { _PyCFunctionFast: mdef.1 },
            ml_flags: METH_FASTCALL,
            ml_doc: std::ptr::null(),
        })
        .collect::<Vec<_>>();
    _methods.push(PY_METHOD_DEF_DUMMY);

    if spec.methods.len() > 0 {
        slots.push(PyType_Slot {
            slot: Py_tp_methods,
            pfunc: _methods.as_ptr() as *mut _,
        });
    }

    slots.push(PY_TYPE_SLOT_DUMMY);

    let sequence_methods = PySequenceMethods {
        sq_length: spec.sequence_length,
        sq_concat: None,
        sq_repeat: None,
        sq_item: spec.sequence_item,
        sq_ass_item: None,
        sq_contains: None,
        sq_inplace_concat: None,
        sq_inplace_repeat: None,
        was_sq_ass_slice: std::ptr::null_mut(),
        was_sq_slice: std::ptr::null_mut(),
    };

    let mapping_methods = PyMappingMethods {
        mp_length: spec.mapping_length,
        mp_subscript: spec.mapping_subscript,
        mp_ass_subscript: None,
    };

    _TypeBuffer {
        name: CString::new(spec.name.clone()).unwrap(),
        spec,
        _member_names,
        _members,
        _method_names,
        _methods,
        slots,
        sequence_methods,
        mapping_methods,
    }
}

const PY_TYPE_SLOT_DUMMY: PyType_Slot = PyType_Slot {
    slot: 0,
    pfunc: std::ptr::null_mut(),
};

const PY_MEMBER_DEF_DUMMY: PyMemberDef = PyMemberDef {
    name: std::ptr::null(),
    type_code: 0,
    offset: 0,
    flags: 0,
    doc: std::ptr::null(),
};

const PY_METHOD_DEF_DUMMY: PyMethodDef = PyMethodDef {
    ml_name: std::ptr::null(),
    ml_meth: unsafe { std::mem::transmute(0usize) },
    ml_flags: 0,
    ml_doc: std::ptr::null(),
};

fn type_from_spec(spec: &PyType_Spec, bases: Vec<PyObj>) -> PyObj {
    unsafe {
        if bases.is_empty() {
            return PyObj::own(PyType_FromSpec(spec as *const _ as *mut _));
        } else {
            let _bases = PyObj::own(PyTuple_New(bases.len() as isize));
            for (i, base) in bases.iter().enumerate() {
                PyTuple_SetItem(
                    _bases.as_ptr(),
                    i as isize,
                    base.clone().into_ptr());
            }
            PyObj::own(PyType_FromSpecWithBases(
                spec as *const _ as *mut _,
                _bases.into_ptr()))
        }
    }
}
