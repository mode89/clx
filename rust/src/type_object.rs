use crate::object::PyObj;
use crate::utils;
use pyo3_ffi::*;
use std::ffi::CString;
use std::collections::LinkedList;
use std::sync::Mutex;

// Define a type with the given specification and return a reference to
// a static instance of the type
macro_rules! static_type {
    ($spec:expr) => {
        crate::utils::lazy_static!(crate::object::PyObj, {
            crate::type_object::new_type($spec)
        })
    };
}

pub(crate) use static_type;

macro_rules! member {
    ($name:expr) => {
        crate::type_object::MemberDef {
            name: $name.to_string(),
            type_code: pyo3_ffi::Py_T_OBJECT_EX,
            offset: None,
            flags: pyo3_ffi::Py_READONLY,
        }
    };
}

pub(crate) use member;

macro_rules! method {
    ($name:expr, $func:ident) => {
        crate::type_object::MethodDef {
            name: $name.to_string(),
            method: pyo3_ffi::PyMethodDefPointer { _PyCFunctionFast: $func },
            flags: pyo3_ffi::METH_FASTCALL,
        }
    };
}

pub(crate) use method;

#[derive(Default)]
pub struct TypeSpec<'a> {
    pub name: &'a str,
    pub bases: Vec<&'static PyObj>,
    pub flags: u64,
    pub size: usize,
    pub dealloc: Option<destructor>,
    pub init: Option<initproc>,
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
    pub members: Vec<MemberDef>,
    pub methods: Vec<MethodDef>,
}

pub struct MemberDef {
    pub name: String,
    pub type_code: i32,
    pub offset: Option<usize>,
    pub flags: i32,
}

pub struct MethodDef {
    pub name: String,
    pub method: PyMethodDefPointer,
    pub flags: i32,
}

pub fn new_type(spec: TypeSpec) -> PyObj {
    let mut type_buffers = utils::lazy_static!(
        Mutex<LinkedList<_TypeBuffer>>, {
            Mutex::new(LinkedList::new())
        }).lock().unwrap();

    let size = spec.size;
    let flags = spec.flags;
    let bases: Vec<PyObj> = spec.bases.iter()
        .map(|x| (*x).clone())
        .collect();

    type_buffers.push_back(_make_type_buffer(spec));
    let tbuf = &type_buffers.back().unwrap();

    let spec = PyType_Spec {
        name: tbuf.name,
        basicsize: size as i32,
        itemsize: 0,
        flags: flags as u32,
        slots: tbuf.slots.as_ptr() as *mut PyType_Slot,
    };

    let otype = type_from_spec(&spec, bases);
    let _type = unsafe { otype.as_ref::<PyTypeObject>() };
    _type.tp_as_sequence = &tbuf.sequence_methods as *const _ as *mut _;
    _type.tp_as_mapping = &tbuf.mapping_methods as *const _ as *mut _;
    otype
}

// Some of the data that is used to create a type is required to be
// staticaly allocated, so we use this structure to hold that data
pub struct _TypeBuffer {
    name: *const i8,
    slots: Vec<PyType_Slot>,
    sequence_methods: PySequenceMethods,
    mapping_methods: PyMappingMethods,
    _members: Vec<PyMemberDef>,
    _methods: Vec<PyMethodDef>,
    _strings: LinkedList<CString>,
}

pub fn _make_type_buffer(spec: TypeSpec) -> _TypeBuffer {
    let mut strings = LinkedList::new();
    let mut alloc_string = |s: &str| {
        let s = CString::new(s).unwrap();
        strings.push_back(s);
        strings.back().unwrap().as_ptr()
    };

    let mut slots: Vec<PyType_Slot> = vec![];

    if let Some(dealloc) = spec.dealloc {
        slots.push(PyType_Slot {
            slot: Py_tp_dealloc,
            pfunc: dealloc as *mut _,
        });
    }

    if let Some(init) = spec.init {
        slots.push(PyType_Slot {
            slot: Py_tp_init,
            pfunc: init as *mut _,
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

    let mut members = spec.members.iter()
        .enumerate()
        .map(|(i, m)| PyMemberDef {
            name: alloc_string(&m.name),
            type_code: m.type_code,
            offset: match m.offset {
                Some(offset) => offset as isize,
                None => (std::mem::size_of::<PyObject>() +
                         i * std::mem::size_of::<PyObj>()) as isize,
            },
            flags: m.flags,
            doc: std::ptr::null(),
        })
        .collect::<Vec<_>>();
    members.push(PY_MEMBER_DEF_DUMMY);

    if spec.members.len() > 0 {
        slots.push(PyType_Slot {
            slot: Py_tp_members,
            pfunc: members.as_ptr() as *mut _,
        });
    }

    let mut methods = spec.methods.iter()
        .map(|m| PyMethodDef {
            ml_name: alloc_string(&m.name),
            ml_meth: m.method,
            ml_flags: m.flags,
            ml_doc: std::ptr::null(),
        })
        .collect::<Vec<_>>();
    methods.push(PY_METHOD_DEF_DUMMY);

    if spec.methods.len() > 0 {
        slots.push(PyType_Slot {
            slot: Py_tp_methods,
            pfunc: methods.as_ptr() as *mut _,
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
        name: alloc_string(&spec.name),
        slots,
        sequence_methods,
        mapping_methods,
        _members: members,
        _methods: methods,
        _strings: strings,
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
