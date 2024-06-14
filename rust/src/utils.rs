use crate::object::PyObj;
use pyo3_ffi::*;
use std::ffi::CString;

pub const PY_TYPE_SLOT_DUMMY: PyType_Slot = PyType_Slot {
    slot: 0,
    pfunc: std::ptr::null_mut(),
};

pub const PY_MEMBER_DEF_DUMMY: PyMemberDef = PyMemberDef {
    name: std::ptr::null(),
    type_code: 0,
    offset: 0,
    flags: 0,
    doc: std::ptr::null(),
};

pub const PY_METHOD_DEF_DUMMY: PyMethodDef = PyMethodDef {
    ml_name: std::ptr::null(),
    ml_meth: unsafe { std::mem::transmute(0usize) },
    ml_flags: 0,
    ml_doc: std::ptr::null(),
};

#[inline]
pub fn intern_string(s: &std::ffi::CStr) -> PyObj {
    unsafe {
        PyObj::own(PyUnicode_InternFromString(s.as_ptr()))
    }
}

#[inline]
pub fn intern_string_in_place(obj: PyObj) -> PyObj {
    unsafe {
        PyUnicode_InternInPlace(&*obj as *const _ as *mut *mut PyObject);
    }
    obj
}

#[inline]
pub fn incref(obj: *mut PyObject) -> *mut PyObject {
    unsafe { Py_XINCREF(obj) };
    obj
}

#[inline]
pub fn ref_false() -> *mut PyObject {
    incref(unsafe { Py_False() })
}

macro_rules! lazy_static {
    ($type:ty, $code:block) => {
        {
            use std::sync::OnceLock;
            static mut CELL: OnceLock<$type> = OnceLock::new();
            static CODE: fn() -> $type = || $code;
            unsafe { CELL.get_or_init(CODE) }
        }
    };
}

pub(crate) use lazy_static;

macro_rules! static_cstring {
    ($val:expr) => {
        {
            use std::ffi::CString;
            crate::utils::lazy_static!(CString, {
                CString::new($val).unwrap()
            })
        }
    };
}

pub(crate) use static_cstring;

macro_rules! static_pystring {
    ($val:expr) => {
        {
            #[allow(unused_unsafe)]
            crate::utils::lazy_static!(PyObj, {
                let cstring = std::ffi::CString::new($val).unwrap();
                unsafe {
                    PyObj::own(PyUnicode_InternFromString(cstring.as_ptr()))
                }
            }).clone()
        }
    };
}

pub(crate) use static_pystring;

macro_rules! module_add_method {
    ($module:expr, $func:expr) => {
        {
            let method = utils::lazy_static!(PyMethodDef, {
                PyMethodDef {
                    ml_name: utils::static_cstring!(stringify!($func))
                        .as_ptr().cast(),
                    ml_meth: PyMethodDefPointer { _PyCFunctionFast: $func },
                    ml_flags: METH_FASTCALL,
                    ml_doc: std::ptr::null(),
                }
            });

            let name = utils::static_cstring!(stringify!($func));
            unsafe {
                PyModule_AddObject(
                    $module,
                    name.as_ptr().cast(),
                    PyCFunction_New(
                        method as *const _ as *mut _,
                        std::ptr::null_mut())
                );
            }
        }
    };
}

pub(crate) use module_add_method;

macro_rules! module_add_type {
    ($module:expr, $type:ty, $type_obj:expr) => {
        {
            let name = utils::static_cstring!(stringify!($type));
            unsafe {
                PyModule_AddObject(
                    $module,
                    name.as_ptr().cast(),
                    $type_obj.clone().into_ptr()
                );
            }
        }
    };
}

pub(crate) use module_add_type;

macro_rules! handle_gil_and_panic {
    ($code:block) => {{
        let gil = unsafe { PyGILState_Ensure() };
        let result = std::panic::catch_unwind(|| {
            $code
        }).unwrap_or_else(|_| {
            crate::utils::raise_exception("panic occurred in Rust code")
        });
        unsafe { PyGILState_Release(gil) };
        result
    }}
}

pub(crate) use handle_gil_and_panic;

macro_rules! handle_gil {
    ($code:block) => {
        {
            let gil = unsafe { PyGILState_Ensure() };
            let result = { $code };
            unsafe { PyGILState_Release(gil) };
            result
        }
    }
}

pub(crate) use handle_gil;

#[allow(unused_macros)]
macro_rules! member_def {
    ($index:expr, $name:expr) => {
        PyMemberDef {
            name: utils::static_cstring!(stringify!($name)).as_ptr().cast(),
            type_code: Py_T_OBJECT_EX,
            offset: (std::mem::size_of::<PyObject>() +
                     std::mem::size_of::<PyObj>() * $index) as isize,
            flags: Py_READONLY,
            doc: std::ptr::null(),
        }
    };
}

#[allow(unused_imports)]
pub(crate) use member_def;

#[allow(unused_macros)]
macro_rules! method_def {
    ($name:expr, $func:expr) => {
        PyMethodDef {
            ml_name: utils::static_cstring!($name).as_ptr().cast(),
            ml_meth: PyMethodDefPointer { _PyCFunctionFast: $func },
            ml_flags: METH_FASTCALL,
            ml_doc: std::ptr::null(),
        }
    };
}

#[allow(unused_imports)]
pub(crate) use method_def;

pub extern "C" fn generic_dealloc<T>(obj: *mut PyObject) {
    unsafe {
        std::ptr::drop_in_place::<T>(obj.cast());
        let obj_type = &*Py_TYPE(obj);
        let free = obj_type.tp_free.unwrap();
        free(obj.cast());
    }
}

pub fn raise_exception(msg: &str) -> *mut PyObject {
    let msg = std::ffi::CString::new(msg).unwrap();
    unsafe {
        PyErr_SetString(PyExc_Exception, msg.as_ptr().cast());
    }
    std::ptr::null_mut()
}

pub fn type_from_spec(spec: &PyType_Spec, bases: Vec<PyObj>) -> PyObj {
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

pub fn seq(obj: &PyObj) -> PyObj {
    if obj.is_none() {
        return obj.clone();
    } else if obj.type_is(crate::list::list_type()) {
        crate::list::list_seq(obj)
    } else {
        obj.call_method0(&static_pystring!("seq"))
    }
}

pub fn first(obj: &PyObj) -> PyObj {
    if obj.is_none() {
        return obj.clone();
    } else if obj.type_is(crate::list::list_type()) {
        crate::list::list_first(obj)
    } else {
        obj.call_method0(&static_pystring!("first"))
    }
}

pub fn next(obj: &PyObj) -> PyObj {
    if obj.is_none() {
        return obj.clone();
    } else if obj.type_is(crate::list::list_type()) {
        crate::list::list_next(obj)
    } else {
        obj.call_method0(&static_pystring!("next"))
    }
}

pub fn sequential_eq(self_: &PyObj, other: &PyObj) -> bool {
    let mut x = seq(self_);
    let mut y = seq(other);
    loop {
        if x.is_none() {
            return y.is_none();
        } else if y.is_none() {
            return false;
        } else if x.is(&y) {
            return true;
        } else if first(&x) != first(&y) {
            return false;
        } else {
            x = next(&x);
            y = next(&y);
        }
    }
}

#[repr(C)]
pub struct SeqIterator {
    ob_base: PyObject,
    seq: PyObj,
}

pub fn seq_iterator_type() -> &'static PyObj {
    static_type!(TypeSpec {
        name: "clx_rust.SeqIterator",
        flags: Py_TPFLAGS_DEFAULT | Py_TPFLAGS_DISALLOW_INSTANTIATION,
        size: std::mem::size_of::<SeqIterator>(),
        dealloc: Some(generic_dealloc::<SeqIterator>),
        iter: Some(seq_iterator_iter),
        next: Some(seq_iterator_next),
        ..Default::default()
    })
}

pub fn seq_iterator(coll: PyObj) -> PyObj {
    let obj = PyObj::alloc(seq_iterator_type());
    unsafe {
        let iter = obj.as_ref::<SeqIterator>();
        std::ptr::write(&mut iter.seq, seq(&coll));
    }
    obj
}

unsafe extern "C" fn seq_iterator_iter(
    self_: *mut PyObject,
) -> *mut PyObject {
    PyObj::borrow(self_).into_ptr()
}

unsafe extern "C" fn seq_iterator_next(
    self_: *mut PyObject,
) -> *mut PyObject {
    let self_ = PyObj::borrow(self_);
    let iter = self_.as_ref::<SeqIterator>();
    let s = &iter.seq;
    if s.is_none() {
        PyErr_SetNone(PyExc_StopIteration);
        std::ptr::null_mut()
    } else {
        let item = first(s);
        (*iter).seq = next(s);
        item.into_ptr()
    }
}

// Define a type with the given specification and return a reference to
// a static instance of the type
macro_rules! static_type {
    ($spec:expr) => {
        {
            use crate::utils;
            utils::lazy_static!(crate::object::PyObj, {
                utils::_make_type(
                    utils::lazy_static!(crate::utils::_TypeBuffer, {
                        utils::_make_type_buffer($spec)
                    }))
            })
        }
    };
}

pub(crate) use static_type;

#[derive(Default)]
pub struct TypeSpec {
    pub name: &'static str,
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
    pub members: Vec<&'static str>,
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
        .map(|x| CString::new(*x).unwrap())
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
        name: CString::new(spec.name).unwrap(),
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
