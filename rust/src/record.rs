use crate::object::PyObj;
use crate::utils;
use crate::type_object as tpo;
use crate::protocols::*;
use pyo3_ffi::*;
use std::collections::HashMap;
use std::sync::Mutex;

pub fn init_module(module: *mut PyObject) {
    utils::module_add_method!(module, define_record, py_define_record);
}

extern "C" fn py_define_record(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        utils::py_assert(nargs > 1,
            "define_record expects at least two arguments")?;

        match unsafe { PyObj::borrow(*args) }.as_cstr() {
            Err(_) => return utils::raise_exception(
                "record name must be a string"),
            Ok(name) => {
                utils::py_assert(!name.is_empty(),
                    "record name must not be empty")?;
                let field_num = (nargs - 1) as usize;

                let mut members: Vec<tpo::MemberDef> = Vec::new();
                let mut kw_members: Vec<PyObj> = Vec::new();
                for i in 0..field_num {
                    let fspec = PyObj::borrow(
                        unsafe { *args.add(i + 1) });
                    utils::py_assert(fspec.is_tuple(),
                        "define_record field spec must be a tuple")?;
                    utils::py_assert(fspec.len()? == 2,
                        "define_record field spec must have two elements")?;
                    let fkw = fspec.get_tuple_item(0)?;
                    utils::py_assert(
                        fkw.type_is(crate::keyword::keyword_type()),
                        "define_record expects keyword as first element \
                        of field spec")?;
                    let fstr = fspec.get_tuple_item(1)?;
                    utils::py_assert(fstr.is_string(),
                        "define_record expects string as second element \
                        of field spec")?;
                    members.push(tpo::MemberDef {
                        name: fstr.as_cstr()?.to_str().unwrap().to_string(),
                        type_code: Py_T_OBJECT_EX,
                        offset: Some(member_offset(i)),
                        flags: Py_READONLY,
                    });
                    kw_members.push(fkw);
                }

                let type_ = tpo::new_type(tpo::TypeSpec {
                    name: name.to_str().unwrap(),
                    flags: Py_TPFLAGS_DEFAULT,
                    size: member_offset(field_num),
                    bases: vec![irecord_type()],
                    init: Some(py_record_init),
                    dealloc: Some(py_record_dealloc),
                    compare: Some(py_record_compare),
                    members,
                    methods: vec![
                        tpo::method!("lookup", py_record_lookup),
                        tpo::method!("assoc", py_record_assoc),
                    ],
                    ..Default::default()
                });

                let mut infos = record_infos().lock().unwrap();
                infos.insert(
                    unsafe { type_.as_ptr() },
                    Box::new(RecordInfo {
                        members: kw_members,
                    })
                );

                Ok(type_)
            }
        }
    })
}

fn record_infos() -> &'static Mutex<RecordInfos> {
    utils::lazy_static!(Mutex<RecordInfos>, {
        Mutex::new(RecordInfos::new())
    })
}

// This struct holds only the header of a record, the members are stored
// in memory directly after the header.
#[repr(C)]
struct RecordBase<'a> {
    ob_base: PyObject,
    info: &'a RecordInfo,
}

struct RecordInfo {
    members: Vec<PyObj>,
}

// Need to wrap the RecordInfo in a Box, because HashMap may relocate its
// elements, rendering any existing references invalid.
type RecordInfos = HashMap<*mut PyTypeObject, Box<RecordInfo>>;

impl RecordBase<'_> {
    #[inline]
    unsafe fn member_ptr(&self, index: usize) -> *mut PyObj {
        let self_ = self as *const RecordBase;
        self_.byte_add(member_offset(index)) as *mut PyObj
    }

    #[inline]
    unsafe fn member(&self, index: usize) -> &PyObj {
        &*self.member_ptr(index)
    }

    #[inline]
    unsafe fn init_member(&self, index: usize, value: PyObj) {
        std::ptr::write(self.member_ptr(index), value);
    }

    #[inline]
    unsafe fn replace_member(&self, index: usize, value: PyObj) {
        let _ = std::ptr::replace(self.member_ptr(index), value);
    }

    #[inline]
    fn clone(&self) -> PyObj {
        let new = PyObj::own(unsafe {
            PyType_GenericAlloc(self.ob_base.ob_type, 0)
        });
        let new_base = unsafe { new.as_ref::<RecordBase>() };
        new_base.info = self.info;
        let member_num = self.info.members.len();
        for i in 0..member_num {
            unsafe { new_base.init_member(i, self.member(i).clone()); }
        }
        new
    }
}

#[inline]
fn member_offset(index: usize) -> usize {
    std::mem::size_of::<RecordBase>() + index * std::mem::size_of::<PyObj>()
}

extern "C" fn py_record_init(
    self_: *mut PyObject,
    args: *mut PyObject,
    _kwargs: *mut PyObject,
) -> i32 {
    utils::handle_gil!({
        match record_init(self_, args) {
            Ok(_) => 0,
            Err(_) => -1,
        }
    })
}

fn record_init(self_: *mut PyObject, args: *mut PyObject) -> Result<(), ()> {
    let infos = record_infos().lock().unwrap();
    let type_ = unsafe { Py_TYPE(self_) };
    if let Some(info) = infos.get(&type_) {
        let self_ = unsafe { &mut *(self_ as *mut RecordBase) };
        self_.info = info;
        let args = PyObj::borrow(args);
        let member_num = info.members.len();
        utils::py_assert(args.len()? == member_num as isize,
            "wrong number of arguments")?;
        for i in 0..member_num {
            unsafe {
                self_.init_member(i, args.get_tuple_item(i as isize)?);
            }
        }
        Ok(())
    } else {
        unsafe { PyErr_BadInternalCall(); Err(()) }
    }
}

extern "C" fn py_record_dealloc(obj: *mut PyObject) {
    utils::handle_gil!({
        let record = unsafe { &mut *(obj as *mut RecordBase) };
        let member_num = record.info.members.len();
        for i in 0..member_num {
            unsafe { std::ptr::drop_in_place(record.member_ptr(i)); }
        }
        unsafe {
            let obj_type = &*Py_TYPE(obj);
            let free = obj_type.tp_free.unwrap();
            free(obj.cast());
        }
    })
}

extern "C" fn py_record_compare(
    self_: *mut PyObject,
    other: *mut PyObject,
    op: i32,
) -> *mut PyObject {
    utils::wrap_body!({
        if unsafe { Py_TYPE(self_) == Py_TYPE(other) } {
            match op {
                pyo3_ffi::Py_EQ => Ok(PyObj::from(record_eq(self_, other))),
                pyo3_ffi::Py_NE => Ok(PyObj::from(!record_eq(self_, other))),
                _ => utils::raise_exception(
                    "record comparison not supported")
            }
        } else {
            Ok(PyObj::from(false))
        }
    })
}

fn record_eq(self_: *mut PyObject, other: *mut PyObject) -> bool {
    let self_ = unsafe { &*(self_ as *const RecordBase) };
    let other = unsafe { &*(other as *const RecordBase) };
    let member_num = self_.info.members.len();
    for i in 0..member_num {
        if unsafe { self_.member(i) != other.member(i) } {
            return false;
        }
    }
    true
}

extern "C" fn py_record_lookup(
    self_: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        utils::py_assert(nargs == 2, "record lookup expects two arguments")?;
        let self_ = unsafe { &*(self_ as *const RecordBase) };
        let key = PyObj::borrow(unsafe { *args });
        let members = &self_.info.members;
        if let Some(index) = members.iter().position(|m| { m.is(&key) }) {
            Ok(unsafe { self_.member(index).clone() })
        } else {
            Ok(PyObj::borrow(unsafe { *args.add(1) }))
        }
    })
}

extern "C" fn py_record_assoc(
    self_: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        utils::py_assert(nargs % 2 == 0,
            "record assoc expects an even number of arguments")?;
        let self_ = unsafe { &*(self_ as *const RecordBase) };
        let new = self_.clone();
        let new_base = unsafe { new.as_ref::<RecordBase>() };
        for i in (0..nargs).step_by(2) {
            let key = PyObj::borrow(unsafe { *args.offset(i) });
            let value = PyObj::borrow(unsafe { *args.offset(i + 1) });
            let members = &self_.info.members;
            if let Some(m) = members.iter().position(|m| { m.is(&key) }) {
                unsafe { new_base.replace_member(m, value); }
            } else {
                return utils::raise_exception("record has no such member");
            }
        }
        Ok(new)
    })
}
