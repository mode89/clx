use crate::object::PyObj;
use crate::type_object as tpo;
use crate::utils;
use std::sync::Mutex;
use pyo3_ffi::*;

pub fn init_module(module: *mut PyObject) {
    utils::module_add_method!(module, atom, py_atom);
    utils::module_add_method!(module, is_atom, py_is_atom);
    utils::module_add_type!(module, Atom, class());
}

#[repr(C)]
pub struct Atom {
    ob_base: PyObject,
    value: Mutex<PyObj>,
}

pub fn class() -> &'static PyObj {
    tpo::static_type!(tpo::TypeSpec {
        name: "lepet_ext.Atom",
        flags: Py_TPFLAGS_DEFAULT,
        size: std::mem::size_of::<Atom>(),
        new: Some(utils::disallowed_new!(class)),
        dealloc: Some(tpo::generic_dealloc::<Atom>),
        methods: vec![
            tpo::method!("deref", py_deref),
            tpo::method!("reset", py_reset),
            tpo::method!("swap", py_swap),
        ],
        ..Default::default()
    })
}

extern "C" fn py_atom(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs == 1 {
            let obj = PyObj::alloc(class());
            unsafe {
                let atom = obj.as_ref::<Atom>();
                std::ptr::write(&mut atom.value,
                    Mutex::new(PyObj::borrow(*args)));
            }
            Ok(obj)
        } else {
            utils::raise_exception("atom() requires exactly one argument")
        }
    })
}

extern "C" fn py_is_atom(
    _self: *mut PyObject,
    args: *mut *mut PyObject,
    _nargs: isize,
) -> *mut PyObject {
    utils::handle_gil!({
        PyObj::from(unsafe {
            Py_TYPE(*args) == class().as_ptr()
        }).into_ptr()
    })
}

extern "C" fn py_deref(
    self_: *mut PyObject,
    _args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs == 0 {
            let self_ = PyObj::borrow(self_);
            let self_ = unsafe { self_.as_ref::<Atom>() };
            let value = self_.value.lock().unwrap();
            Ok(value.clone())
        } else {
            utils::raise_exception("Atom.deref() expects no arguments")
        }
    })
}

extern "C" fn py_reset(
    self_: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs == 1 {
            let self_ = PyObj::borrow(self_);
            let self_ = unsafe { self_.as_ref::<Atom>() };
            let mut value = self_.value.lock().unwrap();
            *value = PyObj::borrow(unsafe { *args });
            Ok(value.clone())
        } else {
            utils::raise_exception(
                "Atom.reset() requires exactly one argument")
        }
    })
}

extern "C" fn py_swap(
    self_: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs > 0 {
            let self_ = PyObj::borrow(self_);
            let self_ = unsafe { self_.as_ref::<Atom>() };
            let f = PyObj::borrow(unsafe { *args });
            let fargs = PyObj::tuple(nargs)?;
            for i in 1..nargs {
                let value = PyObj::borrow(unsafe { *args.offset(i) });
                fargs.set_tuple_item(i, value)?;
            }
            let mut value = self_.value.lock().unwrap();
            fargs.set_tuple_item(0, value.clone())?;
            *value = f.call(fargs)?;
            Ok(value.clone())
        } else {
            utils::raise_exception(
                "Atom.swap() requires at least one argument")
        }
    })
}
