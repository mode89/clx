use crate::object::{PyObj, PyObjSendable};
use crate::type_object as tpo;
use crate::utils;
use pyo3_ffi::*;
use std::cell::RefCell;
use std::collections::LinkedList;
use std::sync::Mutex;
use thread_local::ThreadLocal;

pub fn init_module(module: *mut PyObject) {
    utils::module_add_type!(module, Var, class());
}

#[repr(C)]
pub struct Var {
    ob_base: PyObject,
    stack: Mutex<ThreadLocal<RefCell<LinkedList<PyObjSendable>>>>,
    root: PyObj,
}

pub fn class() -> &'static PyObj {
    tpo::static_type!(tpo::TypeSpec {
        name: "lepet_ext.Var",
        flags: Py_TPFLAGS_DEFAULT,
        size: std::mem::size_of::<Var>(),
        init: Some(py_init),
        dealloc: Some(tpo::generic_dealloc::<Var>),
        methods: vec![
            tpo::method!("deref", py_deref),
            tpo::method!("push", py_push),
            tpo::method!("pop", py_pop),
            tpo::method!("set", py_set),
        ],
        ..Default::default()
    })
}

extern "C" fn py_init(
    self_: *mut PyObject,
    args: *mut PyObject,
    kwargs: *mut PyObject,
) -> i32 {
    utils::handle_gil!({
        if kwargs.is_null() {
            match init(self_, args) {
                Ok(_) => 0,
                Err(_) => -1,
            }
        } else {
            utils::set_exception(
                "Var constructor does not accept keyword arguments");
            -1
        }
    })
}

fn init(self_: *mut PyObject, args: *mut PyObject) -> Result<(), ()> {
    let self_ = unsafe { &mut *(self_ as *mut Var) };
    let args = PyObj::borrow(args);
    let nargs = args.len()?;
    if nargs == 1 {
        self_.root = args.get_tuple_item(0)?;
        self_.stack = Mutex::new(ThreadLocal::new());
        Ok(())
    } else {
        utils::set_exception("Var constructor takes exactly one argument");
        Err(())
    }
}

extern "C" fn py_deref(
    self_: *mut PyObject,
    _args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs == 0 {
            let self_ = unsafe { &mut *(self_ as *mut Var) };
            let stack = self_.stack.lock().unwrap();
            let stack = stack
                .get_or(|| RefCell::new(LinkedList::new()))
                .borrow();
            if stack.is_empty() {
                Ok(self_.root.clone())
            } else {
                Ok(stack.back().unwrap().as_pyobj())
            }
        } else {
            utils::raise_exception("Var.deref() takes no arguments")
        }
    })
}

extern "C" fn py_push(
    self_: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs == 1 {
            let self_ = unsafe { &mut *(self_ as *mut Var) };
            let stack = self_.stack.lock().unwrap();
            let mut stack = stack
                .get_or(|| RefCell::new(LinkedList::new()))
                .borrow_mut();
            let obj = PyObj::borrow(unsafe { *args });
            stack.push_back(PyObjSendable::from(obj));
            Ok(PyObj::none())
        } else {
            utils::raise_exception("Var.push() takes exactly one argument")
        }
    })
}

extern "C" fn py_pop(
    self_: *mut PyObject,
    _args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs == 0 {
            let self_ = unsafe { &mut *(self_ as *mut Var) };
            let stack = self_.stack.lock().unwrap();
            let mut stack = stack
                .get_or(|| RefCell::new(LinkedList::new()))
                .borrow_mut();
            if !stack.is_empty() {
                Ok(stack.pop_back().unwrap().as_pyobj())
            } else {
                utils::raise_exception("Var is not bound")
            }
        } else {
            utils::raise_exception("Var.pop() takes no arguments")
        }
    })
}

extern "C" fn py_set(
    self_: *mut PyObject,
    args: *mut *mut PyObject,
    nargs: isize,
) -> *mut PyObject {
    utils::wrap_body!({
        if nargs == 1 {
            let self_ = unsafe { &mut *(self_ as *mut Var) };
            let stack = self_.stack.lock().unwrap();
            let mut stack = stack
                .get_or(|| RefCell::new(LinkedList::new()))
                .borrow_mut();
            if !stack.is_empty() {
                let prev = stack.pop_back().unwrap().as_pyobj();
                let new = PyObj::borrow(unsafe { *args });
                stack.push_back(PyObjSendable::from(new));
                Ok(prev)
            } else {
                utils::raise_exception("Var is not bound")
            }
        } else {
            utils::raise_exception("Var.alter() takes one argument")
        }
    })
}
