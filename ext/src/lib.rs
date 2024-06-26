mod utils;
mod object;
mod type_object;
mod protocols;
mod symbol;
mod keyword;
mod list;
mod vector;
mod hash_map;
mod cow_set;
mod record;
mod cons;
mod lazy_seq;
mod indexed_seq;
mod atom;
mod var;
mod common;
mod seq_iterator;

use pyo3_ffi::*;

#[no_mangle]
pub extern "C" fn PyInit_lepet_ext() -> *mut PyObject {
    static mut MODULE: PyModuleDef = PyModuleDef {
        m_base: PyModuleDef_HEAD_INIT,
        m_name: "lepet_ext\0".as_ptr().cast(),
        m_doc: std::ptr::null(),
        m_size: -1,
        m_methods: std::ptr::null_mut(),
        m_slots: std::ptr::null_mut(),
        m_traverse: None,
        m_clear: None,
        m_free: None,
    };
    let module = unsafe { PyModule_Create(std::ptr::addr_of_mut!(MODULE)) };

    protocols::init_module(module);
    symbol::init_module(module);
    keyword::init_module(module);
    list::init_module(module);
    vector::init_module(module);
    hash_map::init_module(module);
    cow_set::init_module(module);
    record::init_module(module);
    cons::init_module(module);
    lazy_seq::init_module(module);
    indexed_seq::init_module(module);
    atom::init_module(module);
    var::init_module(module);
    common::init_module(module);

    module
}
