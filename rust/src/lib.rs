mod utils;
mod object;
mod type_object;
mod protocols;
mod symbol;
mod keyword;
mod list;
mod vector;
mod hash_map;
mod record;

use pyo3_ffi::*;

#[no_mangle]
pub extern "C" fn PyInit_clx_rust() -> *mut PyObject {
    static mut MODULE: PyModuleDef = PyModuleDef {
        m_base: PyModuleDef_HEAD_INIT,
        m_name: "clx_rust\0".as_ptr().cast(),
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
    record::init_module(module);

    module
}
