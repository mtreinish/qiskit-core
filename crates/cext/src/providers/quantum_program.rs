use std::ffi::{c_char, c_void, CStr};

use num_complex::{Complex64, Complex32};

use qiskit_providers::{QuantumProgram, math_nodes, Store, OwnedPath, Port, DataTree, tensor::Tensor};

use crate::ExitCode;
use crate::pointers::{const_ptr_as_ref, mut_ptr_as_ref};

#[unsafe(no_mangle)]
pub extern "C" fn qk_quantum_program_new() -> *mut QuantumProgram {
    let program = QuantumProgram::new();
    Box::into_raw(Box::new(program))
}

#[unsafe(no_mangle)]
pub extern "C" fn qk_quantum_program_free(program: *mut QuantumProgram) {
    if !program.is_null() {
        let _program = unsafe { Box::from_raw(program) };
    }
}

#[repr(u8)]
pub enum QuantumProgramNodeType {
    Store,
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
    Power,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    BitwiseNot,
    Parity,
    Mean,
    Variance,
    Std,
    QuantumProgram,
}

#[unsafe(no_mangle)]
pub extern "C" fn qk_quantum_program_port_new(label: *const c_char, path: *mut OwnedPath) -> *mut Port {
    let label = unsafe {CStr::from_ptr(label).to_string_lossy() };
    let path = unsafe { *Box::from_raw(path) };
    Box::into_raw(Box::new(Port::new(label, path)))
}

#[unsafe(no_mangle)]
pub extern "C" fn qk_owned_path_new(size: usize) -> *mut OwnedPath {
    Box::into_raw(Box::new(OwnedPath::with_capacity(size)))
}

#[unsafe(no_mangle)]
pub extern "C" fn qk_owned_path_free(path: *mut OwnedPath) {
    if !path.is_null() {
        let _path: Box<OwnedPath> = unsafe {Box::from_raw(path)};
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn qk_owned_path_append_index(path: *mut OwnedPath, index: usize) {
    let path = unsafe { mut_ptr_as_ref(path) };
    path.push(index.into());
}

#[unsafe(no_mangle)]
pub extern "C" fn qk_owned_path_append_key(path: *mut OwnedPath, key: *const c_char) -> ExitCode {
    let path = unsafe { mut_ptr_as_ref(path) };
    let key = match unsafe { CStr::from_ptr(key) }.to_str() {
        Ok(key) => key,
        Err(_) => return ExitCode::CInputError,
    };
    path.push(key.into());
    ExitCode::Success
}

#[unsafe(no_mangle)]
pub extern "C" fn qk_quantum_program_add_node(program: *mut QuantumProgram, node_type: QuantumProgramNodeType, node: *mut c_void, label: *const c_char) -> ExitCode {
    let program = unsafe { mut_ptr_as_ref(program) };
    let label = unsafe { CStr::from_ptr(label).to_string_lossy() };
    if let Err(_error) = match node_type {
        QuantumProgramNodeType::Store => {
            let node: Box<Store> = unsafe { Box::from_raw(node as *mut Store) };
            program.add_node(label, *node)
        },
        QuantumProgramNodeType::Add => {
            program.add_node(label, math_nodes::binary::Add)
        },
        QuantumProgramNodeType::Subtract => {
            program.add_node(label, math_nodes::binary::Subtract)
        },
        QuantumProgramNodeType::Multiply => {
            program.add_node(label, math_nodes::binary::Multiply)
        },
        QuantumProgramNodeType::Divide => {
            program.add_node(label, math_nodes::binary::Divide)
        },
        QuantumProgramNodeType::Remainder => {
            program.add_node(label, math_nodes::binary::Remainder)
        },
        QuantumProgramNodeType::Power => {
            program.add_node(label, math_nodes::binary::Power)
        },
        QuantumProgramNodeType::BitwiseAnd => {
            program.add_node(label, math_nodes::bitwise::BitwiseAnd)
        },
        QuantumProgramNodeType::BitwiseOr => {
            program.add_node(label, math_nodes::bitwise::BitwiseOr)
        },
        QuantumProgramNodeType::BitwiseXor => {
            program.add_node(label, math_nodes::bitwise::BitwiseXor)
        },
        QuantumProgramNodeType::BitwiseNot => {
            program.add_node(label, math_nodes::bitwise::BitwiseNot)
        },
        QuantumProgramNodeType::Parity => {
            let node: Box<math_nodes::bitwise::Parity> = unsafe { Box::from_raw(node as *mut math_nodes::bitwise::Parity) };
            program.add_node(label, *node)
        },
        QuantumProgramNodeType::Mean => {
            let node: Box<math_nodes::reduction::Mean> = unsafe { Box::from_raw(node as *mut math_nodes::reduction::Mean) };
            program.add_node(label, *node)
        },
        QuantumProgramNodeType::Variance => {
            let node: Box<math_nodes::reduction::Variance> = unsafe { Box::from_raw(node as *mut math_nodes::reduction::Variance) };
            program.add_node(label, *node)
        }
        QuantumProgramNodeType::Std => {
            let node: Box<math_nodes::reduction::Std> = unsafe { Box::from_raw(node as *mut math_nodes::reduction::Std) };
            program.add_node(label, *node)
        }
        QuantumProgramNodeType::QuantumProgram => {
            let node: Box<QuantumProgram> = unsafe { Box::from_raw(node as *mut QuantumProgram) };
            program.add_node(label, *node)
        }
    } {
        ExitCode::DuplicateProgramLabel
    } else {
        ExitCode::Success
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn qk_quantum_program_add_edge(program: *mut QuantumProgram, from: *mut Port, to: *mut Port) -> ExitCode {
    let program = unsafe { mut_ptr_as_ref(program) };
    let from = unsafe {*Box::from_raw(from) };
    let to = unsafe{*Box::from_raw(to) };
    if let Err(_err) = program.add_edge(from, to) {
        ExitCode::ProgramPortAlreadyConnected
    } else {
        ExitCode::Success
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn qk_quantum_program_set_input(program: *mut QuantumProgram, key: *const c_char, port: *mut Port) -> ExitCode {
    let program = unsafe { mut_ptr_as_ref(program) };
    let port = unsafe {*Box::from_raw(port) };
    let key = match unsafe { CStr::from_ptr(key) }.to_str() {
        Ok(key) => key,
        Err(_) => return ExitCode::CInputError,
    };
    if let Err(_err) = program.set_input(key, port) {
        ExitCode::ProgramPortAlreadyConnected
    } else {
        ExitCode::Success
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn qk_quantum_program_set_output(program: *mut QuantumProgram, key: *const c_char, port: *mut Port) -> ExitCode {
    let program = unsafe { mut_ptr_as_ref(program) };
    let port = unsafe {*Box::from_raw(port) };
    let key = match unsafe { CStr::from_ptr(key) }.to_str() {
        Ok(key) => key,
        Err(_) => return ExitCode::CInputError,
    };
    if let Err(_err) = program.set_output(key, port) {
        ExitCode::ProgramPortAlreadyConnected
    } else {
        ExitCode::Success
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn qk_program_node_make_store_double(val: f64) -> *mut Store {
    Box::into_raw(Box::new(Store::new(DataTree::new_leaf(Tensor::from([val])))))
}

#[unsafe(no_mangle)]
pub extern "C" fn qk_program_node_make_store_uint64_t(val: u64) -> *mut Store {
    Box::into_raw(Box::new(Store::new(DataTree::new_leaf(Tensor::from([val])))))
}

#[unsafe(no_mangle)]
pub extern "C" fn qk_program_node_make_store_float(val: f32) -> *mut Store {
    Box::into_raw(Box::new(Store::new(DataTree::new_leaf(Tensor::from([val])))))
}

#[unsafe(no_mangle)]
pub extern "C" fn qk_program_node_make_store_uint32_t(val: u32) -> *mut Store {
    Box::into_raw(Box::new(Store::new(DataTree::new_leaf(Tensor::from([val])))))
}

#[unsafe(no_mangle)]
pub extern "C" fn qk_program_node_make_store_uint16_t(val: u16) -> *mut Store {
    Box::into_raw(Box::new(Store::new(DataTree::new_leaf(Tensor::from([val])))))
}

#[unsafe(no_mangle)]
pub extern "C" fn qk_program_node_make_store_complex64(val: *const Complex64) -> *mut Store {
    let val = unsafe { const_ptr_as_ref(val) };
    Box::into_raw(Box::new(Store::new(DataTree::new_leaf(Tensor::from([*val])))))
}
