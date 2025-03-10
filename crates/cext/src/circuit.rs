// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::exit_codes::CInputError;

use qiskit_circuit::bit::{ShareableClbit, ShareableQubit};
use qiskit_circuit::circuit_data::CircuitData;

/// Check the pointer is not null and is aligned.
fn check_ptr<T>(ptr: *const T) -> Result<(), CInputError> {
    if ptr.is_null() {
        return Err(CInputError::NullPointerError);
    };
    if !ptr.is_aligned() {
        return Err(CInputError::AlignmentError);
    };
    Ok(())
}

/// Casts a const pointer to a reference. Panics is the pointer is null or not aligned.
///
/// # Safety
///
/// This function requires ``ptr`` to be point to an initialized object of type ``T``.
/// While the resulting reference exists, the memory pointed to must not be mutated.
unsafe fn const_ptr_as_ref<'a, T>(ptr: *const T) -> &'a T {
    check_ptr(ptr).unwrap();
    let as_ref = unsafe { ptr.as_ref() };
    as_ref.unwrap() // we know the pointer is not null, hence we can safely unwrap
}

/// @ingroup QkCircuit
/// Construct a new circuit with the given number of qubits and clbits.
///
/// @param num_qubits The number of qubits the circuit contains.
/// @param num_clbits The number of clbits the circuit contains.
///
/// @return A pointer to the created Circuit
///
/// # Example
///
///     QkCircuit *empty = qk_circuit_new(100, 100);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_circuit_new(num_qubits: u32, num_clbits: u32) -> *mut CircuitData {
    let qubits = if num_qubits > 0 {
        Some(
            (0..num_qubits)
                .map(|_| ShareableQubit::new_anonymous())
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };
    let clbits = if num_clbits > 0 {
        Some(
            (0..num_clbits)
                .map(|_| ShareableClbit::new_anonymous())
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };

    let circuit = CircuitData::new(qubits, clbits, None, 0, (0.).into()).unwrap();
    Box::into_raw(Box::new(circuit))
}

/// @ingroup QkCircuit
/// Get the number of qubits the circuit contains.
///
/// @param circuit A pointer to the circuit.
///
/// @return The number of qubits the circuit is defined on.
///
/// # Example
///
///     QkCircuit *qc = qk_circuit_new(100);
///     uint32_t num_qubits = qk_circuit_num_qubits(qc);  // num_qubits==100
///
/// # Safety
///
/// Behavior is undefined ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_num_qubits(circuit: *const CircuitData) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };

    circuit.num_qubits() as u32
}

/// @ingroup QkCircuit
/// Get the number of clbits the circuit contains.
///
/// @param circuit A pointer to the circuit.
///
/// @return The number of qubits the circuit is defined on.
///
/// # Example
///
///     QkCircuit *qc = qk_circuit_new(100);
///     uint32_t num_qubits = qk_circuit_num_clbits(qc);  // num_qubits==100
///
/// # Safety
///
/// Behavior is undefined ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_num_clbits(circuit: *const CircuitData) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };

    circuit.num_clbits() as u32
}

/// @ingroup QkCircuit
/// Free the circuit.
///
/// @param circuit A pointer to the circuit to free.
///
/// # Example
///
///     QkCircuit *qc = qk_circuit_new(100, 100);
///     qk_circuit_free(qc);
///
/// # Safety
///
/// Behavior is undefined if ``circt`` is not either null or a valid pointer to a
/// [CircuitData].
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_free(circuit: *mut CircuitData) {
    if !circuit.is_null() {
        if !circuit.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }

        // SAFETY: We have verified the pointer is non-null and aligned, so it should be
        // readable by Box.
        unsafe {
            let _ = Box::from_raw(circuit);
        }
    }
}
