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

use hashbrown::HashMap;

use crate::pointers::const_ptr_as_ref;

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::{PhysicalQubit, VirtualQubit};
use qiskit_transpiler::passes::vf2_layout_pass;
use qiskit_transpiler::target::Target;

/// The result from ``qk_transpiler_pass_standalone_vf2_layout()``.
pub struct VF2LayoutResult(Option<HashMap<VirtualQubit, PhysicalQubit>>);

/// @ingroup QkVF2LayoutResult
/// Check whether a result was found.
///
/// @param layout a pointer to the layout
///
/// @returns ``true`` if the ``qk_transpiler_pass_standalone_vf2_layout()`` run found a layout
///
/// # Safety
///
/// Behavior is undefined if ``layout`` is not a valid, non-null pointer to a
/// ``QkVF2LayoutResult``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_vf2_layout_has_match(layout: *const VF2LayoutResult) -> bool {
    let layout = unsafe { const_ptr_as_ref(layout) };
    layout.0.is_some()
}

/// @ingroup QkVF2LayoutResult
/// Get the number of virtual qubits in the layout.
///
/// @param layout a pointer to the layout
///
/// @returns The number of virtual qubits in the layout
///
/// # Safety
///
/// Behavior is undefined if ``layout`` is not a valid, non-null pointer to a
/// ``QkVF2LayoutResult``. The result must have a layout found.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_vf2_layout_num_qubits(layout: *const VF2LayoutResult) -> u32 {
    let layout = unsafe { const_ptr_as_ref(layout) };
    let Some(ref layout) = layout.0 else {
        panic!("There was no layout found");
    };
    layout.len() as u32
}

/// @ingroup QkVF2LayoutResult
/// Get the physical qubit for a given virtual qubit
///
/// @param layout a pointer to the layout
/// @param qubit the virtual qubit to get the physical qubit of
///
/// @returns The physical qubit mapped to by the specified virtual qubit
///
/// # Safety
///
/// Behavior is undefined if ``layout`` is not a valid, non-null pointer to a
/// ``QkVF2LayoutResult``. Also qubit must be a valid qubit for the circuit and
/// there must be a result found.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_vf2_layout_map_virtual_qubit(
    layout: *const VF2LayoutResult,
    qubit: u32,
) -> u32 {
    let layout = unsafe { const_ptr_as_ref(layout) };
    let Some(ref layout) = layout.0 else {
        panic!("There was no layout found");
    };
    match layout.get(&VirtualQubit(qubit)) {
        Some(phsyical) => phsyical.0,
        None => panic!("The specified qubit is not in the layout: {}", qubit),
    }
}

/// @ingroup QkVF2LayoutResult
/// Free a ``QkVF2LayoutResult`` object
///
/// @param layout a pointer to the layout to free
///
/// # Example
///
///     QkCircuit *qc = qk_circuit_new(1, 0);
///
/// # Safety
///
/// Behavior is undefined if ``layout`` is not a valid, non-null pointer to a ``QkVF2Layout``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_vf2_layout_free(layout: *mut VF2LayoutResult) {
    if !layout.is_null() {
        if !layout.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }
        // SAFETY: We have verified the pointer is non-null and aligned, so
        // it should be readable by Box.
        unsafe {
            let _ = Box::from_raw(layout);
        }
    }
}

/// @ingroup QkTranspilerPasses
/// Run the VF2Layout pass on a circuit.
///
/// @param circuit A pointer to the circuit to run VF2Layout on
/// @param target The target for the VF2Layout pass
/// @param strict_direction
/// @param time_limit
/// @param max_trials
///
/// @return An array
///
/// # Example
///
///     QkCircuit *qc = qk_circuit_new(1, 0);
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` or ``target`` is not a valid, non-null pointer to a ``QkCircuit`` and ``QkTarget``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_transpiler_pass_standalone_vf2_layout(
    circuit: *const CircuitData,
    target: *const Target,
    strict_direction: bool,
    call_limit: i64,
    time_limit: f64,
    max_trials: i64,
) -> *mut VF2LayoutResult {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };
    let target = unsafe { const_ptr_as_ref(target) };
    let dag = match DAGCircuit::from_circuit_data(
        circuit,
        false,
        None,
        None,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        None,
        None,
    ) {
        Ok(dag) => dag,
        Err(e) => panic!("{}", e),
    };
    let call_limit = if call_limit < 0 {
        None
    } else {
        Some(call_limit as usize)
    };
    let time_limit = if time_limit.is_nan() {
        None
    } else {
        Some(time_limit)
    };
    let max_trials = if max_trials < 0 {
        None
    } else {
        Some(max_trials as usize)
    };
    let layout = match vf2_layout_pass(
        &dag,
        target,
        strict_direction,
        call_limit,
        time_limit,
        max_trials,
        None,
    ) {
        Ok(layout) => layout,
        Err(e) => panic!("{}", e),
    };
    Box::into_raw(Box::new(VF2LayoutResult(layout)))
}
