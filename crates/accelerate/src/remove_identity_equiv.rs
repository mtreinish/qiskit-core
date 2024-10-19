// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use approx::abs_diff_eq;
use ndarray::{aview2, Array2};
use num_complex::Complex64;
use pyo3::prelude::*;
use rustworkx_core::petgraph::stable_graph::NodeIndex;

use crate::nlayout::PhysicalQubit;
use crate::target_transpiler::Target;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::gate_matrix::ONE_QUBIT_IDENTITY;
use qiskit_circuit::operations::Operation;
use qiskit_circuit::operations::OperationRef;
use qiskit_circuit::operations::Param;
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::util::{C_ONE, C_ZERO};

static TWO_QUBIT_IDENTITY: [[Complex64; 4]; 4] = [
    [C_ONE, C_ZERO, C_ZERO, C_ZERO],
    [C_ZERO, C_ONE, C_ZERO, C_ZERO],
    [C_ZERO, C_ZERO, C_ONE, C_ZERO],
    [C_ZERO, C_ZERO, C_ZERO, C_ONE],
];

#[pyfunction]
#[pyo3(signature=(dag, tol=Some(f64::EPSILON), target=None))]
fn remove_identity_equiv(dag: &mut DAGCircuit, tol: Option<f64>, target: Option<&Target>) {
    let mut remove_list: Vec<NodeIndex> = Vec::new();

    let get_tolerance = |inst: &PackedInstruction| -> f64 {
        match tol {
            Some(tol) => tol,
            None => match target {
                Some(target) => {
                    let qargs: Vec<PhysicalQubit> = dag
                        .get_qargs(inst.qubits)
                        .iter()
                        .map(|x| PhysicalQubit::new(x.0))
                        .collect();
                    let error_rate = target.get_error(inst.op.name(), qargs.as_slice());
                    match error_rate {
                        Some(err) => 1. - err,
                        None => f64::EPSILON,
                    }
                }
                None => f64::EPSILON,
            },
        }
    };

    for op_node in dag.op_nodes(false) {
        let inst = dag.dag()[op_node].unwrap_operation();
        match inst.op.view() {
            OperationRef::Standard(gate) => {
                let tol = get_tolerance(inst);
                if gate.num_params() > 0
                    && inst.params_view().iter().all(|x| match x {
                        Param::Float(param) => param.abs() < tol,
                        _ => false,
                    })
                {
                    remove_list.push(op_node);
                }
            }
            OperationRef::Gate(gate) => {
                if let Some(matrix) = gate.matrix(inst.params_view()) {
                    if gate.num_qubits() == 1 {
                        let tol = get_tolerance(inst);
                        if abs_diff_eq!(matrix, aview2(&ONE_QUBIT_IDENTITY), epsilon = tol) {
                            remove_list.push(op_node);
                        }
                    } else if gate.num_qubits() == 2 {
                        let tol = get_tolerance(inst);
                        if abs_diff_eq!(matrix, aview2(&TWO_QUBIT_IDENTITY), epsilon = tol) {
                            remove_list.push(op_node);
                        }
                    } else {
                        let tol = get_tolerance(inst);
                        let identity = Array2::eye(gate.num_qubits().pow(2) as usize);
                        if abs_diff_eq!(matrix, identity, epsilon = tol) {
                            remove_list.push(op_node);
                        }
                    }
                }
            }
            _ => continue,
        }
    }
    for node in remove_list {
        dag.remove_op_node(node);
    }
}

pub fn remove_identity_equiv_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(remove_identity_equiv))?;
    Ok(())
}
