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

use pyo3::import_exception_bound;
use pyo3::prelude::*;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::imports::QUANTUM_REGISTER;
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::Qubit;

import_exception_bound!(qiskit.transpiler.exceptions, TranspilerError);

#[pyfunction]
#[pyo3(signature=(dag, layout, post_layout=None, final_layout=None))]
pub fn apply_layout(
    py: Python,
    dag: DAGCircuit,
    layout: Vec<Qubit>,
    post_layout: Option<Vec<Qubit>>,
    final_layout: Option<Vec<Qubit>>,
) -> PyResult<(DAGCircuit, Option<Vec<Qubit>>, Option<Vec<Qubit>>)> {
    if layout.len() != 1 + layout.iter().max().unwrap().index() {
        return Err(TranspilerError::new_err(
            "The 'layout' must be full (with ancilla).",
        ));
    }
    let reg = QUANTUM_REGISTER.get_bound(py).call1((layout.len(), "q"))?;
    let mut out_dag = DAGCircuit::with_capacity(
        py,
        dag.num_qubits(),
        dag.num_clbits(),
        Some(dag.num_vars()),
        Some(dag.dag().node_count()),
        Some(dag.dag().edge_count()),
    )?;
    out_dag.set_name(dag.name().map(|x| x.clone_ref(py)));
    out_dag.set_metadata(dag.metadata().map(|x| x.clone_ref(py)));
    out_dag.set_calibrations(dag.calibrations().clone());
    out_dag.copy_vars_from(py, &dag)?;
    out_dag.copy_clbits_from(py, &dag)?;
    out_dag.add_qreg(py, &reg)?;
    out_dag.set_global_phase(dag.get_global_phase())?;

    let mut rebuild_dag = |mapping: &[Qubit]| -> PyResult<()> {
        // TODO: Use DAGCircuit::extend() to avoid extra edge ops when there is a pattern to do it
        // with an iterator that interns qubits internally
        for node in dag.topological_op_nodes()? {
            let inst = dag.dag()[node].unwrap_operation();
            let mapped_qubits: Vec<Qubit> = dag
                .get_qargs(inst.qubits)
                .iter()
                .map(|x| mapping[x.index()])
                .collect();
            let mapped_inst = PackedInstruction {
                op: inst.op.clone(),
                qubits: out_dag.qargs_interner.insert_owned(mapped_qubits),
                clbits: inst.clbits,
                params: inst.params.clone(),
                extra_attrs: inst.extra_attrs.clone(),
                #[cfg(feature = "cache_pygates")]
                py_op: inst.py_op.clone(),
            };
            out_dag.push_back(py, mapped_inst)?;
        }
        Ok(())
    };

    match post_layout {
        Some(post_layout) => {
            // Mapping a post layout after we've already set an initial layout
            //
            // First build a new layout object going from:
            // old virtual -> old physical -> new virtual -> new physical
            // to:
            // old virtual -> new physical
            let mut full_layout: Vec<Qubit> = (0..dag.num_qubits() as u32).map(Qubit).collect();
            let mut old_phys_to_virt: Vec<usize> = vec![usize::MAX; dag.num_qubits()];
            for (virt, phys) in layout.iter().enumerate() {
                old_phys_to_virt[phys.index()] = virt;
            }
            for (new_virt, new_phys) in post_layout.iter().enumerate() {
                let old_phys = layout[new_virt];
                full_layout[old_phys.index()] = *new_phys;
            }
            rebuild_dag(&post_layout)?;
            if let Some(final_layout) = final_layout {
                let mut new_final = vec![Qubit(u32::MAX); dag.num_qubits()];
                for (old_virt, old_phys) in final_layout.iter().enumerate() {
                    new_final[full_layout[old_virt].index()] = full_layout[old_phys.index()];
                }
                Ok((out_dag, Some(full_layout), Some(new_final)))
            } else {
                Ok((out_dag, Some(full_layout), None))
            }
        }
        None => {
            rebuild_dag(&layout)?;
            Ok((out_dag, None, None))
        }
    }
}

pub fn apply_layout_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(apply_layout))?;
    Ok(())
}
