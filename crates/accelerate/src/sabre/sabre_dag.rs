// This code is part of Qiskit.
//
// (C) Copyright IBM 2022
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use hashbrown::HashMap;
use hashbrown::HashSet;
use pyo3::prelude::*;
use rustworkx_core::petgraph::prelude::*;

use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::operations::Operation;

use crate::nlayout::VirtualQubit;

/// Named access to the node elements in the [SabreDAG].
#[derive(Clone, Debug)]
pub struct DAGNode {
    pub py_node_id: usize,
    pub qubits: Vec<VirtualQubit>,
    pub directive: bool,
}

/// A DAG representation of the logical circuit to be routed.  This represents the same dataflow
/// dependencies as the Python-space [DAGCircuit], but without any information about _what_ the
/// operations being performed are. Note that all the qubit references here are to "virtual"
/// qubits, that is, the qubits are those specified by the user.  This DAG does not need to be
/// full-width on the hardware.
///
/// Control-flow operations are represented by the presence of the Python [DAGCircuit]'s node id
/// (the [DAGNode.py_node_id] field) as a key in [node_blocks], where the value is an array of the
/// inner dataflow graphs.
#[derive(Clone, Debug)]
pub struct SabreDAG {
    pub num_qubits: usize,
    pub num_clbits: usize,
    pub dag: DiGraph<DAGNode, ()>,
    pub first_layer: Vec<NodeIndex>,
    pub node_blocks: HashMap<usize, Vec<SabreDAG>>,
}

impl SabreDAG {
    pub fn reverse_dag(&self) -> Self {
        let mut out_dag = self.clone();
        out_dag.dag.reverse();
        out_dag.first_layer = out_dag
            .dag
            .node_indices()
            .filter(|idx| {
                out_dag
                    .dag
                    .neighbors_directed(*idx, Incoming)
                    .next()
                    .is_none()
            })
            .collect();
        out_dag
    }
}

pub fn build_sabre_dag(dag: &DAGCircuit) -> PyResult<SabreDAG> {
    let mut out_dag = DiGraph::with_capacity(dag.dag().node_count(), dag.dag().edge_count());
    let mut qubit_pos: Vec<Option<NodeIndex>> = vec![None; dag.num_qubits()];
    let mut clbit_pos: Vec<Option<NodeIndex>> = vec![None; dag.num_clbits()];
    let mut node_blocks = HashMap::new();
    let mut first_layer = Vec::<NodeIndex>::new();

    for node_index in dag.topological_op_nodes().unwrap() {
        let inst = dag.dag()[node_index].unwrap_operation();
        let mut cargs: HashSet<_> = dag.get_cargs(inst.clbits).iter().copied().collect();
        if inst.op.control_flow() {
            let block_dags = Python::with_gil(|py| -> PyResult<Vec<SabreDAG>> {
                if inst.op.name() == "switch_case" {
                    let (extra_cargs, _) = dag.additional_wires(py, inst.op.view())?;
                    cargs.extend(extra_cargs);
                }
                inst.op
                    .blocks()
                    .iter()
                    .map(|block| {
                        let dag = DAGCircuit::from_circuit_data(py, block.clone(), false)?;
                        build_sabre_dag(&dag)
                    })
                    .collect()
            })?;
            node_blocks.insert(node_index.index(), block_dags);
        }
        let new_node = DAGNode {
            py_node_id: node_index.index(),
            qubits: dag
                .get_qargs(inst.qubits)
                .iter()
                .map(|x| VirtualQubit::new(x.0))
                .collect(),
            directive: inst.op.directive(),
        };
        let gate_index = out_dag.add_node(new_node);
        let mut is_front = true;
        for qarg in dag.get_qargs(inst.qubits) {
            let pos = qubit_pos.get_mut(qarg.index()).unwrap();
            if let Some(pred) = *pos {
                is_front = false;
                out_dag.add_edge(pred, gate_index, ());
            }
            *pos = Some(gate_index);
        }
        for carg in cargs {
            let pos = clbit_pos.get_mut(carg.index()).unwrap();
            if let Some(pred) = *pos {
                is_front = false;
                out_dag.add_edge(pred, gate_index, ());
            }
            *pos = Some(gate_index);
        }
        if is_front {
            first_layer.push(gate_index);
        }
    }
    Ok(SabreDAG {
        num_qubits: dag.num_qubits(),
        num_clbits: dag.num_clbits(),
        dag: out_dag,
        first_layer,
        node_blocks,
    })
}

#[cfg(test)]
mod test {
    use super::SabreDAG;
    use crate::nlayout::VirtualQubit;
    use hashbrown::{HashMap, HashSet};

    #[test]
    fn no_panic_on_bad_qubits() {
        let bad_qubits = vec![VirtualQubit::new(0), VirtualQubit::new(2)];
        assert!(SabreDAG::new(
            2,
            0,
            vec![(0, bad_qubits, HashSet::new(), false)],
            HashMap::new()
        )
        .is_err())
    }

    #[test]
    fn no_panic_on_bad_clbits() {
        let good_qubits = vec![VirtualQubit::new(0), VirtualQubit::new(1)];
        assert!(SabreDAG::new(
            2,
            1,
            vec![(0, good_qubits, [0, 1].into(), false)],
            HashMap::new()
        )
        .is_err())
    }
}
