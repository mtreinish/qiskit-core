use qiskit_circuit::dag_circuit::{DAGCircuit, Wire};
use qiskit_circuit::{PhysicalQubit, Qubit, VirtualQubit};
use crate::target::Target;

use hashbrown::HashSet;
use rustworkx_core::petgraph::Direction::Outgoing;
use rustworkx_core::petgraph::graph::NodeIndex;
use rustworkx_core::petgraph::visit::EdgeRef;
use rayon::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::prelude::*;

#[pyfunction]
pub fn degree_matching_layout(dag: &DAGCircuit, target: &Target) -> Vec<PhysicalQubit> {
    let coupling_graph = target.coupling_graph().unwrap();
    let mut output = vec![PhysicalQubit::MAX; dag.num_qubits()];
    let mut circuit_qubit_degrees: Vec<(usize, VirtualQubit)> = dag
        .qubit_io_map()
        .par_iter()
        .enumerate()
        .map(|(qubit_idx, [input_node, _output_node])|  {
            let target_qubit = Qubit::new(qubit_idx);
            let mut next_node = Some(*input_node);
            let mut intersection: HashSet<Qubit> = HashSet::with_capacity(dag.num_qubits());
            while let Some(node_id) = next_node {
                let successor_edges = dag.dag().edges_directed(node_id, Outgoing);
                next_node = None;
                for edge in successor_edges {
                    if let Wire::Qubit(qubit) = edge.weight() {
                        intersection.insert(*qubit);
                        if *qubit == target_qubit {
                            next_node = Some(edge.target())
                        }
                    }
                }
            }
            (intersection.len(), VirtualQubit::new(qubit_idx as u32))
        })
        .collect();
    circuit_qubit_degrees.par_sort_unstable();
    let mut target_degree: Vec<(usize, PhysicalQubit)> = (0..coupling_graph.node_count()).into_par_iter().map(|qubit_idx| {
        (coupling_graph.edges_directed(NodeIndex::new(qubit_idx), Outgoing).count(), PhysicalQubit::new(qubit_idx as u32))
    }).collect();
    target_degree.par_sort_unstable();
    circuit_qubit_degrees.iter().zip(target_degree).for_each(|((_circuit_deg, circuit_qubit), (_target_degree, target_qubit))| {
        output[circuit_qubit.index()] = target_qubit;
    });
    output
}

pub fn degree_matching_layout_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(degree_matching_layout))?;
    Ok(())
}
