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

use ahash::RandomState;
use anyhow::Result;
use hashbrown::{HashSet, HashMap};
use indexmap::IndexSet;
use ndarray::prelude::*;
use rayon::prelude::*;
use rustworkx_core::shortest_path::distance_matrix;
use std::hash::Hash;
use rustworkx_core::petgraph::visit::{IntoNeighborsDirected, NodeCount, NodeIndexable, IntoNodeIdentifiers, GraphProp, IntoEdgeReferences, EdgeCount, EdgeRef};

use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::operations::Operation;
use qiskit_circuit::{PhysicalQubit, VirtualQubit};


pub fn tram_initial_mapping<G>(
    dag: &DAGCircuit,
    cmap: G,
    qubit_t2_times: Option<Vec<f64>>,
    avg_dur: Option<HashMap<&[PhysicalQubit], f64>>,
    phi: f64,
    eta: f64,
) -> Result<Vec<Option<PhysicalQubit>>>
where
    G: Sync + IntoNeighborsDirected + NodeCount + NodeIndexable + IntoNodeIdentifiers + GraphProp + IntoEdgeReferences + EdgeCount,
    G::NodeId: Hash + Eq + Sync,
{
    let num_target_qubits = cmap.node_count();
    let mut initial_layout: Vec<Option<PhysicalQubit>> = vec![None; num_target_qubits as usize];
    let mut heat_map: Array2<f64> =
        Array2::zeros((num_target_qubits as usize, num_target_qubits as usize));
    let total_gate_count = dag
        .op_nodes(false)
        .filter(|(_, inst)| inst.op.num_qubits() == 2)
        .count();
    let mut seen_edges: HashSet<[VirtualQubit; 2]> = HashSet::with_capacity(total_gate_count);
    let mut k = 0;
    let mut virtual_edges: Vec<[VirtualQubit; 2]> = Vec::with_capacity(total_gate_count);
    for node_idx in dag.topological_op_nodes(true) {
        let NodeType::Operation(ref inst) = dag[node_idx] else {
            unreachable!();
        };
        if inst.op.num_qubits() != 2 {
            continue;
        }
        k += 1;
        let qargs = dag.get_qargs(inst.qubits);
        if qargs.len() != 2 {
            continue;
        }
        let virtual_edge = [VirtualQubit::new(qargs[0].0), VirtualQubit::new(qargs[1].0)];
        if !seen_edges.contains(&virtual_edge) {
            seen_edges.insert(virtual_edge);
            virtual_edges.push(virtual_edge);
        }
        let w_k: f64 = (phi * (1. - (k as f64 / total_gate_count as f64))).exp();
        heat_map[[qargs[0].index(), qargs[1].index()]] += w_k;
    }
    let mut dist_matrix = distance_matrix(&cmap, 300, false, f64::NAN);
    let mut physical_edge_list: Vec<[PhysicalQubit; 2]> = Vec::with_capacity(cmap.edge_count());
    if let Some(ref qubit_properties) = qubit_t2_times {
        for edge in cmap.edge_references() {
            let i = cmap.to_index(edge.source());
            let j = cmap.to_index(edge.target());
            let t2_i = qubit_properties[i];
            let t2_j = qubit_properties[j];
            let qargs = [PhysicalQubit::new(i as u32), PhysicalQubit::new(j as u32)];
            let dur: f64 = avg_dur.as_ref().map(|x| x[&qargs.as_slice()]).unwrap_or(0.);
            physical_edge_list.push(qargs);
            let decoherence_i = 1. - (-dur / t2_i).exp();
            let decoherence_j = 1. - (-dur / t2_j).exp();
            let decoherence_term = eta * (decoherence_i + decoherence_j);
            dist_matrix[[i, j]] += decoherence_term;
        }
    } else {
        for edge in cmap.edge_references() {
            let i = cmap.to_index(edge.source());
            let j = cmap.to_index(edge.target());
            let qargs = [PhysicalQubit::new(i as u32), PhysicalQubit::new(j as u32)];
            physical_edge_list.push(qargs);
        }
    }
    let normalize = |err: f64| -> f64 { if err.is_nan() { 0.0 } else { err } };
    physical_edge_list.par_sort_by(|a, b| {
        let a = [a[0].index(), a[1].index()];
        let score_a = normalize(dist_matrix[a]);
        let b = [b[0].index(), b[1].index()];
        let score_b = normalize(dist_matrix[b]);
        score_a.partial_cmp(&score_b).expect("NaNs treated as zero")
    });
    virtual_edges.par_sort_by(|a, b| {
        let a = [a[0].index(), a[1].index()];
        let score_a = normalize(heat_map[a]);
        let b = [b[0].index(), b[1].index()];
        let score_b = normalize(heat_map[b]);
        score_a.partial_cmp(&score_b).expect("NaNs treated as zero")
    });
    let mut unused_physical_qubits: IndexSet<usize, RandomState> =
        (0..num_target_qubits as usize).collect();
    for (virt_edge, phys_edge) in virtual_edges.iter().zip(physical_edge_list) {
        if unused_physical_qubits.contains(&phys_edge[0].index())
            && initial_layout[virt_edge[0].index()].is_none()
        {
            initial_layout[virt_edge[0].index()] = Some(phys_edge[0]);
            unused_physical_qubits.swap_remove(&phys_edge[0].index());
        }
        if unused_physical_qubits.contains(&phys_edge[1].index())
            && initial_layout[virt_edge[1].index()].is_none()
        {
            initial_layout[virt_edge[1].index()] = Some(phys_edge[1]);
            unused_physical_qubits.swap_remove(&phys_edge[1].index());
        }
    }
    if !unused_physical_qubits.is_empty() {
        for phys_qubit in initial_layout.iter_mut().take(num_target_qubits as usize) {
            if phys_qubit.is_none() {
                *phys_qubit = Some(PhysicalQubit::new(
                    unused_physical_qubits.pop().unwrap() as u32
                ));
            }
        }
    }
    Ok(initial_layout)
}

#[cfg(all(test, not(miri)))]
mod tests {
    use super::*;
    use crate::target::InstructionProperties;
    use crate::target::QubitProperties;
    use crate::target::Target;
    use qiskit_circuit::circuit_data::CircuitData;
    use qiskit_circuit::instruction::Parameters;
    use qiskit_circuit::operations::{Operation, Param, StandardGate, StandardInstruction};
    use qiskit_circuit::parameter::parameter_expression::ParameterExpression;
    use qiskit_circuit::parameter::symbol_expr::Symbol;
    use qiskit_circuit::{Clbit, PhysicalQubit, Qubit};
    use rand::distr::Uniform;
    use rand::prelude::*;
    use rand_pcg::Pcg64Mcg;
    use smallvec::smallvec;
    use std::sync::Arc;

    fn build_universal_star_target(num_qubits: u32) -> Target {
        let mut target = Target::default();
        let u_params = Some(Parameters::Params(smallvec![
            Param::ParameterExpression(Arc::new(ParameterExpression::from_symbol(Symbol::new(
                "a", None, None,
            )))),
            Param::ParameterExpression(Arc::new(ParameterExpression::from_symbol(Symbol::new(
                "b", None, None,
            )))),
            Param::ParameterExpression(Arc::new(ParameterExpression::from_symbol(Symbol::new(
                "c", None, None,
            )))),
        ]));
        let mut rng = Pcg64Mcg::seed_from_u64(42);
        let distr = Uniform::try_from(1e-4..5e-3).unwrap();
        let time_distr = Uniform::try_from(5e-6..5e-5).unwrap();

        let props = (0..num_qubits)
            .map(|i| {
                (
                    [PhysicalQubit(i)].into(),
                    Some(InstructionProperties::new(
                        Some(distr.sample(&mut rng)),
                        Some(i as f64 * 2.3e-5),
                    )),
                )
            })
            .collect();
        target
            .add_instruction(StandardGate::U.into(), u_params, None, Some(props))
            .unwrap();
        let props = (0..num_qubits)
            .map(|i| {
                (
                    [PhysicalQubit(i)].into(),
                    Some(InstructionProperties::new(
                        Some(distr.sample(&mut rng)),
                        Some(i as f64 * 2.3e-3),
                    )),
                )
            })
            .collect();
        target
            .add_instruction(StandardInstruction::Measure.into(), None, None, Some(props))
            .unwrap();
        let props = (1..num_qubits)
            .map(|i| {
                (
                    [PhysicalQubit(0), PhysicalQubit(i)].into(),
                    Some(InstructionProperties::new(
                        Some(distr.sample(&mut rng)),
                        Some(i as f64 * 6.2e-4),
                    )),
                )
            })
            .collect();
        target
            .add_instruction(StandardGate::ECR.into(), None, None, Some(props))
            .unwrap();
        let qubit_properties = (0..num_qubits)
            .map(|_| {
                let time: f64 = time_distr.sample(&mut rng);
                QubitProperties {
                    t1: Some(time),
                    t2: Some(time_distr.sample(&mut rng) / 2.),
                    frequency: None,
                }
            })
            .collect::<Vec<_>>();
        target.qubit_properties = Some(qubit_properties);
        target
    }

    #[test]
    fn test_ghz_10() {
        let target = build_universal_star_target(10);
        let circuit = CircuitData::from_packed_operations(
            10,
            0,
            [Ok((
                StandardGate::H.into(),
                smallvec![],
                vec![Qubit(0)],
                vec![],
            ))]
            .into_iter()
            .chain((0..8).map(|i| {
                Ok((
                    StandardGate::CX.into(),
                    smallvec![],
                    vec![Qubit(i), Qubit(i + 1)],
                    vec![],
                ))
            })),
            Param::Float(0.),
        )
        .unwrap();
        let dag = DAGCircuit::from_circuit_data(&circuit, false, None, None, None, None).unwrap();
        let res = tram_initial_mapping(&dag, &target, 0.5, 0.125).unwrap();
        let expected = [0, 9, 4, 1, 2, 3, 5, 6, 7, 8];
        assert_eq!(
            expected
                .into_iter()
                .map(|x| PhysicalQubit::new(x))
                .collect::<Vec<_>>(),
            res
        );
    }
}
