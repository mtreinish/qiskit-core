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
#![allow(clippy::too_many_arguments)]

use hashbrown::HashSet;
use ndarray::prelude::*;
use numpy::IntoPyArray;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use retworkx_core::petgraph::prelude::*;

use crate::getenv_use_multiple_threads;
use crate::nlayout::NLayout;
use crate::sabre_swap::neighbor_table::NeighborTable;
use crate::sabre_swap::sabre_dag::SabreDAG;
use crate::sabre_swap::swap_map::SwapMap;
use crate::sabre_swap::{build_swap_map_inner, Heuristic};

#[pyfunction]
pub fn sabre_layout_and_routing(
    py: Python,
    num_clbits: usize,
    dag_nodes: Vec<(usize, Vec<usize>, HashSet<usize>)>,
    neighbor_table: &NeighborTable,
    distance_matrix: PyReadonlyArray2<f64>,
    heuristic: &Heuristic,
    seed: u64,
    max_iterations: usize,
    num_swap_trials: usize,
    num_layout_trials: usize,
) -> ([NLayout; 2], SwapMap, PyObject) {
    let run_in_parallel = getenv_use_multiple_threads();
    let outer_rng = Pcg64Mcg::seed_from_u64(seed);
    let seed_vec: Vec<u64> = outer_rng
        .sample_iter(&rand::distributions::Standard)
        .take(num_layout_trials)
        .collect();
    let dist = distance_matrix.as_array();
    let result = if run_in_parallel {
        seed_vec
            .into_par_iter()
            .enumerate()
            .map(|(index, seed_trial)| {
                (
                    index,
                    layout_trial(
                        num_clbits,
                        dag_nodes.clone(),
                        neighbor_table,
                        &dist,
                        heuristic,
                        seed_trial,
                        max_iterations,
                        num_swap_trials,
                    ),
                )
            })
            .min_by_key(|(index, result)| {
                (
                    result.1.map.values().map(|x| x.len()).sum::<usize>(),
                    *index,
                )
            })
            .unwrap()
            .1
    } else {
        seed_vec
            .into_iter()
            .map(|seed_trial| {
                layout_trial(
                    num_clbits,
                    dag_nodes.clone(),
                    neighbor_table,
                    &dist,
                    heuristic,
                    seed_trial,
                    max_iterations,
                    num_swap_trials,
                )
            })
            .min_by_key(|result| result.1.map.values().map(|x| x.len()).sum::<usize>())
            .unwrap()
    };
    (result.0, result.1, result.2.into_pyarray(py).into())
}

fn layout_trial(
    num_clbits: usize,
    mut dag_nodes: Vec<(usize, Vec<usize>, HashSet<usize>)>,
    neighbor_table: &NeighborTable,
    distance_matrix: &ArrayView2<f64>,
    heuristic: &Heuristic,
    seed: u64,
    max_iterations: usize,
    num_swap_trials: usize,
) -> ([NLayout; 2], SwapMap, Vec<usize>) {
    // Pick a random initial layout and fully populate ancillas in that layout too
    let num_physical_qubits = distance_matrix.shape()[0];
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let mut physical_qubits: Vec<usize> = (0..num_physical_qubits).collect();
    physical_qubits.shuffle(&mut rng);
    let mut phys_to_logic = vec![0; num_physical_qubits];
    physical_qubits
        .iter()
        .enumerate()
        .for_each(|(logic, phys)| phys_to_logic[*phys] = logic);
    let mut initial_layout = NLayout {
        logic_to_phys: physical_qubits,
        phys_to_logic,
    };
    let mut rev_dag_nodes: Vec<(usize, Vec<usize>, HashSet<usize>)> =
        dag_nodes.iter().rev().cloned().collect();
    for _iter in 0..max_iterations {
        // forward and reverse
        for _direction in 0..2 {
            let dag = apply_layout(&dag_nodes, &initial_layout, num_physical_qubits, num_clbits);
            let mut pass_final_layout = NLayout {
                logic_to_phys: (0..num_physical_qubits).collect(),
                phys_to_logic: (0..num_physical_qubits).collect(),
            };
            build_swap_map_inner(
                num_physical_qubits,
                &dag,
                neighbor_table,
                distance_matrix,
                heuristic,
                seed,
                &mut pass_final_layout,
                num_swap_trials,
                Some(false),
            );
            let final_layout = compose_layout(&initial_layout, &pass_final_layout);
            initial_layout = final_layout;
            std::mem::swap(&mut dag_nodes, &mut rev_dag_nodes);
        }
    }
    let layout_dag = apply_layout(&dag_nodes, &initial_layout, num_physical_qubits, num_clbits);
    let mut final_layout = initial_layout.clone();
    let (swap_map, gate_order) = build_swap_map_inner(
        num_physical_qubits,
        &layout_dag,
        neighbor_table,
        distance_matrix,
        heuristic,
        seed,
        &mut final_layout,
        num_swap_trials,
        Some(false),
    );
    ([initial_layout, final_layout], swap_map, gate_order)
}

fn apply_layout(
    dag_nodes: &[(usize, Vec<usize>, HashSet<usize>)],
    layout: &NLayout,
    num_qubits: usize,
    num_clbits: usize,
) -> SabreDAG {
    let layout_dag_nodes: Vec<(usize, Vec<usize>, HashSet<usize>)> = dag_nodes
        .iter()
        .map(|(node_index, qargs, cargs)| {
            let new_qargs: Vec<usize> = qargs.iter().map(|n| layout.logic_to_phys[*n]).collect();
            (*node_index, new_qargs, cargs.clone())
        })
        .collect();
    build_sabre_dag(layout_dag_nodes, num_qubits, num_clbits)
}

fn build_sabre_dag(
    layout_dag_nodes: Vec<(usize, Vec<usize>, HashSet<usize>)>,
    num_qubits: usize,
    num_clbits: usize,
) -> SabreDAG {
    let mut dag: DiGraph<(usize, Vec<usize>), ()> =
        Graph::with_capacity(layout_dag_nodes.len(), 2 * layout_dag_nodes.len());
    let mut first_layer = Vec::<NodeIndex>::new();
    let mut qubit_pos: Vec<Option<NodeIndex>> = vec![None; num_qubits];
    let mut clbit_pos: Vec<Option<NodeIndex>> = vec![None; num_clbits];
    for node in &layout_dag_nodes {
        let qargs = &node.1;
        let cargs = &node.2;
        let gate_index = dag.add_node((node.0, qargs.clone()));
        let mut is_front = true;
        for x in qargs {
            if let Some(predecessor) = qubit_pos[*x] {
                is_front = false;
                dag.add_edge(predecessor, gate_index, ());
            }
            qubit_pos[*x] = Some(gate_index);
        }
        for x in cargs {
            if let Some(predecessor) = clbit_pos[*x] {
                is_front = false;
                dag.add_edge(predecessor, gate_index, ());
            }
            clbit_pos[*x] = Some(gate_index);
        }
        if is_front {
            first_layer.push(gate_index);
        }
    }
    SabreDAG { dag, first_layer }
}

fn compose_layout(initial_layout: &NLayout, final_layout: &NLayout) -> NLayout {
    let logic_to_phys = initial_layout
        .logic_to_phys
        .iter()
        .map(|n| final_layout.logic_to_phys[*n])
        .collect();
    let phys_to_logic = final_layout
        .phys_to_logic
        .iter()
        .map(|n| initial_layout.phys_to_logic[*n])
        .collect();

    NLayout {
        logic_to_phys,
        phys_to_logic,
    }
}

#[pymodule]
pub fn sabre_layout(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(sabre_layout_and_routing))?;
    Ok(())
}
