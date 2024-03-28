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

use hashbrown::HashMap;
use ndarray::ArrayView2;
use num_complex::Complex64;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;
use smallvec::SmallVec;

use crate::convert_2q_block_matrix::blocks_to_matrix_inner;
use crate::getenv_use_multiple_threads;
use crate::two_qubit_decompose::{TwoQubitBasisDecomposer, TwoQubitGateSequence};

#[pyclass]
pub struct TargetErrorMap {
    error_map: HashMap<String, HashMap<[u32; 2], Option<f64>>>,
}

impl TargetErrorMap {
    pub fn get_error_rate(&self, gate: &str, qubits: [u32; 2]) -> Option<f64> {
        match self.error_map.get(&gate.to_string()) {
            Some(qubit_map) => *qubit_map.get(&qubits).unwrap(),
            None => None,
        }
    }
}

#[pymethods]
impl TargetErrorMap {
    #[new]
    fn new(initial_capacity: usize) -> Self {
        TargetErrorMap {
            error_map: HashMap::with_capacity(initial_capacity),
        }
    }

    fn add_error(&mut self, gate_name: String, qubits: [u32; 2], error_rate: Option<f64>) {
        if !self.error_map.contains_key(&gate_name) {
            let mut new_error_map: HashMap<[u32; 2], Option<f64>> = HashMap::new();
            new_error_map.insert(qubits, error_rate);
            self.error_map.insert(gate_name, new_error_map);
        } else {
            let res = self.error_map.get_mut(&gate_name).unwrap();
            res.insert(qubits, error_rate);
        }
    }
}

#[derive(Clone)]
#[pyclass]
pub struct DecomposerMap {
    decomposer_map: HashMap<[u32; 2], Vec<TwoQubitBasisDecomposer>>,
}

#[pymethods]
impl DecomposerMap {
    #[new]
    fn new(initial_capacity: usize) -> Self {
        DecomposerMap {
            decomposer_map: HashMap::with_capacity(initial_capacity),
        }
    }

    fn add_decomposer(&mut self, qubits: [u32; 2], decomposer: &TwoQubitBasisDecomposer) {
        if !self.decomposer_map.contains_key(&qubits) {
            let decomposer_list = vec![decomposer.clone()];
            self.decomposer_map.insert(qubits, decomposer_list);
        } else {
            let res = self.decomposer_map.get_mut(&qubits).unwrap();
            res.push(decomposer.clone());
        }
    }
}

type InnerBlockType<'a> = Vec<(
    Vec<(ArrayView2<'a, Complex64>, SmallVec<[u8; 2]>)>,
    [u32; 2],
)>;
type BlockInputType<'a> = Vec<(
    Vec<(PyReadonlyArray2<'a, Complex64>, SmallVec<[u8; 2]>)>,
    [u32; 2],
)>;

// TODO: When XX decomposer is ported to rust add an enum that can be used for either
// decomposer type
#[pyfunction]
pub fn optimize_blocks(
    py: Python,
    blocks: BlockInputType,
    decomposers: &DecomposerMap,
    target: &TargetErrorMap,
) -> Vec<Option<(TwoQubitGateSequence, PyObject)>> {
    let run_in_parallel = getenv_use_multiple_threads();
    let blocks: InnerBlockType = blocks
        .iter()
        .map(|(block, qubits)| {
            (
                block
                    .iter()
                    .map(|(unitary, qargs)| (unitary.as_array(), qargs.clone()))
                    .collect::<Vec<(ArrayView2<Complex64>, SmallVec<[u8; 2]>)>>(),
                *qubits,
            )
        })
        .collect();
    if run_in_parallel {
        py.allow_threads(move || {
            blocks
                .into_par_iter()
                .map(|(block, qubits)| {
                    let unitary = blocks_to_matrix_inner(block);
                    let reverse_qubits = [qubits[1], qubits[0]];
                    let forward_decomposer = decomposers.decomposer_map.get(&qubits);
                    let reverse_decomposers = decomposers.decomposer_map.get(&reverse_qubits);
                    let decomposer_lists = match forward_decomposer {
                        Some(decomp) => decomp,
                        None => match reverse_decomposers {
                            Some(decomp) => decomp,
                            None => panic!("invalid qubits: {:?} or {:?}", qubits, reverse_qubits),
                        },
                    };
                    let sequences = decomposer_lists
                        .iter()
                        .filter_map(|decomposer| {
                            let synthesis = decomposer.synthesize(unitary.view(), None, true, None);
                            match synthesis {
                                Ok(s) => Some((s, decomposer.gate_obj.clone())),
                                Err(_) => None,
                            }
                        })
                        .collect();
                    best_synthesis(sequences, qubits, target)
                })
                .collect()
        })
    } else {
        blocks
            .into_iter()
            .map(|(block, qubits)| {
                let unitary = blocks_to_matrix_inner(block);
                let decomposer_lists = decomposers
                    .decomposer_map
                    .get(&qubits)
                    .unwrap_or(&decomposers.decomposer_map[&[qubits[1], qubits[0]]]);
                let sequences = decomposer_lists
                    .iter()
                    .filter_map(|decomposer| {
                        let synthesis = decomposer.synthesize(unitary.view(), None, true, None);
                        match synthesis {
                            Ok(s) => Some((s, decomposer.gate_obj.clone_ref(py))),
                            Err(_) => None,
                        }
                    })
                    .collect();
                best_synthesis(sequences, qubits, target)
            })
            .collect()
    }
}

fn error_for_sequence(
    sequence: &TwoQubitGateSequence,
    qubits: [u32; 2],
    target: &TargetErrorMap,
) -> f64 {
    let mut fidelity = 1.0;
    for inst in &sequence.gates {
        let qubits = if inst.2.len() == 1 {
            [qubits[inst.2[0] as usize], qubits[inst.2[0] as usize]]
        } else {
            [qubits[inst.2[0] as usize], qubits[inst.2[1] as usize]]
        };
        let error_rate = target.get_error_rate(&inst.0, qubits);
        if let Some(error) = error_rate {
            fidelity *= 1. - error
        }
    }
    1. - fidelity
}

fn best_synthesis(
    sequences: Vec<(TwoQubitGateSequence, PyObject)>,
    qubits: [u32; 2],
    target: &TargetErrorMap,
) -> Option<(TwoQubitGateSequence, PyObject)> {
    if sequences.is_empty() {
        return None;
    }
    sequences.into_iter().min_by(|sequence_a, sequence_b| {
        error_for_sequence(&sequence_a.0, qubits, target)
            .partial_cmp(&error_for_sequence(&sequence_b.0, qubits, target))
            .unwrap()
    })
}

#[pymodule]
pub fn optimize_2q_blocks(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<TargetErrorMap>()?;
    m.add_class::<DecomposerMap>()?;
    m.add_wrapped(wrap_pyfunction!(optimize_blocks))?;
    Ok(())
}
