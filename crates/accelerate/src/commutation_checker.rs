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
use smallvec::SmallVec;

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySet, PyTuple};

use qiskit_circuit::circuit_instruction::CircuitInstruction;
use qiskit_circuit::operations::{Operation, OperationType, Param, StandardGate};
use qiskit_circuit::Qubit;

use crate::unitary_compose::compose_unitary;

#[derive(Clone)]
pub enum CommutationLibraryEntry {
    Commutes(bool),
    QubitMapping(HashMap<SmallVec<[Option<Qubit>; 2]>, bool>),
}

impl<'py> FromPyObject<'py> for CommutationLibraryEntry {
    fn extract_bound(b: &Bound<'py, PyAny>) -> Result<Self, PyErr> {
        if let Some(b) = b.extract::<bool>().ok() {
            return Ok(CommutationLibraryEntry::Commutes(b));
        }
        let dict = b.downcast::<PyDict>()?;
        let mut ret = hashbrown::HashMap::with_capacity(dict.len());
        for (k, v) in dict {
            let raw_key: SmallVec<[Option<u32>; 2]> = k.extract()?;
            let v: bool = v.extract()?;
            let key = raw_key
                .into_iter()
                .map(|key| key.map(|x| Qubit(x)))
                .collect();
            ret.insert(key, v);
        }
        Ok(CommutationLibraryEntry::QubitMapping(ret))
    }
}

#[derive(Clone)]
#[pyclass]
pub struct CommutationLibrary {
    pub library: HashMap<[StandardGate; 2], CommutationLibraryEntry>,
}

impl CommutationLibrary {
    fn check_commutation_entries(
        &self,
        first_op: &CircuitInstruction,
        second_op: &CircuitInstruction,
    ) -> Option<bool> {
        None
    }
}

#[pymethods]
impl CommutationLibrary {
    #[new]
    fn new(library: HashMap<[StandardGate; 2], CommutationLibraryEntry>) -> Self {
        CommutationLibrary { library }
    }
}

type CommutationCacheEntry = HashMap<
    (
        SmallVec<[Option<Qubit>; 2]>,
        [SmallVec<[ParameterKey; 3]>; 2],
    ),
    bool,
>;

#[pyclass]
struct CommutationChecker {
    library: CommutationLibrary,
    cache_max_entries: usize,
    cache: HashMap<[String; 2], CommutationCacheEntry>,
    current_cache_entries: usize,
}

#[pymethods]
impl CommutationChecker {
    #[pyo3(signature = (standard_gate_commutations=None, cache_max_entries=1_000_000))]
    #[new]
    fn py_new(
        standard_gate_commutations: Option<CommutationLibrary>,
        cache_max_entries: usize,
    ) -> Self {
        CommutationChecker {
            library: standard_gate_commutations
                .unwrap_or_else(|| CommutationLibrary::new(HashMap::new())),
            cache: HashMap::with_capacity(cache_max_entries),
            cache_max_entries,
            current_cache_entries: 0,
        }
    }

    #[pyo3(signature=(op1, op2, max_num_qubits=3))]
    fn commute(
        &self,
        py: Python,
        op1: &CircuitInstruction,
        op2: &CircuitInstruction,
        max_num_qubits: u32,
    ) -> PyResult<bool> {
        if let Some(commutes) = commutation_precheck(py, op1, op2, max_num_qubits)? {
            return Ok(commutes);
        }
        let reversed = if op1.operation.num_qubits() != op2.operation.num_qubits() {
            op1.operation.num_qubits() < op2.operation.num_qubits()
        } else {
            op1.operation.name() < op2.operation.name()
        };
        let (first_op, second_op) = if reversed { (op2, op1) } else { (op1, op2) };
        if first_op.operation.name() == "annotated" || second_op.operation.name() == "annotated" {
            return Ok(commute_matmul(first_op, second_op));
        }

        if let Some(commutes) = self.library.check_commutation_entries(first_op, second_op) {
            return Ok(commutes);
        }
        let is_commuting = commute_matmul(first_op, second_op);
        // TODO: implement a LRU cache for this
        if self.current_cache_entries >= self.cache_max_entries {
            self.cache.clear();
        }

        let get_relative_placement =
            |first_qargs: Bound<PyTuple>,
             second_qargs: Bound<PyTuple>|
             -> SmallVec<[Option<Qubit>; 2]> { smallvec::smallvec![None] };

        self.cache
            .entry([
                first_op.operation.name().to_string(),
                second_op.operation.name().to_string(),
            ])
            .and_modify(|entries| {
                if first_op.params.is_empty() && second_op.params.is_empty() {
                    let key = (get_relative_placement(first_op, second_op), [None, None]);
                    entries.insert(key, is_commuting);
                    self.current_cache_entries += 1;
                } else {
                }
            })
            .or_insert_with(|| {
                let mut entries = HashMap::with_capacity(1);
                if first_op.params.is_empty() && second_op.params.is_empty() {
                    let key = (get_relative_placement(first_op, second_op), [None, None]);
                    entries.insert(key, is_commuting);
                    self.current_cache_entries += 1;
                } else {
                }
                entries
            });
        Ok(is_commuting)
    }
}

#[derive(Debug, Copy, Clone)]
struct ParameterKey(f64);

impl ParameterKey {
    fn key(&self) -> u64 {
        self.0.to_bits()
    }
}

impl std::hash::Hash for ParameterKey {
    fn hash<H>(&self, state: &mut H)
    where
        H: std::hash::Hasher,
    {
        self.key().hash(state)
    }
}

impl PartialEq for ParameterKey {
    fn eq(&self, other: &ParameterKey) -> bool {
        self.key() == other.key()
    }
}

impl Eq for ParameterKey {}

fn hashable_params(params: &[Param]) -> SmallVec<[ParameterKey; 3]> {
    params
        .iter()
        .map(|x| {
            if let Param::Float(x) = x {
                ParameterKey(*x)
            } else {
                panic!()
            }
        })
        .collect()
}

fn get_qarg_indices(

fn commute_matmul(first_op: &CircuitInstruction, second_op: &CircuitInstruction) -> bool {
    //    let qargs
    let num_qubits = first_op.operation.num_qubits();
    let first_mat = match first_op.operation.matrix(&first_op.params) {
        Some(mat) => mat,
        None => return false,
    };
    let second_mat = match second_op.operation.matrix(&second_op.params) {
        Some(mat) => mat,
        None => return false,
    };
    let [op12, op21] = if first_op.qubits == second_op.qubits {
        [second_mat.dot(&first_mat), first_mat.dot(&second_mat)]
    } else {
        let first_mat = if second_op.qubits.len() > num_qubits {
            let id_op = Array2::eye(second_op.qubits.len());
            id_op.tensor(operator_1)
        } else {
            first_mat
        };
        let op12 = compose_unitary(second_mat, first_mat, second_qarg);
        let op21 = compose_unitary(first_mat, second_mat, second_qarg);
        [op12, op21]
    };
    op12 == op21
}

fn is_commutation_supported(op: &CircuitInstruction) -> bool {
    match op.operation {
        OperationType::Standard(_) | OperationType::Gate(_) => {
            if let Some(attr) = &op.extra_attrs {
                if attr.condition.is_some() {
                    return false;
                }
            }
            true
        }
        _ => false,
    }
}

const SKIPPED_NAMES: [&str; 4] = ["measure", "reset", "delay", "initialize"];

fn is_commutation_skipped(op: &CircuitInstruction, max_qubits: u32) -> bool {
    if op.operation.num_qubits() > max_qubits
        || op.operation.directive()
        || SKIPPED_NAMES.contains(&op.operation.name())
        || op.is_parameterized()
    {
        return true;
    }
    false
}

fn commutation_precheck(
    py: Python,
    op1: &CircuitInstruction,
    op2: &CircuitInstruction,
    max_qubits: u32,
) -> PyResult<Option<bool>> {
    if !is_commutation_supported(op1) || !is_commutation_supported(op2) {
        return Ok(Some(false));
    }
    let qargs_vec: SmallVec<[PyObject; 2]> = op1.qubits.extract(py)?;
    let cargs_vec: SmallVec<[PyObject; 2]> = op1.clbits.extract(py)?;
    // bind(py).iter().map(|x| x.clone_ref(py)).collect();

    let qargs_set = PySet::new_bound(py, &qargs_vec)?;
    let cargs_set = PySet::new_bound(py, &cargs_vec)?;
    if qargs_set
        .call_method1(intern!(py, "isdisjoint"), (op2.qubits.clone_ref(py),))?
        .extract::<bool>()?
        && cargs_set
            .call_method1(intern!(py, "isdisjoint"), (op2.clbits.clone_ref(py),))?
            .extract::<bool>()?
    {
        return Ok(Some(true));
    }

    if is_commutation_skipped(op1, max_qubits) || is_commutation_skipped(op2, max_qubits) {
        return Ok(Some(false));
    }
    Ok(None)
}

#[pymodule]
pub fn commutation_utils(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CommutationLibrary>()?;
    m.add_class::<CommutationChecker>()?;
    Ok(())
}
