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

use pyo3::prelude::*;
use std::fmt;

use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::imports::PAULI_EVOLUTION_GATE;
use qiskit_circuit::interner::{Interned, Interner};
use qiskit_circuit::operations::{
    Operation, OperationRef, Param, PyGate, StandardGate, multiply_param,
};
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::{Clbit, Qubit, VarsMode};

use qiskit_quantum_info::sparse_observable::SparseObservable;

use fixedbitset::FixedBitSet;
use num_complex::Complex64;
use qiskit_quantum_info::sparse_observable::BitTerm;
use smallvec::smallvec;
use std::f64::consts::PI;

use crate::TranspilerError;

// List of gate names supported by the pass: the pass is skipped if the circuit
// contains gate names outside of this list.
static SUPPORTED_GATE_NAMES: &[&str; 19] = &[
    "id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "cx", "cz", "cy", "swap", "iswap", "ecr",
    "dcx", "t", "tdg", "rz",
];

// List of rotation gate names: the pass is skipped if the circuit contains
// no gate names in this list.
static ROTATION_GATE_NAMES: &[&str; 3] = &["t", "tdg", "rz"];

/// Expresses a given circuit as a sequence of Pauli rotations followed by a final Clifford operator.
/// Returns the list of rotations in the sparse format: (sign, paulis, indices).
fn extract_rotations(
    gate: StandardGate,
    qbits: &Interned<[Qubit]>,
    interner: &Interner<[Qubit]>,
    clifford: &mut Clifford,
) -> Option<(bool, Vec<BitTerm>, Vec<Qubit>)> {
    match gate {
        StandardGate::I => {}
        StandardGate::X => clifford.append_x(interner.get(*qbits)[0].index()),
        StandardGate::Y => clifford.append_y(interner.get(*qbits)[0].index()),
        StandardGate::Z => clifford.append_z(interner.get(*qbits)[0].index()),
        StandardGate::H => clifford.append_h(interner.get(*qbits)[0].index()),
        StandardGate::S => clifford.append_s(interner.get(*qbits)[0].index()),
        StandardGate::Sdg => clifford.append_sdg(interner.get(*qbits)[0].index()),
        StandardGate::SX => clifford.append_sx(interner.get(*qbits)[0].index()),
        StandardGate::SXdg => clifford.append_sxdg(interner.get(*qbits)[0].index()),
        StandardGate::CX => clifford.append_cx(
            interner.get(*qbits)[0].index(),
            interner.get(*qbits)[1].index(),
        ),
        StandardGate::CZ => clifford.append_cz(
            interner.get(*qbits)[0].index(),
            interner.get(*qbits)[1].index(),
        ),
        StandardGate::CY => clifford.append_cy(
            interner.get(*qbits)[0].index(),
            interner.get(*qbits)[1].index(),
        ),
        StandardGate::Swap => clifford.append_swap(
            interner.get(*qbits)[0].index(),
            interner.get(*qbits)[1].index(),
        ),
        StandardGate::ISwap => clifford.append_iswap(
            interner.get(*qbits)[0].index(),
            interner.get(*qbits)[1].index(),
        ),
        StandardGate::ECR => clifford.append_ecr(
            interner.get(*qbits)[0].index(),
            interner.get(*qbits)[1].index(),
        ),
        StandardGate::DCX => clifford.append_dcx(
            interner.get(*qbits)[0].index(),
            interner.get(*qbits)[1].index(),
        ),
        StandardGate::RZ => {
            return Some(clifford.get_inverse_z(interner.get(*qbits)[0].index()));
        }
        _ => panic!("Unsupported gate {}", gate.name()),
    };
    None
}

#[pyfunction]
#[pyo3(signature = (dag, fix_clifford=true))]
pub fn run_litinski_transformation(
    py: Python,
    dag: &DAGCircuit,
    fix_clifford: bool,
) -> PyResult<Option<DAGCircuit>> {
    let op_counts = dag.get_op_counts();

    // Skip the pass if there are no rotation gates.
    if op_counts
        .keys()
        .all(|k| !ROTATION_GATE_NAMES.contains(&k.as_str()))
    {
        return Ok(None);
    }

    // Skip the pass if there are unsupported gates.
    if !op_counts
        .keys()
        .all(|k| SUPPORTED_GATE_NAMES.contains(&k.as_str()))
    {
        let unsupported: Vec<_> = op_counts
            .keys()
            .filter(|k| !SUPPORTED_GATE_NAMES.contains(&k.as_str()))
            .collect();

        return Err(TranspilerError::new_err(format!(
            "Unable to run Litinski tranformation as the circuit contains gates not supported by the pass: {:?}",
            unsupported
        )));
    }
    let rotation_count = op_counts
        .iter()
        .filter_map(|(k, v)| {
            if ROTATION_GATE_NAMES.contains(&k.as_str()) {
                Some(v)
            } else {
                None
            }
        })
        .sum();
    let clifford_count = dag.size(false)? - rotation_count;

    let num_qubits = dag.num_qubits();

    // Turn the Qiskit circuit into a vector of (gate name, qubit indices).
    // Additionally, keep track of the rotation angles, an update to the global phase (produced when
    // converting T/Tdg gates to RZ-rotations), and Clifford gates in the circuit.
    let mut angles: Vec<Param> = Vec::with_capacity(rotation_count);
    let mut global_phase_update = 0.;
    let mut clifford_ops: Vec<&PackedInstruction> = Vec::with_capacity(clifford_count);
    let mut clifford = Clifford::identity(num_qubits);
    let rotations: Vec<_> = dag
        .topological_op_nodes()?
        .filter_map(|node_index| {
            let NodeType::Operation(inst) = &dag[node_index] else {
                unreachable!(
                    "Gate instructions should be either Clifford or T/Tdg/RZ at this point."
                );
            };
            let (gate, angle, phase_update) = match inst.op.view() {
                OperationRef::StandardGate(StandardGate::T) => {
                    (StandardGate::RZ, Some(Param::Float(PI / 8.)), PI / 8.)
                }
                OperationRef::StandardGate(StandardGate::Tdg) => {
                    (StandardGate::RZ, Some(Param::Float(-PI / 8.0)), -PI / 8.)
                }
                OperationRef::StandardGate(StandardGate::RZ) => {
                    let param = &inst.params_view()[0];
                    (StandardGate::RZ, Some(multiply_param(param, 0.5)), 0.)
                }
                _ => (inst.op.try_standard_gate().unwrap(), None, 0.),
            };

            global_phase_update += phase_update;

            if let Some(angle) = angle {
                // This is a rotation, save the angle.
                angles.push(angle);
            } else {
                // This is a Clifford operation, save it.
                clifford_ops.push(inst);
            }
            extract_rotations(gate, &inst.qubits, dag.qargs_interner(), &mut clifford)
        })
        .collect();

    // Apply the Litinski transformation.
    // This returns a list of rotations with +1/-1 signs. Since we aim to preserve the
    // global phase of the circuit, we ignore the final Clifford operator, and instead
    // append the Clifford gates from the original circuit.

    let py_evo_cls = PAULI_EVOLUTION_GATE.get_bound(py);
    let no_clbits: Vec<Clbit> = Vec::new();

    let new_dag = dag.copy_empty_like(VarsMode::Alike)?;
    let mut new_dag = new_dag.into_builder();
    new_dag.add_global_phase(&Param::Float(global_phase_update))?;

    // Add Pauli rotation gates to the Qiskit circuit.
    for ((sign, mut paulis, qubits), angle) in rotations.into_iter().zip(angles) {
        let coeffs = vec![Complex64::new(1., 0.)];
        let paulis_len = paulis.len() as u32;
        let boundaries = vec![0, paulis_len as usize];
        paulis.reverse();
        // SAFETY: We made this from a clifford we know it's valid
        let obs: SparseObservable = unsafe {
            SparseObservable::new_unchecked(
                paulis.len() as u32,
                coeffs,
                paulis,
                (0..paulis_len).collect(),
                boundaries,
            )
        };
        let time = if sign {
            multiply_param(&angle, -1.)
        } else {
            angle
        };
        let py_evo = py_evo_cls.call1((obs, time.clone()))?;
        let py_gate = PyGate {
            qubits: qubits.len() as u32,
            clbits: 0,
            params: 1,
            op_name: "PauliEvolution".to_string(),
            gate: py_evo.into(),
        };

        new_dag.apply_operation_back(
            py_gate.into(),
            &qubits,
            &no_clbits,
            Some(smallvec![time]),
            None,
            #[cfg(feature = "cache_pygates")]
            None,
        )?;
    }

    // Add Clifford gates to the Qiskit circuit (when required).
    if fix_clifford {
        for inst in clifford_ops.into_iter() {
            new_dag.push_back(inst.clone())?;
        }
    }

    Ok(Some(new_dag.build()))
}

pub fn litinski_transformation_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(run_litinski_transformation))?;
    Ok(())
}

/// SIMD accelerated Clifford.
struct Clifford {
    /// Number of qubits.
    pub num_qubits: usize,
    /// Matrix with dimensions (2 * num_qubits) x (2 * num_qubits + 1).
    pub tableau: Vec<FixedBitSet>,
}

impl Clifford {
    /// Creates the identity Clifford on num_qubits
    fn identity(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            tableau: (0..2 * num_qubits + 1)
                .map(|i| {
                    let mut row = FixedBitSet::with_capacity(2 * num_qubits);
                    // SAFETY: We know row is large enough since it's larger than the range
                    // i is from
                    unsafe {
                        row.insert_unchecked(i);
                    }
                    row
                })
                .collect(),
        }
    }

    fn get_phase_mut(&mut self) -> &mut FixedBitSet {
        self.tableau.get_mut(2 * self.num_qubits).unwrap()
    }

    fn get_phase(&self) -> &FixedBitSet {
        self.tableau.get(2 * self.num_qubits).unwrap()
    }

    fn get_z(&self, qubit: usize) -> &FixedBitSet {
        self.tableau.get(self.num_qubits + qubit).unwrap()
    }

    fn get_z_mut(&mut self, qubit: usize) -> &mut FixedBitSet {
        self.tableau.get_mut(self.num_qubits + qubit).unwrap()
    }

    /// Modifies the tableau in-place by appending S-gate
    fn append_s(&mut self, qubit: usize) {
        let x_and_z = if let Some(x) = self.tableau.get(qubit) {
            let z = self.get_z(qubit);
            x & z
        } else {
            unreachable!();
        };
        *self.get_phase_mut() ^= x_and_z;
        let xor = self.get_z(qubit) & &self.tableau[qubit];
        *self.get_z_mut(qubit) = xor;
    }

    /// Modifies the tableau in-place by appending Sdg-gate
    #[allow(dead_code)]
    fn append_sdg(&mut self, qubit: usize) {
        let x_and_not_z = if let Some(x) = self.tableau.get(qubit) {
            let mut not_z = self.get_z(qubit).clone();
            not_z.toggle_range(..);
            x & &not_z
        } else {
            unreachable!();
        };
        *self.get_phase_mut() ^= x_and_not_z;
        let xor = &self.tableau[qubit] ^ self.get_z(qubit);
        *self.get_z_mut(qubit) = xor;
    }

    /// Modifies the tableau in-place by appending SX-gate
    fn append_sx(&mut self, qubit: usize) {
        let not_x_and_z = if let Some(x) = self.tableau.get(qubit) {
            let z = self.get_z(qubit);
            let mut not_x = x.clone();
            not_x.toggle_range(..);
            &not_x & z
        } else {
            unreachable!();
        };
        *self.get_phase_mut() ^= not_x_and_z;
        let xor = &self.tableau[qubit] ^ self.get_z(qubit);
        self.tableau[qubit] = xor;
    }

    /// Modifies the tableau in-place by appending SXDG-gate
    fn append_sxdg(&mut self, qubit: usize) {
        let x_and_z = if let Some(x) = self.tableau.get(qubit) {
            let z = self.get_z(qubit);
            x & z
        } else {
            unreachable!();
        };
        *self.get_phase_mut() ^= x_and_z;
        let xor = &self.tableau[qubit] ^ self.get_z(qubit);
        self.tableau[qubit] = xor;
    }

    /// Modifies the tableau in-place by appending H-gate
    fn append_h(&mut self, qubit: usize) {
        let x_and_z = if let Some(x) = self.tableau.get(qubit) {
            let z = self.get_z(qubit);
            x & z
        } else {
            unreachable!();
        };
        *self.get_phase_mut() ^= x_and_z;
        self.tableau.swap(qubit, self.num_qubits + qubit);
    }

    /// Modifies the tableau in-place by appending SWAP-gate
    fn append_swap(&mut self, qubit0: usize, qubit1: usize) {
        self.tableau.swap(qubit0, qubit1);
        self.tableau
            .swap(self.num_qubits + qubit0, self.num_qubits + qubit1);
    }

    /// Modifies the tableau in-place by appending CX-gate
    fn append_cx(&mut self, qubit0: usize, qubit1: usize) {
        let val = if let Some(x0) = self.tableau.get(qubit0) {
            let z0 = self.get_z(qubit0);
            let x1 = &self.tableau[qubit1];
            let z1 = self.get_z(qubit1);
            let x1_xor_z0 = x1 ^ z0;
            let xored = FixedBitSet::with_capacity_and_blocks(x1_xor_z0.len(), x1_xor_z0.zeroes());
            &(&xored & z1) & x0
        } else {
            unreachable!();
        };
        *self.get_phase_mut() ^= val;
        let xor_x = &self.tableau[qubit1] ^ &self.tableau[qubit0];
        let xor_z = self.get_z(qubit0) ^ self.get_z(qubit1);
        self.tableau[qubit1] = xor_x;
        *self.get_z_mut(qubit0) = xor_z;
    }

    /// Modifies the tableau in-place by appending CZ-gate
    fn append_cz(&mut self, qubit0: usize, qubit1: usize) {
        let val = if let Some(x0) = self.tableau.get(qubit0) {
            let z0 = self.get_z(qubit0);
            let x1 = &self.tableau[qubit1];
            let z1 = self.get_z(qubit1);
            let z0_xor_z1 = z0 ^ z1;
            &(x0 & x1) & &z0_xor_z1
        } else {
            unreachable!();
        };
        *self.get_phase_mut() ^= val;
        let xor_z1_x0 = self.get_z(qubit1) ^ &self.tableau[qubit0];
        let xor_z0_x1 = self.get_z(qubit0) ^ &self.tableau[qubit1];
        *self.get_z_mut(qubit1) = xor_z1_x0;
        *self.get_z_mut(qubit0) = xor_z0_x1;
    }

    /// Modifies the tableau in-place by appending CY-gate
    /// (todo: rewrite using native tableau manipulations)
    fn append_cy(&mut self, qubit0: usize, qubit1: usize) {
        self.append_sdg(qubit1);
        self.append_cx(qubit0, qubit1);
        self.append_s(qubit1);
    }

    /// Modifies the tableau in-place by appending X-gate
    fn append_x(&mut self, qubit: usize) {
        let xor = self.get_phase() & self.get_z(qubit);
        *self.get_phase_mut() = xor;
    }

    /// Modifies the tableau in-place by appending Z-gate
    fn append_z(&mut self, qubit: usize) {
        let xor = self.get_phase() & &self.tableau[qubit];
        *self.get_phase_mut() = xor;
    }

    /// Modifies the tableau in-place by appending Y-gate
    fn append_y(&mut self, qubit: usize) {
        let xor = &self.tableau[qubit] & self.get_z(qubit);
        *self.get_phase_mut() ^= xor;
    }

    /// Modifies the tableau in-place by appending iSWAP-gate
    /// (todo: rewrite using native tableau manipulations)
    fn append_iswap(&mut self, qubit0: usize, qubit1: usize) {
        self.append_s(qubit0);
        self.append_s(qubit1);
        self.append_h(qubit0);
        self.append_cx(qubit0, qubit1);
        self.append_cx(qubit1, qubit0);
        self.append_h(qubit1);
    }

    /// Modifies the tableau in-place by appending ECR-gate
    /// (todo: rewrite using native tableau manipulations)
    fn append_ecr(&mut self, qubit0: usize, qubit1: usize) {
        self.append_s(qubit0);
        self.append_sx(qubit1);
        self.append_cx(qubit0, qubit1);
        self.append_x(qubit0);
    }

    /// Modifies the tableau in-place by appending DCX-gate
    /// (todo: rewrite using native tableau manipulations)
    fn append_dcx(&mut self, qubit0: usize, qubit1: usize) {
        self.append_cx(qubit0, qubit1);
        self.append_cx(qubit1, qubit0);
    }

    /// Modifies the tableau in-place by appending V-gate.
    /// This is equivalent to an Sdg gate followed by an H gate.
    #[allow(dead_code)]
    fn append_v(&mut self, qubit: usize) {
        let xor = &self.tableau[qubit] & self.get_z(qubit);
        self.tableau.swap(qubit, self.num_qubits + qubit);
        self.tableau[qubit] = xor;
    }

    /// Modifies the tableau in-place by appending W-gate.
    /// This is equivalent to two V gates.
    #[allow(dead_code)]
    fn append_w(&mut self, qubit: usize) {
        let xor = &self.tableau[qubit] & self.get_z(qubit);
        self.tableau.swap(qubit, self.num_qubits + qubit);
        *self.get_z_mut(qubit) = xor;
    }

    /// Evolving the single-qubit Pauli-Z with Z on qubit qbit.
    /// Returns the evolved Pauli in the sparse format: (sign, paulis, indices).
    fn get_inverse_z(&self, qbit: usize) -> (bool, Vec<BitTerm>, Vec<Qubit>) {
        // Potentially overallocated, but this is temporary in the only use from litinski transform.
        let mut string: Vec<BitTerm> = Vec::with_capacity(self.num_qubits);
        let mut pauli = vec![false; 2 * self.num_qubits];

        let indices = (0..self.num_qubits)
            .filter_map(|i| {
                let x_bit = self.tableau[qbit][i + self.num_qubits];
                let z_bit = self.tableau[qbit][i];
                match (x_bit, z_bit) {
                    (false, false) => None,
                    (true, false) => {
                        string.push(BitTerm::X);
                        pauli[i] = true;
                        Some(Qubit::new(i))
                    }
                    (false, true) => {
                        string.push(BitTerm::Z);
                        pauli[i + self.num_qubits] = true;
                        Some(Qubit::new(i))
                    }
                    (true, true) => {
                        string.push(BitTerm::Y);
                        pauli[i] = true;
                        pauli[i + self.num_qubits] = true;
                        Some(Qubit::new(i))
                    }
                }
            })
            .collect();

        let phase = compute_phase_product_pauli(self, &pauli);
        (phase, string, indices)
    }
}

/// Computes the sign (either +1 or -1) when conjugating a Pauli by a Clifford
fn compute_phase_product_pauli(clifford: &Clifford, pauli: &[bool]) -> bool {
    let phase = pauli.iter().enumerate().fold(false, |acc, (j, &item)| {
        acc ^ (clifford.tableau[2 * clifford.num_qubits][j] & item)
    });

    let mut ifact: u8 = (0..clifford.num_qubits)
        .filter(|&i| pauli[i] & pauli[i + clifford.num_qubits])
        .count() as u8
        % 4;

    for j in 0..clifford.num_qubits {
        let mut x = false;
        let mut z = false;
        for (i, &item) in pauli.iter().enumerate() {
            if item {
                let x1: bool = clifford.tableau[j][i];
                let z1: bool = clifford.tableau[j + clifford.num_qubits][i];

                match (x1, z1, x, z) {
                    (false, true, true, true)
                    | (true, false, false, true)
                    | (true, true, true, false) => {
                        ifact += 1;
                    }
                    (false, true, true, false)
                    | (true, false, true, true)
                    | (true, true, false, true) => {
                        ifact += 3;
                    }
                    _ => {}
                };
                x ^= x1;
                z ^= z1;
                ifact %= 4;
            }
        }
    }
    (((ifact % 4) >> 1) != 0) ^ phase
}

impl fmt::Debug for Clifford {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f)?;
        writeln!(f, "Tableau:")?;
        for i in 0..2 * self.num_qubits {
            for j in 0..2 * self.num_qubits + 1 {
                write!(f, "{} ", self.tableau[j][i] as u8)?;
            }
            writeln!(f)?;
        }
        writeln!(f)?;
        Ok(())
    }
}
