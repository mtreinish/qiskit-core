// This code is part of Qiskit.
//
// (C) Copyright IBM 2022
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#[cfg(feature = "py")]
use pyo3::intern;
#[cfg(feature = "py")]
use pyo3::prelude::*;

use nalgebra::{Matrix2, Matrix4, MatrixViewMut4};
use ndarray::{Array2, ArrayView2};
use num_complex::Complex64;
#[cfg(feature = "py")]
use numpy::PyReadonlyArray2;
use rustworkx_core::petgraph::stable_graph::NodeIndex;

#[cfg(feature = "py")]
use crate::QiskitError;
use qiskit_circuit::Qubit;
use qiskit_circuit::dag_circuit::DAGCircuit;
#[cfg(feature = "py")]
use qiskit_circuit::imports::QI_OPERATOR;
use qiskit_circuit::interner::Interner;
#[cfg(feature = "py")]
use qiskit_circuit::operations::PyInstruction;
use qiskit_circuit::operations::{ArrayType, OperationRef};
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_quantum_info::versor_u2::{VersorSU2, VersorU2, VersorU2Error};
use thiserror::Error;

#[cfg(feature = "py")]
#[inline]
fn matrix4_from_pyreadonly(array: &PyReadonlyArray2<Complex64>) -> Matrix4<Complex64> {
    Matrix4::new(
        *array.get((0, 0)).unwrap(),
        *array.get((0, 1)).unwrap(),
        *array.get((0, 2)).unwrap(),
        *array.get((0, 3)).unwrap(),
        *array.get((1, 0)).unwrap(),
        *array.get((1, 1)).unwrap(),
        *array.get((1, 2)).unwrap(),
        *array.get((1, 3)).unwrap(),
        *array.get((2, 0)).unwrap(),
        *array.get((2, 1)).unwrap(),
        *array.get((2, 2)).unwrap(),
        *array.get((2, 3)).unwrap(),
        *array.get((3, 0)).unwrap(),
        *array.get((3, 1)).unwrap(),
        *array.get((3, 2)).unwrap(),
        *array.get((3, 3)).unwrap(),
    )
}

#[cfg(feature = "py")]
#[inline]
fn matrix2_from_pyreadonly(array: &PyReadonlyArray2<Complex64>) -> Matrix2<Complex64> {
    Matrix2::new(
        *array.get((0, 0)).unwrap(),
        *array.get((0, 1)).unwrap(),
        *array.get((1, 0)).unwrap(),
        *array.get((1, 1)).unwrap(),
    )
}

#[derive(Debug, Error)]
enum MatrixFromInstError {
    #[error("Parameterized gates can't be consolidated")]
    ParameterizedGate,
    #[error("Can't compute matrix of non-unitary op")]
    NonUnitary,
    #[cfg(feature = "py")]
    FromPython(#[from] PyErr),
}

#[cfg(feature = "py")]
impl From<MatrixFromInstError> for PyErr {
    fn from(err: MatrixFromInstError) -> PyErr {
        match err {
            MatrixFromInstError::FromPython(e) => e,
            _ => QiskitError::new_err(err.to_string()),
        }
    }
}

#[inline]
pub fn get_matrix_from_inst(
    inst: &PackedInstruction,
) -> Result<Array2<Complex64>, MatrixFromInstError> {
    if let Some(mat) = inst.try_matrix() {
        return Ok(mat);
    }
    #[cfg(feature = "py")]
    {
        if inst.op.try_standard_gate().is_some() {
            return Err(MatrixFromInstError::ParameterizedGate);
        }
        #[cfg(feature = "py")]
        let Some(ob) = inst.op.try_py_custom().and_then(PyInstruction::gate_object) else {
            return Err(MatrixFromInstError::NonUnitary);
        };
        // If the operation is a custom python gate, we will acquire the gil and use an
        // Operator. Otherwise, using op.matrix() should work.  A user should not be
        // able to reach this condition in Rust standalone mode.
        #[cfg(feature = "py")]
        Python::attach(|py| {
            Ok(QI_OPERATOR
                .get_bound(py)
                .call1((ob,))?
                .getattr(intern!(py, "data"))?
                .extract::<PyReadonlyArray2<Complex64>>()?
                .as_array()
                .to_owned())
        })
    }
    #[cfg(not(feature = "py"))]
    Err(MatrixFromInstError::ParameterizedGate)
}

#[inline]
pub fn get_2q_matrix_from_inst(
    inst: &PackedInstruction,
) -> Result<Matrix4<Complex64>, MatrixFromInstError> {
    if let Some(mat) = inst.try_matrix_as_nalgebra_2q() {
        return Ok(mat);
    }
    #[cfg(feature = "py")]
    {
        if inst.op.try_standard_gate().is_some() {
            return Err(MatrixFromInstError::ParameterizedGate);
        }
        let Some(ob) = inst.op.try_py_custom().and_then(PyInstruction::gate_object) else {
            return Err(MatrixFromInstError::NonUnitary);
        };
        // If the operation is a custom python gate, we will acquire the gil and use an
        // Operator. Otherwise, using op.matrix() should work.  A user should not be
        // able to reach this condition in Rust standalone mode.
        Python::attach(|py| {
            let res = QI_OPERATOR
                .get_bound(py)
                .call1((ob.clone_ref(py),))?
                .getattr(intern!(py, "data"))?
                .extract()?;
            Ok(matrix4_from_pyreadonly(&res))
        })
    }
    #[cfg(not(feature = "py"))]
    Err(MatrixFromInstError::ParameterizedGate)
}

#[inline]
pub fn get_1q_matrix_from_inst(
    inst: &PackedInstruction,
) -> Result<Matrix2<Complex64>, MatrixFromInstError> {
    if let Some(mat) = inst.try_matrix_as_nalgebra_1q() {
        return Ok(mat);
    }
    #[cfg(feature = "py")]
    {
        if inst.op.try_standard_gate().is_some() {
            return Err(MatrixFromInstError::ParameterizedGate);
        }
        let Some(ob) = inst.op.try_py_custom().and_then(PyInstruction::gate_object) else {
            return Err(MatrixFromInstError::NonUnitary);
        };
        // If the operation is a custom python gate, we will acquire the gil and use an
        // Operator. Otherwise, using op.matrix() should work.  A user should not be
        // able to reach this condition in Rust standalone mode.
        Python::attach(|py| {
            let res = QI_OPERATOR
                .get_bound(py)
                .call1((ob.clone_ref(py),))?
                .getattr(intern!(py, "data"))?
                .extract()?;
            Ok(matrix2_from_pyreadonly(&res))
        })
    }
    #[cfg(not(feature = "py"))]
    Err(MatrixFromInstError::ParameterizedGate)
}

/// Quaternion-based collect of two parallel runs of 1q gates.
#[derive(Clone, Debug)]
struct Separable1q {
    phase: f64,
    qubits: [VersorSU2; 2],
}
impl Separable1q {
    /// Construct an initial state from a single qubit operation.
    #[inline]
    fn from_qubit(n: usize, versor: VersorU2) -> Self {
        let mut qubits: [VersorSU2; 2] = Default::default();
        qubits[n] = versor.su2;
        Self {
            phase: versor.phase,
            qubits,
        }
    }

    /// Apply a new gate to one of the two qubits.
    #[inline]
    fn apply_on_qubit(&mut self, n: usize, versor: &VersorU2) {
        self.phase += versor.phase;
        self.qubits[n] = versor.su2 * self.qubits[n];
    }

    /// Construct the two-qubit gate matrix implied by these two runs.
    #[inline]
    fn matrix_into(&self, mut target: MatrixViewMut4<Complex64>) {
        let q0 = VersorU2 {
            phase: self.phase,
            su2: self.qubits[0],
        }
        .matrix_contiguous();

        // This is the manually unrolled Kronecker product.
        let [iz, iy, ix, scalar] = *self.qubits[1].0.quaternion().coords.as_ref();
        let q1_row = [Complex64::new(scalar, iz), Complex64::new(iy, ix)];
        for out_row in 0..2 {
            for out_col in 0..4 {
                target[(out_row, out_col)] = q1_row[(out_col & 2) >> 1] * q0[out_row][out_col & 1];
            }
        }
        let q1_row = [Complex64::new(-iy, ix), Complex64::new(scalar, -iz)];
        for out_row in 0..2 {
            for out_col in 0..4 {
                target[(out_row + 2, out_col)] =
                    q1_row[(out_col & 2) >> 1] * q0[out_row][out_col & 1];
            }
        }
    }
}

#[derive(Error, Debug)]
pub enum VersorFromGateError {
    #[error(transparent)]
    VersorError(#[from] VersorU2Error),
    #[error(transparent)]
    MatrixFromGate(#[from] MatrixFromInstError),
}

#[cfg(feature = "py")]
impl From<VersorFromGateError> for PyErr {
    fn from(err: VersorFromGateError) -> PyErr {
        match err {
            VersorFromGateError::VersorError(e) => e.into(),
            VersorFromGateError::MatrixFromGate(e) => e.into(),
        }
    }
}

/// Extract a versor representation of an arbitrary 1q DAG instruction.
fn versor_from_1q_gate(inst: &PackedInstruction) -> Result<VersorU2, VersorFromGateError> {
    let tol = 1e-12;
    match inst.op.view() {
        OperationRef::StandardGate(gate) => Ok(gate.versor_u2(inst.params_view())?),
        OperationRef::Unitary(gate) => match &gate.array {
            ArrayType::NDArray(arr) => Ok(VersorU2::from_ndarray_unchecked(&arr.view())),
            ArrayType::OneQ(arr) => Ok(VersorU2::from_nalgebra_unchecked(arr)),
            ArrayType::TwoQ(_) => Err(VersorFromGateError::VersorError(VersorU2Error::MultiQubit)),
        },
        _ => Ok(VersorU2::from_ndarray(
            &get_matrix_from_inst(inst)?.view(),
            tol,
        )?),
    }
}

/// Return the matrix Operator resulting from a block of Instructions.
///
/// # Panics
///
/// If any node in `op_list` is not a 1q or 2q gate.
pub fn blocks_to_matrix(
    dag: &DAGCircuit,
    op_list: &[NodeIndex],
    block_index_map: [Qubit; 2],
) -> Result<Matrix4<Complex64>, VersorFromGateError> {
    let inst_iter = op_list.iter().map(|node| dag[*node].unwrap_operation());
    instructions_to_matrix(inst_iter, block_index_map, dag.qargs_interner())
}

pub fn instructions_to_matrix<'a>(
    op_list: impl Iterator<Item = &'a PackedInstruction>,
    block_index_map: [Qubit; 2],
    qargs_interner: &'a Interner<[Qubit]>,
) -> Result<Matrix4<Complex64>, VersorFromGateError> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Qarg {
        Q0 = 0,
        Q1 = 1,
        Q01 = 2,
        Q10 = 3,
    }
    let qarg_vals = [Qarg::Q0, Qarg::Q1, Qarg::Q01, Qarg::Q10];
    let interned_default = qargs_interner.get_default();
    let interned_map = [
        &[block_index_map[0]],
        &[block_index_map[1]],
        block_index_map.as_slice(),
        &[block_index_map[1], block_index_map[0]],
    ]
    .map(|qubits| qargs_interner.try_key(qubits).unwrap_or(interned_default));
    let qarg_lookup = |qargs| {
        qarg_vals[interned_map
            .iter()
            .position(|key| qargs == *key)
            .expect("not a 1q or 2q gate in the qargs block")]
    };

    let mut work: Matrix4<Complex64> = Matrix4::zeros();
    let mut qubits_1q: Option<Separable1q> = None;
    let mut output_matrix: Option<Matrix4<Complex64>> = None;
    for inst in op_list {
        let qarg = qarg_lookup(inst.qubits);
        match qarg {
            Qarg::Q0 | Qarg::Q1 => {
                let versor = versor_from_1q_gate(inst)?;
                match qubits_1q.as_mut() {
                    Some(sep) => sep.apply_on_qubit(qarg as usize, &versor),
                    None => qubits_1q = Some(Separable1q::from_qubit(qarg as usize, versor)),
                };
            }
            Qarg::Q01 | Qarg::Q10 => {
                let mut matrix = get_2q_matrix_from_inst(inst)?;
                if qarg == Qarg::Q10 {
                    change_basis_inplace(matrix.as_view_mut());
                }
                if let Some(sep) = qubits_1q.take() {
                    sep.matrix_into(work.as_view_mut());
                    matrix *= work;
                }
                output_matrix = if let Some(state) = output_matrix {
                    Some(matrix * state)
                } else {
                    Some(matrix)
                };
            }
        }
    }

    match (
        qubits_1q.map(|sep| {
            sep.matrix_into(work.as_view_mut());
            work
        }),
        output_matrix,
    ) {
        (Some(sep), Some(state)) => Ok(sep * state),
        (None, Some(state)) => Ok(state),
        (Some(sep), None) => Ok(sep),
        // This shouldn't actually ever trigger, because we expect blocks to be non-empty, but it's
        // trivial to handle anyway.
        (None, None) => Ok(Matrix4::identity()),
    }
}

/// Switches the order of qubits in a two qubit operation.
#[inline]
pub fn change_basis(matrix: ArrayView2<Complex64>) -> Array2<Complex64> {
    let mut trans_matrix: Array2<Complex64> = matrix.reversed_axes().to_owned();
    for index in 0..trans_matrix.ncols() {
        trans_matrix.swap([1, index], [2, index]);
    }
    trans_matrix = trans_matrix.reversed_axes();
    for index in 0..trans_matrix.ncols() {
        trans_matrix.swap([1, index], [2, index]);
    }
    trans_matrix
}

/// Change the qubit order of a 2q matrix in place.
#[inline]
pub fn change_basis_inplace(mut matrix: MatrixViewMut4<Complex64>) {
    // SAFETY: The bounds are all valid for a 4x4 matrix
    unsafe {
        matrix.swap_unchecked((0, 1), (0, 2));
        matrix.swap_unchecked((3, 1), (3, 2));
        matrix.swap_unchecked((1, 0), (2, 0));
        matrix.swap_unchecked((1, 3), (2, 3));
        matrix.swap_unchecked((1, 1), (2, 2));
        matrix.swap_unchecked((1, 2), (2, 1));
    }
}
