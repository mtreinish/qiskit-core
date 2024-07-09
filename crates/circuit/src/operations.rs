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

use std::f64::consts::PI;

use crate::circuit_data::CircuitData;
use crate::circuit_instruction::ExtraInstructionAttributes;
use crate::imports::get_std_gate_class;
use crate::imports::{PARAMETER_EXPRESSION, QUANTUM_CIRCUIT};
use crate::{gate_matrix, Qubit};

use ndarray::{aview2, Array2};
use num_complex::Complex64;
use smallvec::smallvec;

use numpy::IntoPyArray;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyTuple};
use pyo3::{intern, IntoPy, Python};

#[derive(Clone, Debug)]
pub enum Param {
    ParameterExpression(PyObject),
    Float(f64),
    Obj(PyObject),
}

impl<'py> FromPyObject<'py> for Param {
    fn extract_bound(b: &Bound<'py, PyAny>) -> Result<Self, PyErr> {
        Ok(
            if b.is_instance(PARAMETER_EXPRESSION.get_bound(b.py()))?
                || b.is_instance(QUANTUM_CIRCUIT.get_bound(b.py()))?
            {
                Param::ParameterExpression(b.clone().unbind())
            } else if let Ok(val) = b.extract::<f64>() {
                Param::Float(val)
            } else {
                Param::Obj(b.clone().unbind())
            },
        )
    }
}

impl IntoPy<PyObject> for Param {
    fn into_py(self, py: Python) -> PyObject {
        match &self {
            Self::Float(val) => val.to_object(py),
            Self::ParameterExpression(val) => val.clone_ref(py),
            Self::Obj(val) => val.clone_ref(py),
        }
    }
}

impl ToPyObject for Param {
    fn to_object(&self, py: Python) -> PyObject {
        match self {
            Self::Float(val) => val.to_object(py),
            Self::ParameterExpression(val) => val.clone_ref(py),
            Self::Obj(val) => val.clone_ref(py),
        }
    }
}

/// Trait for generic circuit operations these define the common attributes
/// needed for something to be addable to the circuit struct
pub trait Operation {
    fn name(&self) -> &str;
    fn num_qubits(&self) -> u32;
    fn num_clbits(&self) -> u32;
    fn num_params(&self) -> u32;
    fn control_flow(&self) -> bool;
    fn matrix(&self, params: &[Param]) -> Option<Array2<Complex64>>;
    fn definition(&self, params: &[Param]) -> Option<CircuitData>;
    fn standard_gate(&self) -> Option<StandardGate>;
    fn directive(&self) -> bool;
}

/// Unpacked view object onto a `PackedOperation`.  This is the return value of
/// `PackedInstruction::op`, and in turn is a view object onto a `PackedOperation`.
///
/// This is the main way that we interact immutably with general circuit operations from Rust space.
pub enum OperationRef<'a> {
    Standard(StandardGate),
    Gate(&'a PyGate),
    Instruction(&'a PyInstruction),
    Operation(&'a PyOperation),
}

impl<'a> Operation for OperationRef<'a> {
    #[inline]
    fn name(&self) -> &str {
        match self {
            Self::Standard(standard) => standard.name(),
            Self::Gate(gate) => gate.name(),
            Self::Instruction(instruction) => instruction.name(),
            Self::Operation(operation) => operation.name(),
        }
    }
    #[inline]
    fn num_qubits(&self) -> u32 {
        match self {
            Self::Standard(standard) => standard.num_qubits(),
            Self::Gate(gate) => gate.num_qubits(),
            Self::Instruction(instruction) => instruction.num_qubits(),
            Self::Operation(operation) => operation.num_qubits(),
        }
    }
    #[inline]
    fn num_clbits(&self) -> u32 {
        match self {
            Self::Standard(standard) => standard.num_clbits(),
            Self::Gate(gate) => gate.num_clbits(),
            Self::Instruction(instruction) => instruction.num_clbits(),
            Self::Operation(operation) => operation.num_clbits(),
        }
    }
    #[inline]
    fn num_params(&self) -> u32 {
        match self {
            Self::Standard(standard) => standard.num_params(),
            Self::Gate(gate) => gate.num_params(),
            Self::Instruction(instruction) => instruction.num_params(),
            Self::Operation(operation) => operation.num_params(),
        }
    }
    #[inline]
    fn control_flow(&self) -> bool {
        match self {
            Self::Standard(standard) => standard.control_flow(),
            Self::Gate(gate) => gate.control_flow(),
            Self::Instruction(instruction) => instruction.control_flow(),
            Self::Operation(operation) => operation.control_flow(),
        }
    }
    #[inline]
    fn matrix(&self, params: &[Param]) -> Option<Array2<Complex64>> {
        match self {
            Self::Standard(standard) => standard.matrix(params),
            Self::Gate(gate) => gate.matrix(params),
            Self::Instruction(instruction) => instruction.matrix(params),
            Self::Operation(operation) => operation.matrix(params),
        }
    }
    #[inline]
    fn definition(&self, params: &[Param]) -> Option<CircuitData> {
        match self {
            Self::Standard(standard) => standard.definition(params),
            Self::Gate(gate) => gate.definition(params),
            Self::Instruction(instruction) => instruction.definition(params),
            Self::Operation(operation) => operation.definition(params),
        }
    }
    #[inline]
    fn standard_gate(&self) -> Option<StandardGate> {
        match self {
            Self::Standard(standard) => standard.standard_gate(),
            Self::Gate(gate) => gate.standard_gate(),
            Self::Instruction(instruction) => instruction.standard_gate(),
            Self::Operation(operation) => operation.standard_gate(),
        }
    }
    #[inline]
    fn directive(&self) -> bool {
        match self {
            Self::Standard(standard) => standard.directive(),
            Self::Gate(gate) => gate.directive(),
            Self::Instruction(instruction) => instruction.directive(),
            Self::Operation(operation) => operation.directive(),
        }
    }
}

#[derive(Clone, Debug, Copy, Eq, PartialEq, Hash)]
#[repr(u8)]
#[pyclass(module = "qiskit._accelerate.circuit")]
pub enum StandardGate {
    ZGate = 0,
    YGate = 1,
    XGate = 2,
    CZGate = 3,
    CYGate = 4,
    CXGate = 5,
    CCXGate = 6,
    RXGate = 7,
    RYGate = 8,
    RZGate = 9,
    ECRGate = 10,
    SwapGate = 11,
    SXGate = 12,
    GlobalPhaseGate = 13,
    IGate = 14,
    HGate = 15,
    PhaseGate = 16,
    UGate = 17,
    SGate = 18,
    SdgGate = 19,
    TGate = 20,
    TdgGate = 21,
    SXdgGate = 22,
    ISwapGate = 23,
    XXMinusYYGate = 24,
    XXPlusYYGate = 25,
    U1Gate = 26,
    U2Gate = 27,
    U3Gate = 28,
    CRXGate = 29,
    CRYGate = 30,
    CRZGate = 31,
    RGate = 32,
    CHGate = 33,
    CPhaseGate = 34,
    CSGate = 35,
    CSdgGate = 36,
    CSXGate = 37,
    CSwapGate = 38,
    CUGate = 39,
    CU1Gate = 40,
    CU3Gate = 41,
    C3XGate = 42,
    C3SXGate = 43,
    C4XGate = 44,
    DCXGate = 45,
    CCZGate = 46,
    RCCXGate = 47,
    RC3XGate = 48,
    RXXGate = 49,
    RYYGate = 50,
    RZZGate = 51,
    RZXGate = 52,
}

unsafe impl ::bytemuck::CheckedBitPattern for StandardGate {
    type Bits = u8;

    fn is_valid_bit_pattern(bits: &Self::Bits) -> bool {
        *bits < 53
    }
}
unsafe impl ::bytemuck::NoUninit for StandardGate {}

impl ToPyObject for StandardGate {
    fn to_object(&self, py: Python) -> Py<PyAny> {
        (*self).into_py(py)
    }
}

static STANDARD_GATE_NUM_QUBITS: [u32; STANDARD_GATE_SIZE] = [
    1, 1, 1, 2, 2, 2, 3, 1, 1, 1, // 0-9
    2, 2, 1, 0, 1, 1, 1, 1, 1, 1, // 10-19
    1, 1, 1, 2, 2, 2, 1, 1, 1, 2, // 20-29
    2, 2, 1, 2, 2, 2, 2, 2, 3, 2, // 30-39
    2, 2, 4, 4, 5, 2, 3, 3, 4, 2, // 40-49
    2, 2, 2, // 50-52
];

static STANDARD_GATE_NUM_PARAMS: [u32; STANDARD_GATE_SIZE] = [
    0, 0, 0, 0, 0, 0, 0, 1, 1, 1, // 0-9
    0, 0, 0, 1, 0, 0, 1, 3, 0, 0, // 10-19
    0, 0, 0, 0, 2, 2, 1, 2, 3, 1, // 20-29
    1, 1, 2, 0, 1, 0, 0, 0, 0, 4, // 30-39
    1, 3, 0, 0, 0, 0, 0, 0, 0, 1, // 40-49
    1, 1, 1, // 50-52
];

static STANDARD_GATE_NUM_CTRL_QUBITS: [u32; STANDARD_GATE_SIZE] = [
    0, 0, 0, 1, 1, 1, 2, 0, 0, 0, // 0-9
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 10-19
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, // 20-29
    1, 1, 0, 1, 1, 1, 1, 1, 1, 1, // 30-39
    1, 1, 3, 3, 4, 0, 2, 0, 0, 0, // 40-49
    0, 0, 0, // 50-52
];

static STANDARD_GATE_NAME: [&str; STANDARD_GATE_SIZE] = [
    "z",            // 0
    "y",            // 1
    "x",            // 2
    "cz",           // 3
    "cy",           // 4
    "cx",           // 5
    "ccx",          // 6
    "rx",           // 7
    "ry",           // 8
    "rz",           // 9
    "ecr",          // 10
    "swap",         // 11
    "sx",           // 12
    "global_phase", // 13
    "id",           // 14
    "h",            // 15
    "p",            // 16
    "u",            // 17
    "s",            // 18
    "sdg",          // 19
    "t",            // 20
    "tdg",          // 21
    "sxdg",         // 22
    "iswap",        // 23
    "xx_minus_yy",  // 24
    "xx_plus_yy",   // 25
    "u1",           // 26
    "u2",           // 27
    "u3",           // 28
    "crx",          // 29
    "cry",          // 30
    "crz",          // 31
    "r",            // 32
    "ch",           // 33
    "cp",           // 34
    "cs",           // 35
    "csdg",         // 36
    "csx",          // 37
    "cswap",        // 38
    "cu",           // 39
    "cu1",          // 40
    "cu3",          // 41
    "mcx",          // 42 ("c3x")
    "c3sx",         // 43
    "mcx",          // 44 ("c4x")
    "dcx",          // 45
    "ccz",          // 46
    "rccx",         // 47
    "rcccx",        // 48 ("rc3x")
    "rxx",          // 49
    "ryy",          // 50
    "rzz",          // 51
    "rzx",          // 52
];

impl StandardGate {
    pub fn create_py_op(
        &self,
        py: Python,
        params: Option<&[Param]>,
        extra_attrs: Option<&ExtraInstructionAttributes>,
    ) -> PyResult<Py<PyAny>> {
        let gate_class = get_std_gate_class(py, *self)?;
        let args = match params.unwrap_or(&[]) {
            &[] => PyTuple::empty_bound(py),
            params => PyTuple::new_bound(py, params),
        };
        if let Some(extra) = extra_attrs {
            let kwargs = [
                ("label", extra.label.to_object(py)),
                ("unit", extra.unit.to_object(py)),
                ("duration", extra.duration.to_object(py)),
            ]
            .into_py_dict_bound(py);
            let mut out = gate_class.call_bound(py, args, Some(&kwargs))?;
            if let Some(ref condition) = extra.condition {
                out = out.call_method0(py, "to_mutable")?;
                out.setattr(py, "condition", condition)?;
            }
            Ok(out)
        } else {
            gate_class.call_bound(py, args, None)
        }
    }

    pub fn num_ctrl_qubits(&self) -> u32 {
        STANDARD_GATE_NUM_CTRL_QUBITS[*self as usize]
    }
}

#[pymethods]
impl StandardGate {
    pub fn copy(&self) -> Self {
        *self
    }

    // These pymethods are for testing:
    pub fn _to_matrix(&self, py: Python, params: Vec<Param>) -> Option<PyObject> {
        self.matrix(&params)
            .map(|x| x.into_pyarray_bound(py).into())
    }

    pub fn _num_params(&self) -> u32 {
        self.num_params()
    }

    pub fn _get_definition(&self, params: Vec<Param>) -> Option<CircuitData> {
        self.definition(&params)
    }

    #[getter]
    pub fn get_num_qubits(&self) -> u32 {
        self.num_qubits()
    }

    #[getter]
    pub fn get_num_clbits(&self) -> u32 {
        self.num_clbits()
    }

    #[getter]
    pub fn get_num_params(&self) -> u32 {
        self.num_params()
    }

    #[getter]
    pub fn get_name(&self) -> &str {
        self.name()
    }
}

// This must be kept up-to-date with `StandardGate` when adding or removing
// gates from the enum
//
// Remove this when std::mem::variant_count() is stabilized (see
// https://github.com/rust-lang/rust/issues/73662 )
pub const STANDARD_GATE_SIZE: usize = 53;

impl Operation for StandardGate {
    fn name(&self) -> &str {
        STANDARD_GATE_NAME[*self as usize]
    }

    fn num_qubits(&self) -> u32 {
        STANDARD_GATE_NUM_QUBITS[*self as usize]
    }

    fn num_params(&self) -> u32 {
        STANDARD_GATE_NUM_PARAMS[*self as usize]
    }

    fn num_clbits(&self) -> u32 {
        0
    }

    fn control_flow(&self) -> bool {
        false
    }

    fn directive(&self) -> bool {
        false
    }

    fn matrix(&self, params: &[Param]) -> Option<Array2<Complex64>> {
        match self {
            Self::ZGate => match params {
                [] => Some(aview2(&gate_matrix::Z_GATE).to_owned()),
                _ => None,
            },
            Self::YGate => match params {
                [] => Some(aview2(&gate_matrix::Y_GATE).to_owned()),
                _ => None,
            },
            Self::XGate => match params {
                [] => Some(aview2(&gate_matrix::X_GATE).to_owned()),
                _ => None,
            },
            Self::CZGate => match params {
                [] => Some(aview2(&gate_matrix::CZ_GATE).to_owned()),
                _ => None,
            },
            Self::CYGate => match params {
                [] => Some(aview2(&gate_matrix::CY_GATE).to_owned()),
                _ => None,
            },
            Self::CXGate => match params {
                [] => Some(aview2(&gate_matrix::CX_GATE).to_owned()),
                _ => None,
            },
            Self::CCXGate => match params {
                [] => Some(aview2(&gate_matrix::CCX_GATE).to_owned()),
                _ => None,
            },
            Self::RXGate => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::rx_gate(*theta)).to_owned()),
                _ => None,
            },
            Self::RYGate => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::ry_gate(*theta)).to_owned()),
                _ => None,
            },
            Self::RZGate => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::rz_gate(*theta)).to_owned()),
                _ => None,
            },
            Self::CRXGate => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::crx_gate(*theta)).to_owned()),
                _ => None,
            },
            Self::CRYGate => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::cry_gate(*theta)).to_owned()),
                _ => None,
            },
            Self::CRZGate => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::crz_gate(*theta)).to_owned()),
                _ => None,
            },
            Self::ECRGate => match params {
                [] => Some(aview2(&gate_matrix::ECR_GATE).to_owned()),
                _ => None,
            },
            Self::SwapGate => match params {
                [] => Some(aview2(&gate_matrix::SWAP_GATE).to_owned()),
                _ => None,
            },
            Self::SXGate => match params {
                [] => Some(aview2(&gate_matrix::SX_GATE).to_owned()),
                _ => None,
            },
            Self::SXdgGate => match params {
                [] => Some(aview2(&gate_matrix::SXDG_GATE).to_owned()),
                _ => None,
            },
            Self::GlobalPhaseGate => match params {
                [Param::Float(theta)] => {
                    Some(aview2(&gate_matrix::global_phase_gate(*theta)).to_owned())
                }
                _ => None,
            },
            Self::IGate => match params {
                [] => Some(aview2(&gate_matrix::ONE_QUBIT_IDENTITY).to_owned()),
                _ => None,
            },
            Self::HGate => match params {
                [] => Some(aview2(&gate_matrix::H_GATE).to_owned()),
                _ => None,
            },
            Self::PhaseGate => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::phase_gate(*theta)).to_owned()),
                _ => None,
            },
            Self::UGate => match params {
                [Param::Float(theta), Param::Float(phi), Param::Float(lam)] => {
                    Some(aview2(&gate_matrix::u_gate(*theta, *phi, *lam)).to_owned())
                }
                _ => None,
            },
            Self::SGate => match params {
                [] => Some(aview2(&gate_matrix::S_GATE).to_owned()),
                _ => None,
            },
            Self::SdgGate => match params {
                [] => Some(aview2(&gate_matrix::SDG_GATE).to_owned()),
                _ => None,
            },
            Self::TGate => match params {
                [] => Some(aview2(&gate_matrix::T_GATE).to_owned()),
                _ => None,
            },
            Self::TdgGate => match params {
                [] => Some(aview2(&gate_matrix::TDG_GATE).to_owned()),
                _ => None,
            },
            Self::ISwapGate => match params {
                [] => Some(aview2(&gate_matrix::ISWAP_GATE).to_owned()),
                _ => None,
            },
            Self::XXMinusYYGate => match params {
                [Param::Float(theta), Param::Float(beta)] => {
                    Some(aview2(&gate_matrix::xx_minus_yy_gate(*theta, *beta)).to_owned())
                }
                _ => None,
            },
            Self::XXPlusYYGate => match params {
                [Param::Float(theta), Param::Float(beta)] => {
                    Some(aview2(&gate_matrix::xx_plus_yy_gate(*theta, *beta)).to_owned())
                }
                _ => None,
            },
            Self::U1Gate => match params[0] {
                Param::Float(val) => Some(aview2(&gate_matrix::u1_gate(val)).to_owned()),
                _ => None,
            },
            Self::U2Gate => match params {
                [Param::Float(phi), Param::Float(lam)] => {
                    Some(aview2(&gate_matrix::u2_gate(*phi, *lam)).to_owned())
                }
                _ => None,
            },
            Self::U3Gate => match params {
                [Param::Float(theta), Param::Float(phi), Param::Float(lam)] => {
                    Some(aview2(&gate_matrix::u3_gate(*theta, *phi, *lam)).to_owned())
                }
                _ => None,
            },
            Self::CUGate => match params {
                [Param::Float(theta), Param::Float(phi), Param::Float(lam), Param::Float(gamma)] => {
                    Some(aview2(&gate_matrix::cu_gate(*theta, *phi, *lam, *gamma)).to_owned())
                }
                _ => None,
            },
            Self::CU1Gate => match params[0] {
                Param::Float(lam) => Some(aview2(&gate_matrix::cu1_gate(lam)).to_owned()),
                _ => None,
            },
            Self::CU3Gate => match params {
                [Param::Float(theta), Param::Float(phi), Param::Float(lam)] => {
                    Some(aview2(&gate_matrix::cu3_gate(*theta, *phi, *lam)).to_owned())
                }
                _ => None,
            },
            Self::C3XGate => match params {
                [] => Some(aview2(&gate_matrix::C3X_GATE).to_owned()),
                _ => None,
            },
            Self::C3SXGate => match params {
                [] => Some(aview2(&gate_matrix::C3SX_GATE).to_owned()),
                _ => None,
            },
            Self::CCZGate => match params {
                [] => Some(aview2(&gate_matrix::CCZ_GATE).to_owned()),
                _ => None,
            },
            Self::CHGate => match params {
                [] => Some(aview2(&gate_matrix::CH_GATE).to_owned()),
                _ => None,
            },
            Self::CPhaseGate => match params {
                [Param::Float(lam)] => Some(aview2(&gate_matrix::cp_gate(*lam)).to_owned()),
                _ => None,
            },
            Self::CSGate => match params {
                [] => Some(aview2(&gate_matrix::CS_GATE).to_owned()),
                _ => None,
            },
            Self::CSdgGate => match params {
                [] => Some(aview2(&gate_matrix::CSDG_GATE).to_owned()),
                _ => None,
            },
            Self::CSXGate => match params {
                [] => Some(aview2(&gate_matrix::CSX_GATE).to_owned()),
                _ => None,
            },
            Self::CSwapGate => match params {
                [] => Some(aview2(&gate_matrix::CSWAP_GATE).to_owned()),
                _ => None,
            },
            Self::RGate => match params {
                [Param::Float(theta), Param::Float(phi)] => {
                    Some(aview2(&gate_matrix::r_gate(*theta, *phi)).to_owned())
                }
                _ => None,
            },
            Self::DCXGate => match params {
                [] => Some(aview2(&gate_matrix::DCX_GATE).to_owned()),
                _ => None,
            },
            Self::C4XGate => todo!(),
            Self::RXXGate => match params[0] {
                Param::Float(theta) => Some(aview2(&gate_matrix::rxx_gate(theta)).to_owned()),
                _ => None,
            },
            Self::RYYGate => match params[0] {
                Param::Float(theta) => Some(aview2(&gate_matrix::ryy_gate(theta)).to_owned()),
                _ => None,
            },
            Self::RZZGate => match params[0] {
                Param::Float(theta) => Some(aview2(&gate_matrix::rzz_gate(theta)).to_owned()),
                _ => None,
            },
            Self::RZXGate => match params[0] {
                Param::Float(theta) => Some(aview2(&gate_matrix::rzx_gate(theta)).to_owned()),
                _ => None,
            },
            Self::RCCXGate => match params {
                [] => Some(aview2(&gate_matrix::RCCX_GATE).to_owned()),
                _ => None,
            },
            Self::RC3XGate => match params {
                [] => Some(aview2(&gate_matrix::RC3X_GATE).to_owned()),
                _ => None,
            },
        }
    }

    fn definition(&self, params: &[Param]) -> Option<CircuitData> {
        match self {
            Self::ZGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::PhaseGate,
                            smallvec![Param::Float(PI)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::YGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::UGate,
                            smallvec![
                                Param::Float(PI),
                                Param::Float(PI / 2.),
                                Param::Float(PI / 2.),
                            ],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::XGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::UGate,
                            smallvec![Param::Float(PI), Param::Float(0.), Param::Float(PI)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CZGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::HGate, smallvec![], q1.clone()),
                            (Self::CXGate, smallvec![], q0_1),
                            (Self::HGate, smallvec![], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CYGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::SdgGate, smallvec![], q1.clone()),
                            (Self::CXGate, smallvec![], q0_1),
                            (Self::SGate, smallvec![], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CXGate => None,
            Self::CCXGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q2 = smallvec![Qubit(2)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                let q0_2 = smallvec![Qubit(0), Qubit(2)];
                let q1_2 = smallvec![Qubit(1), Qubit(2)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        3,
                        [
                            (Self::HGate, smallvec![], q2.clone()),
                            (Self::CXGate, smallvec![], q1_2.clone()),
                            (Self::TdgGate, smallvec![], q2.clone()),
                            (Self::CXGate, smallvec![], q0_2.clone()),
                            (Self::TGate, smallvec![], q2.clone()),
                            (Self::CXGate, smallvec![], q1_2),
                            (Self::TdgGate, smallvec![], q2.clone()),
                            (Self::CXGate, smallvec![], q0_2),
                            (Self::TGate, smallvec![], q1.clone()),
                            (Self::TGate, smallvec![], q2.clone()),
                            (Self::HGate, smallvec![], q2),
                            (Self::CXGate, smallvec![], q0_1.clone()),
                            (Self::TGate, smallvec![], q0),
                            (Self::TdgGate, smallvec![], q1),
                            (Self::CXGate, smallvec![], q0_1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::RXGate => Python::with_gil(|py| -> Option<CircuitData> {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::RGate,
                            smallvec![theta.clone(), FLOAT_ZERO],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::RYGate => Python::with_gil(|py| -> Option<CircuitData> {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::RGate,
                            smallvec![theta.clone(), Param::Float(PI / 2.0)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::RZGate => Python::with_gil(|py| -> Option<CircuitData> {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::PhaseGate,
                            smallvec![theta.clone()],
                            smallvec![Qubit(0)],
                        )],
                        multiply_param(theta, -0.5, py),
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CRXGate => Python::with_gil(|py| -> Option<CircuitData> {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(PI / 2.)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::UGate,
                                smallvec![
                                    multiply_param(theta, -0.5, py),
                                    Param::Float(0.0),
                                    Param::Float(0.0)
                                ],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::UGate,
                                smallvec![
                                    multiply_param(theta, 0.5, py),
                                    Param::Float(-PI / 2.),
                                    Param::Float(0.0)
                                ],
                                smallvec![Qubit(1)],
                            ),
                        ],
                        Param::Float(0.0),
                    )
                    .expect("Unexpected Qiskit Python bug!"),
                )
            }),
            Self::CRYGate => Python::with_gil(|py| -> Option<CircuitData> {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                Self::RYGate,
                                smallvec![multiply_param(theta, 0.5, py)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::RYGate,
                                smallvec![multiply_param(theta, -0.5, py)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        ],
                        Param::Float(0.0),
                    )
                    .expect("Unexpected Qiskit Python bug!"),
                )
            }),
            Self::CRZGate => Python::with_gil(|py| -> Option<CircuitData> {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                Self::RZGate,
                                smallvec![multiply_param(theta, 0.5, py)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::RZGate,
                                smallvec![multiply_param(theta, -0.5, py)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        ],
                        Param::Float(0.0),
                    )
                    .expect("Unexpected Qiskit Python bug!"),
                )
            }),
            Self::ECRGate => todo!("Add when we have RZX"),
            Self::SwapGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (Self::CXGate, smallvec![], smallvec![Qubit(1), Qubit(0)]),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::SXGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [
                            (Self::SdgGate, smallvec![], smallvec![Qubit(0)]),
                            (Self::HGate, smallvec![], smallvec![Qubit(0)]),
                            (Self::SdgGate, smallvec![], smallvec![Qubit(0)]),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::SXdgGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [
                            (Self::SGate, smallvec![], smallvec![Qubit(0)]),
                            (Self::HGate, smallvec![], smallvec![Qubit(0)]),
                            (Self::SGate, smallvec![], smallvec![Qubit(0)]),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::GlobalPhaseGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(py, 0, [], params[0].clone())
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::IGate => None,
            Self::HGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::UGate,
                            smallvec![Param::Float(PI / 2.), Param::Float(0.), Param::Float(PI)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::PhaseGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::UGate,
                            smallvec![Param::Float(0.), Param::Float(0.), params[0].clone()],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::UGate => None,
            Self::U1Gate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::PhaseGate,
                            params.iter().cloned().collect(),
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::U2Gate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::UGate,
                            smallvec![Param::Float(PI / 2.), params[0].clone(), params[1].clone()],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::U3Gate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::UGate,
                            params.iter().cloned().collect(),
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::SGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::PhaseGate,
                            smallvec![Param::Float(PI / 2.)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::SdgGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::PhaseGate,
                            smallvec![Param::Float(-PI / 2.)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::TGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::PhaseGate,
                            smallvec![Param::Float(PI / 4.)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::TdgGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::PhaseGate,
                            smallvec![Param::Float(-PI / 4.)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::ISwapGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::SGate, smallvec![], smallvec![Qubit(0)]),
                            (Self::SGate, smallvec![], smallvec![Qubit(1)]),
                            (Self::HGate, smallvec![], smallvec![Qubit(0)]),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (Self::CXGate, smallvec![], smallvec![Qubit(1), Qubit(0)]),
                            (Self::HGate, smallvec![], smallvec![Qubit(1)]),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::XXMinusYYGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                let theta = &params[0];
                let beta = &params[1];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                Self::RZGate,
                                smallvec![multiply_param(beta, -1.0, py)],
                                q1.clone(),
                            ),
                            (Self::RZGate, smallvec![Param::Float(-PI / 2.)], q0.clone()),
                            (Self::SXGate, smallvec![], q0.clone()),
                            (Self::RZGate, smallvec![Param::Float(PI / 2.)], q0.clone()),
                            (Self::SGate, smallvec![], q1.clone()),
                            (Self::CXGate, smallvec![], q0_1.clone()),
                            (
                                Self::RYGate,
                                smallvec![multiply_param(theta, 0.5, py)],
                                q0.clone(),
                            ),
                            (
                                Self::RYGate,
                                smallvec![multiply_param(theta, -0.5, py)],
                                q1.clone(),
                            ),
                            (Self::CXGate, smallvec![], q0_1),
                            (Self::SdgGate, smallvec![], q1.clone()),
                            (Self::RZGate, smallvec![Param::Float(-PI / 2.)], q0.clone()),
                            (Self::SXdgGate, smallvec![], q0.clone()),
                            (Self::RZGate, smallvec![Param::Float(PI / 2.)], q0),
                            (Self::RZGate, smallvec![beta.clone()], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::XXPlusYYGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q1_0 = smallvec![Qubit(1), Qubit(0)];
                let theta = &params[0];
                let beta = &params[1];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::RZGate, smallvec![beta.clone()], q0.clone()),
                            (Self::RZGate, smallvec![Param::Float(-PI / 2.)], q1.clone()),
                            (Self::SXGate, smallvec![], q1.clone()),
                            (Self::RZGate, smallvec![Param::Float(PI / 2.)], q1.clone()),
                            (Self::SGate, smallvec![], q0.clone()),
                            (Self::CXGate, smallvec![], q1_0.clone()),
                            (
                                Self::RYGate,
                                smallvec![multiply_param(theta, -0.5, py)],
                                q1.clone(),
                            ),
                            (
                                Self::RYGate,
                                smallvec![multiply_param(theta, -0.5, py)],
                                q0.clone(),
                            ),
                            (Self::CXGate, smallvec![], q1_0),
                            (Self::SdgGate, smallvec![], q0.clone()),
                            (Self::RZGate, smallvec![Param::Float(-PI / 2.)], q1.clone()),
                            (Self::SXdgGate, smallvec![], q1.clone()),
                            (Self::RZGate, smallvec![Param::Float(PI / 2.)], q1),
                            (Self::RZGate, smallvec![multiply_param(beta, -1.0, py)], q0),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CUGate => Python::with_gil(|py| -> Option<CircuitData> {
                let param_second_p = radd_param(
                    multiply_param(&params[2], 0.5, py),
                    multiply_param(&params[1], 0.5, py),
                    py,
                );
                let param_third_p = radd_param(
                    multiply_param(&params[2], 0.5, py),
                    multiply_param(&params[1], -0.5, py),
                    py,
                );
                let param_first_u = radd_param(
                    multiply_param(&params[1], -0.5, py),
                    multiply_param(&params[2], -0.5, py),
                    py,
                );
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                Self::PhaseGate,
                                smallvec![params[3].clone()],
                                smallvec![Qubit(0)],
                            ),
                            (
                                Self::PhaseGate,
                                smallvec![param_second_p],
                                smallvec![Qubit(0)],
                            ),
                            (
                                Self::PhaseGate,
                                smallvec![param_third_p],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::UGate,
                                smallvec![
                                    multiply_param(&params[0], -0.5, py),
                                    Param::Float(0.),
                                    param_first_u
                                ],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::UGate,
                                smallvec![
                                    multiply_param(&params[0], 0.5, py),
                                    params[1].clone(),
                                    Param::Float(0.)
                                ],
                                smallvec![Qubit(1)],
                            ),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CHGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::SGate, smallvec![], q1.clone()),
                            (Self::HGate, smallvec![], q1.clone()),
                            (Self::TGate, smallvec![], q1.clone()),
                            (Self::CXGate, smallvec![], q0_1),
                            (Self::TdgGate, smallvec![], q1.clone()),
                            (Self::HGate, smallvec![], q1.clone()),
                            (Self::SdgGate, smallvec![], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CU1Gate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                Self::U1Gate,
                                smallvec![multiply_param(&params[0], 0.5, py)],
                                smallvec![Qubit(0)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::U1Gate,
                                smallvec![multiply_param(&params[0], -0.5, py)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::U1Gate,
                                smallvec![multiply_param(&params[0], 0.5, py)],
                                smallvec![Qubit(1)],
                            ),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CPhaseGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                Self::PhaseGate,
                                smallvec![multiply_param(&params[0], 0.5, py)],
                                q0,
                            ),
                            (Self::CXGate, smallvec![], q0_1.clone()),
                            (
                                Self::PhaseGate,
                                smallvec![multiply_param(&params[0], -0.5, py)],
                                q1.clone(),
                            ),
                            (Self::CXGate, smallvec![], q0_1),
                            (
                                Self::PhaseGate,
                                smallvec![multiply_param(&params[0], 0.5, py)],
                                q1,
                            ),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CU3Gate => Python::with_gil(|py| -> Option<CircuitData> {
                let param_first_u1 = radd_param(
                    multiply_param(&params[2], 0.5, py),
                    multiply_param(&params[1], 0.5, py),
                    py,
                );
                let param_second_u1 = radd_param(
                    multiply_param(&params[2], 0.5, py),
                    multiply_param(&params[1], -0.5, py),
                    py,
                );
                let param_first_u3 = radd_param(
                    multiply_param(&params[1], -0.5, py),
                    multiply_param(&params[2], -0.5, py),
                    py,
                );
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::U1Gate, smallvec![param_first_u1], smallvec![Qubit(0)]),
                            (
                                Self::U1Gate,
                                smallvec![param_second_u1],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::U3Gate,
                                smallvec![
                                    multiply_param(&params[0], -0.5, py),
                                    Param::Float(0.),
                                    param_first_u3
                                ],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::U3Gate,
                                smallvec![
                                    multiply_param(&params[0], 0.5, py),
                                    params[1].clone(),
                                    Param::Float(0.)
                                ],
                                smallvec![Qubit(1)],
                            ),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CSGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::PhaseGate, smallvec![Param::Float(PI / 4.)], q0),
                            (Self::CXGate, smallvec![], q0_1.clone()),
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(-PI / 4.)],
                                q1.clone(),
                            ),
                            (Self::CXGate, smallvec![], q0_1),
                            (Self::PhaseGate, smallvec![Param::Float(PI / 4.)], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::C3XGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        4,
                        [
                            (Self::HGate, smallvec![], smallvec![Qubit(3)]),
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(PI / 8.)],
                                smallvec![Qubit(0)],
                            ),
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(PI / 8.)],
                                smallvec![Qubit(1)],
                            ),
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(PI / 8.)],
                                smallvec![Qubit(2)],
                            ),
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(PI / 8.)],
                                smallvec![Qubit(3)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(-PI / 8.)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (Self::CXGate, smallvec![], smallvec![Qubit(1), Qubit(2)]),
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(-PI / 8.)],
                                smallvec![Qubit(2)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(2)]),
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(PI / 8.)],
                                smallvec![Qubit(2)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(1), Qubit(2)]),
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(-PI / 8.)],
                                smallvec![Qubit(2)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(2)]),
                            (Self::CXGate, smallvec![], smallvec![Qubit(2), Qubit(3)]),
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(-PI / 8.)],
                                smallvec![Qubit(3)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(1), Qubit(3)]),
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(PI / 8.)],
                                smallvec![Qubit(3)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(2), Qubit(3)]),
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(-PI / 8.)],
                                smallvec![Qubit(3)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(3)]),
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(PI / 8.)],
                                smallvec![Qubit(3)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(2), Qubit(3)]),
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(-PI / 8.)],
                                smallvec![Qubit(3)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(1), Qubit(3)]),
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(PI / 8.)],
                                smallvec![Qubit(3)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(2), Qubit(3)]),
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(-PI / 8.)],
                                smallvec![Qubit(3)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(3)]),
                            (Self::HGate, smallvec![], smallvec![Qubit(3)]),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CSdgGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::PhaseGate, smallvec![Param::Float(-PI / 4.)], q0),
                            (Self::CXGate, smallvec![], q0_1.clone()),
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(PI / 4.)],
                                q1.clone(),
                            ),
                            (Self::CXGate, smallvec![], q0_1),
                            (Self::PhaseGate, smallvec![Param::Float(-PI / 4.)], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::C3SXGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        4,
                        [
                            (Self::HGate, smallvec![], smallvec![Qubit(3)]),
                            (
                                Self::CU1Gate,
                                smallvec![Param::Float(PI / 8.)],
                                smallvec![Qubit(0), Qubit(3)],
                            ),
                            (Self::HGate, smallvec![], smallvec![Qubit(3)]),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (Self::HGate, smallvec![], smallvec![Qubit(3)]),
                            (
                                Self::CU1Gate,
                                smallvec![Param::Float(-PI / 8.)],
                                smallvec![Qubit(1), Qubit(3)],
                            ),
                            (Self::HGate, smallvec![], smallvec![Qubit(3)]),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (Self::HGate, smallvec![], smallvec![Qubit(3)]),
                            (
                                Self::CU1Gate,
                                smallvec![Param::Float(PI / 8.)],
                                smallvec![Qubit(1), Qubit(3)],
                            ),
                            (Self::HGate, smallvec![], smallvec![Qubit(3)]),
                            (Self::CXGate, smallvec![], smallvec![Qubit(1), Qubit(2)]),
                            (Self::HGate, smallvec![], smallvec![Qubit(3)]),
                            (
                                Self::CU1Gate,
                                smallvec![Param::Float(-PI / 8.)],
                                smallvec![Qubit(2), Qubit(3)],
                            ),
                            (Self::HGate, smallvec![], smallvec![Qubit(3)]),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(2)]),
                            (Self::HGate, smallvec![], smallvec![Qubit(3)]),
                            (
                                Self::CU1Gate,
                                smallvec![Param::Float(PI / 8.)],
                                smallvec![Qubit(2), Qubit(3)],
                            ),
                            (Self::HGate, smallvec![], smallvec![Qubit(3)]),
                            (Self::CXGate, smallvec![], smallvec![Qubit(1), Qubit(2)]),
                            (Self::HGate, smallvec![], smallvec![Qubit(3)]),
                            (
                                Self::CU1Gate,
                                smallvec![Param::Float(-PI / 8.)],
                                smallvec![Qubit(2), Qubit(3)],
                            ),
                            (Self::HGate, smallvec![], smallvec![Qubit(3)]),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(2)]),
                            (Self::HGate, smallvec![], smallvec![Qubit(3)]),
                            (
                                Self::CU1Gate,
                                smallvec![Param::Float(PI / 8.)],
                                smallvec![Qubit(2), Qubit(3)],
                            ),
                            (Self::HGate, smallvec![], smallvec![Qubit(3)]),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CSXGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::HGate, smallvec![], q1.clone()),
                            (Self::CPhaseGate, smallvec![Param::Float(PI / 2.)], q0_1),
                            (Self::HGate, smallvec![], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CCZGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        3,
                        [
                            (Self::HGate, smallvec![], smallvec![Qubit(2)]),
                            (
                                Self::CCXGate,
                                smallvec![],
                                smallvec![Qubit(0), Qubit(1), Qubit(2)],
                            ),
                            (Self::HGate, smallvec![], smallvec![Qubit(2)]),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CSwapGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        3,
                        [
                            (Self::CXGate, smallvec![], smallvec![Qubit(2), Qubit(1)]),
                            (
                                Self::CCXGate,
                                smallvec![],
                                smallvec![Qubit(0), Qubit(1), Qubit(2)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(2), Qubit(1)]),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::RGate => Python::with_gil(|py| -> Option<CircuitData> {
                let theta_expr = clone_param(&params[0], py);
                let phi_expr1 = add_param(&params[1], -PI / 2., py);
                let phi_expr2 = multiply_param(&phi_expr1, -1.0, py);
                let defparams = smallvec![theta_expr, phi_expr1, phi_expr2];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(Self::UGate, defparams, smallvec![Qubit(0)])],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::C4XGate => todo!(),
            Self::DCXGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (Self::CXGate, smallvec![], smallvec![Qubit(1), Qubit(0)]),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::RXXGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_q1 = smallvec![Qubit(0), Qubit(1)];
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::HGate, smallvec![], q0.clone()),
                            (Self::HGate, smallvec![], q1.clone()),
                            (Self::CXGate, smallvec![], q0_q1.clone()),
                            (Self::RZGate, smallvec![theta.clone()], q1.clone()),
                            (Self::CXGate, smallvec![], q0_q1),
                            (Self::HGate, smallvec![], q1),
                            (Self::HGate, smallvec![], q0),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::RYYGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_q1 = smallvec![Qubit(0), Qubit(1)];
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::RXGate, smallvec![Param::Float(PI / 2.)], q0.clone()),
                            (Self::RXGate, smallvec![Param::Float(PI / 2.)], q1.clone()),
                            (Self::CXGate, smallvec![], q0_q1.clone()),
                            (Self::RZGate, smallvec![theta.clone()], q1.clone()),
                            (Self::CXGate, smallvec![], q0_q1),
                            (Self::RXGate, smallvec![Param::Float(-PI / 2.)], q0),
                            (Self::RXGate, smallvec![Param::Float(-PI / 2.)], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::RZZGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q1 = smallvec![Qubit(1)];
                let q0_q1 = smallvec![Qubit(0), Qubit(1)];
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::CXGate, smallvec![], q0_q1.clone()),
                            (Self::RZGate, smallvec![theta.clone()], q1),
                            (Self::CXGate, smallvec![], q0_q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::RZXGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q1 = smallvec![Qubit(1)];
                let q0_q1 = smallvec![Qubit(0), Qubit(1)];
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::HGate, smallvec![], q1.clone()),
                            (Self::CXGate, smallvec![], q0_q1.clone()),
                            (Self::RZGate, smallvec![theta.clone()], q1.clone()),
                            (Self::CXGate, smallvec![], q0_q1),
                            (Self::HGate, smallvec![], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::RCCXGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q2 = smallvec![Qubit(2)];
                let q0_2 = smallvec![Qubit(0), Qubit(2)];
                let q1_2 = smallvec![Qubit(1), Qubit(2)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        3,
                        [
                            (
                                Self::U2Gate,
                                smallvec![Param::Float(0.), Param::Float(PI)],
                                q2.clone(),
                            ),
                            (Self::U1Gate, smallvec![Param::Float(PI / 4.)], q2.clone()),
                            (Self::CXGate, smallvec![], q1_2.clone()),
                            (Self::U1Gate, smallvec![Param::Float(-PI / 4.)], q2.clone()),
                            (Self::CXGate, smallvec![], q0_2),
                            (Self::U1Gate, smallvec![Param::Float(PI / 4.)], q2.clone()),
                            (Self::CXGate, smallvec![], q1_2),
                            (Self::U1Gate, smallvec![Param::Float(-PI / 4.)], q2.clone()),
                            (
                                Self::U2Gate,
                                smallvec![Param::Float(0.), Param::Float(PI)],
                                q2,
                            ),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::RC3XGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        4,
                        [
                            (
                                Self::U2Gate,
                                smallvec![Param::Float(0.), Param::Float(PI)],
                                smallvec![Qubit(3)],
                            ),
                            (
                                Self::U1Gate,
                                smallvec![Param::Float(PI / 4.)],
                                smallvec![Qubit(3)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(2), Qubit(3)]),
                            (
                                Self::U1Gate,
                                smallvec![Param::Float(-PI / 4.)],
                                smallvec![Qubit(3)],
                            ),
                            (
                                Self::U2Gate,
                                smallvec![Param::Float(0.), Param::Float(PI)],
                                smallvec![Qubit(3)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(3)]),
                            (
                                Self::U1Gate,
                                smallvec![Param::Float(PI / 4.)],
                                smallvec![Qubit(3)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(1), Qubit(3)]),
                            (
                                Self::U1Gate,
                                smallvec![Param::Float(-PI / 4.)],
                                smallvec![Qubit(3)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(3)]),
                            (
                                Self::U1Gate,
                                smallvec![Param::Float(PI / 4.)],
                                smallvec![Qubit(3)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(1), Qubit(3)]),
                            (
                                Self::U1Gate,
                                smallvec![Param::Float(-PI / 4.)],
                                smallvec![Qubit(3)],
                            ),
                            (
                                Self::U2Gate,
                                smallvec![Param::Float(0.), Param::Float(PI)],
                                smallvec![Qubit(3)],
                            ),
                            (
                                Self::U1Gate,
                                smallvec![Param::Float(PI / 4.)],
                                smallvec![Qubit(3)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(2), Qubit(3)]),
                            (
                                Self::U1Gate,
                                smallvec![Param::Float(-PI / 4.)],
                                smallvec![Qubit(3)],
                            ),
                            (
                                Self::U2Gate,
                                smallvec![Param::Float(0.), Param::Float(PI)],
                                smallvec![Qubit(3)],
                            ),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
        }
    }

    fn standard_gate(&self) -> Option<StandardGate> {
        Some(*self)
    }
}

const FLOAT_ZERO: Param = Param::Float(0.0);

// Return explicitly requested copy of `param`, handling
// each variant separately.
fn clone_param(param: &Param, py: Python) -> Param {
    match param {
        Param::Float(theta) => Param::Float(*theta),
        Param::ParameterExpression(theta) => Param::ParameterExpression(theta.clone_ref(py)),
        Param::Obj(_) => unreachable!(),
    }
}

fn multiply_param(param: &Param, mult: f64, py: Python) -> Param {
    match param {
        Param::Float(theta) => Param::Float(theta * mult),
        Param::ParameterExpression(theta) => Param::ParameterExpression(
            theta
                .clone_ref(py)
                .call_method1(py, intern!(py, "__rmul__"), (mult,))
                .expect("Multiplication of Parameter expression by float failed."),
        ),
        Param::Obj(_) => unreachable!(),
    }
}

fn add_param(param: &Param, summand: f64, py: Python) -> Param {
    match param {
        Param::Float(theta) => Param::Float(*theta + summand),
        Param::ParameterExpression(theta) => Param::ParameterExpression(
            theta
                .clone_ref(py)
                .call_method1(py, intern!(py, "__add__"), (summand,))
                .expect("Sum of Parameter expression and float failed."),
        ),
        Param::Obj(_) => unreachable!(),
    }
}

fn radd_param(param1: Param, param2: Param, py: Python) -> Param {
    match [param1, param2] {
        [Param::Float(theta), Param::Float(lambda)] => Param::Float(theta + lambda),
        [Param::ParameterExpression(theta), Param::ParameterExpression(lambda)] => {
            Param::ParameterExpression(
                theta
                    .clone_ref(py)
                    .call_method1(py, intern!(py, "__radd__"), (lambda,))
                    .expect("Parameter expression addition failed"),
            )
        }
        _ => unreachable!(),
    }
}

/// This class is used to wrap a Python side Instruction that is not in the standard library
#[derive(Clone, Debug)]
// We bit-pack pointers to this, so having a known alignment even on 32-bit systems is good.
#[repr(align(8))]
pub struct PyInstruction {
    pub qubits: u32,
    pub clbits: u32,
    pub params: u32,
    pub op_name: String,
    pub instruction: PyObject,
}

impl Operation for PyInstruction {
    fn name(&self) -> &str {
        self.op_name.as_str()
    }
    fn num_qubits(&self) -> u32 {
        self.qubits
    }
    fn num_clbits(&self) -> u32 {
        self.clbits
    }
    fn num_params(&self) -> u32 {
        self.params
    }
    fn control_flow(&self) -> bool {
        false
    }
    fn matrix(&self, _params: &[Param]) -> Option<Array2<Complex64>> {
        None
    }
    fn definition(&self, _params: &[Param]) -> Option<CircuitData> {
        Python::with_gil(|py| -> Option<CircuitData> {
            match self.instruction.getattr(py, intern!(py, "definition")) {
                Ok(definition) => {
                    let res: Option<PyObject> = definition.call0(py).ok()?.extract(py).ok();
                    match res {
                        Some(x) => {
                            let out: CircuitData =
                                x.getattr(py, intern!(py, "data")).ok()?.extract(py).ok()?;
                            Some(out)
                        }
                        None => None,
                    }
                }
                Err(_) => None,
            }
        })
    }
    fn standard_gate(&self) -> Option<StandardGate> {
        None
    }

    fn directive(&self) -> bool {
        Python::with_gil(|py| -> bool {
            match self.instruction.getattr(py, intern!(py, "_directive")) {
                Ok(directive) => {
                    let res: bool = directive.extract(py).unwrap();
                    res
                }
                Err(_) => false,
            }
        })
    }
}

/// This class is used to wrap a Python side Gate that is not in the standard library
#[derive(Clone, Debug)]
// We bit-pack pointers to this, so having a known alignment even on 32-bit systems is good.
#[repr(align(8))]
pub struct PyGate {
    pub qubits: u32,
    pub clbits: u32,
    pub params: u32,
    pub op_name: String,
    pub gate: PyObject,
}

impl Operation for PyGate {
    fn name(&self) -> &str {
        self.op_name.as_str()
    }
    fn num_qubits(&self) -> u32 {
        self.qubits
    }
    fn num_clbits(&self) -> u32 {
        self.clbits
    }
    fn num_params(&self) -> u32 {
        self.params
    }
    fn control_flow(&self) -> bool {
        false
    }
    fn matrix(&self, _params: &[Param]) -> Option<Array2<Complex64>> {
        Python::with_gil(|py| -> Option<Array2<Complex64>> {
            match self.gate.getattr(py, intern!(py, "to_matrix")) {
                Ok(to_matrix) => {
                    let res: Option<PyObject> = to_matrix.call0(py).ok()?.extract(py).ok();
                    match res {
                        Some(x) => {
                            let array: PyReadonlyArray2<Complex64> = x.extract(py).ok()?;
                            Some(array.as_array().to_owned())
                        }
                        None => None,
                    }
                }
                Err(_) => None,
            }
        })
    }
    fn definition(&self, _params: &[Param]) -> Option<CircuitData> {
        Python::with_gil(|py| -> Option<CircuitData> {
            match self.gate.getattr(py, intern!(py, "definition")) {
                Ok(definition) => {
                    let res: Option<PyObject> = definition.call0(py).ok()?.extract(py).ok();
                    match res {
                        Some(x) => {
                            let out: CircuitData =
                                x.getattr(py, intern!(py, "data")).ok()?.extract(py).ok()?;
                            Some(out)
                        }
                        None => None,
                    }
                }
                Err(_) => None,
            }
        })
    }
    fn standard_gate(&self) -> Option<StandardGate> {
        Python::with_gil(|py| -> Option<StandardGate> {
            match self.gate.getattr(py, intern!(py, "_standard_gate")) {
                Ok(stdgate) => match stdgate.extract(py) {
                    Ok(out_gate) => out_gate,
                    Err(_) => None,
                },
                Err(_) => None,
            }
        })
    }
    fn directive(&self) -> bool {
        false
    }
}

/// This class is used to wrap a Python side Operation that is not in the standard library
#[derive(Clone, Debug)]
// We bit-pack pointers to this, so having a known alignment even on 32-bit systems is good.
#[repr(align(8))]
pub struct PyOperation {
    pub qubits: u32,
    pub clbits: u32,
    pub params: u32,
    pub op_name: String,
    pub operation: PyObject,
}

impl Operation for PyOperation {
    fn name(&self) -> &str {
        self.op_name.as_str()
    }
    fn num_qubits(&self) -> u32 {
        self.qubits
    }
    fn num_clbits(&self) -> u32 {
        self.clbits
    }
    fn num_params(&self) -> u32 {
        self.params
    }
    fn control_flow(&self) -> bool {
        false
    }
    fn matrix(&self, _params: &[Param]) -> Option<Array2<Complex64>> {
        None
    }
    fn definition(&self, _params: &[Param]) -> Option<CircuitData> {
        None
    }
    fn standard_gate(&self) -> Option<StandardGate> {
        None
    }

    fn directive(&self) -> bool {
        Python::with_gil(|py| -> bool {
            match self.operation.getattr(py, intern!(py, "_directive")) {
                Ok(directive) => {
                    let res: bool = directive.extract(py).unwrap();
                    res
                }
                Err(_) => false,
            }
        })
    }
}
