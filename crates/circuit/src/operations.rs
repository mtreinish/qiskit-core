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

use crate::circuit_data::CircuitData;
use crate::gate_matrix;
use ndarray::{aview2, Array2};
use num_complex::Complex64;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::{intern, IntoPy, Python};
use smallvec::SmallVec;

/// Valid types for OperationType
#[derive(FromPyObject, Clone, Debug)]
pub enum OperationType {
    Standard(StandardGate),
    Instruction(PyInstruction),
    Gate(PyGate),
    Operation(PyOperation),
}

impl Operation for OperationType {
    fn name(&self) -> &str {
        match self {
            Self::Standard(op) => op.name(),
            Self::Gate(op) => op.name(),
            Self::Instruction(op) => op.name(),
            Self::Operation(op) => op.name(),
        }
    }

    fn num_qubits(&self) -> u32 {
        match self {
            Self::Standard(op) => op.num_qubits(),
            Self::Gate(op) => op.num_qubits(),
            Self::Instruction(op) => op.num_qubits(),
            Self::Operation(op) => op.num_qubits(),
        }
    }
    fn num_clbits(&self) -> u32 {
        match self {
            Self::Standard(op) => op.num_clbits(),
            Self::Gate(op) => op.num_clbits(),
            Self::Instruction(op) => op.num_clbits(),
            Self::Operation(op) => op.num_clbits(),
        }
    }

    fn num_params(&self) -> u32 {
        match self {
            Self::Standard(op) => op.num_params(),
            Self::Gate(op) => op.num_params(),
            Self::Instruction(op) => op.num_params(),
            Self::Operation(op) => op.num_params(),
        }
    }
    fn matrix(&self, params: Option<SmallVec<[Param; 3]>>) -> Option<Array2<Complex64>> {
        match self {
            Self::Standard(op) => op.matrix(params),
            Self::Gate(op) => op.matrix(params),
            Self::Instruction(op) => op.matrix(params),
            Self::Operation(op) => op.matrix(params),
        }
    }

    fn control_flow(&self) -> bool {
        match self {
            Self::Standard(op) => op.control_flow(),
            Self::Gate(op) => op.control_flow(),
            Self::Instruction(op) => op.control_flow(),
            Self::Operation(op) => op.control_flow(),
        }
    }

    fn definition(&self, params: Option<SmallVec<[Param; 3]>>) -> Option<CircuitData> {
        match self {
            Self::Standard(op) => op.definition(params),
            Self::Gate(op) => op.definition(params),
            Self::Instruction(op) => op.definition(params),
            Self::Operation(op) => op.definition(params),
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
    fn matrix(&self, params: Option<SmallVec<[Param; 3]>>) -> Option<Array2<Complex64>>;
    fn definition(&self, params: Option<SmallVec<[Param; 3]>>) -> Option<CircuitData>;
    // fn control_flow_body(&self) -> Option<CircuitData>;
}

#[derive(FromPyObject, Clone, Debug)]
pub enum Param {
    Float(f64),
    ParameterExpression(PyObject),
}

impl IntoPy<PyObject> for Param {
    fn into_py(self, py: Python) -> PyObject {
        match &self {
            Self::Float(val) => val.to_object(py),
            Self::ParameterExpression(val) => val.clone_ref(py),
        }
    }
}

impl ToPyObject for Param {
    fn to_object(&self, py: Python) -> PyObject {
        match self {
            Self::Float(val) => val.to_object(py),
            Self::ParameterExpression(val) => val.clone_ref(py),
        }
    }
}

#[derive(Clone, Debug, Copy, Eq, PartialEq, Hash)]
#[pyclass]
pub enum StandardGate {
    // Pauli Gates
    ZGate,
    YGate,
    XGate,
    // Controlled Pauli Gates
    CZGate,
    CYGate,
    CXGate,
    CCXGate,
    RXGate,
    RYGate,
    RZGate,
    ECRGate,
    SwapGate,
    SXGate,
    GlobalPhaseGate,
    IGate,
    HGate,
    PhaseGate,
}

#[pymethods]
impl StandardGate {
    pub fn copy(&self) -> Self {
        *self
    }
}

// This must be kept up-to-date with `StandardGate` when adding or removing
// gates from the enum
//
// Remove this when std::mem::variant_count() is stabilized (see
// https://github.com/rust-lang/rust/issues/73662 )
pub const STANDARD_GATE_SIZE: usize = 17;

impl Operation for StandardGate {
    fn name(&self) -> &str {
        match self {
            Self::ZGate => "z",
            Self::YGate => "y",
            Self::XGate => "x",
            Self::CZGate => "cz",
            Self::CYGate => "cy",
            Self::CXGate => "cx",
            Self::CCXGate => "ccx",
            Self::RXGate => "rx",
            Self::RYGate => "ry",
            Self::RZGate => "rz",
            Self::ECRGate => "ecr",
            Self::SwapGate => "swap",
            Self::SXGate => "sx",
            Self::GlobalPhaseGate => "global_phase",
            Self::IGate => "i",
            Self::HGate => "h",
            Self::PhaseGate => "p",
        }
    }

    fn num_qubits(&self) -> u32 {
        match self {
            Self::ZGate => 1,
            Self::YGate => 1,
            Self::XGate => 1,
            Self::CZGate => 2,
            Self::CYGate => 2,
            Self::CXGate => 2,
            Self::CCXGate => 3,
            Self::RXGate => 1,
            Self::RYGate => 1,
            Self::RZGate => 1,
            Self::ECRGate => 2,
            Self::SwapGate => 2,
            Self::SXGate => 1,
            Self::GlobalPhaseGate => 0,
            Self::IGate => 1,
            Self::HGate => 1,
            Self::PhaseGate => 1,
        }
    }

    fn num_params(&self) -> u32 {
        match self {
            Self::ZGate => 0,
            Self::YGate => 0,
            Self::XGate => 0,
            Self::CZGate => 0,
            Self::CYGate => 0,
            Self::CXGate => 0,
            Self::CCXGate => 0,
            Self::RXGate => 1,
            Self::RYGate => 1,
            Self::RZGate => 1,
            Self::ECRGate => 0,
            Self::SwapGate => 0,
            Self::SXGate => 0,
            Self::GlobalPhaseGate => 1,
            Self::IGate => 0,
            Self::HGate => 0,
            Self::PhaseGate => 1,
        }
    }

    fn num_clbits(&self) -> u32 {
        0
    }

    fn control_flow(&self) -> bool {
        false
    }

    fn matrix(&self, params: Option<SmallVec<[Param; 3]>>) -> Option<Array2<Complex64>> {
        match self {
            Self::ZGate => Some(aview2(&gate_matrix::ZGATE).to_owned()),
            Self::YGate => Some(aview2(&gate_matrix::YGATE).to_owned()),
            Self::XGate => Some(aview2(&gate_matrix::XGATE).to_owned()),
            Self::CZGate => Some(aview2(&gate_matrix::CZGATE).to_owned()),
            Self::CYGate => Some(aview2(&gate_matrix::CYGATE).to_owned()),
            Self::CXGate => Some(aview2(&gate_matrix::CXGATE).to_owned()),
            Self::CCXGate => Some(aview2(&gate_matrix::CCXGATE).to_owned()),
            Self::RXGate => {
                let theta = &params.unwrap()[0];
                match theta {
                    Param::Float(theta) => Some(aview2(&gate_matrix::rx_gate(*theta)).to_owned()),
                    _ => None,
                }
            }
            Self::RYGate => {
                let theta = &params.unwrap()[0];
                match theta {
                    Param::Float(theta) => Some(aview2(&gate_matrix::ry_gate(*theta)).to_owned()),
                    _ => None,
                }
            }
            Self::RZGate => {
                let theta = &params.unwrap()[0];
                match theta {
                    Param::Float(theta) => Some(aview2(&gate_matrix::rz_gate(*theta)).to_owned()),
                    _ => None,
                }
            }
            Self::ECRGate => Some(aview2(&gate_matrix::ECRGATE).to_owned()),
            Self::SwapGate => Some(aview2(&gate_matrix::SWAPGATE).to_owned()),
            Self::SXGate => Some(aview2(&gate_matrix::SXGATE).to_owned()),
            Self::GlobalPhaseGate => {
                let theta = &params.unwrap()[0];
                match theta {
                    Param::Float(theta) => {
                        Some(aview2(&gate_matrix::global_phase_gate(*theta)).to_owned())
                    }
                    _ => None,
                }
            }
            Self::IGate => Some(aview2(&gate_matrix::ONE_QUBIT_IDENTITY).to_owned()),
            Self::HGate => Some(aview2(&gate_matrix::HGATE).to_owned()),
            Self::PhaseGate => {
                let theta = &params.unwrap()[0];
                match theta {
                    Param::Float(theta) => {
                        Some(aview2(&gate_matrix::phase_gate(*theta)).to_owned())
                    }
                    _ => None,
                }
            }
        }
    }

    fn definition(&self, _params: Option<SmallVec<[Param; 3]>>) -> Option<CircuitData> {
        // TODO: Add definition for completeness. This shouldn't be necessary in practice
        // though because nothing will rely on this in practice.
        match self {
            Self::ZGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::build_new_from(
                        py,
                        1,
                        0,
                        &[(
                            OperationType::Standard(Self::PhaseGate),
                            &[Param::Float(std::f64::consts::PI)],
                            &[0],
                        )],
                    )
                    .expect("TIS_VALID"),
                )
            }),
            _ => None,
        }
    }
}

/// This class is used to wrap a Python side Instruction that is not in the standard library
#[derive(Clone, Debug)]
#[pyclass]
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
    fn matrix(&self, _params: Option<SmallVec<[Param; 3]>>) -> Option<Array2<Complex64>> {
        None
    }
    fn definition(&self, _params: Option<SmallVec<[Param; 3]>>) -> Option<CircuitData> {
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
}

/// This class is used to wrap a Python side Gate that is not in the standard library
#[derive(Clone, Debug)]
#[pyclass]
pub struct PyGate {
    pub qubits: u32,
    pub clbits: u32,
    pub params: u32,
    pub op_name: String,
    pub gate: PyObject,
}

#[pymethods]
impl PyGate {
    #[new]
    fn new(op_name: String, qubits: u32, clbits: u32, params: u32, gate: PyObject) -> Self {
        PyGate {
            qubits,
            clbits,
            params,
            op_name,
            gate,
        }
    }
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
    fn matrix(&self, _params: Option<SmallVec<[Param; 3]>>) -> Option<Array2<Complex64>> {
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
    fn definition(&self, _params: Option<SmallVec<[Param; 3]>>) -> Option<CircuitData> {
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
}

/// This class is used to wrap a Python side Operation that is not in the standard library
#[derive(Clone, Debug)]
#[pyclass]
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
    fn matrix(&self, _params: Option<SmallVec<[Param; 3]>>) -> Option<Array2<Complex64>> {
        None
    }
    fn definition(&self, _params: Option<SmallVec<[Param; 3]>>) -> Option<CircuitData> {
        None
    }
}
