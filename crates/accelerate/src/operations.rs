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

use crate::gate_matrix;
use ndarray::{aview2, Array2};
use num_complex::Complex64;
use pyo3::prelude::*;
use pyo3::Python;
use smallvec::SmallVec;
use numpy::IntoPyArray;

/// Trait for generic circuit operations these define the common attributes
/// needed for something to be addable to the circuit struct
pub trait Operation {
    fn name(&self) -> &str;
    fn num_qubits(&self) -> u32;
    fn num_clbits(&self) -> u32;
    fn num_params(&self) -> u32;
    fn control_flow(&self) -> bool;
}

pub enum Param {
    Float(f64),
    Parameter(String),
    ParameterExpression(PyObject),
}

/// Gate is a specialized operation that represents a unitary operation
pub trait Gate: Operation {
    fn matrix(&self, params: Option<SmallVec<[Param; 3]>>) -> Option<Array2<Complex64>>;
//    fn control(&self, ctrl_state: u32) -> impl Gate;
//    fn inverse(&self) -> impl Gate;
//    fn power(&self) -> impl Gate;
}

#[derive(Clone)]
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
}

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
        }
    }

    fn num_clbits(&self) -> u32 {
        0
    }

    fn control_flow(&self) -> bool {
        false
    }
}

impl Gate for StandardGate {
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
                    Param::Float(theta) => Some(aview2(&gate_matrix::rx_gate(theta.clone())).to_owned()),
                    _ => None,
                }
            }
            Self::RYGate => {
                let theta = &params.unwrap()[0];
                match theta {
                    Param::Float(theta) => Some(aview2(&gate_matrix::ry_gate(theta.clone())).to_owned()),
                    _ => None,
                }
            }
            Self::RZGate => {
                let theta = &params.unwrap()[0];
                match theta {
                    Param::Float(theta) => Some(aview2(&gate_matrix::rz_gate(theta.clone())).to_owned()),
                    _ => None,
                }
            }
            Self::ECRGate => Some(aview2(&gate_matrix::ECRGATE).to_owned()),
            Self::SwapGate => Some(aview2(&gate_matrix::SWAPGATE).to_owned()),
            Self::SXGate => Some(aview2(&gate_matrix::SXGATE).to_owned()),
            Self::GlobalPhaseGate => {
                let theta = &params.unwrap()[0];
                match theta {
                    Param::Float(theta) => Some(aview2(&gate_matrix::global_phase_gate(theta.clone())).to_owned()),
                    _ => None,
                }
            }
            Self::IGate => Some(aview2(&gate_matrix::ONE_QUBIT_IDENTITY).to_owned()),
            Self::HGate => Some(aview2(&gate_matrix::HGATE).to_owned()),
        }
    }

//    fn control(&self, ctrl_state: u32) -> impl Gate {
//        match self {
//            Self::ZGate => Self::CZGate,
//            Self::YGate => Self::CYGate,
//            Self::XGate => Self::CXGate,
//            Self::CZGate => todo!("Add variants"),
//            Self::CYGate => todo!("Add variants"),
//            Self::CXGate => Self::CCXGate,
//            Self::CCXGate => todo!("TODO: Add mcx function"),
//            Self::RXGate => todo!("TODO: Add crx variant"),
//            Self::RYGate => todo!("TODO: Add cry variant"),
//            Self::RZGate => todo!("TODO: Add crz variant"),
//            Self::ECRGate => todo!("TODO: Add arbitrary control function"),
//            Self::SwapGate => todo!("TODO: Add cswap variant"),
//            Self::SXGate => todo!("TODO: Add csx variant"),
//            Self::GlobalPhaseGate => todo!("TODO: Add arbitrary control function"),
//            Self::IGate => todo!("TODO: Add arbitrary control function"),
//        }
//    }
//
//    fn inverse(&self) -> impl Gate {
//        todo!("inverse");
//    }
//
//    fn power(&self) -> impl Gate {
//        todo!("power");
//    }
}

#[pyclass(subclass)]
pub struct PyStandardGate {
    pub gate: StandardGate,
}

#[pymethods]
impl PyStandardGate {
    #[new]
    fn new(gate: StandardGate) -> Self {
        PyStandardGate { gate }
    }

    #[getter]
    fn num_qubits(&self) -> u32 {
        self.gate.num_qubits()
    }

    #[getter]
    fn num_clbits(&self) -> u32 {
        self.gate.num_qubits()
    }

    #[getter]
    fn name(&self) -> &str {
        self.gate.name()
    }

    fn to_matrix(&self, py: Python, params: Option<SmallVec<[f64; 3]>>) -> PyObject {
        let params = params.map(|param_vec| param_vec.into_iter().map(|x| Param::Float(x)).collect());
        let matrix = self.gate.matrix(params);
        match matrix {
            Some(matrix) => matrix.into_pyarray(py).into(),
            None => py.None(),
        }
    }
}

impl Operation for PyStandardGate {
    fn num_qubits(&self) -> u32 {
        self.gate.num_qubits()
    }

    fn num_clbits(&self) -> u32 {
        self.gate.num_qubits()
    }

    fn name(&self) -> &str {
        self.gate.name()
    }

    fn num_params(&self) -> u32 {
        self.gate.num_params()
    }

    fn control_flow(&self) -> bool {
        false
    }
}

impl Gate for PyStandardGate {
    fn matrix(&self, param: Option<SmallVec<[Param; 3]>>) -> Option<Array2<Complex64>> {
        self.gate.matrix(param)
    }
}
