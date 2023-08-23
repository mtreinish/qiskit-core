// This code is part of Qiskit.
//
// (C) Copyright IBM 2023
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;

/// A container for required swaps before a gate qubit
#[pyclass(module = "qiskit._accelerate.sabre_swap", sequence)]
#[derive(Clone, Debug)]
pub struct SabreGateOrder {
    pub order: Vec<(Option<usize>, Vec<usize>)>,
}

#[pymethods]
impl SabreGateOrder {
    #[new]
    pub fn new(size: usize) -> Self {
        SabreGateOrder {
            order: Vec::with_capacity(size),
        }
    }

    // Sequence Protocol below
    pub fn __len__(&self) -> usize {
        self.order.len()
    }

    pub fn __getitem__(&self, object: usize) -> PyResult<(Option<usize>, Vec<usize>)> {
        match self.order.get(object) {
            Some(val) => Ok(val.clone()),
            None => Err(PyIndexError::new_err(format!(
                "Node index {object} not in swap mapping",
            ))),
        }
    }

    pub fn __str__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.order))
    }
}
