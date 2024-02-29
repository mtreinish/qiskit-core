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

use std::num::NonZeroUsize;

use lru::LruCache;
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::gc::PyVisit;
use pyo3::prelude::*;
use pyo3::PyTraverseError;

#[derive(Clone)]
#[pyclass(mapping)]
struct LRUDict {
    inner_dict: LruCache<isize, PyObject>,
}

#[pymethods]
impl LRUDict {
    #[new]
    pub fn new(maxsize: usize) -> PyResult<Self> {
        let max_size = match NonZeroUsize::new(maxsize) {
            Some(size) => size,
            None => return Err(PyValueError::new_err("maxsize must be non-zero")),
        };
        Ok(LRUDict {
            inner_dict: LruCache::new(max_size),
        })
    }

    fn get(&mut self, py: Python, key: &PyAny, default: Option<PyObject>) -> PyResult<PyObject> {
        let hash = key.hash()?;
        match self.inner_dict.get(&hash) {
            Some(obj) => Ok(obj.clone_ref(py)),
            None => match default {
                Some(default) => Ok(default.clone_ref(py)),
                None => Ok(py.None()),
            }
        }
    }

    fn __len__(&self) -> usize {
        self.inner_dict.len()
    }

    fn __contains__(&self, key: &PyAny) -> PyResult<bool> {
        let hash = key.hash()?;
        Ok(self.inner_dict.contains(&hash))
    }

    fn __getitem__(&mut self, key: &PyAny) -> PyResult<&PyObject> {
        let hash = key.hash()?;
        match self.inner_dict.get(&hash) {
            Some(obj) => Ok(obj),
            None => Err(PyKeyError::new_err(format!("{} not found", key.str()?))),
        }
    }

    fn __setitem__(&mut self, key: &PyAny, value: PyObject) -> PyResult<()> {
        let hash = key.hash()?;
        self.inner_dict.push(hash, value);
        Ok(())
    }

    fn __traverse__(&self, visit: PyVisit) -> Result<(), PyTraverseError> {
        for (_hash, obj) in &self.inner_dict {
            visit.call(obj)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.inner_dict.clear();
    }
}

#[pymodule]
pub fn lru_dict(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<LRUDict>()?;
    Ok(())
}
