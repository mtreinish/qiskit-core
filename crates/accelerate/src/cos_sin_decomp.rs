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

// Based on the Cosine Sine decomposition TKET uses for unitary matrices:
//
// https://github.com/CQCL/tket/blob/a61bbac1c07487521acd8da463023f3c85c799e8/tket/src/Utils/CosSinDecomposition.cpp

use numpy::IntoPyArray;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

use faer_ext::{IntoFaerComplex, IntoNdarrayComplex};
use ndarray::prelude::*;
use num_complex::{Complex64, ComplexFloat};

type CosSinDecompReturn = (
    Array2<Complex64>,
    Array2<Complex64>,
    Array2<Complex64>,
    Array2<Complex64>,
    Array2<Complex64>,
    Array2<f64>,
);

/// Comute the cosine-sin decomposition of a Unitary matrix
///
/// # Args
///
/// - `u`: The input unitary to decompose
///
/// # Returns
///
/// The tuple (l0, l1, r0, r1, c, s) which represents the decomposition
///
/// [l0   ] [c -s] [r0   ]
/// [   l1] [s  c] [   r1]
///
/// where l0, l1, r0 and r1 are unitaries of equal size, c and s are diagonal
/// matrices with non-negative entries, the diagonal entries of c are in
/// non-decreasimg order, and
///
/// c^2 + s^2 = I
pub fn cos_sin_decomposition(u: ArrayView2<Complex64>) -> CosSinDecompReturn {
    let n = u.shape()[0] / 2;
    // Upper left corner
    let u00 = u.slice(s![0..n, 0..n]);
    // Upper right corner
    let u01 = u.slice(s![n.., 0..n]);
    // Bottom left corner
    let u10 = u.slice(s![0..n, n..]);
    // bottom right corner
    let u11 = u.slice(s![n.., n..]);
    let svd = u00.into_faer_complex().svd();
    // Faer's reverse_rows() breaks the ndarray conversion trait and causes
    // a panic. After reversing the rows create an owned copy of the reversed
    // matrix to avoid the panic.
    let u_rev = svd.u().reverse_rows().to_owned();
    let l0 = u_rev.as_ref().into_ndarray_complex();
    let r0_dag = svd.v().reverse_rows();
    let mut c_vec = svd
        .s_diagonal()
        .iter()
        .map(|x| x.to_num_complex())
        .collect::<Vec<Complex64>>();
    c_vec.reverse();
    let c = Array2::from_diag(&Array1::from_vec(c_vec));
    let r0_rev = svd.v().reverse_rows().adjoint().to_owned();
    let r0 = r0_rev.as_ref().into_ndarray_complex();
    let qr = (u10.into_faer_complex() * r0_dag).qr();
    let q = qr.compute_q();
    let mut l1 = q.as_ref().into_ndarray_complex().to_owned();
    let r = qr.compute_r();
    let mut s = r.as_ref().into_ndarray_complex().to_owned();
    // Now u10 r0* = l1 S; l1 is unitary, and S is upper triangular.
    //
    // Claim: S is diagonal.
    // Proof: Since u is unitary, we have
    //     I = u00* u00 + u10* u10
    //       = (l0 c r0)* (l0 c r0) + (l1 S r0)* (l1 S r0)
    //       = r0* c l0* l0 c r0 + r0* S* l1* l1 S r0
    //       = r0* c^2 r0 + r0* S* S r0
    //       = r0* (c^2 + S* S) r0
    // So I = c^2 + (S* S), so (S* S) = I - c^2 is a diagonal matrix with non-
    // increasing entries in the range [0,1). As S is upper triangular, this
    // implies that S must be diagonal. (Proof by induction over the dimension n
    // of S: consider the two cases S_00 = 0 and S_00 != 0 and reduce to the n-1
    // case.)
    //
    // We want S to be real. This is not guaranteed, though since it is diagonal
    // it can be made so by adjusting l1.
    if s.iter().any(|x| x.im != 0.) {
        for j in 0..n {
            let z = s[[j, j]];
            let r = z.abs();
            if r > f64::EPSILON {
                let w = z.conj() / r;
                s[[j, j]] *= w;
                l1.column_mut(j).mapv_inplace(|x| x / w);
            }
        }
    }

    // Now s is real and diagonal, and c^2 + S^2 = I
    let mut s_real: Array2<f64> = s.mapv(|x| x.re);
    // Make all entries in s_real non-negative.
    for j in 0..n {
        let val = s_real.get_mut((j, j)).unwrap();
        if *val < 0. {
            *val = -*val;
            l1.column_mut(j).mapv_inplace(|x| -x);
        }
    }
    // Finally compute r1, being careful not to divide by small things.
    let mut r1 = Array2::zeros((n, n));
    let l0_adjoint = l0.t().mapv(|x| x.conj());
    let l1_adjoint = l1.t().mapv(|x| x.conj());
    for i in 0..n {
        let mut row = r1.row_mut(i);
        if s_real[[i, i]] > c[[i, i]].re {
            let mut new_row = l0_adjoint.dot(&u01);
            new_row.mapv_inplace(|x| -x / s[[i, i]]);
            row.iter_mut()
                .zip(new_row.into_iter())
                .for_each(|(x, y)| *x = y);
        } else {
            let mut new_row = l1_adjoint.dot(&u11);
            new_row.mapv_inplace(|x| x / s[[i, i]]);
            row.iter_mut()
                .zip(new_row.into_iter())
                .for_each(|(x, y)| *x = y);
        }
    }

    (l0.to_owned(), l1, r0.to_owned(), r1, c, s_real)
}

// TODO: Remove this function and all the python interface when QSD is ported to rust
#[pyfunction]
pub fn cossin<'py>(py: Python<'py>, u: PyReadonlyArray2<Complex64>) -> PyResult<Bound<'py, PyAny>> {
    let mat = u.as_array();
    let res = cos_sin_decomposition(mat);
    Ok((
        (res.0.into_pyarray(py), res.1.into_pyarray(py)),
        res.5.into_pyarray(py),
        (res.2.into_pyarray(py), res.3.into_pyarray(py)),
    )
        .into_pyobject(py)?
        .into_any())
}

pub fn cos_sin_decomp(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(cossin))?;
    Ok(())
}
