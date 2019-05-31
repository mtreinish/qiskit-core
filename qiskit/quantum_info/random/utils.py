# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""
Methods to create random unitaries, states, etc.
"""

import math
import numpy as np
from scipy.stats import unitary_group

from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info.operators import Operator
from qiskit.exceptions import QiskitError


# TODO: return a QuantumState object
def random_state(dim, seed=None):
    """
    Return a random quantum state from the uniform (Haar) measure on
    state space.

    Args:
        dim (int): the dim of the state space
        seed (int): Optional. To set a random seed.

    Returns:
        ndarray:  state(2**num) a random quantum state.
    """
    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.RandomState(seed)
    # Random array over interval (0, 1]
    x = rng.rand(dim)
    x += x == 0
    x = -np.log(x)
    sumx = sum(x)
    phases = rng.rand(dim)*2.0*np.pi
    return np.sqrt(x/sumx)*np.exp(1j*phases)


def random_unitary(dim, seed=None):
    """
    Return a random dim x dim unitary Operator from the Haar measure.

    Args:
        dim (int): the dim of the state space.
        seed (int): Optional. To set a random seed.

    Returns:
        Operator: (dim, dim) unitary operator.

    Raises:
        QiskitError: if dim is not a positive power of 2.
    """
    if seed is not None:
        np.random.seed(seed)
    if dim == 0 or not math.log2(dim).is_integer():
        raise QiskitError("Desired unitary dimension not a positive power of 2.")
    return Operator(unitary_group.rvs(dim))


# TODO: return a DensityMatrix object.
def random_density_matrix(length, rank=None, method='Hilbert-Schmidt', seed=None):
    """
    Generate a random density matrix rho.

    Args:
        length (int): the length of the density matrix.
        rank (int or None): the rank of the density matrix. The default
            value is full-rank.
        method (string): the method to use.
            'Hilbert-Schmidt': sample rho from the Hilbert-Schmidt metric.
            'Bures': sample rho from the Bures metric.
        seed (int): Optional. To set a random seed.
    Returns:
        ndarray: rho (length, length) a density matrix.
    Raises:
        QiskitError: if the method is not valid.
    """
    if method == 'Hilbert-Schmidt':
        return __random_density_hs(length, rank, seed)
    elif method == 'Bures':
        return __random_density_bures(length, rank, seed)
    else:
        raise QiskitError('Error: unrecognized method {}'.format(method))


def __ginibre_matrix(nrow, ncol=None, seed=None):
    """
    Return a normally distributed complex random matrix.

    Args:
        nrow (int): number of rows in output matrix.
        ncol (int): number of columns in output matrix.
        seed (int): Optional. To set a random seed.
    Returns:
        ndarray: A complex rectangular matrix where each real and imaginary
            entry is sampled from the normal distribution.
    """
    if ncol is None:
        ncol = nrow
    if seed is not None:
        np.random.seed(seed)
    G = np.random.normal(size=(nrow, ncol)) + \
        np.random.normal(size=(nrow, ncol)) * 1j
    return G


def __random_density_hs(N, rank=None, seed=None):
    """
    Generate a random density matrix from the Hilbert-Schmidt metric.

    Args:
        N (int): the length of the density matrix.
        rank (int or None): the rank of the density matrix. The default
            value is full-rank.
        seed (int): Optional. To set a random seed.
    Returns:
        ndarray: rho (N,N  a density matrix.
    """
    G = __ginibre_matrix(N, rank, seed)
    G = G.dot(G.conj().T)
    return G / np.trace(G)


def __random_density_bures(N, rank=None, seed=None):
    """
    Generate a random density matrix from the Bures metric.

    Args:
        N (int): the length of the density matrix.
        rank (int or None): the rank of the density matrix. The default
            value is full-rank.
        seed (int): Optional. To set a random seed.
    Returns:
        ndarray: rho (N,N) a density matrix.
    """
    P = np.eye(N) + random_unitary(N).data
    G = P.dot(__ginibre_matrix(N, rank, seed))
    G = G.dot(G.conj().T)
    return G / np.trace(G)


def _get_random_qubits(register, n_qubits, rng):
    qubits = []
    while len(qubits) < n_qubits:
        qubit = rng.choice(register, 1)[0]
        if qubit not in qubits:
            qubits.append(qubit)

    return qubits


def random_circuit(n_qubits, depth=20, seed=None):
    standard_gates_1q = ['z', 'y', 'x', 'h', 'iden', 's', 'sdg', 't', 'tdg',
                         'ch']
    param_1_gates_1q = ['rx', 'rz', 'ry', 'u1', 'u0']
    param_2_gates_1q = ['u2']
    param_3_gates_1q = ['u3']
    standard_gates_2q = ['cx', 'cy', 'cz', 'swap', 'ch']
    param_1_gates_2q = ['rzz', 'crz', 'cu1']
    param_3_gates_2q = ['cu3']
    standard_gates_3q = ['ccx', 'cswap']

    qr = QuantumRegister(n_qubits)
    qc = QuantumCircuit(qr)
    # Determine gates to be used
    valid_gates = [standard_gates_1q, param_1_gates_1q, param_2_gates_1q,
                   param_3_gates_1q]
    if n_qubits >= 2:
        valid_gates.append(standard_gates_2q)
        valid_gates.append(param_1_gates_2q)
        valid_gates.append(param_3_gates_2q)
    if n_qubits >= 3:
        valid_gates.append(standard_gates_3q)

    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.RandomState(seed)

    for i in range(depth):
        gate_list = rng.choice(valid_gates, 1)[0]
        gate = rng.choice(gate_list, 1)[0]
        if gate in standard_gates_3q:
            qubits = _get_random_qubits(qr, 3, rng)
            getattr(qc, gate)(*qubits)
        elif gate in standard_gates_2q:
            qubits = _get_random_qubits(qr, 2, rng)
            getattr(qc, gate)(*qubits)
        elif gate in standard_gates_1q:
            qubits = _get_random_qubits(qr, 1, rng)
            getattr(qc, gate)(*qubits)
        elif gate in param_1_gates_2q:
            qubits = _get_random_qubits(qr, 2, rng)
            func = getattr(qc, gate)
            param = rng.random_sample()
            func(param, *qubits)
        elif gate in param_3_gates_2q:
            qubits = _get_random_qubits(qr, 2, rng)
            func = getattr(qc, gate)
            params = [rng.random_sample() for _ in range(3)]
            func(*params, *qubits)
        elif gate in param_3_gates_1q:
            qubits = _get_random_qubits(qr, 1, rng)
            func = getattr(qc, gate)
            params = [rng.random_sample() for _ in range(3)]
            func(*params, *qubits)
        elif gate in param_2_gates_1q:
            qubits = _get_random_qubits(qr, 1, rng)
            func = getattr(qc, gate)
            params = [rng.random_sample() for _ in range(2)]
            func(*params, *qubits)
        else:
            qubits = _get_random_qubits(qr, 1, rng)
            func = getattr(qc, gate)
            param = rng.random_sample()
            func(param, *qubits)
    return qc
