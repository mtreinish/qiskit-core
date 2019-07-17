# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import copy
import logging

import numpy as np
from qiskit.quantum_info import Pauli
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.qasm import pi

from qiskit.aqua import AquaError

logger = logging.getLogger(__name__)


class WeightedPauli(Pauli):

    def __init__(self, z=None, x=None, label=None, weight=0.0):
        super().__init__(z, x, label)
        self._weight = weight

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, new_value):
        self._weight = new_value


def measure_pauli_z(data, pauli):
    """
    Appropriate post-rotations on the state are assumed.

    Args:
        data (dict): a dictionary of the form data = {'00000': 10} ({str: int})
        pauli (Pauli): a Pauli object

    Returns:
        float: Expected value of paulis given data
    """

    observable = 0
    tot = sum(data.values())
    for key in data:
        value = 1
        for j in range(pauli.numberofqubits):
            if ((pauli.x[j] or pauli.z[j]) and
                    key[pauli.numberofqubits - j - 1] == '1'):
                value = -value
        # print(key, data[key])
        observable = observable + value * data[key] / tot
    return observable

    # observable = 0.0
    # num_shots = sum(data.values())
    # p_z_or_x = np.logical_or(pauli.z, pauli.x)
    # for key, value in data.items():
    #     bitstr = np.asarray(list(key))[::-1].astype(np.bool)
    #     # pylint: disable=no-member
    #     sign = -1.0 if np.logical_xor.reduce(np.logical_and(bitstr, p_z_or_x)) else 1.0
    #     observable += sign * value
    # observable /= num_shots
    # return observable


def covariance(data, pauli_1, pauli_2, avg_1, avg_2):
    """
    Compute the covariance matrix element between two
    Paulis, given the measurement outcome.
    Appropriate post-rotations on the state are assumed.

    Args:
        data (dict): a dictionary of the form data = {'00000': 10} ({str:int})
        pauli_1 (Pauli): a Pauli class member
        pauli_2 (Pauli): a Pauli class member
        avg_1 (float): expectation value of pauli_1 on `data`
        avg_2 (float): expectation value of pauli_2 on `data`

    Returns:
        float: the element of the covariance matrix between two Paulis
    """
    cov = 0.0
    num_shots = sum(data.values())

    if num_shots == 1:
        return cov

    p1_z_or_x = np.logical_or(pauli_1.z, pauli_1.x)
    p2_z_or_x = np.logical_or(pauli_2.z, pauli_2.x)
    for key, value in data.items():
        bitstr = np.asarray(list(key))[::-1].astype(np.bool)
        # pylint: disable=no-member
        sign_1 = -1.0 if np.logical_xor.reduce(np.logical_and(bitstr, p1_z_or_x)) else 1.0
        sign_2 = -1.0 if np.logical_xor.reduce(np.logical_and(bitstr, p2_z_or_x)) else 1.0
        cov += (sign_1 - avg_1) * (sign_2 - avg_2) * value
    cov /= (num_shots - 1)
    return cov


def row_echelon_F2(matrix_in):
    """
    Computes the row Echelon form of a binary matrix on the binary
    finite field

    Args:
        matrix_in (numpy.ndarray): binary matrix

    Returns:
        numpy.ndarray : matrix_in in Echelon row form
    """
    size = matrix_in.shape

    for i in range(size[0]):
        pivot_index = 0
        for j in range(size[1]):
            if matrix_in[i, j] == 1:
                pivot_index = j
                break
        for k in range(size[0]):
            if k != i and matrix_in[k, pivot_index] == 1:
                matrix_in[k, :] = np.mod(matrix_in[k, :] + matrix_in[i, :], 2)

    matrix_out_temp = copy.deepcopy(matrix_in)
    indices = []
    matrix_out = np.zeros(size)

    for i in range(size[0] - 1):
        if np.array_equal(matrix_out_temp[i, :], np.zeros(size[1])):
            indices.append(i)
    for row in np.sort(indices)[::-1]:
        matrix_out_temp = np.delete(matrix_out_temp, (row), axis=0)

    matrix_out[0:size[0] - len(indices), :] = matrix_out_temp
    matrix_out = matrix_out.astype(int)

    return matrix_out


def kernel_F2(matrix_in):
    """
    Computes the kernel of a binary matrix on the binary finite field

    Args:
        matrix_in (numpy.ndarray): binary matrix

    Returns:
        [numpy.ndarray]: the list of kernel vectors
    """
    size = matrix_in.shape
    kernel = []
    matrix_in_id = np.vstack((matrix_in, np.identity(size[1])))
    matrix_in_id_ech = (row_echelon_F2(matrix_in_id.transpose())).transpose()

    for col in range(size[1]):
        if (np.array_equal(matrix_in_id_ech[0:size[0], col], np.zeros(size[0])) and not
                np.array_equal(matrix_in_id_ech[size[0]:, col], np.zeros(size[1]))):
            kernel.append(matrix_in_id_ech[size[0]:, col])

    return kernel


def suzuki_expansion_slice_pauli_list(pauli_list, lam_coef, expansion_order):
    """
    Similar to _suzuki_expansion_slice_matrix, with the difference that this method
    computes the list of pauli terms for a single slice of the suzuki expansion,
    which can then be fed to construct_evolution_circuit to build the QuantumCircuit.
    #TODO: polish the docstring
    Args:
        pauli_list (list[list[complex, Pauli]]): the weighted pauli list??
        lam_coef (float): ???
        expansion_order (int): ???
    """
    if expansion_order == 1:
        half = [[lam_coef / 2 * c, p] for c, p in pauli_list]
        return half + list(reversed(half))
    else:
        pk = (4 - 4 ** (1 / (2 * expansion_order - 1))) ** -1
        side_base = suzuki_expansion_slice_pauli_list(
            pauli_list,
            lam_coef * pk,
            expansion_order - 1
        )
        side = side_base * 2
        middle = suzuki_expansion_slice_pauli_list(
            pauli_list,
            lam_coef * (1 - 4 * pk),
            expansion_order - 1
        )
        return side + middle + side


def check_commutativity(op_1, op_2, anti=False):
    """
    Check the commutativity between two operators.

    Args:
        op_1 (WeightedPauliOperator):
        op_2 (WeightedPauliOperator):
        anti (bool): if True, check anti-commutativity, otherwise check commutativity.

    Returns:
        bool: whether or not two operators are commuted or anti-commuted.
    """
    com = op_1 * op_2 - op_2 * op_1 if not anti else op_1 * op_2 + op_2 * op_1
    com.remove_zero_weights()
    return True if com.is_empty() else False


def evolution_instruction(pauli_list, evo_time, num_time_slices,
                          controlled=False, power=1,
                          use_basis_gates=True, shallow_slicing=False):
    """
    Construct the evolution circuit according to the supplied specification.

    Args:
        pauli_list (list([[complex, Pauli]])): The list of pauli terms corresponding to a single time slice to be evolved
        evo_time (complex | float): The evolution time
        num_time_slices (int): The number of time slices for the expansion
        controlled (bool, optional): Controlled circuit or not
        power (int, optional): The power to which the unitary operator is to be raised
        use_basis_gates (bool, optional): boolean flag for indicating only using basis gates when building circuit.
        shallow_slicing (bool, optional): boolean flag for indicating using shallow qc.data reference repetition for slicing

    Returns:
        InstructionSet: The InstructionSet corresponding to specified evolution.
    """

    if not isinstance(power, (int, np.int)) or power < 1:
        raise AquaError("power must be an integer and greater or equal to 1.")

    state_registers = QuantumRegister(pauli_list[0][1].numberofqubits)
    if controlled:
        ancillary_registers = QuantumRegister(1)
        qc_slice = QuantumCircuit(state_registers, ancillary_registers, name='Controlled-Evolution^{}'.format(power))
    else:
        qc_slice = QuantumCircuit(state_registers, name='Evolution^{}'.format(power))

    # for each pauli [IXYZ]+, record the list of qubit pairs needing CX's
    cnot_qubit_pairs = [None] * len(pauli_list)
    # for each pauli [IXYZ]+, record the highest index of the nontrivial pauli gate (X,Y, or Z)
    top_XYZ_pauli_indices = [-1] * len(pauli_list)

    for pauli_idx, pauli in enumerate(reversed(pauli_list)):
        n_qubits = pauli[1].numberofqubits
        # changes bases if necessary
        nontrivial_pauli_indices = []
        for qubit_idx in range(n_qubits):
            # pauli I
            if not pauli[1].z[qubit_idx] and not pauli[1].x[qubit_idx]:
                continue

            if cnot_qubit_pairs[pauli_idx] is None:
                nontrivial_pauli_indices.append(qubit_idx)

            if pauli[1].x[qubit_idx]:
                # pauli X
                if not pauli[1].z[qubit_idx]:
                    if use_basis_gates:
                        qc_slice.u2(0.0, pi, state_registers[qubit_idx])
                    else:
                        qc_slice.h(state_registers[qubit_idx])
                # pauli Y
                elif pauli[1].z[qubit_idx]:
                    if use_basis_gates:
                        qc_slice.u3(pi / 2, -pi / 2, pi / 2, state_registers[qubit_idx])
                    else:
                        qc_slice.rx(pi / 2, state_registers[qubit_idx])
            # pauli Z
            elif pauli[1].z[qubit_idx] and not pauli[1].x[qubit_idx]:
                pass
            else:
                raise ValueError('Unrecognized pauli: {}'.format(pauli[1]))

        if len(nontrivial_pauli_indices) > 0:
            top_XYZ_pauli_indices[pauli_idx] = nontrivial_pauli_indices[-1]

        # insert lhs cnot gates
        if cnot_qubit_pairs[pauli_idx] is None:
            cnot_qubit_pairs[pauli_idx] = list(zip(
                sorted(nontrivial_pauli_indices)[:-1],
                sorted(nontrivial_pauli_indices)[1:]
            ))

        for pair in cnot_qubit_pairs[pauli_idx]:
            qc_slice.cx(state_registers[pair[0]], state_registers[pair[1]])

        # insert Rz gate
        if top_XYZ_pauli_indices[pauli_idx] >= 0:
            lam = (2.0 * pauli[0] * evo_time / num_time_slices).real
            if not controlled:

                if use_basis_gates:
                    qc_slice.u1(lam, state_registers[top_XYZ_pauli_indices[pauli_idx]])
                else:
                    qc_slice.rz(lam, state_registers[top_XYZ_pauli_indices[pauli_idx]])
            else:
                # unitary_power = (2 ** ctl_idx) if unitary_power is None else unitary_power
                if use_basis_gates:
                    qc_slice.u1(lam / 2, state_registers[top_XYZ_pauli_indices[pauli_idx]])
                    qc_slice.cx(ancillary_registers[0], state_registers[top_XYZ_pauli_indices[pauli_idx]])
                    qc_slice.u1(-lam / 2, state_registers[top_XYZ_pauli_indices[pauli_idx]])
                    qc_slice.cx(ancillary_registers[0], state_registers[top_XYZ_pauli_indices[pauli_idx]])
                else:
                    qc_slice.crz(lam, ancillary_registers[0],
                                 state_registers[top_XYZ_pauli_indices[pauli_idx]])

        # insert rhs cnot gates
        for pair in reversed(cnot_qubit_pairs[pauli_idx]):
            qc_slice.cx(state_registers[pair[0]], state_registers[pair[1]])

        # revert bases if necessary
        for qubit_idx in range(n_qubits):
            if pauli[1].x[qubit_idx]:
                # pauli X
                if not pauli[1].z[qubit_idx]:
                    if use_basis_gates:
                        qc_slice.u2(0.0, pi, state_registers[qubit_idx])
                    else:
                        qc_slice.h(state_registers[qubit_idx])
                # pauli Y
                elif pauli[1].z[qubit_idx]:
                    if use_basis_gates:
                        qc_slice.u3(-pi / 2, -pi / 2, pi / 2, state_registers[qubit_idx])
                    else:
                        qc_slice.rx(-pi / 2, state_registers[qubit_idx])
    # repeat the slice
    if shallow_slicing:
        logger.info('Under shallow slicing mode, the qc.data reference is repeated shallowly. '
                    'Thus, changing gates of one slice of the output circuit might affect other slices.')
        qc_slice.barrier(state_registers)
        qc_slice.data *= (num_time_slices * power)
        qc = qc_slice
    else:
        qc = QuantumCircuit()
        for _ in range(num_time_slices * power):
            qc += qc_slice
            qc.barrier(state_registers)
    return qc.to_instruction()
