#!python
#cython: language_level = 3
#distutils: language = c++

# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

cimport cython
from libc.math cimport asin, atan2, acos, sqrt, cos, sin, M_PI
from libcpp cimport bool

import numpy as np
from scipy.optimize import fsolve

from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info.operators.predicates import matrix_equal

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple convert_u2_to_su2(complex[:, ::1] u2_matrix):
    cdef complex det = np.linalg.det(u2_matrix)
    cdef complex z = 1 / np.sqrt(det)
    cdef complex[:, ::1] su2_matrix = np.multiply(z, u2_matrix)
    cdef double phase = np.arctan2(np.imag(z), np.real(z))
    return su2_matrix, phase

cdef double[::1] compute_euler_angles(double[:, ::1] matrix):
    cdef double theta
    cdef double psi
    cdef double phi
    cdef double[3] output
    cdef double[:, ::1] round_matrix = np.round(matrix, decimals=7)
    if matrix[2][0] != 1 and matrix[2][1] != -1:
        theta = -asin(matrix[2][0])
        psi = atan2(matrix[2][1] / cos(theta),
                    matrix[2][2] / cos(theta))
        phi = atan2(matrix[1][0] / cos(theta),
                    matrix[0][0] / cos(theta))
    else:
        phi = 0
        if matrix[2][0] == 1:
            theta = M_PI/2
            psi = phi + atan2(matrix[0][1], matrix[0][2])
        else:
            theta = -M_PI/2
            psi = -phi + atan2(-matrix[0][1], -matrix[0][2])
    output = [phi, theta, psi]
    return output

cdef complex[:, ::1] compute_su2_euler(double phi, double theta, double psi):
    cdef complex[:, ::1] uz_phi = np.array(
        [[np.exp(-0.5j * phi), 0],
         [0, np.exp(0.5j * phi)]], dtype=complex)
    cdef complex[:, ::1] uy_theta = np.array(
        [[cos(theta / 2), sin(theta / 2)],
         [-sin(theta / 2), cos(theta / 2)]], dtype=complex)
    cdef complex[:, ::1] ux_psi = np.array(
        [[cos(psi / 2), sin(psi / 2) * 1j],
         [sin(psi / 2) * 1j, cos(psi / 2)]], dtype=complex)
    return np.dot(uz_phi, np.dot(uy_theta, ux_psi))

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:, ::1] convert_su2_to_so3(complex[:, ::1] matrix):
    cdef double rotation[3][3]
    cdef double a = matrix[0][0].real
    cdef double b = matrix[0][0].imag
    cdef double c = -matrix[0][1].real
    cdef double d = -matrix[0][1].imag
    rotation[0][:] = [a ** 2- b ** 2 - c ** 2 + d ** 2, 2 * a * b + 2 * c * d, -2 * a * c + 2 * b * d]
    rotation[1][:] = [-2 * a * b + 2 * c * d, a ** 2 - b ** 2 + c ** 2 - d ** 2, 2 * a * d + 2 * b * c]
    rotation[2][:] = [2 * a * c + 2 * b * d, 2 * b * c - 2 * a * d, a ** 2 + b ** 2 - c ** 2 - d ** 2]
    return rotation

cdef double compute_trace_so3(double[:, ::1] matrix):
    cdef double[::1] trace = np.matrix.trace(np.asarray(matrix, dtype=complex))
    cdef double trace_rounded = np.min(trace, 3.)
    return trace_rounded

cdef double solve_decomposition_angle(double[:, ::1] matrix):
    cdef double trace = compute_trace_so3(matrix)
    cdef double angle = acos((1/2)*(trace-1))

    def _objective_function(double phi):
        cdef double rhs = 2 * sin(phi / 2) ** 2
        rhs *= sqrt(1 - sin(phi / 2) ** 4)
        cdef lhs = sin(angle / 2)
        return rhs - lhs

    return fsolve(_objective_function, angle)[0]

cdef double[:, ::1] compute_rotation(double[::1] from_vector,
                                     double[::1] to_vector):
    from_vector = from_vector / np.linalg.norm(from_vector)
    to_vector = to_vector / np.linalg.norm(to_vector)

    cdef double[:, ::1] dot = np.dot(from_vector, to_vector)
    cdef double[:, ::1] cross = cross_product_matrix(np.cross(from_vector, to_vector))
    return np.identity(3) + cross + np.dot(cross, cross) / np.add(1, dot)

cdef double[:, ::1] cross_product_matrix(double[::1] v):
    cdef double cross[3][3]
    cross[0][:] = [0.0, -v[2], v[1]]
    cross[1][:] = [v[2], 0.0, -v[0]]
    cross[2][:] = [-v[1], v[0], 0.0]
    return cross

cdef double[:, ::1] compute_commutator_so3(double[:, ::1] a, double[:, ::1] b):
    cdef double[:, ::1] a_dagger = np.ascontiguousarray(np.conj(a).T)
    cdef double[:, ::1] b_dagger = np.ascontiguousarray(np.conj(b).T)

    return np.dot(np.dot(np.dot(a, b), a_dagger), b_dagger)

cdef double[:, ::1] compute_rotation_from_angle_and_axis(double angle,
                                                         double[::1] axis):
    if axis.shape[0] != 3:
        raise ValueError(f'Axis must be a 1d array of length 3, but has shape {axis.shape}.')

    if abs(np.linalg.norm(axis) - 1.0) > 1e-4:
        raise ValueError(f'Axis must have a norm of 1, but has {np.linalg.norm(axis)}.')

    res = np.multiply(cos(angle), np.identity(3)) + np.multiply(sin(angle),
                                                                cross_product_matrix(axis))
    res += (1 - cos(angle)) * np.outer(axis, axis)
    return res

cdef double[::1] compute_rotation_axis(double[:, ::1] matrix):
    cdef double trace = compute_trace_so3(matrix)
    cdef double theta = acos(0.5 * (trace - 1))
    cdef double x, y, z
    if sin(theta) > 1e-10:
        x = 1 / (2 * sin(theta)) * (matrix[2][1] - matrix[1][2])
        y = 1 / (2 * sin(theta)) * (matrix[0][2] - matrix[2][0])
        z = 1 / (2 * sin(theta)) * (matrix[1][0] - matrix[0][1])
    else:
        x = 1.0
        y = 0.0
        z = 0.0
    cdef double out_array[3]
    out_array[:] = [x, y, z]
    return out_array

cdef double[:, ::1] compute_rotation_between(double[::1] from_vector,
                                             double[::1] to_vector):
    cdef double[::1] norm_from_vector = from_vector / np.linalg.norm(from_vector)
    cdef double[::1] norm_to_vector = to_vector / np.linalg.norm(to_vector)
    cdef double dot = np.dot(from_vector, to_vector)
    cdef double[:, ::1] cross = cross_product_matrix(np.cross(norm_from_vector,
                                                              norm_to_vector))
    return np.identity(3) + cross + np.dot(cross, cross) / np.add(1, dot)



cpdef tuple commutator_decompose(double[:, ::1] u_so3, bool check_input=False):
    cdef double angle = solve_decomposition_angle(u_so3)
    cdef double axis[3]
    axis[:] = [1, 0, 0]
    cdef double[:, ::1] vx = compute_rotation_from_angle_and_axis(angle, axis)
    axis[:] = [0, 1, 0]
    cdef double[:, ::1] wy = compute_rotation_from_angle_and_axis(angle, axis)

    cdef double[:, ::1] commutator = compute_commutator_so3(vx, wy)
    cdef double[::1] u_so3_axis = compute_rotation_axis(u_so3)
    cdef double[::1] commutator_axis = compute_rotation_axis(commutator)
    cdef double[:, ::1] sim_matrix = compute_rotation_between(commutator_axis, u_so3_axis)
    cdef double[:, ::1] sim_matrix_dagger = np.ascontiguousarray(np.conj(sim_matrix).T)
    cdef double[:, ::1] v = np.dot(np.dot(sim_matrix, vx), sim_matrix_dagger)
    cdef double[:, ::1] w = np.dot(np.dot(sim_matrix, wy), sim_matrix_dagger)

    return GateSequence.from_matrix(v), GateSequence.from_matrix(w)

cdef complex[:, ::1] convert_so3_to_su2(double[:, ::1] matrix):
    cdef double[::1] angles = compute_euler_angles(matrix)
    return compute_su2_euler(angles[0], angles[1], angles[2])

cdef class GateSequence:
    # Attributes
    cdef public double global_phase
    cdef public double[:, ::1] product
    cdef public object gates

    def __cinit__(self, object gates):
        cdef complex[:, ::1] u2_matrix = np.identity(2, dtype=complex)
        cdef complex[:, ::1] su2_matrix
        cdef double[:, ::1] so3_matrix
        cdef double global_phase
        if gates is None:
            gates = []
        self.gates = gates
        for gate in gates:
            u2_matrix = gate.to_matrix().dot(u2_matrix)
        su2_matrix, global_phase = convert_u2_to_su2(u2_matrix)
        so3_matrix = convert_su2_to_so3(su2_matrix)
        self.global_phase = global_phase
        self.product = so3_matrix

    def __eq__(self, GateSequence other):
        if not len(self.gates) == len(other.gates):
            return False

        for gate1, gate2 in zip(self.gates, other.gates):
            if gate1 != gate2:
                return False

        if self.global_phase != other.global_phase:
            return False

        return True

    cpdef object to_circuit(self):
        if len(self.gates) == 0 and not np.allclose(self.product, np.identity(3)):
            circuit = QuantumCircuit(1, global_phase=self.global_phase)
            su2 = convert_so3_to_su2(self.product)
            circuit.unitary(su2, [0])
            return circuit

        circuit = QuantumCircuit(1, global_phase=self.global_phase)
        for gate in self.gates:
            circuit.append(gate, [0])

        return circuit


    cpdef GateSequence append(self, object gate):
        """Append gate to the sequence of gates.
        Args:
            gate: The gate to be appended.
        Returns:
            GateSequence with ``gate`` appended.
        """
        cdef double[:, ::1] so3
        cdef complex[:, ::1] su2
        cdef double phase
        # TODO: this recomputes the product whenever we append something, which could be more
        # efficient by storing the current matrix and just multiplying the input gate to it
        # self.product = convert_su2_to_so3(self._compute_product(self.gates))
        su2, phase = convert_u2_to_su2(gate.to_matrix())
        so3 = convert_su2_to_so3(su2)

        self.product = np.dot(so3, self.product)
        self.global_phase = self.global_phase + phase
        self.gates.append(gate)
        return self

    cpdef void adjoint(self):
        adjoint = GateSequence(None)
        adjoint.gates = [gate.inverse() for gate in reversed(self.gates)]
        adjoint.product = np.conj(self.product).T
        adjoint.global_phase = -self.global_phase

    cpdef GateSequence copy(self):
        return GateSequence(self.gates.copy())

    cpdef GateSequence dot(self, GateSequence other):
        """Compute the dot-product with another gate sequence.
        Args:
            other: The other gate sequence.
        Returns:
            The dot-product as gate sequence.
        """
        composed = GateSequence(None)
        composed.gates = other.gates + self.gates
        composed.product = np.dot(self.product, other.product)
        composed.global_phase = self.global_phase + other.global_phase
        return composed

    @classmethod
    def from_matrix(cls, object matrix):
        instance = cls(None)
        if matrix.shape == (2, 2):
            instance.product = convert_su2_to_so3(matrix)
        elif matrix.shape == (3, 3):
            instance.product = matrix
        else:
            raise ValueError(f'Matrix must have shape (3, 3) or (2, 2) but has {matrix.shape}.')
        instance.gates = []
        return instance


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef remove_inverse_follows_gate(GateSequence sequence):
    cdef int index = 0
    while index < len(sequence.gates) - 1:
        if sequence.gates[index + 1] == sequence.gates[index].inverse():
            # remove gates at index and index + 1 (pop shifts the whole sequence, so we apply it
            # twice for the value `index`)
            sequence.gates.pop(index)
            sequence.gates.pop(index)
            # take a step back to see if we have uncovered a new pair, e.g.
            # [h, s, sdg, h] at index = 1 removes s, sdg but if we continue at index 1
            # we miss the uncovered [h, h] pair at indices 0 and 1
            if index > 0:
                index -= 1
        else:
            # next index
            index += 1

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef bool check_candidate(GateSequence candidate, object sequences):
    # check if a matrix representation already exists
    for existing in sequences:
        # eliminate global phase
        if matrix_equal(existing.product, candidate.product, ignore_phase=True):
            # is the new sequence less or more efficient?
            if len(candidate.gates) >= len(existing.gates):
                return False
            return True
    return True


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def generate_approximations(object products):
    cdef list sequences = []
    for item in products:
        candidate = GateSequence(item)
        remove_inverse_follows_gate(candidate)
        if check_candidate(candidate, sequences):
            sequences.append(candidate)
    return sequences
