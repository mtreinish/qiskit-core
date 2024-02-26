# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-function-docstring

"""Test the StarPreRouting pass"""

import unittest
from test import QiskitTestCase

import ddt

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.transpiler.passes.routing.star_prerouting import StarPreRouting
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.utils.optionals import HAS_AER


@ddt.ddt
class TestStarPreRouting(QiskitTestCase):
    """Tests the StarPreRouting pass"""

    def test_double_ghz(self):
        qc = QuantumCircuit(10)
        qc.h(0)
        qc.cx(0, range(1, 5))
        qc.h(9)
        qc.cx(9, range(8, 4, -1))
        qc.measure_all()
        result = StarPreRouting()(qc)
        cr = ClassicalRegister(10, "meas")
        expected = QuantumCircuit(10)
        expected.add_register(cr)
        expected.h(0)
        expected.h(9)
        expected.cx(0, 1)
        expected.cx(0, 2)
        expected.swap(0, 2)
        expected.cx(2, 3)
        expected.swap(2, 3)
        expected.cx(3, 4)
        expected.swap(3, 4)
        expected.cx(9, 8)
        expected.cx(9, 7)
        expected.swap(9, 7)
        expected.cx(7, 6)
        expected.swap(7, 6)
        expected.cx(6, 5)
        expected.swap(6, 5)
        expected.barrier()
        expected.measure(4, 0)
        expected.measure(1, 1)
        expected.measure(0, 2)
        expected.measure(2, 3)
        expected.measure(3, 4)
        expected.measure(6, 5)
        expected.measure(7, 6)
        expected.measure(9, 7)
        expected.measure(8, 8)
        expected.measure(5, 9)

        self.assertEqual(expected, result)

    def test_linear_ghz_no_change(self):
        qc = QuantumCircuit(6)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.cx(3, 4)
        qc.cx(4, 5)
        qc.measure_all()
        result = StarPreRouting()(qc)
        self.assertEqual(result, qc)

    def test_no_star(self):
        qc = QuantumCircuit(6)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(3, 2)
        qc.cx(1, 4)
        qc.cx(0, 3)
        qc.cx(2, 1)
        qc.measure_all()
        result = StarPreRouting()(qc)
        self.assertEqual(result, qc)

    def test_10q_bv(self):
        num_qubits = 10
        qc = QuantumCircuit(num_qubits, num_qubits - 1)
        qc.x(num_qubits - 1)
        qc.h(qc.qubits)
        for i in range(num_qubits - 1):
            qc.cx(i, num_qubits - 1)
        qc.barrier()
        qc.h(qc.qubits[:-1])
        for i in range(num_qubits - 1):
            qc.measure(i, i)
        result = StarPreRouting()(qc)

        expected = QuantumCircuit(num_qubits, num_qubits - 1)
        expected.x(num_qubits - 1)
        expected.h(qc.qubits)
        expected.cx(0, num_qubits - 1)
        expected.cx(1, num_qubits - 1)
        expected.swap(1, num_qubits - 1)
        for i in range(1, num_qubits - 2):
            expected.cx(i + 1, i)
            expected.swap(i + 1, i)
        expected.barrier()
        expected.h(qc.qubits[:-2])
        expected.h(qc.qubits[-1])
        expected.measure(0, 0)
        expected.measure(9, 1)
        for i in range(2, num_qubits - 1):
            expected.measure(i - 1, i)
        self.assertEqual(result, expected)

    # Skip level 3 because of unitary synth introducing non-clifford gates
    @unittest.skipUnless(HAS_AER, "Aer required for clifford simulation")
    @ddt.data(0, 1, 2)
    def test_100q_grid_full_path(self, opt_level):
        from qiskit_aer import AerSimulator

        num_qubits = 100
        coupling_map = CouplingMap.from_grid(10, 10)
        qc = QuantumCircuit(num_qubits, num_qubits - 1)
        qc.x(num_qubits - 1)
        qc.h(qc.qubits)
        for i in range(num_qubits - 1):
            qc.cx(i, num_qubits - 1)
        qc.barrier()
        qc.h(qc.qubits[:-1])
        for i in range(num_qubits - 1):
            qc.measure(i, i)
        pm = generate_preset_pass_manager(
            opt_level, basis_gates=["h", "cx", "x"], coupling_map=coupling_map
        )
        pm.pre_layout = PassManager(StarPreRouting())
        result = pm.run(qc)
        counts_before = AerSimulator().run(qc).result().get_counts()
        counts_after = AerSimulator().run(result).result().get_counts()
        self.assertEqual(counts_before, counts_after)

    def test_10q_bv_no_barrier(self):
        num_qubits = 10
        qc = QuantumCircuit(num_qubits, num_qubits - 1)
        qc.x(num_qubits - 1)
        qc.h(qc.qubits)
        for i in range(num_qubits - 1):
            qc.cx(i, num_qubits - 1)
        qc.h(qc.qubits[:-1])
        for i in range(num_qubits - 1):
            qc.measure(i, i)
        result = StarPreRouting()(qc)

        expected = QuantumCircuit(num_qubits, num_qubits - 1)
        expected.x(num_qubits - 1)
        expected.h(qc.qubits)
        expected.cx(0, num_qubits - 1)
        expected.cx(1, num_qubits - 1)
        expected.swap(1, num_qubits - 1)
        for i in range(1, num_qubits - 2):
            expected.cx(i + 1, i)
            expected.swap(i + 1, i)
        expected.h(qc.qubits[:-2])
        expected.h(qc.qubits[-1])
        expected.measure(0, 0)
        expected.measure(9, 1)
        for i in range(1, num_qubits - 2):
            expected.measure(i, i + 1)
        self.assertEqual(result, expected)

    # Skip level 3 because of unitary synth introducing non-clifford gates
    @unittest.skipUnless(HAS_AER, "Aer required for clifford simulation")
    @ddt.data(0, 1, 2)
    def test_100q_grid_full_path_no_barrier(self, opt_level):
        from qiskit_aer import AerSimulator

        num_qubits = 100
        coupling_map = CouplingMap.from_grid(10, 10)
        qc = QuantumCircuit(num_qubits, num_qubits - 1)
        qc.x(num_qubits - 1)
        qc.h(qc.qubits)
        for i in range(num_qubits - 1):
            qc.cx(i, num_qubits - 1)
        qc.h(qc.qubits[:-1])
        for i in range(num_qubits - 1):
            qc.measure(i, i)
        pm = generate_preset_pass_manager(
            opt_level, basis_gates=["h", "cx", "x"], coupling_map=coupling_map
        )
        pm.pre_layout = PassManager(StarPreRouting())
        result = pm.run(qc)
        counts_before = AerSimulator().run(qc).result().get_counts()
        counts_after = AerSimulator().run(result).result().get_counts()
        self.assertEqual(counts_before, counts_after)

    def test_hadamard_ordering(self):
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cx(0, 1)
        qc.h(0)
        qc.cx(0, 2)
        qc.h(0)
        qc.cx(0, 3)
        qc.h(0)
        qc.cx(0, 4)
        result = StarPreRouting()(qc)
        expected = QuantumCircuit(5)
        expected.h(0)
        expected.cx(0, 1)
        expected.h(0)
        expected.cx(0, 2)
        expected.swap(0, 2)
        expected.h(2)
        expected.cx(2, 3)
        expected.swap(2, 3)
        expected.h(3)
        expected.cx(3, 4)
        expected.swap(3, 4)
        self.assertEqual(expected, result)
