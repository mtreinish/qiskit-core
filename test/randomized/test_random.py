
import os
import tempfile

import numpy as np

from qiskit.compiler import assemble
from qiskit.compiler import transpile
from qiskit.quantum_info import Operator
from qiskit.quantum_info.random import random_circuit
from qiskit.transpiler.passes import unroller
from qiskit.transpiler.passmanager import PassManager

try:
    from qiskit import Aer
    HAS_AER = True
except ImportError:
    from qiskit import BasicAer
    HAS_AER = False

from unittest import TestCase


class RandomCircuitTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.has_aer = HAS_AER
        if cls.has_aer:
            cls.backend = Aer.get_backend('unitary_simulator')
        else:
            cls.backend = BasicAer.get_backend('unitary_simulator')
        cls.qubits = int(os.getenv('QISKIT_RANDOM_QUBITS', 5))
        cls.depth = int(os.getenv('QISKIT_RANDOM_DEPTH', 42))
        cls.qasm_dir = os.getenv('QISKIT_RANDOM_QASMDIR', None)

    def compare_circuits(self, circuit_a, circuit_b, qasm_path, t_qasm_path,
                         unitary_compare=False):
        msg = ("Random circuit transpilation failed, to see the input "
               "circuit's qasm equivalent look at: %s, and the transpiler "
               "output qasm path is: %s" % (qasm_path, t_qasm_path))
        if unitary_compare:
            # Unroll circuit_a since it's randomly defined gates might be
            # outside the basis set
            basis_gates = None
            if hasattr(self.backend, 'configuration'):
                basis_gates = getattr(self.backend.configuration(),
                                      'basis_gates', None)
            if basis_gates:
                pm = PassManager()
                pm.append(unroller.Unroller(basis_gates))
                circuit_a = transpile(circuit_a, pass_manager=pm)

            unitary_a = self.backend.run(
                assemble(circuit_a, self.backend)).result().get_unitary()
            unitary_b = self.backend.run(
                assemble(circuit_b, self.backend)).result().get_unitary()
            self.assertTrue(np.array_equal(unitary_a, unitary_b), msg)
        self.assertEqual(Operator(circuit_a), Operator(circuit_b), msg)

    def test_random_circuits_lvl_3(self):
        _, qasm_path = tempfile.mkstemp(suffix='.qasm',
                                        prefix='qiskit-random-level-3-',
                                        dir=self.qasm_dir)
        circuit = random_circuit(self.qubits, self.depth, self.seed)
        with open(qasm_path, 'w') as fd:
            fd.write(circuit.qasm())
        transpiled_circuit = transpile(circuit, self.backend,
                                       optimization_level=3)
        trans_qasm_path = qasm_path + '-transpiled'
        with open(trans_qasm_path, 'w') as fd:
            fd.write(transpiled_circuit.qasm())

        self.compare_circuits(circuit, transpiled_circuit, qasm_path,
                              trans_qasm_path)
        os.remove(qasm_path)
        os.remove(trans_qasm_path)

    def test_random_circuits_lvl_2(self):
        _, qasm_path = tempfile.mkstemp(suffix='.qasm',
                                        prefix='qiskit-random-level-2-',
                                        dir=self.qasm_dir)
        circuit = random_circuit(self.qubits, self.depth, self.seed)
        with open(qasm_path, 'w') as fd:
            fd.write(circuit.qasm())
        transpiled_circuit = transpile(circuit, self.backend,
                                       optimization_level=2)
        trans_qasm_path = qasm_path + '-transpiled'
        with open(trans_qasm_path, 'w') as fd:
            fd.write(transpiled_circuit.qasm())
        self.compare_circuits(circuit, transpiled_circuit, qasm_path,
                              trans_qasm_path)
        os.remove(qasm_path)
        os.remove(trans_qasm_path)

    def test_random_circuits_lvl_1(self):
        _, qasm_path = tempfile.mkstemp(suffix='.qasm',
                                        prefix='qiskit-random-level-1-',
                                        dir=self.qasm_dir)
        circuit = random_circuit(self.qubits, self.depth, self.seed)
        with open(qasm_path, 'w') as fd:
            fd.write(circuit.qasm())
        transpiled_circuit = transpile(circuit, self.backend,
                                       optimization_level=1)
        trans_qasm_path = qasm_path + '-transpiled'
        with open(trans_qasm_path, 'w') as fd:
            fd.write(transpiled_circuit.qasm())
        self.compare_circuits(circuit, transpiled_circuit, qasm_path,
                              trans_qasm_path)
        os.remove(qasm_path)
        os.remove(trans_qasm_path)

    def test_random_circuits_lvl_0(self):
        _, qasm_path = tempfile.mkstemp(suffix='.qasm',
                                        prefix='qiskit-random-level-0-',
                                        dir=self.qasm_dir)
        circuit = random_circuit(self.qubits, self.depth, self.seed)
        with open(qasm_path, 'w') as fd:
            fd.write(circuit.qasm())
        transpiled_circuit = transpile(circuit, self.backend,
                                       optimization_level=0)
        trans_qasm_path = qasm_path + '-transpiled'
        with open(trans_qasm_path, 'w') as fd:
            fd.write(transpiled_circuit.qasm())
        self.compare_circuits(circuit, transpiled_circuit, qasm_path,
                              trans_qasm_path)
        os.remove(qasm_path)
        os.remove(trans_qasm_path)
