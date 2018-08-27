# -*- coding: utf-8 -*

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import os

import qiskit


class TranspilerBenchSuite:

    def _build_cx_circuit(self):
        cx_register = qiskit.QuantumRegister(2)
        cx_circuit = qiskit.QuantumCircuit(cx_register)
        cx_circuit.h(cx_register[0])
        cx_circuit.h(cx_register[0])
        cx_circuit.cx(cx_register[0], cx_register[1])
        cx_circuit.cx(cx_register[0], cx_register[1])
        cx_circuit.cx(cx_register[0], cx_register[1])
        cx_circuit.cx(cx_register[0], cx_register[1])
        return cx_circuit

    def _build_single_gate_circuit(self):
        single_register = qiskit.QuantumRegister(1)
        single_gate_circuit = qiskit.QuantumCircuit(single_register)
        single_gate_circuit.h(single_register[0])
        return single_gate_circuit

    def setup(self):
        self.local_qasm_simulator = qiskit.get_backend('local_qasm_simulator')
        self.single_gate_circuit = self._build_single_gate_circuit()
        self.cx_circuit = self._build_cx_circuit()
        self.qasm_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'qasm'))
        pea_3_pi_8_path = os.path.join(self.qasm_path, 'pea_3_pi_8.qasm')
        self.pea_3_pi_8 = qiskit.load_qasm_file(pea_3_pi_8_path)

    def time_single_gate_transpile(self):
        qiskit.compile(self.single_gate_circuit, self.local_qasm_simulator)

    def time_cx_transpile(self):
        qiskit.compile(self.cx_circuit, self.local_qasm_simulator)

    def time_pea_3_pi_8(self):
        qiskit.compile(self.pea_3_pi_8, self.local_qasm_simulator)
