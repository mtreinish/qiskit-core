# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Quantum State Tomography.
Generates many small circuits, thus good for profiling compiler overhead.
Number of circuits grows like 3^n_qubits
"""

import numpy as np

# import qiskit modules
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import execute
# import tomography libary and other useful tools
import qiskit.tools.qcvv.tomography as tomo
from qiskit.tools.qi.qi import state_fidelity, purity
from qiskit.tools.qi.qi import outer, random_unitary_matrix

# circuit that outputs the target state
def target_prep(state, target, n_qubits):
    # quantum circuit to make an entangled cat state
    if state == 'cat':
        n_qubits = int(np.log2(target.size))
        qr = QuantumRegister(n_qubits, 'qr')
        cr = ClassicalRegister(n_qubits, 'cr')
        circ = QuantumCircuit(qr, cr, name='cat')
        circ.h(qr[0])
        for i in range(1, n_qubits):
            circ.cx(qr[0], qr[i])

    # quantum circuit to prepare arbitrary given state
    elif state == 'random':
        n_qubits = int(np.log2(target.size))
        qr = QuantumRegister(n_qubits, 'qr')
        cr = ClassicalRegister(n_qubits, 'cr')
        circ = QuantumCircuit(qr, cr, name='random')
        circ.initialize(target, [qr[i] for i in range(n_qubits)])

    return circ


# add basis measurements to the circuit for tomography
# XX..X, XX..Y, .., ZZ..Z
def add_tomo_circuits(circ):
    # Construct state tomography set for measurement of qubits in the register
    qr = next(iter(circ.get_qregs().values()))
    cr = next(iter(circ.get_cregs().values()))
    tomo_set = tomo.state_tomography_set(list(range(qr.size)))

    # Add the state tomography measurement circuits
    tomo_circuits = tomo.create_tomography_circuits(circ, qr, cr, tomo_set)

    return tomo_set, tomo_circuits


class StateTomographyBench:
    params = [2, 3, 4, 5]
    timeout = 240.0

    def time_state_tomography_cat(self, n_qubits):
        # cat target state: [1. 0. 0. ... 0. 0. 1.]/sqrt(2.)
        target = np.zeros(pow(2, n_qubits))
        target[0] = 1
        target[pow(2, n_qubits)-1] = 1.0
        target /= np.sqrt(2.0)
        self._state_tomography(target, 'cat', n_qubits)


    def time_state_tomography_random(self, n_qubits):
        # random target state: first column of a random unitary
        target = random_unitary_matrix(pow(2, n_qubits))[0]
        self._state_tomography(target, 'random', n_qubits)

    # perform quantum state tomography and assess quality of reconstructed vector
    def _state_tomography(self, target, state, n_qubits, shots=1):
        # Use the local qasm simulator
        backend = 'local_qasm_simulator'

        # Prepared target state and assess quality
        prep_circ = target_prep(state, target, n_qubits)
        prep_result = execute(prep_circ, backend='local_statevector_simulator').result()
        prep_state = prep_result.get_statevector(prep_circ)
        F_prep = state_fidelity(prep_state, target)
        print('Prepared state fidelity =', F_prep)

        # Run state tomography simulation and fit data to reconstruct circuit
        tomo_set, tomo_circuits = add_tomo_circuits(prep_circ)
        tomo_result = execute(tomo_circuits, backend=backend, shots=shots).result()
        tomo_data = tomo.tomography_data(tomo_result, prep_circ.name, tomo_set)
        rho_fit = tomo.fit_tomography_data(tomo_data)

        # calculate fidelity and purity of fitted state
        F_fit = state_fidelity(rho_fit, target)
        pur = purity(rho_fit)
        print('Fitted state fidelity =', F_fit)
        print('Fitted state purity =', str(pur))
