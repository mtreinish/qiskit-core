# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Variational Quantum Eigensolver (VQE).
Generates many small circuits, thus good for profiling compiler overhead.
"""

from functools import partial
import os

import numpy as np
from scipy import linalg as la

# import qiskit modules
import qiskit
from qiskit.tools.apps import optimization


def cost_function(H, n_qubits, depth, entangler_map, shots, device, theta, qp=None):
    trial_circuit = optimization.trial_circuit_ryrz(
        n_qubits, depth, theta, entangler_map, measurement=False)

    if not qp:
        energy, circuits = optimization.eval_hamiltonian(
            H, trial_circuit, shots, device)
        return energy.real, circuits
    else:
        energy = optimization.eval_hamiltonian(qp, H, trial_circuit, shots,
                                               device).real
        return energy




class TimeVqeSuite:

    timeout = 360.0

    def setup(self):
        if hasattr(qiskit, 'QuantumProgram'):
            self.use_quantum_program = True
        else:
            self.use_quantum_program = False

    def time_vqe_H2(self):
        n_qubits = 2
        Z1 = 1
        Z2 = 1
        min_distance = 0.2
        max_distance = 4
        if not self.use_quantum_program:
            self._vqe(n_qubits, Z1, Z2, min_distance, max_distance, 'H2')
        else:
            self._vqe_with_quantum_program(n_qubits, Z1, Z2, min_distance,
                                           max_distance, 'H2')

    def time_vqe_LiH(self):
        n_qubits = 4
        Z1 = 1
        Z2 = 3
        min_distance = 0.5
        max_distance = 5
        if not self.use_quantum_program:
            self._vqe(n_qubits, Z1, Z2, min_distance, max_distance, 'LiH')
        else:
            self._vqe_with_quantum_program(n_qubits, Z1, Z2, min_distance,
                                           max_distance, 'LiH')


    def _vqe_with_quantum_program(self, n_qubits, Z1, Z2, min_distance,
                                  max_distance, molecule, depth=0,
                                  max_trials=200, shots=1):
        # Read Hamiltonian
        ham_name = os.path.join(os.path.dirname(__file__),
                                molecule + '/' + molecule + 'Equilibrium.txt')
        pauli_list = optimization.Hamiltonian_from_file(ham_name)
        H = optimization.make_Hamiltonian(pauli_list)

        # Exact Energy
        exact = np.amin(la.eig(H)[0]).real
        print('The exact ground state energy is: {}'.format(exact))

        qp = qiskit.QuantumProgram()
        # Optimization
        try:
            device = 'local_qiskit_simulator'
            qp.get_backend_configuration(device)
        except LookupError:
            device = 'local_qasm_simulator'

        if shots != 1:
            H = optimization.group_paulis(pauli_list)

        entangler_map = qp.get_backend_configuration(device)['coupling_map']
        if entangler_map == 'all-to-all':
            entangler_map = {i: [j for j in range(n_qubits) if j != i] for i in range(n_qubits)}

        initial_theta = np.random.randn(2 * n_qubits * depth)   # initial angles
        initial_c = 0.01                                        # first theta perturbations
        target_update = 2 * np.pi * 0.1                         # aimed update on first trial
        save_step = 20                                          # print optimization trajectory

        cost = partial(cost_function, H, n_qubits, depth, entangler_map, shots, device, qp=qp)

        SPSA_params = optimization.SPSA_calibration(
            cost, initial_theta, initial_c, target_update, stat=25)
        optimization.SPSA_optimization(cost, initial_theta,
                                       SPSA_params,
                                       max_trials, save_step,
                                       last_avg=1)

    def _vqe(self, n_qubits, Z1, Z2, min_distance, max_distance, molecule,
             depth=0, max_trials=200, shots=1):
        # Read Hamiltonian
        ham_name = os.path.join(os.path.dirname(__file__),
                                molecule + '/' + molecule + 'Equilibrium.txt')
        pauli_list = optimization.Hamiltonian_from_file(ham_name)
        H = optimization.make_Hamiltonian(pauli_list)

        # Exact Energy
        exact = np.amin(la.eig(H)[0]).real
        print('The exact ground state energy is: {}'.format(exact))

        # Optimization
        device = 'local_qasm_simulator'
        if shots == 1:
            device = 'local_statevector_simulator'

        if 'statevector' not in device:
            H = optimization.group_paulis(pauli_list)

        entangler_map = qiskit.get_backend(device).configuration['coupling_map']

        if entangler_map == 'all-to-all':
            entangler_map = {i: [j for j in range(n_qubits) if j != i] for i in range(n_qubits)}
        else:
            entangler_map = qiskit.mapper.coupling_list2dict(entangler_map)

        initial_theta = np.random.randn(2 * n_qubits * depth)   # initial angles
        initial_c = 0.01                                        # first theta perturbations
        target_update = 2 * np.pi * 0.1                         # aimed update on first trial
        save_step = 20                                          # print optimization trajectory

        cost = partial(cost_function, H, n_qubits, depth, entangler_map, shots, device)

        SPSA_params, _ = optimization.SPSA_calibration(
            cost, initial_theta, initial_c, target_update, stat=25)
        optimization.SPSA_optimization(
            cost, initial_theta, SPSA_params, max_trials, save_step,
            last_avg=1)
