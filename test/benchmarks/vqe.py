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
from qiskit import mapper

# import optimization tools
from qiskit.tools.apps.optimization import trial_circuit_ryrz, SPSA_optimization, SPSA_calibration
from qiskit.tools.apps.optimization import Hamiltonian_from_file, make_Hamiltonian
from qiskit.tools.apps.optimization import eval_hamiltonian, group_paulis
from qiskit import get_backend


def cost_function(H, n_qubits, depth, entangler_map, shots, device, theta):
    trial_circuit = trial_circuit_ryrz(n_qubits, depth, theta, entangler_map,
                                       meas_string=None, measurement=False)

    energy, circuits = eval_hamiltonian(H, trial_circuit, shots, device)

    return energy.real, circuits


class TimeVqeSuite:

    def setup(self):
        pass

    def time_vqe_H2(self):
        n_qubits = 2
        Z1 = 1
        Z2 = 1
        min_distance = 0.2
        max_distance = 4
        self._vqe(n_qubits, Z1, Z2, min_distance, max_distance, 'H2')

    def time_vqe_LiH(self):
        n_qubits = 4
        Z1 = 1
        Z2 = 3
        min_distance = 0.5
        max_distance = 5
        self._vqe(n_qubits, Z1, Z2, min_distance, max_distance, 'LiH')


    def _vqe(self, n_qubits, Z1, Z2, min_distance, max_distance, molecule,
             depth=0, max_trials=200, shots=1):
        # Read Hamiltonian
        ham_name = os.path.join(os.path.dirname(__file__),
                                molecule + '/' + molecule + 'Equilibrium.txt')
        pauli_list = Hamiltonian_from_file(ham_name)
        H = make_Hamiltonian(pauli_list)

        # Exact Energy
        exact = np.amin(la.eig(H)[0]).real
        print('The exact ground state energy is: {}'.format(exact))

        # Optimization
        device = 'local_qasm_simulator'
        if shots == 1:
            device = 'local_statevector_simulator'

        if 'statevector' not in device:
            H = group_paulis(pauli_list)

        entangler_map = get_backend(device).configuration['coupling_map']

        if entangler_map == 'all-to-all':
            entangler_map = {i: [j for j in range(n_qubits) if j != i] for i in range(n_qubits)}
        else:
            entangler_map = mapper.coupling_list2dict(entangler_map)

        initial_theta = np.random.randn(2 * n_qubits * depth)   # initial angles
        initial_c = 0.01                                        # first theta perturbations
        target_update = 2 * np.pi * 0.1                         # aimed update on first trial
        save_step = 20                                          # print optimization trajectory

        cost = partial(cost_function, H, n_qubits, depth, entangler_map, shots, device)

        SPSA_params, circuits_cal = SPSA_calibration(cost, initial_theta, initial_c,
                                                     target_update, stat=25)
        output, circuits_opt = SPSA_optimization(cost, initial_theta, SPSA_params, max_trials,
                                             save_step, last_avg=1)
