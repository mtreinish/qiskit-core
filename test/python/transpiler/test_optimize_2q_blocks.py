# This code is part of Qiskit.
#
# (C) Copyright IBM 2024
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the optimize-2q-gate pass"""

import unittest

import ddt
import numpy as np
import rustworkx as rx

from qiskit.circuit import QuantumRegister, QuantumCircuit, ClassicalRegister, Parameter
from qiskit.circuit.library.standard_gates import (
    UGate,
    SXGate,
    PhaseGate,
    U3Gate,
    U2Gate,
    U1Gate,
    RZGate,
    RXGate,
    RYGate,
    HGate,
)
from qiskit.circuit.random import random_circuit
from qiskit.compiler import transpile
from qiskit.transpiler.passes import Optimize2qBlocks
from test import QiskitTestCase  # pylint: disable=wrong-import-order
from qiskit.providers.fake_provider import GenericBackendV2


@ddt.ddt
class TestOptimize2qBlocks(QiskitTestCase):
    """Test for 2q gate optimizations."""

    def test_run_pass(self):
        """Test running pass."""
        backend = GenericBackendV2(
            10,
            ["cx", "rz", "sx", "x"],
            coupling_map=rx.generators.directed_mesh_graph(10).edge_list(),
        )

        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.cz(1, 0)
        qc.t(0)
        qc.y(0)
        qc.cz(0, 1)
        qc.cx(1, 0)
        qc.h(2)
        qc.cx(2, 1)
        qc.cz(1, 2)
        qc.t(2)
        qc.y(2)
        qc.cz(2, 1)
        qc.cx(1, 2)
        qc.measure_all()
        opt_pass = Optimize2qBlocks(target=backend.target)
        print(opt_pass(qc))
