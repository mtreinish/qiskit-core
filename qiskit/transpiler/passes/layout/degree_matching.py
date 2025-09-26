# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Choose a Layout by matching circuit qubit degree with target qubit degree."""


import numpy as np
import rustworkx

from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit._accelerate.degree_matching import degree_matching_layout


class DegreeMatchingLayout(AnalysisPass):
    """Choose a Layout by finding the most connected subset of qubits.

    This pass associates a physical qubit (int) to each virtual qubit
    of the circuit (Qubit).

    Note:
        Even though a ``'layout'`` is not strictly a property of the DAG,
        in the transpiler architecture it is best passed around between passes
        by being set in ``property_set``.
    """

    def __init__(self, target):
        """DenseLayout initializer.

        Args:
            target (Target): A target representing the target backend.
        """
        super().__init__()
        self.target = target

    def run(self, dag):
        """Run the DegreeMatchingLayout pass."""
        let_layout = degree_matching_layout(dag, self.target)
        self.property_set["layout"] = Layout.from_intlist(let_layout)
