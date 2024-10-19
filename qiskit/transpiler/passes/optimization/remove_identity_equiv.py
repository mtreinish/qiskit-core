# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Transpiler pass to drop gates with negligible effects."""

from __future__ import annotations

from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.target import Target
from qiskit.transpiler.basepasses import TransformationPass
from qiskit._accelerate.remove_identity_equiv import remove_identity_equiv


class RemoveIdentityEquivalent(TransformationPass):
    """Remove gates with negligible effects.

    Removes gates whose effect is close to an identity operation, up to the specified
    tolerance.
    """

    def __init__(
        self, *, approximation_degree: float | None = 1.0, target: None | Target = None
    ) -> None:
        """Initialize the transpiler pass.

        Args:
            approximation_degree: The degree to approximate the the equivalence check. A value of 1
            defaults to 1e-16, and 0 is the maximum approximation where a tolerance of 1e-5 is used.
            For all other values in between the tolerance is computed as
            ``min(1e-16 / approximation_degree, 1e-5)``.
            target: If ``approximation_degree`` is set to ``None`` and a :class:`.Target` is provided
                for this field the tolerance for determining whether an operation is equivalent to
                identity will be set to the reported error rate in the target.
        """
        super().__init__()
        if approximation_degree is not None:
            atol = min(1e-16 / approximation_degree, 1e-5)
        else:
            atol = None
        self._atol = atol
        self._target = target

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        remove_identity_equiv(dag, self._atol, self._target)
        return dag
