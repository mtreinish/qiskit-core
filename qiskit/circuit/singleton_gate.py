# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Singleton metaclass.
"""
import copy
from typing import Union

from qiskit.circuit.gate import Gate
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit


class SingletonGate(Gate):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if "label" in kwargs or "_condition" in kwargs:
            return object.__new__(cls)
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, *args, _condition=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._condition = _condition

    def c_if(self, classical, val):
        if not isinstance(classical, (ClassicalRegister, Clbit)):
            raise CircuitError("c_if must be used with a classical register or classical bit")
        if val < 0:
            raise CircuitError("condition value should be non-negative")
        if isinstance(classical, Clbit):
            # Casting the conditional value as Boolean when
            # the classical condition is on a classical bit.
            val = bool(val)
        instance = type(self)(label=self.label, _condition=(classical, val))
        return instance

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, name: str):
        raise NotImplementedError(
            f"This gate class {type(self)} does not support manually setting a "
            "label on an instance. Instead you must set the label when instantiating a new object."
        )

    @property
    def condition(self):
        return self._condition

    @condition.setter
    def condition(self, name: str):
        raise NotImplementedError(
            f"This gate class {type(self)} does not support manually setting a "
            "condition on an instance. Instead you must set the label when instantiating a new object "
            "or via the .c_if() method"
        )

    def __deepcopy__(self, _memo=None):
        if self.condition is None and self.label is None:
            return self
        else:
            return type(self)(label=self.label, _condition=self.condition)

    def copy(self, name=None):
        if name is not None and self.condition is None and self.label is None:
            raise QiskitError("A custom name can not be set on a copy of a singleton gate")
        return super().copy()
