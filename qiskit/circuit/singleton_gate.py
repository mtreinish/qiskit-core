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
from typing import Union

from qiskit.circuit.gate import Gate
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit


class SingletonGate(Gate):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if "label" in kwargs or "condition" in kwargs:
            return super().__new__(cls)
        #        bject.__new__(cls, *args, **kwargs)
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance

    @classmethod
    def c_if(cls, classical, val):
        if not isinstance(classical, (ClassicalRegister, Clbit)):
            raise CircuitError("c_if must be used with a classical register or classical bit")
        if val < 0:
            raise CircuitError("condition value should be non-negative")
        if isinstance(classical, Clbit):
            # Casting the conditional value as Boolean when
            # the classical condition is on a classical bit.
            val = bool(val)
        instance = cls()
        instance._condition = (classical, val)
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
    def condition(self) -> str:
        return self._condition

    @condition.setter
    def condition(self, name: str):
        raise NotImplementedError(
            f"This gate class {type(self)} does not support manually setting a "
            "condition on an instance. Instead you must set the label when instantiating a new object "
            "or via the .c_if() method"
        )


class SingletonControlledGate(type):
    _instance = None

    def __call__(cls, *args, **kwargs):
        if "label" in kwargs or "condition" in kwargs or "ctrl_state" in kwargs:
            return object.__new__(cls, *args, **kwargs)
        if cls is None:
            cls._instances[cls] = super(SingletonControlledGateMetaClass, cls).__call__(
                *args, **kwargs
            )
        return cls._instances[cls]

    def c_if(cls, classical, val):
        if not isinstance(classical, (ClassicalRegister, Clbit)):
            raise CircuitError("c_if must be used with a classical register or classical bit")
        if val < 0:
            raise CircuitError("condition value should be non-negative")
        if isinstance(classical, Clbit):
            # Casting the conditional value as Boolean when
            # the classical condition is on a classical bit.
            val = bool(val)
        return super(SingletonGateMeta, cls).__call__(*args, condition=(classical, val), **kwargs)

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
    def condition(self) -> str:
        return self._label

    @condition.setter
    def condition(self, name: str):
        raise NotImplementedError(
            f"This gate class {type(self)} does not support manually setting a "
            "condition on an instance. Instead you must set the label when instantiating a new object "
            "or via the .c_if() method"
        )

    @property
    def definition(self) -> QuantumCircuit:
        """Return definition in terms of other basic gates. If the gate has
        open controls, as determined from `self.ctrl_state`, the returned
        definition is conjugated with X without changing the internal
        `_definition`.
        """
        if self._open_ctrl:
            closed_gate = self.copy()
            closed_gate.ctrl_state = None
            bit_ctrl_state = bin(self.ctrl_state)[2:].zfill(self.num_ctrl_qubits)
            qreg = QuantumRegister(self.num_qubits, "q")
            qc_open_ctrl = QuantumCircuit(qreg)
            for qind, val in enumerate(bit_ctrl_state[::-1]):
                if val == "0":
                    qc_open_ctrl.x(qind)
            qc_open_ctrl.append(closed_gate, qargs=qreg[:])
            for qind, val in enumerate(bit_ctrl_state[::-1]):
                if val == "0":
                    qc_open_ctrl.x(qind)
            return qc_open_ctrl
        else:
            return super().definition

    @definition.setter
    def definition(self, excited_def: "QuantumCircuit"):
        raise NotImplementedError(
            f"This gate class {type(self)} does not support manually setting a "
            "definition on an instance."
        )

    @property
    def name(self) -> str:
        if self._open_ctrl:
            return f"{self._name}_o{self.ctrl_state}"
        else:
            return self._name

    @name.setter
    def name(self, name_str):
        raise NotImplementedError(
            f"This gate class {type(self)} does not support manually setting a "
            "name on an instance."
        )

    @property
    def num_ctrl_qubits(self):
        return self._num_ctrl_qubits

    @num_ctrl_qubits.setter
    def num_ctrl_qubits(self, num_ctrl_qubits):
        raise NotImplementedError(
            f"This gate class {type(self)} does not support manually setting a "
            "number of control qubits on an instance."
        )

    @property
    def ctrl_state(self) -> int:
        return self._ctrl_state

    @ctrl_state.setter
    def ctrl_state(self, ctrl_state: Union[int, str, None]):
        raise NotImplementedError(
            f"This gate class {type(self)} does not support manually setting a "
            "ctrl_state on an instance."
        )

    @property
    def params(self):
        if self.base_gate:
            return self.base_gate.params
        else:
            raise CircuitError("Controlled gate does not define base gate for extracting params")

    @params.setter
    def params(self, parameters):
        raise NotImplementedError(
            f"This gate class {type(self)} does not support manually setting "
            "params on an instance."
        )
