# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Instruction collection.
"""
from qiskit.terra import _instruction
from qiskit.terra import _qiskiterror

__all__ = ['InstructionSet']


class InstructionSet(object):
    """Instruction collection."""

    def __init__(self):
        """New collection of instructions."""
        self.instructions = set([])

    def add(self, gate):
        """Add instruction to set."""
        if not isinstance(gate, _instruction.Instruction):
            raise _qiskiterror.QISKitError("attempt to add non-Instruction" +
                                           " to InstructionSet")
        self.instructions.add(gate)

    def inverse(self):
        """Invert all instructions."""
        for instruction in self.instructions:
            instruction.inverse()
        return self

    def q_if(self, *qregs):
        """Add controls to all instructions."""
        for gate in self.instructions:
            gate.q_if(*qregs)
        return self

    def c_if(self, classical, val):
        """Add classical control register to all instructions."""
        for gate in self.instructions:
            gate.c_if(classical, val)
        return self
