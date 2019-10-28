# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Exception for errors raised by the transpiler.
"""
import warnings
from qiskit.exceptions import QiskitError


class TranspilerAccessError(QiskitError):
    """Exception of access error in the transpiler passes."""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(' '.join(message))
        warnings.warn('The exception TranspilerAccessError is deprecated as of 0.11.0 '
                      'and will be removed no earlier than 3 months after that release '
                      'date. You should use the exception TranspilerError instead.',
                      DeprecationWarning, stacklevel=2)


class TranspilerError(TranspilerAccessError):
    """Exceptions raised during transpilation."""


class CouplingError(QiskitError):
    """Base class for errors raised by the coupling graph object."""

    def __init__(self, *msg):
        """Set the error message."""
        super().__init__(*msg)
        self.msg = ' '.join(msg)

    def __str__(self):
        """Return the message."""
        return repr(self.msg)


class LayoutError(QiskitError):
    """Errors raised by the layout object."""

    def __init__(self, *msg):
        """Set the error message."""
        super().__init__(*msg)
        self.msg = ' '.join(msg)

    def __str__(self):
        """Return the message."""
        return repr(self.msg)
