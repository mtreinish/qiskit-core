# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Module for working with results."""

from qiskit.terra.result import _result
from qiskit.terra.result import _resulterror

__all__ = _result.__all__ + _resulterror.__all__
