# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,anomalous-backslash-in-string

from collections import OrderedDict

import pylatex


class Equation(pylatex.base_classes.Environment):
    _latex_name = 'equation*'


class QcircuitCommand(pylatex.base_classes.CommandBase):
    _latex_name = 'Qcircuit'

    def __init__(self, arguments=None, column_sep=None, row_sep=None):
        self.circuit = arguments
        self.size_info = {}
        if not column_sep is None:
            self.size_info['c_sep'] = column_sep
        else:
            self.size_info['c_sep'] = 0.5
        if not row_sep is None:
            self.size_info['r_sep'] = row_sep
        else:
            self.size_info['r_sep'] = 0.0
        super().__init__()

    def dumps(self):
        return r'\%s @C%.1fem @R%.1fem @!R {%s}' % (
            self.latex_name, self.size_info['c_sep'], self.size_info['r_sep'],
            self.circuit)


class Qcircuit(pylatex.base_classes.Environment):
    _latex_name = 'Qcircuit'

    def add_row(self, *cells):
        """Add a row of cells to the table.

        Args
        ----
        cells: iterable, such as a `list` or `tuple`
            There's two ways to use this method. The first method is to pass
            the content of each cell as a separate argument. The second method
            is to pass a single argument that is an iterable that contains each
            contents.
        color: str
            The name of the color used to highlight the row
        mapper: callable or `list`
            A function or a list of functions that should be called on all
            entries of the list after converting them to a string,
            for instance bold
        strict: bool
            Check for correct count of cells in row or not.
        """

        if len(cells) == 1:
            cells = cells[0]

        # Propagiate packages used in cells
        out_cells = []
        for c in cells:
            if isinstance(c, pylatex.base_classes.LatexObject):
                for p in c.packages:
                    self.packages.add(p)
                out_cells.append(c)
            elif isinstance(c, list):
                out_cells.append(pylatex.utils.dumps_list(c))
            else:
                out_cells.append(c)

        self.append(pylatex.utils.dumps_list(
            out_cells, escape=False, token='&',
            mapper=None) + pylatex.NoEscape(r'\\'))

    def dumps_content(self, **kwargs):
        r"""Represent the content of the circuit in LaTeX syntax.

        Args
        ----
        \*\*kwargs:
            Arguments that can be passed to `~.dumps_list`
        Returns
        -------
        string:
            A LaTeX string
        """

        content = ''
        content += super().dumps_content(**kwargs)

        return pylatex.NoEscape(content)

    def dumps(self):
        content = self.dumps_content()
        if not content.strip() and self.omit_if_empty:
            return ''
        arguments = pylatex.NoEscape(content) + self.content_separator
        output = QcircuitCommand(arguments)

        return pylatex.NoEscape(output.dumps())


class Gate(pylatex.base_classes.CommandBase):
    _latex_name = 'gate'


class Measure(pylatex.base_classes.CommandBase):
    _latex_name = 'meter'

    def __init__(self, cbit_label):
        super(Measure, self).__init__()
        self.cbit_label = cbit_label


class Barrier(pylatex.base_classes.CommandBase):
    _latex_name = 'barrier'
    horizontal_offset = None

    def set_offset(self, offset='-0.95em'):
        self._set_parameters(offset, 'options')
        self.horizontal_offset = offset


class Lstick(pylatex.base_classes.CommandBase):
    _latex_name = 'lstick'


class Ket(pylatex.base_classes.CommandBase):
    _latex_name = 'ket'


class QuantumWire(pylatex.base_classes.CommandBase):
    _latex_name = 'qw'


class ClassicWire(pylatex.base_classes.CommandBase):
    _latex_name = 'cw'


class ClassicWireX(pylatex.base_classes.CommandBase):
    _latex_name = 'cwx'
    span = None

    def __init__(self, label):
        super(ClassicWireX, self).__init__()
        self.label = label

    def set_span(self, span):
        self._set_parameters(span, 'options')
        self.span = span


class LatexCircuit(object):

    def __init__(self, ):
        self.circuit_list = {'qbits': OrderedDict(), 'cbits': OrderedDict()}

    def add_qbit(self, qbit_label):
        self.circuit_list['qbits'][qbit_label] = []

    def add_cbit(self, cbit_label):
        self.circuit_list['cbits'][cbit_label] = []

    def add_gate(self, qbit_label, gate_type):
        gate = Gate(arguments=gate_type)
        self.circuit_list['qbits'][qbit_label].append(gate)

    def add_barrier(self, qbit_label, span=None, horizontal_offset=None):
        if span is None:
            span = len(self.circuit_list['qbits'])
        barrier = Barrier(span)
        end_gate = self.circuit_list['qbits'][qbit_label][-1]
        if horizontal_offset:
            barrier.set_offset(horizontal_offset)
        self.circuit_list['qbits'][qbit_label][-1] = [end_gate, barrier]

    def add_meter(self, qbit_label, cbit_label):
        meter = Measure(cbit_label)
        self.circuit_list['qbits'][qbit_label].append(meter)

    def add_qw(self, qbit_label):
        self.circuit_list['qbits'][qbit_label].append(QuantumWire())

    def add_cw(self, cbit_label):
        self.circuit_list['cbits'][cbit_label].append(ClassicWire())

    def add_reset(self, qbit_label):
        reset = pylatex.NoEscape(
            "\\push{\\rule{.6em}{0em}\\ket{0}\\rule{.2em}{0em}}")
        self.circuit_list['qbits'][qbit_label].append(reset)
        self.add_qw(qbit_label)

    def add_cwx(self, cbit_label, end_label):
        cwx = ClassicWireX(end_label)
        end_gate = self.circuit_list['cbits'][cbit_label][-1]
        self.circuit_list['cbits'][cbit_label][-1] = [end_gate, cwx]

    def fill_circuit(self, width):
        for qbit in self.circuit_list['qbits']:
            current_length = len(self.circuit_list['qbits'][qbit]) + 1
            if current_length < width:
                for _ in range(width - (current_length + 1)):
                    self.add_qw(qbit)
        for cbit in self.circuit_list['cbits']:
            current_length = len(self.circuit_list['cbits'][cbit]) + 1
            if current_length < width:
                for _ in range(width - current_length):
                    self.add_cw(cbit)

    def generate_circuit_latex(self, doc, width=None):
        if width:
            self.fill_circuit(width)
        with doc.create(Qcircuit()) as circuit:
            for qbit in self.circuit_list['qbits']:
                row = [qbit] + self.circuit_list['qbits'][qbit]
                circuit.add_row(row)
            for cbit in self.circuit_list['cbits']:
                row = [cbit] + self.circuit_list['cbits'][cbit]
                circuit.add_row(row)

        
