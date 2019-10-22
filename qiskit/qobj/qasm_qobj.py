# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import os

import json
import fastjsonschema

from qiskit.validation.jsonschema import validate_json_against_schema


path_part = 'schemas/qobj_schema.json'
path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    path_part)
with open(path, 'r') as fd:
    json_schema = json.loads(fd.read())
validator = fastjsonschema.compile(json_schema)


class QasmQobjInstruction:
    def __init__(self, name, params=None, qubits=None, register=None,
                 memory=None, condition=None, label=None, mask=None,
                 relation=None, val=None):
        super(QasmQobjInstruction, self).__init__()
        self._data = {}
        self._data['name'] = name
        if params:
            self._data['params'] = params
        if qubits:
            self._data['qubits'] = qubits
        if register:
            self._data['register'] = register
        if memory:
            self._data['memory'] = memory
        if condition:
            self._data['_condition'] = condition
        if label:
            self._data['label'] = label
        if mask:
            self._data['mask'] = mask
        if relation:
            self._data['relation'] = relation
        if val:
            self._data['val'] = val

    @property
    def name(self):
        return self._data['name']

    @name.setter
    def name(self, value):
        self._data['name'] = value

    @property
    def params(self):
        param = self._data.get('params')
        if param is None:
            raise AttributeError
        return param

    @params.setter
    def params(self, value):
        self._data['params'] = value

    @property
    def qubits(self):
        qbits = self._data.get('qubits')
        if qbits is None:
            raise AttributeError
        return qbits

    @qubits.setter
    def qubits(self, value):
        self._data['qubits'] = value

    @property
    def register(self):
        reg = self._data.get('register')
        if reg is None:
            raise AttributeError
        return reg

    @register.setter
    def register(self, value):
        self._data['register'] = value

    @property
    def memory(self):
        mem = self._data.get('memory')
        if mem is None:
            raise AttributeError
        return mem

    @memory.setter
    def memory(self, value):
        self._data['memory'] = value

    @property
    def _condition(self):
        cond = self._data.get('_condition')
        if cond is None:
            raise AttributeError
        return cond

    @_condition.setter
    def _condition(self, value):
        self._data['_condition'] = value

    @_condition.deleter
    def _condition(self):
        del self._data['_condition']

    @property
    def conditional(self):
        cond = self._data.get('conditional')
        if cond is None:
            raise AttributeError
        return cond

    @conditional.setter
    def conditional(self, value):
        self._data['conditional'] = value

    @property
    def label(self):
        return self._data.get('label')

    @label.setter
    def label(self, value):
        self._data['label'] = value

    @property
    def mask(self):
        return self._data.get('mask')

    @mask.setter
    def mask(self, value):
        self._data['mask'] = value

    @property
    def relation(self):
        return self._data.get('relation')

    @relation.setter
    def relation(self, value):
        self._data['relation'] = value

    @property
    def val(self):
        return self._data.get('val')

    @val.setter
    def val(self, value):
        self._data['val'] = value

    def to_dict(self):
        return self._data

    @classmethod
    def from_dict(cls, data):
        name = data.pop('name')
        return cls(name, **data)

    def __eq__(self, other):
        return_val = False
        if isinstance(other, QasmQobjInstruction):
            if self.to_dict() == other.to_dict():
                return_val = True
        return return_val


class QasmQobjExperiment:
    def __init__(self, config=None, header=None, instructions=None):
        """A fast qasm qobj experiment

        Args:
            config (QasmQobjExperimentConfig): An experiment config dict
            header (QasmQobjExperimentHeader): Am experiment header dict
            instruction (list): A list of :class:`QasmQobjInstruction` objects

        """
        super(QasmQobjExperiment, self).__init__()
        self.config = config or QasmQobjExperimentConfig()
        self.header = header or QasmQobjExperimentHeader()
        self.instructions = instructions or []

    def to_dict(self):
        out_dict = {
            'config': self.config.to_dict(),
            'header': self.header.to_dict(),
            'instructions': [x.to_dict() for x in self.instructions]
        }
        return out_dict

    @classmethod
    def from_dict(cls, data):
        config = None
        if 'config' in data:
            config = QasmQobjExperimentConfig.from_dict(data.pop('config'))
        header = None
        if 'header' in data:
            header = QasmQobjExperimentHeader.from_dict(data.pop('header'))
        instructions = None
        if 'instructions' in data:
            instructions = [
                QasmQobjInstruction.from_dict(
                    inst) for inst in data.pop('instructions')]
        return cls(config, header, instructions)

    def __eq__(self, other):
        return_val = False
        if isinstance(other, QasmQobjExperiment):
            if self.to_dict() == other.to_dict():
                return_val = True
        return return_val


class QasmQobjConfig:
    _data = {}

    def __init__(self, shots=None, max_credits=None, seed_simulator=None,
                 memory=None, parameter_binds=None, n_qubits=None, **kwargs):
        """Model for RunConfig.

        Please note that this class only describes the required fields. For the
        full description of the model, please check ``RunConfigSchema``.

        Attributes:
            shots (int): the number of shots.
            max_credits (int): the max_credits to use on the IBMQ public devices.
            seed_simulator (int): the seed to use in the simulator
            memory (bool): whether to request memory from backend (per-shot readouts)
            parameter_binds (list[dict]): List of parameter bindings
            memory_slots (int): The number of memory slots on the device
            n_qubits (int): The number of qubits in the device
        """
        self._data = {}
        if shots is not None:
            self._data['shots'] = int(shots)

        if max_credits is not None:
            self._data['max_credits'] = int(max_credits)

        if seed_simulator is not None:
            self._data['seed_simulator'] = int(seed_simulator)

        if memory is not None:
            self._data['memory'] = bool(memory)

        if parameter_binds is not None:
            self._data['parameter_binds'] = parameter_binds

        if n_qubits is not None:
            self._data['n_qubits'] = n_qubits

        if kwargs:
            self._data.update(kwargs)

    @property
    def shots(self):
        return self._data.get('shots')

    @shots.setter
    def shots(self, value):
        self._data['shots'] = value

    @property
    def max_credits(self):
        return self._data.get('max_credits')

    @max_credits.setter
    def max_credits(self, value):
        self._data['max_credits'] = value

    @property
    def memory(self):
        return self._data.get('memory')

    @memory.setter
    def memory(self, value):
        self._data['memory'] = value

    @property
    def parameter_binds(self):
        return self._data.get('parameter_binds')

    @parameter_binds.setter
    def parameter_binds(self, value):
        self._data['parameter_binds'] = value

    @property
    def memory_slots(self):
        self._data.get('memory_slots')

    @memory_slots.setter
    def memory_slots(self, value):
        self._data['memory_slots'] = value

    @property
    def n_qubits(self):
        nqubits = self._data.get('n_qubits')
        if nqubits is None:
            raise AttributeError
        return nqubits

    @n_qubits.setter
    def n_qubits(self, value):
        self._data['n_qubits'] = value

    def to_dict(self):
        return self._data

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def __getattr__(self, name):
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError

    def __eq__(self, other):
        return_val = False
        if isinstance(other, QasmQobjConfig):
            if self.to_dict() == other.to_dict():
                return_val = True
        return return_val


class QasmQobjExperimentHeader:
    _data = {}

    def __init__(self, **kwargs):
        self._data = kwargs

    def __getstate__(self):
        return self._data

    def __setstate__(self, state):
        self._data = state

    def __getattr__(self, attr):
        try:
            return self._data[attr]
        except KeyError:
            raise AttributeError

    def __setattr__(self, name, value):
        if not hasattr(self, name):
            self._data[name] = value
        else:
            super().__setattr__(name, value)

    def to_dict(self):
        return self._data

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def __eq__(self, other):
        return_val = False
        if isinstance(other, QasmQobjExperimentHeader):
            if self.to_dict() == other.to_dict():
                return_val = True
        return return_val



class QasmQobjExperimentConfig:
    _data = {}
    def __init__(self, **kwargs):
        self._data = kwargs

    def __getstate__(self):
        return self._data

    def __setstate__(self, state):
        self._data = state

    def __getattr__(self, attr):
        try:
            return self._data[attr]
        except KeyError:
            raise AttributeError

    def __setattr__(self, name, value):
        if not hasattr(self, name):
            self._data[name] = value
        else:
            super().__setattr__(name, value)

    def to_dict(self):
        return self._data

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def __eq__(self, other):
        return_val = False
        if isinstance(other, QasmQobjExperimentConfig):
            if self.to_dict() == other.to_dict():
                return_val = True
        return return_val

class QobjHeader:
    _data = {}

    def __init__(self, **kwargs):
        self._data = kwargs

    def __getstate__(self):
        return self._data

    def __setstate__(self, state):
        self._data = state

    def __getattr__(self, attr):
        try:
            return self._data[attr]
        except KeyError:
            raise AttributeError

    def __setattr__(self, name, value):
        if not hasattr(self, name):
            self._data[name] = value
        else:
            super().__setattr__(name, value)

    def to_dict(self):
        return self._data

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def __eq__(self, other):
        return_val = False
        if isinstance(other, QobjHeader):
            if self.to_dict() == other.to_dict():
                return_val = True
        return return_val

class QobjExperimentHeader(QobjHeader):
    pass


class QasmQobj:
    def __init__(self, qobj_id=None, config=None, experiments=None,
                 header=None):
        """A Qasm Qobj object that's fast

        Args:
            qobj_id str: An identifier for the qobj
            config QasmQobjRunConfig: A config for the entire run
            experiments list: A list of lists of :class:`QasmQobjExperiment`
                objects representing an experiment

        """
        self.header = header or QobjHeader()
        self.config = config or QasmQobjConfig()
        self.experiments = experiments or []
        self.qobj_id = qobj_id

    def to_dict(self, validate=False):
        out_dict = {
            'qobj_id': self.qobj_id,
            'header': self.header.to_dict(),
            'config': self.config.to_dict(),
            'schema_version': '1.1.0',
            'type': 'QASM',
            'experiments': [x.to_dict() for x in self.experiments]
        }
        if validate:
            validator(out_dict)
        return out_dict

    @classmethod
    def from_dict(cls, data):
        config = None
        if 'config' in data:
            config = QasmQobjConfig.from_dict(data['config'])
        experiments = None
        if 'experiments' in data:
            experiments = [
                QasmQobjExperiment.from_dict(
                    exp) for exp in data['experiments']]
        header = None
        if 'header' in data:
            header = QobjHeader.from_dict(data['header'])

        return cls(qobj_id=data.get('qobj_id'), config=config,
                   experiments=experiments, header=header)

    def __eq__(self, other):
        return_val = False
        if isinstance(other, QasmQobj):
            if self.to_dict() == other.to_dict():
                return_val = True
        return return_val
