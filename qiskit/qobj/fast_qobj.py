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
code = fastjsonschema.compile_to_code(json_schema)
with open('/tmp/jsonschema.py', 'w') as f:
    f.write(code)


class FastQasmInstruction:
    def __init__(self, name, params=None, qubits=None, register=None,
                 memory=None, condition=None, label=None, mask=None,
                 relation=None, val=None):
        super(FastQasmInstruction, self).__init__()
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


class FastQasmExperiment:
    def __init__(self, config, header, instructions):
        """A fast qasm qobj experiment

        Args:
            config (dict): An experiment config dict
            header (dict): Am experiment header dict
            instruction (list): A list of :class:`FastQasmInstruction` objects

        """
        super(FastQasmExperiment, self).__init__()
        self.config = config
        self.header = header
        self.instructions = instructions

    def to_dict(self):
        out_dict = {
            'config': self.config,
            'header': self.header,
            'instructions': [x.to_dict() for x in self.instructions]
        }
        return out_dict

    @classmethod
    def from_dict(cls, data):
        config = data.pop('name')
        header = data.pop('header')
        instructions = data.pop('instructions')
        return cls(config, header, instructions)


class FastRunConfig:

    def __init__(self, shots=None, max_credits=None, seed_simulator=None,
                 memory=None, parameter_binds=None, **kwargs):
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
        self._data.get('n_qubits')

    @n_qubits.setter
    def n_qubits(self, value):
        self._data['n_qubits'] = value

    def to_dict(self):
        return self._data

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


class FastQasmQobj:
    def __init__(self, qobj_id=None, config=None, experiments=None,
                 header=None):
        """A Qasm Qobj object that's fast

        Args:
            qobj_id str: An identifier for the qobj
            config FastRunConfig: A config for the entire run
            experiments list: A list of lists of :class:`FastQasmExperiment`
                objects representing an experiment

        """
        self.header = header or {}
        self.config = config or FastRunConfig()
        self.experiments = experiments or []
        self.qobj_id = qobj_id

    def to_dict(self, validate=False):
        out_dict = {
            'qobj_id': self.qobj_id,
            'header': self.header,
            'config': self.config.to_dict(),
            'schema_version': '1.1.0',
            'type': 'QASM',
            'experiments': [x.to_dict() for x in self.experiments]
        }
        if validate:
            validator(out_dict)
        return out_dict
