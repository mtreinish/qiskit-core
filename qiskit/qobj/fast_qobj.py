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
                 memory=None, condition=None):
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

    @property
    def name(self):
        return self._data['name']

    @name.setter
    def name(self, value):
        self._data['name'] = value

    @property
    def params(self):
        return self._data.get('params')

    @params.setter
    def params(self, value):
        self._data['params'] = value

    @property
    def qubits(self):
        return self._data['qubits']

    @qubits.setter
    def qubits(self, value):
        self._data['qubits'] = value

    @property
    def register(self):
        return self._data.get('register')

    @register.setter
    def register(self, value):
        self._data['register'] = value

    @property
    def memory(self):
        return self._data['memory']

    @memory.setter
    def memory(self, value):
        self._data['memory'] = value

    @property
    def _condition(self):
        return self._data.get('_condition')

    @_condition.setter
    def _condition(self, value):
        self._data['_condition'] = value

    def to_dict(self):
        return self._data


class FastQasmExperiment:
    def __init__(self, config, header, instructions):
        """A fast qasm qobj experiment

        Args:
            config (dict): An experiment config dict
            header (dict): Am experiment header dict
            instruction (list): A list of :class:`FastQasmInstruction` objects

        """
        super(FastQasmExperiment, self).__init__()
        self._data = {}
        self.config = config
        self.header = header
        self.instructions = instructions

    def to_dict(self):
        out_dict = {
            'config': self.config,
            'header': self.header,
            'instructions': [x.to_dict for x in self.instructions]
        }
        return out_dict


class FastQasmQobj:
    def __init__(self, qobj_id=None, config=None, experiments=None,
                 header=None):
        """A Qasm Qobj object that's fast

        Args:
            qobj_id str: An identifier for the qobj
            config dict: A config for the entire run
            experiments list: A list of lists of :class:`FastQasmExperiment`
                objects representing an experiment

        """
        self.header = header or {}
        self.config = config or {}
        self.experiments = experiments or []
        self.qobj_id = qobj_id

    def to_dict(self, validate=True):
        out_dict = {
            'qobj_id': self.qobj_id,
            'header': self.header,
            'config': self.config,
            'schema_version': '1.1.0',
            'type': 'QASM',
            'experiments': [x.to_dict for x in self.experiments]
        }
        if validate:
            validator(out_dict)
        return out_dict
