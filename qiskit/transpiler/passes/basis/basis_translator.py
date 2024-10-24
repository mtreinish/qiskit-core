# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Translates gates to a target basis using a given equivalence library."""

import time
import logging

from functools import singledispatchmethod
from collections import defaultdict

from qiskit.circuit import (
    ControlFlowOp,
    QuantumCircuit,
    ParameterExpression,
)
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.circuit.controlflow import CONTROL_FLOW_OP_NAMES
from qiskit._accelerate.basis.basis_translator import basis_search, compose_transforms

logger = logging.getLogger(__name__)


class BasisTranslator(TransformationPass):
    """Translates gates to a target basis by searching for a set of translations
    from a given EquivalenceLibrary.

    This pass operates in several steps:

    * Determine the source basis from the input circuit.
    * Perform a Dijkstra search over basis sets, starting from the device's
      target_basis new gates are being generated using the rules from the provided
      EquivalenceLibrary and the search stops if all gates in the source basis have
      been generated.
    * The found path, as a set of rules from the EquivalenceLibrary, is composed
      into a set of gate replacement rules.
    * The composed replacement rules are applied in-place to each op node which
      is not already in the target_basis.

    If the target keyword argument is specified and that
    :class:`~qiskit.transpiler.Target` objects contains operations
    which are non-global (i.e. they are defined only for a subset of qubits),
    as calculated by :meth:`~qiskit.transpiler.Target.get_non_global_operation_names`,
    this pass will attempt to match the output translation to those constraints.
    For 1 qubit operations this is straightforward, the pass will perform a
    search using the union of the set of global operations with the set of operations
    defined solely on that qubit. For multi-qubit gates this is a bit more involved,
    while the behavior is initially similar to the single qubit case, just using all
    the qubits the operation is run on (where order is not significant) isn't sufficient.
    We also need to consider any potential local qubits defined on subsets of the
    quantum arguments for the multi-qubit operation. This means the target used for the
    search of a non-global multi-qubit gate is the union of global operations, non-global
    multi-qubit gates sharing the same qubits, and any non-global gates defined on
    any subset of the qubits used.


    .. note::

        In the case of non-global operations it is possible for a single
        execution of this pass to output an incomplete translation if any
        non-global gates are defined on qubits that are a subset of a larger
        multi-qubit gate. For example, if you have a ``u`` gate only defined on
        qubit 0 and an ``x`` gate only on qubit 1 it is possible when
        translating a 2 qubit operation on qubit 0 and 1 that the output might
        have ``u`` on qubit 1 and ``x`` on qubit 0. Typically running this pass
        a second time will correct these issues.

    .. _translation_errors:

    Translation Errors
    ------------------

    This pass will error if there is no path to translate an input gate to
    the specified basis. However, during a typical/default preset passmanager
    this pass gets run multiple times at different stages of the compilation
    pipeline. This means that potentially the input gates that are getting
    translated were not in the input circuit to :func:`~.transpile` as they
    were generated by an intermediate transform in the circuit.

    When this error occurs it typically means that either the target basis
    is not universal or there are additional equivalence rules needed in the
    :class:`~.EquivalenceLibrary` instance being used by the
    :class:`~.BasisTranslator` pass. You can refer to
    :ref:`custom_basis_gates` for details on adding custom equivalence rules.
    """

    def __init__(self, equivalence_library, target_basis, target=None, min_qubits=0):
        """Initialize a BasisTranslator instance.

        Args:
            equivalence_library (EquivalenceLibrary): The equivalence library
                which will be used by the BasisTranslator pass. (Instructions in
                this library will not be unrolled by this pass.)
            target_basis (list[str]): Target basis names to unroll to, e.g. ``['u3', 'cx']``.
            target (Target): The backend compilation target
            min_qubits (int): The minimum number of qubits for operations in the input
                dag to translate.
        """

        super().__init__()
        self._equiv_lib = equivalence_library
        self._target_basis = target_basis
        self._target = target
        self._non_global_operations = None
        self._qargs_with_non_global_operation = {}
        self._min_qubits = min_qubits
        if target is not None:
            self._non_global_operations = self._target.get_non_global_operation_names()
            self._qargs_with_non_global_operation = defaultdict(set)
            for gate in self._non_global_operations:
                for qarg in self._target[gate]:
                    self._qargs_with_non_global_operation[qarg].add(gate)

    def run(self, dag):
        """Translate an input DAGCircuit to the target basis.

        Args:
            dag (DAGCircuit): input dag

        Raises:
            TranspilerError: if the target basis cannot be reached

        Returns:
            DAGCircuit: translated circuit.
        """
        if self._target_basis is None and self._target is None:
            return dag

        qarg_indices = {qubit: index for index, qubit in enumerate(dag.qubits)}

        # Names of instructions assumed to supported by any backend.
        if self._target is None:
            basic_instrs = ["measure", "reset", "barrier", "snapshot", "delay", "store"]
            target_basis = set(self._target_basis)
            source_basis = set(self._extract_basis(dag))
            qargs_local_source_basis = {}
        else:
            basic_instrs = ["barrier", "snapshot", "store"]
            target_basis = self._target.keys() - set(self._non_global_operations)
            source_basis, qargs_local_source_basis = self._extract_basis_target(dag, qarg_indices)

        target_basis = set(target_basis).union(basic_instrs)
        # If the source basis is a subset of the target basis and we have no circuit
        # instructions on qargs that have non-global operations there is nothing to
        # translate and we can exit early.
        source_basis_names = {x[0] for x in source_basis}
        if source_basis_names.issubset(target_basis) and not qargs_local_source_basis:
            return dag

        logger.info(
            "Begin BasisTranslator from source basis %s to target basis %s.",
            source_basis,
            target_basis,
        )

        # Search for a path from source to target basis.
        search_start_time = time.time()
        basis_transforms = basis_search(self._equiv_lib, source_basis, target_basis)

        qarg_local_basis_transforms = {}
        for qarg, local_source_basis in qargs_local_source_basis.items():
            expanded_target = set(target_basis)
            # For any multiqubit operation that contains a subset of qubits that
            # has a non-local operation, include that non-local operation in the
            # search. This matches with the check we did above to include those
            # subset non-local operations in the check here.
            if len(qarg) > 1:
                for non_local_qarg, local_basis in self._qargs_with_non_global_operation.items():
                    if qarg.issuperset(non_local_qarg):
                        expanded_target |= local_basis
            else:
                expanded_target |= self._qargs_with_non_global_operation[tuple(qarg)]

            logger.info(
                "Performing BasisTranslator search from source basis %s to target "
                "basis %s on qarg %s.",
                local_source_basis,
                expanded_target,
                qarg,
            )
            local_basis_transforms = basis_search(
                self._equiv_lib, local_source_basis, expanded_target
            )

            if local_basis_transforms is None:
                raise TranspilerError(
                    "Unable to translate the operations in the circuit: "
                    f"{[x[0] for x in local_source_basis]} to the backend's (or manually "
                    f"specified) target basis: {list(expanded_target)}. This likely means the "
                    "target basis is not universal or there are additional equivalence rules "
                    "needed in the EquivalenceLibrary being used. For more details on this "
                    "error see: "
                    "https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes."
                    "BasisTranslator#translation-errors"
                )

            qarg_local_basis_transforms[qarg] = local_basis_transforms

        search_end_time = time.time()
        logger.info(
            "Basis translation path search completed in %.3fs.", search_end_time - search_start_time
        )

        if basis_transforms is None:
            raise TranspilerError(
                "Unable to translate the operations in the circuit: "
                f"{[x[0] for x in source_basis]} to the backend's (or manually specified) target "
                f"basis: {list(target_basis)}. This likely means the target basis is not universal "
                "or there are additional equivalence rules needed in the EquivalenceLibrary being "
                "used. For more details on this error see: "
                "https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes."
                "BasisTranslator#translation-errors"
            )

        # Compose found path into a set of instruction substitution rules.

        compose_start_time = time.time()
        instr_map = compose_transforms(basis_transforms, source_basis, dag)
        extra_instr_map = {
            qarg: compose_transforms(transforms, qargs_local_source_basis[qarg], dag)
            for qarg, transforms in qarg_local_basis_transforms.items()
        }

        compose_end_time = time.time()
        logger.info(
            "Basis translation paths composed in %.3fs.", compose_end_time - compose_start_time
        )

        # Replace source instructions with target translations.

        replace_start_time = time.time()

        def apply_translation(dag, wire_map):
            is_updated = False
            out_dag = dag.copy_empty_like()
            for node in dag.topological_op_nodes():
                node_qargs = tuple(wire_map[bit] for bit in node.qargs)
                qubit_set = frozenset(node_qargs)
                if node.name in target_basis or len(node.qargs) < self._min_qubits:
                    if node.name in CONTROL_FLOW_OP_NAMES:
                        flow_blocks = []
                        for block in node.op.blocks:
                            dag_block = circuit_to_dag(block)
                            updated_dag, is_updated = apply_translation(
                                dag_block,
                                {
                                    inner: wire_map[outer]
                                    for inner, outer in zip(block.qubits, node.qargs)
                                },
                            )
                            if is_updated:
                                flow_circ_block = dag_to_circuit(updated_dag)
                            else:
                                flow_circ_block = block
                            flow_blocks.append(flow_circ_block)
                        node.op = node.op.replace_blocks(flow_blocks)
                    out_dag.apply_operation_back(node.op, node.qargs, node.cargs, check=False)
                    continue
                if (
                    node_qargs in self._qargs_with_non_global_operation
                    and node.name in self._qargs_with_non_global_operation[node_qargs]
                ):
                    out_dag.apply_operation_back(node.op, node.qargs, node.cargs, check=False)
                    continue

                if dag._has_calibration_for(node):
                    out_dag.apply_operation_back(node.op, node.qargs, node.cargs, check=False)
                    continue
                if qubit_set in extra_instr_map:
                    self._replace_node(out_dag, node, extra_instr_map[qubit_set])
                elif (node.name, node.num_qubits) in instr_map:
                    self._replace_node(out_dag, node, instr_map)
                else:
                    raise TranspilerError(f"BasisTranslator did not map {node.name}.")
                is_updated = True
            return out_dag, is_updated

        out_dag, _ = apply_translation(dag, qarg_indices)
        replace_end_time = time.time()
        logger.info(
            "Basis translation instructions replaced in %.3fs.",
            replace_end_time - replace_start_time,
        )

        return out_dag

    def _replace_node(self, dag, node, instr_map):
        target_params, target_dag = instr_map[node.name, node.num_qubits]
        if len(node.params) != len(target_params):
            raise TranspilerError(
                "Translation num_params not equal to op num_params."
                f"Op: {node.params} {node.name} Translation: {target_params}\n{target_dag}"
            )
        if node.params:
            parameter_map = dict(zip(target_params, node.params))
            for inner_node in target_dag.topological_op_nodes():
                new_node = DAGOpNode.from_instruction(inner_node._to_circuit_instruction())
                new_node.qargs = tuple(
                    node.qargs[target_dag.find_bit(x).index] for x in inner_node.qargs
                )
                new_node.cargs = tuple(
                    node.cargs[target_dag.find_bit(x).index] for x in inner_node.cargs
                )

                if not new_node.is_standard_gate():
                    new_node.op = new_node.op.copy()
                if any(isinstance(x, ParameterExpression) for x in inner_node.params):
                    new_params = []
                    for param in new_node.params:
                        if not isinstance(param, ParameterExpression):
                            new_params.append(param)
                        else:
                            bind_dict = {x: parameter_map[x] for x in param.parameters}
                            if any(isinstance(x, ParameterExpression) for x in bind_dict.values()):
                                new_value = param
                                for x in bind_dict.items():
                                    new_value = new_value.assign(*x)
                            else:
                                new_value = param.bind(bind_dict)
                            if not new_value.parameters:
                                new_value = new_value.numeric()
                            new_params.append(new_value)
                    new_node.params = new_params
                    if not new_node.is_standard_gate():
                        new_node.op.params = new_params
                dag._apply_op_node_back(new_node)

            if isinstance(target_dag.global_phase, ParameterExpression):
                old_phase = target_dag.global_phase
                bind_dict = {x: parameter_map[x] for x in old_phase.parameters}
                if any(isinstance(x, ParameterExpression) for x in bind_dict.values()):
                    new_phase = old_phase
                    for x in bind_dict.items():
                        new_phase = new_phase.assign(*x)
                else:
                    new_phase = old_phase.bind(bind_dict)
                if not new_phase.parameters:
                    new_phase = new_phase.numeric()
                    if isinstance(new_phase, complex):
                        raise TranspilerError(f"Global phase must be real, but got '{new_phase}'")
                dag.global_phase += new_phase

        else:
            for inner_node in target_dag.topological_op_nodes():
                new_node = DAGOpNode.from_instruction(
                    inner_node._to_circuit_instruction(),
                )
                new_node.qargs = tuple(
                    node.qargs[target_dag.find_bit(x).index] for x in inner_node.qargs
                )
                new_node.cargs = tuple(
                    node.cargs[target_dag.find_bit(x).index] for x in inner_node.cargs
                )
                if not new_node.is_standard_gate:
                    new_node.op = new_node.op.copy()
                # dag_op may be the same instance as other ops in the dag,
                # so if there is a condition, need to copy
                if getattr(node.op, "condition", None):
                    new_node_op = new_node.op.to_mutable()
                    new_node_op.condition = node.op.condition
                    new_node.op = new_node_op
                dag._apply_op_node_back(new_node)
            if target_dag.global_phase:
                dag.global_phase += target_dag.global_phase

    @singledispatchmethod
    def _extract_basis(self, circuit):
        return circuit

    @_extract_basis.register
    def _(self, dag: DAGCircuit):
        for node in dag.op_nodes():
            if not dag._has_calibration_for(node) and len(node.qargs) >= self._min_qubits:
                yield (node.name, node.num_qubits)
            if node.name in CONTROL_FLOW_OP_NAMES:
                for block in node.op.blocks:
                    yield from self._extract_basis(block)

    @_extract_basis.register
    def _(self, circ: QuantumCircuit):
        for instruction in circ.data:
            operation = instruction.operation
            if (
                not circ._has_calibration_for(instruction)
                and len(instruction.qubits) >= self._min_qubits
            ):
                yield (operation.name, operation.num_qubits)
            if isinstance(operation, ControlFlowOp):
                for block in operation.blocks:
                    yield from self._extract_basis(block)

    def _extract_basis_target(
        self, dag, qarg_indices, source_basis=None, qargs_local_source_basis=None
    ):
        if source_basis is None:
            source_basis = set()
        if qargs_local_source_basis is None:
            qargs_local_source_basis = defaultdict(set)
        for node in dag.op_nodes():
            qargs = tuple(qarg_indices[bit] for bit in node.qargs)
            if dag._has_calibration_for(node) or len(node.qargs) < self._min_qubits:
                continue
            # Treat the instruction as on an incomplete basis if the qargs are in the
            # qargs_with_non_global_operation dictionary or if any of the qubits in qargs
            # are a superset for a non-local operation. For example, if the qargs
            # are (0, 1) and that's a global (ie no non-local operations on (0, 1)
            # operation but there is a non-local operation on (1,) we need to
            # do an extra non-local search for this op to ensure we include any
            # single qubit operation for (1,) as valid. This pattern also holds
            # true for > 2q ops too (so for 4q operations we need to check for 3q, 2q,
            # and 1q operations in the same manner)
            if qargs in self._qargs_with_non_global_operation or any(
                frozenset(qargs).issuperset(incomplete_qargs)
                for incomplete_qargs in self._qargs_with_non_global_operation
            ):
                qargs_local_source_basis[frozenset(qargs)].add((node.name, node.num_qubits))
            else:
                source_basis.add((node.name, node.num_qubits))
            if node.name in CONTROL_FLOW_OP_NAMES:
                for block in node.op.blocks:
                    block_dag = circuit_to_dag(block)
                    source_basis, qargs_local_source_basis = self._extract_basis_target(
                        block_dag,
                        {
                            inner: qarg_indices[outer]
                            for inner, outer in zip(block.qubits, node.qargs)
                        },
                        source_basis=source_basis,
                        qargs_local_source_basis=qargs_local_source_basis,
                    )
        return source_basis, qargs_local_source_basis
