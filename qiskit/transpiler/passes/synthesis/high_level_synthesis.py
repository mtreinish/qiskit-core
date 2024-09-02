# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
High-level-synthesis transpiler pass.
"""

from __future__ import annotations

import typing
from functools import partial
from collections.abc import Callable

import numpy as np

from qiskit.circuit.annotated_operation import Modifier
from qiskit.circuit.operation import Operation
from qiskit.circuit.instruction import Instruction
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import ControlledGate, EquivalenceLibrary, equivalence
from qiskit.transpiler.passes.utils import control_flow
from qiskit.transpiler.target import Target
from qiskit.transpiler.coupling import CouplingMap
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.transpiler.exceptions import TranspilerError

from qiskit.circuit.annotated_operation import (
    AnnotatedOperation,
    InverseModifier,
    ControlModifier,
    PowerModifier,
)

from .plugin import HighLevelSynthesisPluginManager
from .qubit_tracker import QubitTracker

if typing.TYPE_CHECKING:
    from qiskit.dagcircuit import DAGOpNode


class HLSConfig:
    """The high-level-synthesis config allows to specify a list of "methods" used by
    :class:`~.HighLevelSynthesis` transformation pass to synthesize different types
    of higher-level objects.

    A higher-level object is an object of type :class:`~.Operation` (e.g., :class:`.Clifford` or
    :class:`.LinearFunction`).  Each object is referred to by its :attr:`~.Operation.name` field
    (e.g., ``"clifford"`` for :class:`.Clifford` objects), and the applicable synthesis methods are
    tied to this name.

    In the config, each method is specified in one of several ways:

    1. a tuple consisting of the name of a known synthesis plugin and a dictionary providing
       additional arguments for the algorithm.
    2. a tuple consisting of an instance of :class:`.HighLevelSynthesisPlugin` and additional
       arguments for the algorithm.
    3. a single string of a known synthesis plugin
    4. a single instance of :class:`.HighLevelSynthesisPlugin`.

    The following example illustrates different ways how a config file can be created::

        from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig
        from qiskit.transpiler.passes.synthesis.high_level_synthesis import ACGSynthesisPermutation

        # All the ways to specify hls_config are equivalent
        hls_config = HLSConfig(permutation=[("acg", {})])
        hls_config = HLSConfig(permutation=["acg"])
        hls_config = HLSConfig(permutation=[(ACGSynthesisPermutation(), {})])
        hls_config = HLSConfig(permutation=[ACGSynthesisPermutation()])

    The names of the synthesis plugins should be declared in ``entry-points`` table for
    ``qiskit.synthesis`` in ``pyproject.toml``, in the form
    <higher-level-object-name>.<synthesis-method-name>.

    The standard higher-level-objects are recommended to have a synthesis method
    called "default", which would be called automatically when synthesizing these objects,
    without having to explicitly set these methods in the config.

    To avoid synthesizing a given higher-level-object, one can give it an empty list of methods.

    For an explicit example of using such config files, refer to the documentation for
    :class:`~.HighLevelSynthesis`.

    For an overview of the complete process of using high-level synthesis, see
    :ref:`using-high-level-synthesis-plugins`.
    """

    def __init__(
        self,
        use_default_on_unspecified: bool = True,
        plugin_selection: str = "sequential",
        plugin_evaluation_fn: Callable[[QuantumCircuit], int] | None = None,
        **kwargs,
    ):
        """Creates a high-level-synthesis config.

        Args:
            use_default_on_unspecified: if True, every higher-level-object without an
                explicitly specified list of methods will be synthesized using the "default"
                algorithm if it exists.
            plugin_selection: if set to ``"sequential"`` (default), for every higher-level-object
                the synthesis pass will consider the specified methods sequentially, stopping
                at the first method that is able to synthesize the object. If set to ``"all"``,
                all the specified methods will be considered, and the best synthesized circuit,
                according to ``plugin_evaluation_fn`` will be chosen.
            plugin_evaluation_fn: a callable that evaluates the quality of the synthesized
                quantum circuit; a smaller value means a better circuit. If ``None``, the
                quality of the circuit its size (i.e. the number of gates that it contains).
            kwargs: a dictionary mapping higher-level-objects to lists of synthesis methods.
        """
        self.use_default_on_unspecified = use_default_on_unspecified
        self.plugin_selection = plugin_selection
        self.plugin_evaluation_fn = (
            plugin_evaluation_fn if plugin_evaluation_fn is not None else lambda qc: qc.size()
        )
        self.methods = {}

        for key, value in kwargs.items():
            self.set_methods(key, value)

    def set_methods(self, hls_name, hls_methods):
        """Sets the list of synthesis methods for a given higher-level-object. This overwrites
        the lists of methods if also set previously."""
        self.methods[hls_name] = hls_methods


class HighLevelSynthesis(TransformationPass):
    r"""Synthesize higher-level objects and unroll custom definitions.

    The input to this pass is a DAG that may contain higher-level objects,
    including abstract mathematical objects (e.g., objects of type :class:`.LinearFunction`),
    annotated operations (objects of type :class:`.AnnotatedOperation`), and
    custom gates.

    In the most common use-case when either ``basis_gates`` or ``target`` is specified,
    all higher-level objects are synthesized, so the output is a :class:`.DAGCircuit`
    without such objects.
    More precisely, every gate in the output DAG is either directly supported by the target,
    or is in ``equivalence_library``.

    The abstract mathematical objects are synthesized using synthesis plugins, applying
    synthesis methods specified in the high-level-synthesis config (refer to the documentation
    for :class:`~.HLSConfig`).

    As an example, let us assume that ``op_a`` and ``op_b`` are names of two higher-level objects,
    that ``op_a``-objects have two synthesis methods ``default`` which does require any additional
    parameters and ``other`` with two optional integer parameters ``option_1`` and ``option_2``,
    that ``op_b``-objects have a single synthesis method ``default``, and ``qc`` is a quantum
    circuit containing ``op_a`` and ``op_b`` objects. The following code snippet::

        hls_config = HLSConfig(op_b=[("other", {"option_1": 7, "option_2": 4})])
        pm = PassManager([HighLevelSynthesis(hls_config=hls_config)])
        transpiled_qc = pm.run(qc)

    shows how to run the alternative synthesis method ``other`` for ``op_b``-objects, while using the
    ``default`` methods for all other high-level objects, including ``op_a``-objects.

    The annotated operations (consisting of a base operation and a list of inverse, control and power
    modifiers) are synthesizing recursively, first synthesizing the base operation, and then applying
    synthesis methods for creating inverted, controlled, or powered versions of that).

    The custom gates are synthesized by recursively unrolling their definitions, until every gate
    is either supported by the target or is in the equivalence library.

    When neither ``basis_gates`` nor ``target`` is specified, the pass synthesizes only the top-level
    abstract mathematical objects and annotated operations, without descending into the gate
    ``definitions``. This is consistent with the older behavior of the pass, allowing to synthesize
    some higher-level objects using plugins and leaving the other gates untouched.

    The high-level-synthesis passes information about available auxiliary qubits, and whether their
    state is clean (defined as :math:`|0\rangle`) or dirty (unknown state) to the synthesis routine
    via the respective arguments ``"num_clean_ancillas"`` and ``"num_dirty_ancillas"``.
    If ``qubits_initially_zero`` is ``True`` (default), the qubits are assumed to be in the
    :math:`|0\rangle` state. When appending a synthesized block using auxiliary qubits onto the
    circuit, we first use the clean auxiliary qubits.

    .. note::

        Synthesis methods are assumed to maintain the state of the auxiliary qubits.
        Concretely this means that clean auxiliary qubits must still be in the :math:`|0\rangle`
        state after the synthesized block, while dirty auxiliary qubits are re-used only
        as dirty qubits.

    """

    def __init__(
        self,
        hls_config: HLSConfig | None = None,
        coupling_map: CouplingMap | None = None,
        target: Target | None = None,
        use_qubit_indices: bool = False,
        equivalence_library: EquivalenceLibrary | None = None,
        basis_gates: list[str] | None = None,
        min_qubits: int = 0,
        qubits_initially_zero: bool = True,
    ):
        r"""
        HighLevelSynthesis initializer.

        Args:
            hls_config: Optional, the high-level-synthesis config that specifies synthesis methods
                and parameters for various high-level-objects in the circuit. If it is not specified,
                the default synthesis methods and parameters will be used.
            coupling_map: Optional, directed graph represented as a coupling map.
            target: Optional, the backend target to use for this pass. If it is specified,
                it will be used instead of the coupling map.
            use_qubit_indices: a flag indicating whether this synthesis pass is running before or after
                the layout is set, that is, whether the qubit indices of higher-level-objects correspond
                to qubit indices on the target backend.
            equivalence_library: The equivalence library used (instructions in this library will not
                be unrolled by this pass).
            basis_gates: Optional, target basis names to unroll to, e.g. `['u3', 'cx']`.
                Ignored if ``target`` is also specified.
            min_qubits: The minimum number of qubits for operations in the input
                dag to translate.
            qubits_initially_zero: Indicates whether the qubits are initially in the state
                :math:`|0\rangle`. This allows the high-level-synthesis to use clean auxiliary qubits
                (i.e. in the zero state) to synthesize an operation.
        """
        super().__init__()

        if hls_config is not None:
            self.hls_config = hls_config
        else:
            # When the config file is not provided, we will use the "default" method
            # to synthesize Operations (when available).
            self.hls_config = HLSConfig(True)

        self.hls_plugin_manager = HighLevelSynthesisPluginManager()
        self._coupling_map = coupling_map
        self._target = target
        self._use_qubit_indices = use_qubit_indices
        self.qubits_initially_zero = qubits_initially_zero
        if target is not None:
            self._coupling_map = self._target.build_coupling_map()
        self._equiv_lib = equivalence_library
        self._basis_gates = basis_gates
        self._min_qubits = min_qubits

        self._top_level_only = self._basis_gates is None and self._target is None

        # include path for when target exists but target.num_qubits is None (BasicSimulator)
        if not self._top_level_only and (self._target is None or self._target.num_qubits is None):
            basic_insts = {"measure", "reset", "barrier", "snapshot", "delay", "store"}
            self._device_insts = basic_insts | set(self._basis_gates)
        else:
            self._device_insts = set()

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the HighLevelSynthesis pass on `dag`.

        Args:
            dag: input dag.

        Returns:
            Output dag with higher-level operations synthesized.

        Raises:
            TranspilerError: when the transpiler is unable to synthesize the given DAG
            (for instance, when the specified synthesis method is not available).
        """
        qubits = tuple(dag.find_bit(q).index for q in dag.qubits)
        if self.qubits_initially_zero:
            clean, dirty = set(qubits), set()
        else:
            clean, dirty = set(), set(qubits)

        tracker = QubitTracker(qubits=qubits, clean=clean, dirty=dirty)
        return self._run(dag, tracker)

    def _run(self, dag: DAGCircuit, tracker: QubitTracker) -> DAGCircuit:
        # Check if HighLevelSynthesis can be skipped.
        all_skipped = True
        for node in dag.topological_op_nodes():
            qubits = tuple(dag.find_bit(q).index for q in node.qargs)
            if not (
                dag.has_calibration_for(node)
                or len(node.qargs) < self._min_qubits
                or node.is_directive()
                or self._definitely_skip_node(node, qubits)
            ):
                all_skipped = False
                break
        if all_skipped:
            return dag

        # Start by analyzing the nodes in the DAG. This for-loop is a first version of a potentially
        # more elaborate approach to find good operation/ancilla allocations. It greedily iterates
        # over the nodes, checking whether we can synthesize them, while keeping track of the
        # qubit states. It does not trade-off allocations and just gives all available qubits
        # to the current operation (a "the-first-takes-all" approach).
        synthesized_nodes = {}

        for node in dag.topological_op_nodes():
            qubits = tuple(dag.find_bit(q).index for q in node.qargs)
            synthesized = None
            used_qubits = None

            # check if synthesis for the operation can be skipped
            if (
                dag.has_calibration_for(node)
                or len(node.qargs) < self._min_qubits
                or node.is_directive()
                or self._definitely_skip_node(node, qubits)
            ):
                pass

            # next check control flow
            elif node.is_control_flow():
                dag.substitute_node(
                    node,
                    control_flow.map_blocks(partial(self._run, tracker=tracker.copy()), node.op),
                    propagate_condition=False,
                )

            # now we are free to synthesize
            else:
                # this returns the synthesized operation and the qubits it acts on -- note that this
                # may be different from the original qubits, since we may use auxiliary qubits
                synthesized, used_qubits = self._synthesize_operation(node.op, qubits, tracker)

            # if the synthesis changed the operation (i.e. it is not None), store the result
            # and mark the operation qubits as used
            if synthesized is not None:
                synthesized_nodes[node] = (synthesized, used_qubits)
                tracker.used(qubits)  # assumes that auxiliary are returned in the same state

            # if the synthesis did not change anything, just update the qubit tracker
            # other cases can be added: swaps, controlled gates (e.g. if control is 0), ...
            else:
                if node.op.name in ["id", "delay", "barrier"]:
                    pass  # tracker not updated, these are no-ops
                elif node.op.name == "reset":
                    tracker.reset(qubits)  # reset qubits to 0
                else:
                    tracker.used(qubits)  # any other op used the clean state up

        # we did not change anything just return the input
        if len(synthesized_nodes) == 0:
            return dag

        # Otherwise, we will rebuild with the new operations. Note that we could also
        # check if no operation changed in size and substitute in-place, but rebuilding is
        # generally as fast or faster, unless very few operations are changed.
        out = dag.copy_empty_like()
        index_to_qubit = dict(enumerate(dag.qubits))

        for node in dag.topological_op_nodes():
            if node in synthesized_nodes:
                op, qubits = synthesized_nodes[node]
                qargs = tuple(index_to_qubit[index] for index in qubits)
                if isinstance(op, Operation):
                    out.apply_operation_back(op, qargs, cargs=[])
                    continue

                if isinstance(op, QuantumCircuit):
                    op = circuit_to_dag(op, copy_operations=False)

                if isinstance(op, DAGCircuit):
                    qubit_map = {
                        qubit: index_to_qubit[index] for index, qubit in zip(qubits, op.qubits)
                    }
                    clbit_map = dict(zip(op.clbits, node.cargs))
                    for sub_node in op.op_nodes():
                        out.apply_operation_back(
                            sub_node.op,
                            tuple(qubit_map[qarg] for qarg in sub_node.qargs),
                            tuple(clbit_map[carg] for carg in sub_node.cargs),
                        )
                    out.global_phase += op.global_phase
                else:
                    raise RuntimeError(f"Unexpected synthesized type: {type(op)}")
            else:
                out.apply_operation_back(node.op, node.qargs, node.cargs, check=False)

        return out

    def _synthesize_operation(
        self,
        operation: Operation,
        qubits: tuple[int],
        tracker: QubitTracker,
    ) -> tuple[QuantumCircuit | Operation | DAGCircuit | None, list[int] | None]:
        # Try to synthesize the operation. We'll go through the following options:
        #  (1) Annotations: if the operator is annotated, synthesize the base operation
        #       and then apply the modifiers. Returns a circuit (e.g. applying a power)
        #       or operation (e.g adding control on an X gate).
        #  (2) High-level objects: try running the battery of high-level synthesis plugins (e.g.
        #       if the operation is a Clifford). Returns a circuit.
        #  (3) Unrolling custom definitions: try defining the operation if it is not yet
        #       in the set of supported instructions. Returns a circuit.
        # If any of the above were triggered, we will recurse and go again through these steps
        # until no further change occurred. At this point, we convert circuits to DAGs (the final
        # possible return type). If there was no change, we just return ``None``.
        synthesized = None

        # Try synthesizing via AnnotatedOperation. This is faster than an isinstance check
        # but a bit less safe since someone could create operations with a ``modifiers`` attribute.
        if len(modifiers := getattr(operation, "modifiers", [])) > 0:
            # The base operation must be synthesized without using potential control qubits
            # used in the modifiers.
            num_ctrl = sum(
                mod.num_ctrl_qubits for mod in modifiers if isinstance(mod, ControlModifier)
            )
            baseop_qubits = qubits[num_ctrl:]  # reminder: control qubits are the first ones
            baseop_tracker = tracker.copy(drop=qubits[:num_ctrl])  # no access to control qubits

            # get qubits of base operation
            synthesized_base_op, _ = self._synthesize_operation(
                operation.base_op, baseop_qubits, baseop_tracker
            )
            if synthesized_base_op is None:
                synthesized_base_op = operation.base_op
            elif isinstance(synthesized_base_op, DAGCircuit):
                synthesized_base_op = dag_to_circuit(synthesized_base_op)

            synthesized = self._apply_annotations(synthesized_base_op, operation.modifiers)

        # If it was no AnnotatedOperation, try synthesizing via HLS or by unrolling.
        else:
            # Try synthesis via HLS -- which will return ``None`` if unsuccessful.
            indices = qubits if self._use_qubit_indices else None
            if len(hls_methods := self._methods_to_try(operation.name)) > 0:
                synthesized = self._synthesize_op_using_plugins(
                    hls_methods,
                    operation,
                    indices,
                    tracker.num_clean(qubits),
                    tracker.num_dirty(qubits),
                )

            # If HLS did not apply, or was unsuccessful, try unrolling custom definitions.
            if synthesized is None and not self._top_level_only:
                synthesized = self._unroll_custom_definition(operation, indices)

        if synthesized is None:
            # if we didn't synthesize, there was nothing to unroll, so just set the used qubits
            used_qubits = qubits

        else:
            # if it has been synthesized, recurse and finally store the decomposition
            if isinstance(synthesized, Operation):
                re_synthesized, qubits = self._synthesize_operation(
                    synthesized, qubits, tracker.copy()
                )
                if re_synthesized is not None:
                    synthesized = re_synthesized
                used_qubits = qubits

            elif isinstance(synthesized, QuantumCircuit):
                aux_qubits = tracker.borrow(synthesized.num_qubits - len(qubits), qubits)
                used_qubits = qubits + tuple(aux_qubits)
                as_dag = circuit_to_dag(synthesized, copy_operations=False)

                # map used qubits to subcircuit
                new_qubits = [as_dag.find_bit(q).index for q in as_dag.qubits]
                qubit_map = dict(zip(used_qubits, new_qubits))

                synthesized = self._run(as_dag, tracker.copy(qubit_map))
                if synthesized.num_qubits() != len(used_qubits):
                    raise RuntimeError(
                        f"Mismatching number of qubits, using {synthesized.num_qubits()} "
                        f"but have {len(used_qubits)}."
                    )

            else:
                raise RuntimeError(f"Unexpected synthesized type: {type(synthesized)}")

        if synthesized is not None and used_qubits is None:
            raise RuntimeError("Failed to find qubit indices on", synthesized)

        return synthesized, used_qubits

    def _unroll_custom_definition(
        self, inst: Instruction, qubits: list[int] | None
    ) -> QuantumCircuit | None:
        # check if the operation is already supported natively
        if not (isinstance(inst, ControlledGate) and inst._open_ctrl):
            # include path for when target exists but target.num_qubits is None (BasicSimulator)
            inst_supported = self._instruction_supported(inst.name, qubits)
            if inst_supported or (self._equiv_lib is not None and self._equiv_lib.has_entry(inst)):
                return None  # we support this operation already

        # if not, try to get the definition
        try:
            definition = inst.definition
        except (TypeError, AttributeError) as err:
            raise TranspilerError(f"HighLevelSynthesis was unable to define {inst.name}.") from err

        if definition is None:
            raise TranspilerError(f"HighLevelSynthesis was unable to synthesize {inst}.")

        return definition

    def _methods_to_try(self, name: str):
        """Get a sequence of methods to try for a given op name."""
        if (methods := self.hls_config.methods.get(name)) is not None:
            # the operation's name appears in the user-provided config,
            # we use the list of methods provided by the user
            return methods
        if (
            self.hls_config.use_default_on_unspecified
            and "default" in self.hls_plugin_manager.method_names(name)
        ):
            # the operation's name does not appear in the user-specified config,
            # we use the "default" method when instructed to do so and the "default"
            # method is available
            return ["default"]
        return []

    def _synthesize_op_using_plugins(
        self,
        hls_methods: list,
        op: Operation,
        qubits: list[int] | None,
        num_clean_ancillas: int = 0,
        num_dirty_ancillas: int = 0,
    ) -> QuantumCircuit | None:
        """
        Attempts to synthesize op using plugin mechanism.

        The arguments ``num_clean_ancillas`` and ``num_dirty_ancillas`` specify
        the number of clean and dirty qubits available to synthesize the given
        operation. A synthesis method does not need to use these additional qubits.

        Returns either the synthesized circuit or None (which may occur
        when no synthesis methods is available or specified, or when there is
        an insufficient number of auxiliary qubits).
        """
        hls_plugin_manager = self.hls_plugin_manager

        best_decomposition = None
        best_score = np.inf

        for method in hls_methods:
            # There are two ways to specify a synthesis method. The more explicit
            # way is to specify it as a tuple consisting of a synthesis algorithm and a
            # list of additional arguments, e.g.,
            #   ("kms", {"all_mats": 1, "max_paths": 100, "orig_circuit": 0}), or
            #   ("pmh", {}).
            # When the list of additional arguments is empty, one can also specify
            # just the synthesis algorithm, e.g.,
            #   "pmh".
            if isinstance(method, tuple):
                plugin_specifier, plugin_args = method
            else:
                plugin_specifier = method
                plugin_args = {}

            # There are two ways to specify a synthesis algorithm being run,
            # either by name, e.g. "kms" (which then should be specified in entry_points),
            # or directly as a class inherited from HighLevelSynthesisPlugin (which then
            # does not need to be specified in entry_points).
            if isinstance(plugin_specifier, str):
                if plugin_specifier not in hls_plugin_manager.method_names(op.name):
                    raise TranspilerError(
                        f"Specified method: {plugin_specifier} not found in available "
                        f"plugins for {op.name}"
                    )
                plugin_method = hls_plugin_manager.method(op.name, plugin_specifier)
            else:
                plugin_method = plugin_specifier

            # Set the number of available clean and dirty auxiliary qubits via plugin args.
            plugin_args["num_clean_ancillas"] = num_clean_ancillas
            plugin_args["num_dirty_ancillas"] = num_dirty_ancillas

            decomposition = plugin_method.run(
                op,
                coupling_map=self._coupling_map,
                target=self._target,
                qubits=qubits,
                **plugin_args,
            )

            # The synthesis methods that are not suited for the given higher-level-object
            # will return None.
            if decomposition is not None:
                if self.hls_config.plugin_selection == "sequential":
                    # In the "sequential" mode the first successful decomposition is
                    # returned.
                    best_decomposition = decomposition
                    break

                # In the "run everything" mode we update the best decomposition
                # discovered
                current_score = self.hls_config.plugin_evaluation_fn(decomposition)
                if current_score < best_score:
                    best_decomposition = decomposition
                    best_score = current_score

        return best_decomposition

    def _apply_annotations(
        self, synthesized: Operation | QuantumCircuit, modifiers: list[Modifier]
    ) -> QuantumCircuit:
        """
        Recursively synthesizes annotated operations.
        Returns either the synthesized operation or None (which occurs when the operation
        is not an annotated operation).
        """
        for modifier in modifiers:
            if isinstance(modifier, InverseModifier):
                # Both QuantumCircuit and Gate have inverse method
                synthesized = synthesized.inverse()

            elif isinstance(modifier, ControlModifier):
                # Both QuantumCircuit and Gate have control method, however for circuits
                # it is more efficient to avoid constructing the controlled quantum circuit.
                if isinstance(synthesized, QuantumCircuit):
                    synthesized = synthesized.to_gate()

                synthesized = synthesized.control(
                    num_ctrl_qubits=modifier.num_ctrl_qubits,
                    label=None,
                    ctrl_state=modifier.ctrl_state,
                    annotated=False,
                )

                if isinstance(synthesized, AnnotatedOperation):
                    raise TranspilerError(
                        "HighLevelSynthesis failed to synthesize the control modifier."
                    )

            elif isinstance(modifier, PowerModifier):
                # QuantumCircuit has power method, and Gate needs to be converted
                # to a quantum circuit.
                if not isinstance(synthesized, QuantumCircuit):
                    synthesized = _instruction_to_circuit(synthesized)

                synthesized = synthesized.power(modifier.power)

            else:
                raise TranspilerError(f"Unknown modifier {modifier}.")

        return synthesized

    def _definitely_skip_node(self, node: DAGOpNode, qubits: tuple[int] | None) -> bool:
        """Fast-path determination of whether a node can certainly be skipped (i.e. nothing will
        attempt to synthesise it) without accessing its Python-space `Operation`.

        This is tightly coupled to `_recursively_handle_op`; it exists as a temporary measure to
        avoid Python-space `Operation` creation from a `DAGOpNode` if we wouldn't do anything to the
        node (which is _most_ nodes)."""
        return (
            # The fast path is just for Rust-space standard gates (which excludes
            # `AnnotatedOperation`).
            node.is_standard_gate()
            # At the moment we don't consider fast-path handling for controlled gates over 3 or
            # more controlled qubits. However, we should not abort early for CX/CZ/etc.
            and not (node.is_controlled_gate() and node.num_qubits >= 3)
            # If there are plugins to try, they need to be tried.
            and not self._methods_to_try(node.name)
            # If all the above constraints hold, and it's already supported or the basis translator
            # can handle it, we'll leave it be.
            and (
                self._instruction_supported(node.name, qubits)
                # This uses unfortunately private details of `EquivalenceLibrary`, but so does the
                # `BasisTranslator`, and this is supposed to just be temporary til this is moved
                # into Rust space.
                or (
                    self._equiv_lib is not None
                    and equivalence.Key(name=node.name, num_qubits=node.num_qubits)
                    in self._equiv_lib._key_to_node_index
                )
            )
        )

    def _instruction_supported(self, name: str, qubits: tuple[int] | None) -> bool:
        # include path for when target exists but target.num_qubits is None (BasicSimulator)
        if self._target is None or self._target.num_qubits is None:
            return name in self._device_insts
        return self._target.instruction_supported(operation_name=name, qargs=qubits)


def _instruction_to_circuit(inst: Instruction) -> QuantumCircuit:
    circuit = QuantumCircuit(inst.num_qubits, inst.num_clbits)
    circuit.append(inst, circuit.qubits, circuit.clbits)
    return circuit
