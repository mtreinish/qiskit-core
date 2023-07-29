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

"""Layout selection using the SABRE bidirectional search approach from Li et al.
"""

from collections import defaultdict
import copy
import logging
import time

import numpy as np
import rustworkx as rx

from qiskit.converters import dag_to_circuit
from qiskit.transpiler.passes.layout.set_layout import SetLayout
from qiskit.transpiler.passes.layout.full_ancilla_allocation import FullAncillaAllocation
from qiskit.transpiler.passes.layout.enlarge_with_ancilla import EnlargeWithAncilla
from qiskit.transpiler.passes.layout.apply_layout import ApplyLayout
from qiskit.transpiler.passes.layout import disjoint_utils
from qiskit.transpiler.passes.layout import vf2_utils
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit._accelerate.nlayout import NLayout
from qiskit._accelerate.sabre_layout import sabre_layout_and_routing
from qiskit._accelerate.sabre_swap import (
    Heuristic,
    NeighborTable,
)
from qiskit.transpiler.passes.routing.sabre_swap import _build_sabre_dag, _apply_sabre_result
from qiskit.transpiler.target import Target
from qiskit.transpiler.coupling import CouplingMap
from qiskit.tools.parallel import CPU_COUNT
from qiskit.circuit.controlflow import ControlFlowOp, ForLoopOp
from qiskit.converters import circuit_to_dag

logger = logging.getLogger(__name__)


class SabreLayout(TransformationPass):
    """Choose a Layout via iterative bidirectional routing of the input circuit.

    The algorithm does a full routing of the circuit (via the `routing_pass`
    method) to end up with a `final_layout`. This final_layout is then used as
    the initial_layout for routing the reverse circuit. The algorithm iterates a
    number of times until it finds an initial_layout that reduces full routing cost.

    Prior to running the SABRE algorithm this transpiler pass will try to find the layout
    for deepest layer that is has an isomorphic subgraph in the coupling graph. This is
    done by progressively using the algorithm from :class:`~.VF2Layout` on the circuit
    until a mapping is not found. This partial layout is then used to seed the SABRE algorithm
    and then random physical bits are selected for the remaining elements in the mapping.

    This method exploits the reversibility of quantum circuits, and tries to
    include global circuit information in the choice of initial_layout.

    By default this pass will run both layout and routing and will transform the
    circuit so that the layout is applied to the input dag (meaning that the output
    circuit will have ancilla qubits allocated for unused qubits on the coupling map
    and the qubits will be reordered to match the mapped physical qubits) and then
    routing will be applied (inserting :class:`~.SwapGate`s to account for limited
    connectivity). This is unlike most other layout passes which are :class:`~.AnalysisPass`
    objects and just find an initial layout and set that on the property set. This is
    done because by default the pass will run parallel seed trials with different random
    seeds for selecting the random initial layout and then selecting the routed output
    which results in the least number of swap gates needed.

    You can use the ``routing_pass`` argument to have this pass operate as a typical
    layout pass. When specified this will use the specified routing pass to select an
    initial layout only and will not run multiple seed trials.

    **References:**

    [1] Li, Gushu, Yufei Ding, and Yuan Xie. "Tackling the qubit mapping problem
    for NISQ-era quantum devices." ASPLOS 2019.
    `arXiv:1809.02573 <https://arxiv.org/pdf/1809.02573.pdf>`_
    """

    def __init__(
        self,
        coupling_map,
        routing_pass=None,
        seed=None,
        max_iterations=3,
        swap_trials=None,
        layout_trials=None,
        skip_routing=False,
        vf2_partial_layout=True,
        vf2_call_limit=None,
        vf2_time_limit=None,
        vf2_max_trials=None,
    ):
        """SabreLayout initializer.

        Args:
            coupling_map (Union[CouplingMap, Target]): directed graph representing a coupling map.
            routing_pass (BasePass): the routing pass to use while iterating.
                If specified this pass operates as an :class:`~.AnalysisPass` and
                will only populate the ``layout`` field in the property set and
                the input dag is returned unmodified. This argument is mutually
                exclusive with the ``swap_trials`` and the ``layout_trials``
                arguments and if this is specified at the same time as either
                argument an error will be raised.
            seed (int): seed for setting a random first trial layout.
            max_iterations (int): number of forward-backward iterations.
            swap_trials (int): The number of trials to run of
                :class:`~.SabreSwap` for each iteration. This is equivalent to
                the ``trials`` argument on :class:`~.SabreSwap`. If this is not
                specified (and ``routing_pass`` isn't set) by default the number
                of physical CPUs on your local system will be used. For
                reproducibility between environments it is best to set this
                to an explicit number because the output will potentially depend
                on the number of trials run. This option is mutually exclusive
                with the ``routing_pass`` argument and an error will be raised
                if both are used.
            layout_trials (int): The number of random seed trials to run
                layout with. When > 1 the trial that resuls in the output with
                the fewest swap gates will be selected. If this is not specified
                (and ``routing_pass`` is not set) then the number of local
                physical CPUs will be used as the default value. This option is
                mutually exclusive with the ``routing_pass`` argument and an error
                will be raised if both are used.
            skip_routing (bool): If this is set ``True`` and ``routing_pass`` is not used
                then routing will not be applied to the output circuit.  Only the layout
                will be returned in the property set. This is a tradeoff to run custom
                routing with multiple layout trials, as using this option will cause
                SabreLayout to run the routing stage internally but not use that result.
            vf2_partial_layout (bool): Run vf2 partial layout
            vf2_call_limit (int): The number of state visits to attempt in each execution of
                VF2 to attempt to find a partial layout.
            vf2_time_limit (float): The total time limit in seconds to run VF2 to find a partial
                layout
            vf2_max_trials (int): The maximum number of trials to run VF2 to find
                a partial layout. If this is not specified the number of trials will be limited
                based on the number of edges in the interaction graph or the coupling graph
                (whichever is larger) if no other limits are set. If set to a value <= 0 no
                limit on the number of trials will be set.

        Raises:
            TranspilerError: If both ``routing_pass`` and ``swap_trials`` or
            both ``routing_pass`` and ``layout_trials`` are specified
        """
        super().__init__()
        if isinstance(coupling_map, Target):
            self.target = coupling_map
            self.coupling_map = self.target.build_coupling_map()
        else:
            self.target = None
            self.coupling_map = coupling_map
        self._neighbor_table = None
        if routing_pass is not None and (swap_trials is not None or layout_trials is not None):
            raise TranspilerError("Both routing_pass and swap_trials can't be set at the same time")
        self.routing_pass = routing_pass
        self.seed = seed
        self.max_iterations = max_iterations
        self.trials = swap_trials
        if swap_trials is None:
            self.swap_trials = CPU_COUNT
        else:
            self.swap_trials = swap_trials
        if layout_trials is None:
            self.layout_trials = CPU_COUNT
        else:
            self.layout_trials = layout_trials
        self.skip_routing = skip_routing
        if self.coupling_map is not None:
            if not self.coupling_map.is_symmetric:
                # deepcopy is needed here if we don't own the coupling map (i.e. we were passed it
                # directly) to avoid modifications updating shared references in passes which
                # require directional constraints
                if isinstance(coupling_map, CouplingMap):
                    self.coupling_map = copy.deepcopy(self.coupling_map)
                self.coupling_map.make_symmetric()
            self._neighbor_table = NeighborTable(rx.adjacency_matrix(self.coupling_map.graph))
        self.avg_error_map = None
        self.vf2_partial_layout = vf2_partial_layout
        self.call_limit = vf2_call_limit
        self.time_limit = vf2_time_limit
        self.max_trials = vf2_max_trials

    def run(self, dag):
        """Run the SabreLayout pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to find layout for.

        Returns:
           DAGCircuit: The output dag if swap mapping was run
            (otherwise the input dag is returned unmodified).

        Raises:
            TranspilerError: if dag wider than self.coupling_map
        """
        if len(dag.qubits) > self.coupling_map.size():
            raise TranspilerError("More virtual qubits exist than physical.")

        # Choose a random initial_layout.
        if self.routing_pass is not None:
            if not self.coupling_map.is_connected():
                raise TranspilerError(
                    "The routing_pass argument cannot be used with disjoint coupling maps."
                )
            if self.seed is None:
                seed = np.random.randint(0, np.iinfo(np.int32).max)
            else:
                seed = self.seed
            rng = np.random.default_rng(seed)

            physical_qubits = rng.choice(self.coupling_map.size(), len(dag.qubits), replace=False)
            physical_qubits = rng.permutation(physical_qubits)
            initial_layout = Layout({q: dag.qubits[i] for i, q in enumerate(physical_qubits)})

            self.routing_pass.fake_run = True

            # Do forward-backward iterations.
            circ = dag_to_circuit(dag)
            rev_circ = circ.reverse_ops()
            for _ in range(self.max_iterations):
                for _ in ("forward", "backward"):
                    pm = self._layout_and_route_passmanager(initial_layout)
                    new_circ = pm.run(circ)

                    # Update initial layout and reverse the unmapped circuit.
                    pass_final_layout = pm.property_set["final_layout"]
                    final_layout = self._compose_layouts(
                        initial_layout, pass_final_layout, new_circ.qregs
                    )
                    initial_layout = final_layout
                    circ, rev_circ = rev_circ, circ

                # Diagnostics
                logger.info("new initial layout")
                logger.info(initial_layout)

            for qreg in dag.qregs.values():
                initial_layout.add_register(qreg)
            self.property_set["layout"] = initial_layout
            self.routing_pass.fake_run = False
            return dag
        # Combined
        if self.target is not None:
            # This is a special case SABRE only works with a bidirectional coupling graph
            # which we explicitly can't create from the target. So do this manually here
            # to avoid altering the shared state with the unfiltered indices.
            target = self.target.build_coupling_map(filter_idle_qubits=True)
            target.make_symmetric()
        else:
            target = self.coupling_map
        layout_components = disjoint_utils.run_pass_over_connected_components(
            dag,
            target,
            self._inner_run,
        )
        initial_layout_dict = {}
        final_layout_dict = {}
        for (
            layout_dict,
            final_dict,
            component_map,
            _sabre_result,
            _circuit_to_dag_dict,
            local_dag,
        ) in layout_components:
            initial_layout_dict.update({k: component_map[v] for k, v in layout_dict.items()})
            final_layout_dict.update({component_map[k]: component_map[v] for k, v in final_dict})
        self.property_set["layout"] = Layout(initial_layout_dict)
        # If skip_routing is set then return the layout in the property set
        # and throwaway the extra work we did to compute the swap map.
        # We also skip routing here if there is more than one connected
        # component we ran layout on. We can only reliably route the full dag
        # in this case if there is any dependency between the components
        # (typically shared classical data or barriers).
        if self.skip_routing or len(layout_components) > 1:
            return dag
        # After this point the pass is no longer an analysis pass and the
        # output circuit returned is transformed with the layout applied
        # and swaps inserted
        dag = self._apply_layout_no_pass_manager(dag)
        mapped_dag = dag.copy_empty_like()
        self.property_set["final_layout"] = Layout(
            {dag.qubits[k]: v for (k, v) in final_layout_dict.items()}
        )
        canonical_register = dag.qregs["q"]
        original_layout = NLayout.generate_trivial_layout(self.coupling_map.size())
        for (
            _layout_dict,
            _final_layout_dict,
            component_map,
            sabre_result,
            circuit_to_dag_dict,
            local_dag,
        ) in layout_components:
            _apply_sabre_result(
                mapped_dag,
                local_dag,
                initial_layout_dict,
                canonical_register,
                original_layout,
                sabre_result,
                circuit_to_dag_dict,
                component_map,
            )
        disjoint_utils.combine_barriers(mapped_dag, retain_uuid=False)
        return mapped_dag

    def _inner_run(self, dag, coupling_map):
        if not coupling_map.is_symmetric:
            # deepcopy is needed here to avoid modifications updating
            # shared references in passes which require directional
            # constraints
            coupling_map = copy.deepcopy(coupling_map)
            coupling_map.make_symmetric()
        neighbor_table = NeighborTable(rx.adjacency_matrix(coupling_map.graph))
        dist_matrix = coupling_map.distance_matrix
        original_qubit_indices = {bit: index for index, bit in enumerate(dag.qubits)}
        sabre_dag, circuit_to_dag_dict = _build_sabre_dag(
            dag,
            coupling_map.size(),
            original_qubit_indices,
        )
        partial_layout = None
        if self.vf2_partial_layout:
            partial_layout_virtual_bits = self._vf2_partial_layout(
                dag, coupling_map
            ).get_virtual_bits()
            partial_layout = [partial_layout_virtual_bits.get(i, None) for i in dag.qubits]

        ((initial_layout, final_layout), sabre_result) = sabre_layout_and_routing(
            sabre_dag,
            neighbor_table,
            dist_matrix,
            Heuristic.Decay,
            self.max_iterations,
            self.swap_trials,
            self.layout_trials,
            self.seed,
            partial_layout,
        )

        # Apply initial layout selected.
        layout_dict = {}
        num_qubits = len(dag.qubits)
        for k, v in initial_layout.layout_mapping():
            if k < num_qubits:
                layout_dict[dag.qubits[k]] = v
        final_layout_dict = final_layout.layout_mapping()
        component_mapping = {x: coupling_map.graph[x] for x in coupling_map.graph.node_indices()}
        return (
            layout_dict,
            final_layout_dict,
            component_mapping,
            sabre_result,
            circuit_to_dag_dict,
            dag,
        )

    def _apply_layout_no_pass_manager(self, dag):
        """Apply and embed a layout into a dagcircuit without using a ``PassManager`` to
        avoid circuit<->dag conversion.
        """
        ancilla_pass = FullAncillaAllocation(self.coupling_map)
        ancilla_pass.property_set = self.property_set
        dag = ancilla_pass.run(dag)
        enlarge_pass = EnlargeWithAncilla()
        enlarge_pass.property_set = ancilla_pass.property_set
        dag = enlarge_pass.run(dag)
        apply_pass = ApplyLayout()
        apply_pass.property_set = enlarge_pass.property_set
        dag = apply_pass.run(dag)
        return dag

    def _layout_and_route_passmanager(self, initial_layout):
        """Return a passmanager for a full layout and routing.

        We use a factory to remove potential statefulness of passes.
        """
        layout_and_route = [
            SetLayout(initial_layout),
            FullAncillaAllocation(self.coupling_map),
            EnlargeWithAncilla(),
            ApplyLayout(),
            self.routing_pass,
        ]
        pm = PassManager(layout_and_route)
        return pm

    def _compose_layouts(self, initial_layout, pass_final_layout, qregs):
        """Return the real final_layout resulting from the composition
        of an initial_layout with the final_layout reported by a pass.

        The routing passes internally start with a trivial layout, as the
        layout gets applied to the circuit prior to running them. So the
        "final_layout" they report must be amended to account for the actual
        initial_layout that was selected.
        """
        trivial_layout = Layout.generate_trivial_layout(*qregs)
        qubit_map = Layout.combine_into_edge_map(initial_layout, trivial_layout)
        final_layout = {v: pass_final_layout._v2p[qubit_map[v]] for v in initial_layout._v2p}
        return Layout(final_layout)

    # TODO: Migrate this to rust as part of sabre_layout.rs after
    # https://github.com/Qiskit/rustworkx/issues/741 is implemented and released
    def _vf2_partial_layout(self, dag, coupling_map):
        """Find a partial layout using vf2 on the deepest subgraph that is isomorphic to
        the coupling graph."""
        im_graph_node_map = {}
        reverse_im_graph_node_map = {}
        im_graph = rx.PyGraph(multigraph=False)
        logger.debug("Buidling interaction graphs")
        largest_im_graph = None
        best_mapping = None
        first_mapping = None
        if self.avg_error_map is None:
            self.avg_error_map = vf2_utils.build_average_error_map(self.target, None, coupling_map)

        cm_graph, cm_nodes = vf2_utils.shuffle_coupling_graph(coupling_map, self.seed, False)
        # To avoid trying to over optimize the result by default limit the number
        # of trials based on the size of the graphs. For circuits with simple layouts
        # like an all 1q circuit we don't want to sit forever trying every possible
        # mapping in the search space if no other limits are set
        if self.max_trials is None and self.call_limit is None and self.time_limit is None:
            im_graph_edge_count = len(im_graph.edge_list())
            cm_graph_edge_count = len(coupling_map.graph.edge_list())
            self.max_trials = max(im_graph_edge_count, cm_graph_edge_count) + 15

        start_time = time.time()

        # A more efficient search pattern would be to do a binary search
        # and find, but to conserve memory and avoid a large number of
        # unecessary graphs this searchs from the beginning and continues
        # until there is no vf2 match
        def _visit(dag, weight, wire_map):
            for node in dag.topological_op_nodes():
                nonlocal largest_im_graph
                largest_im_graph = im_graph.copy()
                if getattr(node.op, "_directive", False):
                    continue
                if isinstance(node.op, ControlFlowOp):
                    if isinstance(node.op, ForLoopOp):
                        inner_weight = len(node.op.params[0]) * weight
                    else:
                        inner_weight = weight
                    for block in node.op.blocks:
                        inner_wire_map = {
                            inner: wire_map[outer] for outer, inner in zip(node.qargs, block.qubits)
                        }
                        _visit(circuit_to_dag(block), inner_weight, inner_wire_map)
                    continue
                len_args = len(node.qargs)
                qargs = [wire_map[q] for q in node.qargs]
                if len_args == 1:
                    if qargs[0] not in im_graph_node_map:
                        weights = defaultdict(int)
                        weights[node.name] += weight
                        im_graph_node_map[qargs[0]] = im_graph.add_node(weights)
                        reverse_im_graph_node_map[im_graph_node_map[qargs[0]]] = qargs[0]
                    else:
                        im_graph[im_graph_node_map[qargs[0]]][node.op.name] += weight
                if len_args == 2:
                    if qargs[0] not in im_graph_node_map:
                        im_graph_node_map[qargs[0]] = im_graph.add_node(defaultdict(int))
                        reverse_im_graph_node_map[im_graph_node_map[qargs[0]]] = qargs[0]
                    if qargs[1] not in im_graph_node_map:
                        im_graph_node_map[qargs[1]] = im_graph.add_node(defaultdict(int))
                        reverse_im_graph_node_map[im_graph_node_map[qargs[1]]] = qargs[1]
                    edge = (im_graph_node_map[qargs[0]], im_graph_node_map[qargs[1]])
                    if im_graph.has_edge(*edge):
                        im_graph.get_edge_data(*edge)[node.name] += weight
                    else:
                        weights = defaultdict(int)
                        weights[node.name] += weight
                        im_graph.add_edge(*edge, weights)
                if len_args > 2:
                    raise TranspilerError(
                        "Encountered an instruction operating on more than 2 qubits, this pass "
                        "only functions with 1 or 2 qubit operations."
                    )
                vf2_mapping = rx.vf2_mapping(
                    cm_graph,
                    im_graph,
                    subgraph=True,
                    id_order=False,
                    induced=False,
                    call_limit=self.call_limit,
                )
                try:
                    nonlocal first_mapping
                    first_mapping = next(vf2_mapping)
                except StopIteration:
                    break
                nonlocal best_mapping
                best_mapping = vf2_mapping
                elapsed_time = time.time() - start_time
                if (
                    self.time_limit is not None
                    and best_mapping is not None
                    and elapsed_time >= self.time_limit
                ):
                    logger.debug(
                        "SabreLayout VF2 heuristic has taken %s which exceeds configured max time: %s",
                        elapsed_time,
                        self.time_limit,
                    )
                    break

        _visit(dag, 1, {bit: bit for bit in dag.qubits})
        logger.debug("Finding best mappings of largest partial subgraph")
        im_graph = largest_im_graph

        def mapping_to_layout(layout_mapping):
            return Layout({reverse_im_graph_node_map[k]: v for k, v in layout_mapping.items()})

        layout_mapping = {im_i: cm_nodes[cm_i] for cm_i, im_i in first_mapping.items()}
        chosen_layout = mapping_to_layout(layout_mapping)
        for reg in dag.qregs.values():
            chosen_layout.add_register(reg)
        return chosen_layout
