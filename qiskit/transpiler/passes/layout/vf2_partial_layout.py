# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""VF2Layout pass to find a layout using subgraph isomorphism"""
from collections import defaultdict
import logging
import time

import retworkx as rx

from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.layout import vf2_utils
from qiskit.circuit import ControlFlowOp, ForLoopOp
from qiskit.converters import circuit_to_dag


logger = logging.getLogger(__name__)


class VF2PartialLayout(AnalysisPass):
    """A subgraph isomorphism based Layout pass.

    This pass uses rustworkx's VF2++ subgraph isomorphism mapper to find
    the largest subgraph of the interaction graph that has an isomorphic
    subgraph in the coupling graph. It operates by finding the largest depth
    interaction graph that still has an isomorphic subgraph in the coupling
    graph and the selecting the lowest noise qubits for those virtual qubits
    to operate on. If there are any leftover circuit qubits they are assigned
    an unused physical qubits. You typically want to combine this with the
    :class:`~.VF2Layout` pass to do a search for a perfect layout first to
    avoid performing this search.

    This pass only works with input :class:`~.DAGCircuit` that are composed of
    1 and 2 qubit non-directive operations. If the target supports operations
    on > 2 qubits a different pass should be used.

    This pass was inspired by the description of ``GraphPlacemnt`` in the
    `pytket manual <https://cqcl.github.io/pytket/manual/manual_compiler.html#placement>`__. [#f1]_

    .. [#f1] GraphPlacement will try to identify a subgraph isomorphism between
        the graph of interacting logical qubits (up to some depth into the
        Circuit) and the connectivity graph of the physical qubits.
    """

    def __init__(
        self,
        coupling_map=None,
        seed=None,
        call_limit=None,
        time_limit=None,
        properties=None,
        max_trials=None,
        target=None,
    ):
        """Initialize a ``VF2Layout`` pass instance

        Args:
            coupling_map (CouplingMap): Directed graph representing a coupling map.
            seed (int): Sets the seed of the PRNG. -1 Means no node shuffling.
            call_limit (int): The number of state visits to attempt in each execution of
                VF2.
            time_limit (float): The total time limit in seconds to run ``VF2Layout``
            properties (BackendProperties): The backend properties for the backend. If
                :meth:`~qiskit.providers.models.BackendProperties.readout_error` is available
                it is used to score the layout.
            max_trials (int): The maximum number of trials to run VF2 to find
                a layout. If this is not specified the number of trials will be limited
                based on the number of edges in the interaction graph or the coupling graph
                (whichever is larger) if no other limits are set. If set to a value <= 0 no
                limit on the number of trials will be set.
            target (Target): A target representing the backend device to run ``VF2Layout`` on.
                If specified it will supersede a set value for ``properties`` and
                ``coupling_map``.

        Raises:
            TypeError: At runtime, if neither ``coupling_map`` or ``target`` are provided.
        """
        super().__init__()
        self.target = target
        if target is not None:
            self.coupling_map = self.target.build_coupling_map()
        else:
            self.coupling_map = coupling_map
        self.properties = properties
        self.seed = seed
        self.call_limit = call_limit
        self.time_limit = time_limit
        self.max_trials = max_trials
        self.avg_error_map = None

    def run(self, dag):
        """run the layout method"""
        if self.coupling_map is None:
            raise TranspilerError("coupling_map or target must be specified.")
        if self.avg_error_map is None:
            self.avg_error_map = vf2_utils.build_average_error_map(
                self.target, self.properties, self.coupling_map
            )

        interaction_graphs = []
        im_graph_node_map = {}
        reverse_im_graph_node_map = {}
        im_graph = rx.PyGraph(multigraph=False)

        def _visit(dag, weight, wire_map):
            for node in dag.topological_op_nodes():
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
                        interaction_graphs.append(im_graph.copy())
                if len_args > 2:
                    raise TranspilerError(
                        "Encountered an instruction operating on more than 2 qubits, this pass "
                        "only functions with 1 or 2 qubit operations."
                    )

        _visit(dag, 1, {bit: bit for bit in dag.qubits})

        cm_graph, cm_nodes = vf2_utils.shuffle_coupling_graph(self.coupling_map, self.seed, False)
        # To avoid trying to over optimize the result by default limit the number
        # of trials based on the size of the graphs. For circuits with simple layouts
        # like an all 1q circuit we don't want to sit forever trying every possible
        # mapping in the search space if no other limits are set
        if self.max_trials is None and self.call_limit is None and self.time_limit is None:
            im_graph_edge_count = len(im_graph.edge_list())
            cm_graph_edge_count = len(self.coupling_map.graph.edge_list())
            self.max_trials = max(im_graph_edge_count, cm_graph_edge_count) + 15

        index = len(interaction_graphs) // 2
        end_index = len(interaction_graphs)
        best_mapping = None
        best_index = None
        start_time = time.time()
        while index < end_index:
            vf2_mapping = rx.vf2_mapping(
                cm_graph,
                interaction_graphs[index],
                subgraph=True,
                id_order=False,
                induced=False,
                call_limit=self.call_limit,
            )
            try:
                first_mapping = next(vf2_mapping)
            except StopIteration:
                end_index = index
                index = end_index // 2
            else:
                best_mapping = vf2_mapping
                best_index = index
                offset = (end_index - index) // 2
                # Skip a full subgraph because vf2layout would have found this already
                # or we tried and failed that index already
                if offset == 0:
                    break
                index = index + offset
            elapsed_time = time.time() - start_time
            if (
                self.time_limit is not None
                and best_mapping is not None
                and elapsed_time >= self.time_limit
            ):
                logger.debug(
                    "VF2Layout has taken %s which exceeds configured max time: %s",
                    elapsed_time,
                    self.time_limit,
                )
                break
        if best_mapping is None:
            raise TranspilerError(
                "Compilation target doesn't have any qubit connectivity no mapping is possible"
            )

        logger.debug("Finding best mappings of largest partial subgraph")
        im_graph = interaction_graphs[best_index]
        chosen_layout = Layout(
            {
                reverse_im_graph_node_map[im_i]: cm_nodes[cm_i]
                for cm_i, im_i in first_mapping.items()
            }
        )
        chosen_layout_score = vf2_utils.score_layout(
            self.avg_error_map,
            chosen_layout,
            im_graph_node_map,
            reverse_im_graph_node_map,
            im_graph,
            False,
        )
        trials = 1
        for mapping in best_mapping:
            trials += 1
            logger.debug("Running trial: %s", trials)
            layout = Layout(
                {reverse_im_graph_node_map[im_i]: cm_nodes[cm_i] for cm_i, im_i in mapping.items()}
            )
            # If the graphs have the same number of nodes we don't need to score or do multiple
            # trials as the score heuristic currently doesn't weigh nodes based on gates on a
            # qubit so the scores will always all be the same
            if len(cm_graph) == len(im_graph):
                chosen_layout = layout
                break
            layout_score = vf2_utils.score_layout(
                self.avg_error_map,
                layout,
                im_graph_node_map,
                reverse_im_graph_node_map,
                im_graph,
                False,
            )
            logger.debug("Trial %s has score %s", trials, layout_score)
            if chosen_layout is None:
                chosen_layout = layout
                chosen_layout_score = layout_score
            elif layout_score < chosen_layout_score:
                logger.debug(
                    "Found layout %s has a lower score (%s) than previous best %s (%s)",
                    layout,
                    layout_score,
                    chosen_layout,
                    chosen_layout_score,
                )
                chosen_layout = layout
                chosen_layout_score = layout_score
            if self.max_trials is not None and self.max_trials > 0 and trials >= self.max_trials:
                logger.debug("Trial %s is >= configured max trials %s", trials, self.max_trials)
                break
            elapsed_time = time.time() - start_time
            if self.time_limit is not None and elapsed_time >= self.time_limit:
                logger.debug(
                    "VF2Layout has taken %s which exceeds configured max time: %s",
                    elapsed_time,
                    self.time_limit,
                )
                break

        # If any qubits are not in layout just add them in order
        chosen_layout_virtual_bits = chosen_layout.get_virtual_bits()
        for qubit in dag.qubits:
            if qubit not in chosen_layout_virtual_bits:
                chosen_layout.add(qubit)
        self.property_set["layout"] = chosen_layout
        for reg in dag.qregs.values():
            self.property_set["layout"].add_register(reg)
