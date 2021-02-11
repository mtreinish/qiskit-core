#!python
#cython: language_level = 3
#distutils: language = c++

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

cimport cython
from cpython cimport array
from libcpp.unordered_set cimport unordered_set as cset
from libcpp.vector cimport vector
from libcpp.list cimport list as cpplist
from libcpp cimport bool
from libcpp.algorithm cimport sort as stdsort

from ..stochastic_swap.utils cimport NLayout, EdgeCollection

from copy import copy
import time
from itertools import cycle
import numpy as np

from qiskit.dagcircuit import DAGNode
from qiskit.circuit.library.standard_gates import SwapGate

EXTENDED_SET_SIZE = 20     # Size of lookahead window. TODO: set dynamically to len(current_layout)
EXTENDED_SET_WEIGHT = 0.5  # Weight of lookahead window compared to front_layer.

DECAY_RATE = 0.001         # Decay cooefficient for penalizing serial swaps.
DECAY_RESET_INTERVAL = 5   # How often to reset all decay rates to 1.

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def heuristic_search(cpplist[unsigned int] front_layer, object dag,
                     double[::1] qubits_decay, unsigned int num_qubits,
                     const double[:, ::1] adj_matrix,
                     NLayout current_layout,
                     const double[:, ::1] cdist,
                     unsigned int heuristic,
                     object rng):
    """ A single iteration of the tchastic swap mapping routine.

    Args:
        front_layer (unsigned int): The node ids in thenumber of physical qubits.
        dag (DAGCircuit): The DAGCircuit object for the numeric (integer) representation of 
                              the initial_layout.

        int_qubit_subset (ndarray): Int ndarray listing qubits in set.
        cdist (ndarray): Array of doubles that gives the distance graph.
        edges (ndarray): Int array of edges in coupling map.
        scale (ndarray): A double array that holds the perturbed cdist2 array.
        rng (default_rng): An instance of the NumPy default_rng.

    Returns:
        double: Best distance achieved in this trial.
        EdgeCollection: Collection of optimal edges found.
        NLayout: The optimal layout found.
        int: The number of depth steps required in mapping.
    """

    cdef cset[unsigned int] applied_gates
    cdef unsigned int num_search_steps = 0
    cdef vector[unsigned int] execute_gate_list
    cdef cpplist[unsigned int] extended_set
    cdef vector[unsigned int *] swap_candidates
    cdef double[::1] swap_scores
    cdef unsigned int count
    cdef list best_swaps = []
    cdef dict qubit_map = {
            qubit: qubit.index for qubit in dag.qubits}
    cdef dict qarg_cache
    cdef double start_time
    cdef double stop_time
    # Preserve input DAG's name, regs, wire_map, etc. but replace the graph.
    mapped_dag = dag._copy_circuit_metadata()
    while not front_layer.empty():
        start = time.time()
        print([x for x in front_layer])
        print([x for x in execute_gate_list])
        execute_gate_list.clear()
        for node_id in front_layer:
            node = dag._multi_graph[node_id]
            print(node)
            print(node.qargs)
            if len(node.qargs) == 2:
                v0, v1 = node.qargs
                print(current_layout.logic_to_phys[qubit_map[v0]])
                print(current_layout.logic_to_phys[qubit_map[v1]])
                print(np.array(adj_matrix))
                print("edge: %s" % adj_matrix[current_layout.logic_to_phys[qubit_map[v1]],
                                              current_layout.logic_to_phys[qubit_map[v0]]])
                if current_layout.logic_to_phys[qubit_map[v0]] == 10 or current_layout.logic_to_phys[qubit_map[v1]] == 10:
                    raise Exception()
                if adj_matrix[current_layout.logic_to_phys[qubit_map[v1]],
                              current_layout.logic_to_phys[qubit_map[v0]]] != 0:
                    execute_gate_list.push_back(node_id)
            else:
                execute_gate_list.push_back(node_id)
        if not execute_gate_list.empty():
            for node_id in execute_gate_list:
                node = dag._multi_graph[node_id]
                new_node = _transform_gate_for_layout(node, current_layout,
                                                      qubit_map)
                mapped_dag.apply_operation_back(new_node.op,
                                                new_node.qargs,
                                                new_node.cargs,
                                                new_node.condition)
                front_layer.remove(node_id)
                applied_gates.insert(node_id)
                for successor in dag.quantum_successors(node):
                    if successor.type != 'op':
                        continue
                    if is_resolved(applied_gates, successor, dag):
                        front_layer.push_back(successor._node_id)

                if node.qargs:
                    _reset_qubits_decay(num_qubits, qubits_decay)
            continue

        extended_set = obtain_extended_set(dag, front_layer)
        swap_candidates = obtain_swaps(adj_matrix, front_layer,
                                        current_layout, dag, qubit_map)

        swap_scores = np.zeros(swap_candidates.size(), dtype=np.float64)
        count = 0
        for swap_qubits in swap_candidates:
            trial_layout = current_layout.copy()
            trial_layout.swap(swap_qubits[0], swap_qubits[1])
            score = score_heuristic(heuristic, cdist, front_layer,
                                    extended_set, trial_layout, qubits_decay,
                                    swap_qubits, dag, qubit_map)
            swap_scores[count] = score
            count += 1
        min_score = np.amin(swap_scores)
        count = 0
        best_swaps = []
        for score in swap_scores:
            if score == min_score:
                best_swaps.append([swap_candidates[count][0], swap_candidates[count][1]])
            count += 1
        best_swaps.sort(key=lambda x: (x[0], x[1]))
        best_swap = rng.choice(best_swaps)
        swap_node = DAGNode(op=SwapGate(),
                            qargs=[dag.qubits[best_swap[0]], dag.qubits[best_swap[1]]],
                            type='op')
        swap_node = _transform_gate_for_layout(swap_node, current_layout,
                                               qubit_map)
        mapped_dag.apply_operation_back(swap_node.op, swap_node.qargs)
        current_layout.swap(best_swap[0], best_swap[1])
        num_search_steps += 1
        if num_search_steps % DECAY_RESET_INTERVAL == 0:
            _reset_qubits_decay(num_qubits, qubits_decay)
        else:
            qubits_decay[best_swap[0]] += DECAY_RATE
            qubits_decay[best_swap[1]] += DECAY_RATE
        stop_time = time.time()
        print("Iteration time: %s" % str(stop_time - start_time))
    return mapped_dag, current_layout

cdef double score_heuristic(unsigned int heuristic, const double[:, ::1] cdist,
                            cpplist[unsigned int] front_layer,
                            cpplist[unsigned int] extended_set, NLayout layout,
                            double[::1] qubits_decay, unsigned int[2] swap_qubits,
                            object dag, dict qubit_map):
    """Return a heuristic score for a trial layout.

    Assuming a trial layout has resulted from a SWAP, we now assign a cost
    to it. The goodness of a layout is evaluated based on how viable it makes
    the remaining virtual gates that must be applied.
    """
    cdef unsigned int u, v
    cdef double first_cost, second_cost
    cdef unsigned int front
    cdef double sum = 0
    cdef unsigned int q0, q1
    if heuristic == 1:
        for node in front_layer:
            q = dag._multi_graph[node].qargs
            q0 = layout.logic_to_phys[qubit_map[q[0]]]
            q1 = layout.logic_to_phys[qubit_map[q[1]]]
            sum += cdist[q0, q1]
        return sum
    elif heuristic == 2:
        first_cost = score_heuristic(1, cdist, front_layer, [], layout, qubits_decay, swap_qubits, dag, qubit_map)
        first_cost /= front_layer.size()

        second_cost = score_heuristic(1, cdist, extended_set, [], layout, qubits_decay, swap_qubits, dag, qubit_map)
        second_cost = 0.0 if extended_set.empty() else second_cost / extended_set.size()
        return first_cost + EXTENDED_SET_WEIGHT * second_cost
    elif heuristic == 3:
        return max(qubits_decay[swap_qubits[0]], qubits_decay[swap_qubits[1]]) * \
                score_heuristic(2, cdist, front_layer, extended_set, layout, qubits_decay, swap_qubits, dag, qubit_map)


cdef vector[unsigned int *] obtain_swaps(
        const double[:, ::1] adj_matrix, cpplist[unsigned int] front_layer,
        NLayout current_layout, object dag, dict qubit_map):
    """Return a set of candidate swaps that affect qubits in front_layer.

    For each virtual qubit in front_layer, find its current location
    on hardware and the physical qubits in that neighborhood. Every SWAP
    on virtual qubits that corresponds to one of those physical couplings
    is a candidate SWAP.

    Candidate swaps are sorted so SWAP(i,j) and SWAP(j,i) are not duplicated.
    """
    cdef vector[unsigned int *] candidate_swaps
    cdef unsigned int[2] swap_array
    for node in front_layer:
        for virtual in dag._multi_graph[node].qargs:
            physical = current_layout.logic_to_phys[qubit_map[virtual]]

            for neighbor in range(adj_matrix[physical, :].size):
                if adj_matrix[physical, neighbor] == 0.0:
                    continue
                virtual_neighbor = current_layout.phys_to_logic[neighbor]
                swap = array.array('I', sorted([qubit_map[virtual],
                                                virtual_neighbor]))
                swap_array = swap
                candidate_swaps.push_back(swap_array)

    return candidate_swaps



cdef cpplist[unsigned int] obtain_extended_set(object dag, cpplist[unsigned int] front_layer):
    """Populate extended_set by looking ahead a fixed number of gates.
    For each existing element add a successor until reaching limit.
    """
    # TODO: use layers instead of bfs_successors so long range successors aren't included.
    cdef cpplist[unsigned int] extended_set
    bfs_successors_pernode = [dag.bfs_successors(dag._multi_graph[n]) for n in front_layer]
    cdef bool[:] node_lookahead_exhausted = np.array([False] * front_layer.size(),
                                                     dtype=np.bool8)
    for i, node_successor_generator in cycle(enumerate(bfs_successors_pernode)):
        if all(node_lookahead_exhausted) or extended_set.size() >= EXTENDED_SET_SIZE:
            break

        try:
            _, successors = next(node_successor_generator)
            successors = list(filter(lambda x: x.type == 'op' and len(x.qargs) == 2,
                                     successors))
        except StopIteration:
            node_lookahead_exhausted[i] = True
            continue

        successors = iter(successors)
        while extended_set.size() < EXTENDED_SET_SIZE:
            try:
                extended_set.push_back(next(successors)._node_id)
            except StopIteration:
                break

    return extended_set

def _reset_qubits_decay(unsigned int num_qubits, double[::1] qubits_decay):
    for i in range(num_qubits):
        qubits_decay[i] = 1

def _transform_gate_for_layout(op_node, layout, qubit_map):
    """Return node implementing a virtual op on given layout."""
    mapped_op_node = copy(op_node)

    device_qreg = op_node.qargs[0].register
    premap_qargs = [qubit_map[x] for x in op_node.qargs]
    mapped_qargs = map(lambda x: device_qreg[layout.phys_to_logic[x]],
                       premap_qargs)
    mapped_op_node.qargs = list(mapped_qargs)

    return mapped_op_node


cdef bool is_resolved(cset[unsigned int] applied_gates, object node, object dag):
    """Return True if all of a node's predecessors in dag are applied."""
    predecessors = dag.quantum_predecessors(node)
    predecessors = filter(lambda x: x.type == 'op', predecessors)
    cdef unsigned int tmp
    for n in predecessors:
        tmp = n._node_id
        if not applied_gates.count(tmp):
            return False
    return True

