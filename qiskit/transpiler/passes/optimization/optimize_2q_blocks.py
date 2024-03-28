# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Optimize chains of single-qubit gates using Euler 1q decomposer"""

import logging
import math

import numpy as np

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.synthesis.one_qubit import one_qubit_decompose
from qiskit._accelerate import euler_one_qubit_decomposer
from qiskit.circuit.library.standard_gates import (
    UGate,
    PhaseGate,
    U3Gate,
    U2Gate,
    U1Gate,
    RXGate,
    RYGate,
    RZGate,
    RGate,
    SXGate,
    XGate,
    CXGate,
)
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.transpiler.passmanager import PassManager
from qiskit._accelerate.optimize_2q_blocks import optimize_blocks, TargetErrorMap, DecomposerMap
from qiskit.exceptions import QiskitError
from qiskit.synthesis.two_qubit import TwoQubitBasisDecomposer


logger = logging.getLogger(__name__)


GATE_NAME_MAP = {
    "cx": CXGate,
    "rx": RXGate,
    "sx": SXGate,
    "x": XGate,
    "rz": RZGate,
    "u": UGate,
    "p": PhaseGate,
    "u1": U1Gate,
    "u2": U2Gate,
    "u3": U3Gate,
}


class Optimize2qBlocks(TransformationPass):

    def __init__(
        self,
        target=None,
        unitary_synthesis_method=None,
        unitary_synthesis_plugin_config=None,
        approximation_degree=None,
    ):
        """Optimize2qBlocks.

        Args:
            target (Optional[Target]): The :class:`~.Target` object corresponding to the compilation
                target. When specified, any argument specified for ``basis_gates`` is ignored.
        """
        super().__init__()
        if target is not None:
            self._target_map = TargetErrorMap(len(target))
            self._decomposers = DecomposerMap(target.num_qubits)
        else:
            self._target_map = TargetErrorMap(0)
            self._decomposers = DecomposerMap(0)
        qubits_set = set()
        self._target = target
        if target is not None:
            for gate_name, props_dict in target.items():
                for qubits, props in props_dict.items():
                    if len(qubits) == 2:
                        self._target_map.add_error(gate_name, qubits, props.error)
                        qubits_set.add(qubits)
            # TODO: Add real logic for building decomposer list:
            decomposer = TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSXX", pulse_optimize=True)
            for qubits in qubits_set:
                self._decomposers.add_decomposer(qubits, decomposer._inner_decomposer)
        self._unitary_synthesis_method = unitary_synthesis_method
        self._unitary_synthesis_config = unitary_synthesis_plugin_config
        self._approximation_degree = approximation_degree

    @control_flow.trivial_recurse
    def run(self, dag):
        """Run the Optimize1qGatesDecomposition pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        if self._target is None:
            raise Exception("AHHH")
        blocks = dag.collect_2q_runs()
        all_block_gates = set()
        blocks_as_unitaries = []
        if self._unitary_synthesis_method is not None:
            from qiskit.transpiler.passes import (
                ConsolidateBlocks,
                UnitarySynthesis,
                Collect2qBlocks,
            )

            consolidate_pass = ConsolidateBlocks()
            consolidate_pass.property_set["block_list"] = blocks
            consolidated_dag = consolidate_pass.run(dag)
            synth_pass = UnitarySynthesis(
                target=target,
                approximation_degree=approximation_degree,
                method=unitary_synthesis_method,
                plugin_config=unitary_synthesis_config,
            )
            return synth_pass.run(consolidated_dag)
        else:
            blocks_to_sub = []
            block_unitaries = []
            for block in blocks:
                block_qargs = set()
                for nd in block:
                    block_qargs |= set(nd.qargs)
                block_index_map = self._block_qargs_to_indices(dag, block_qargs)
                block_details = []
                for node in block:
                    try:
                        op_matrix = node.op.to_matrix()
                    except QiskitError:
                        op_matrix = Operator(node.op).data
                    q_list = [block_index_map[qubit] for qubit in node.qargs]
                    block_details.append((op_matrix, q_list))
                block_unitaries.append(
                    (block_details, [dag.find_bit(q).index for q in block_qargs])
                )
                blocks_to_sub.append(block)

            sequences = optimize_blocks(
                block_unitaries,
                self._decomposers,
                self._target_map,
            )
            for block, sequence in zip(blocks_to_sub, sequences):
                if sequence is None:
                    continue
                qr = QuantumRegister(2)
                new_dag = DAGCircuit()
                new_dag.add_qreg(qr)
                new_dag.global_phase = sequence[0].global_phase
                for name, params, qubits in sequence[0]:
                    if name == "USER_GATE":
                        new_dag.apply_operation_back(
                            sequence[1], tuple(new_dag.qubits[x] for x in qubits), check=False
                        )
                    else:
                        gate = GATE_NAME_MAP[name](*params)
                        new_dag.apply_operation_back(
                            gate, tuple(new_dag.qubits[x] for x in qubits), check=False
                        )

                block_qargs = set()
                for nd in block:
                    block_qargs |= set(nd.qargs)
                block_index_map = self._block_qargs_to_indices(dag, block_qargs)
                dummy_op = UnitaryGate(np.eye(4))
                new_node = dag.replace_block_with_op(
                    block, dummy_op, block_index_map, cycle_check=False
                )
                dag.substitute_node_with_dag(new_node, new_dag)

        return dag

    def _block_qargs_to_indices(self, dag, block_qargs):
        block_indices = [dag.find_bit(q).index for q in block_qargs]
        ordered_block_indices = {bit: index for index, bit in enumerate(sorted(block_indices))}
        block_positions = {q: ordered_block_indices[dag.find_bit(q).index] for q in block_qargs}
        return block_positions
