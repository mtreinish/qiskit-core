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

"""Pass manager for optimization level 1, providing light optimization.

Level 1 pass manager: light optimization by simple adjacent gate collapsing.
"""
from __future__ import annotations
from qiskit.transpiler.basepasses import BasePass
from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.passmanager import StagedPassManager
from qiskit.transpiler import ConditionalController, FlowController
from qiskit.transpiler.passes import CXCancellation
from qiskit.transpiler.passes import FixedPoint
from qiskit.transpiler.passes import Depth
from qiskit.transpiler.passes import Size
from qiskit.transpiler.passes import Optimize1qGatesDecomposition
from qiskit.transpiler.passes import GatesInBasis
from qiskit.transpiler.preset_passmanagers import common
from qiskit.transpiler.preset_passmanagers.plugin import (
    PassManagerStagePluginManager,
)


def level_1_pass_manager(pass_manager_config: PassManagerConfig) -> StagedPassManager:
    """Level 1 pass manager: light optimization by simple adjacent gate collapsing.

    This pass manager applies the user-given initial layout. If none is given,
    and a trivial layout (i-th virtual -> i-th physical) makes the circuit fit
    the coupling map, that is used.
    Otherwise, the circuit is mapped to the most densely connected coupling subgraph,
    and swaps are inserted to map. Any unused physical qubit is allocated as ancilla space.
    The pass manager then unrolls the circuit to the desired basis, and transforms the
    circuit to match the coupling map. Finally, optimizations in the form of adjacent
    gate collapse and redundant reset removal are performed.

    Args:
        pass_manager_config: configuration of the pass manager.

    Returns:
        a level 1 pass manager.

    Raises:
        TranspilerError: if the passmanager config is invalid.
    """
    plugin_manager = PassManagerStagePluginManager()
    basis_gates = pass_manager_config.basis_gates
    coupling_map = pass_manager_config.coupling_map
    initial_layout = pass_manager_config.initial_layout
    init_method = pass_manager_config.init_method
    # Unlike other presets, the layout and routing defaults aren't set here because they change
    # based on whether the input circuit has control flow.
    layout_method = pass_manager_config.layout_method or "default"
    routing_method = pass_manager_config.routing_method or "sabre"
    translation_method = pass_manager_config.translation_method or "translator"
    optimization_method = pass_manager_config.optimization_method
    scheduling_method = pass_manager_config.scheduling_method or "default"
    backend_properties = pass_manager_config.backend_properties
    approximation_degree = pass_manager_config.approximation_degree
    unitary_synthesis_method = pass_manager_config.unitary_synthesis_method
    unitary_synthesis_plugin_config = pass_manager_config.unitary_synthesis_plugin_config
    target = pass_manager_config.target
    hls_config = pass_manager_config.hls_config

    # Choose routing pass
    routing_pm = plugin_manager.get_passmanager_stage(
        "routing", routing_method, pass_manager_config, optimization_level=1
    )

    # Build optimization loop: merge 1q rotations and cancel CNOT gates iteratively
    # until no more change in depth
    _depth_check = [Depth(recurse=True), FixedPoint("depth")]
    _size_check = [Size(recurse=True), FixedPoint("size")]

    def _opt_control(property_set):
        return (not property_set["depth_fixed_point"]) or (not property_set["size_fixed_point"])

    _opt = [Optimize1qGatesDecomposition(basis=basis_gates, target=target), CXCancellation()]

    unroll_3q = None
    # Build full pass manager
    if coupling_map or initial_layout:
        unroll_3q = common.generate_unroll_3q(
            target,
            basis_gates,
            approximation_degree,
            unitary_synthesis_method,
            unitary_synthesis_plugin_config,
            hls_config,
        )
        layout = plugin_manager.get_passmanager_stage(
            "layout", layout_method, pass_manager_config, optimization_level=1
        )
        routing = routing_pm

    else:
        layout = None
        routing = None

    if translation_method not in {"translator", "synthesis", "unroller"}:
        translation = plugin_manager.get_passmanager_stage(
            "translation", translation_method, pass_manager_config, optimization_level=1
        )
    else:
        translation = common.generate_translation_passmanager(
            target,
            basis_gates,
            translation_method,
            approximation_degree,
            coupling_map,
            backend_properties,
            unitary_synthesis_method,
            unitary_synthesis_plugin_config,
            hls_config,
        )

    if (coupling_map and not coupling_map.is_symmetric) or (
        target is not None and target.get_non_global_operation_names(strict_direction=True)
    ):
        pre_optimization = common.generate_pre_op_passmanager(
            target, coupling_map, remove_reset_in_zero=True
        )
    else:
        pre_optimization = common.generate_pre_op_passmanager(remove_reset_in_zero=True)
    if optimization_method is None:
        optimization = PassManager()
        unroll = [pass_ for x in translation.passes() for pass_ in x["passes"]]
        # Build nested Flow controllers
        def _unroll_condition(property_set):
            return not property_set["all_gates_in_basis"]

        # Check if any gate is not in the basis, and if so, run unroll passes
        _unroll_if_out_of_basis: list[BasePass | FlowController] = [
            GatesInBasis(basis_gates, target=target),
            ConditionalController(unroll, condition=_unroll_condition),
        ]

        optimization.append(_depth_check + _size_check)
        opt_loop = _opt + _unroll_if_out_of_basis + _depth_check + _size_check
        optimization.append(opt_loop, do_while=_opt_control)
    else:
        optimization = plugin_manager.get_passmanager_stage(
            "optimization", optimization_method, pass_manager_config, optimization_level=1
        )

    sched = plugin_manager.get_passmanager_stage(
        "scheduling", scheduling_method, pass_manager_config, optimization_level=1
    )

    init = common.generate_control_flow_options_check(
        layout_method=layout_method,
        routing_method=routing_method,
        translation_method=translation_method,
        optimization_method=optimization_method,
        scheduling_method=scheduling_method,
        basis_gates=basis_gates,
        target=target,
    )
    if init_method is not None:
        init += plugin_manager.get_passmanager_stage(
            "init", init_method, pass_manager_config, optimization_level=1
        )
    elif unroll_3q is not None:
        init += unroll_3q

    return StagedPassManager(
        init=init,
        layout=layout,
        routing=routing,
        translation=translation,
        pre_optimization=pre_optimization,
        optimization=optimization,
        scheduling=sched,
    )
