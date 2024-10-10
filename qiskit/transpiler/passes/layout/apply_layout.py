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

"""Transform a circuit with virtual qubits into a circuit with physical qubits."""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit._accelerate.apply_layout import apply_layout


class ApplyLayout(TransformationPass):
    """Transform a circuit with virtual qubits into a circuit with physical qubits.

    Transforms a DAGCircuit with virtual qubits into a DAGCircuit with physical qubits
    by applying the Layout given in `property_set`.
    Requires either of passes to set/select Layout, e.g. `SetLayout`, `TrivialLayout`.
    Assumes the Layout has full physical qubits.

    If a post layout pass is run and sets the ``post_layout`` property set field with
    a new layout to use after ``ApplyLayout`` has already run once this pass will
    compact the layouts so that we apply
    ``original_virtual`` -> ``existing_layout`` -> ``new_layout`` -> ``new_physical``
    so that the output circuit and layout combination become:
    ``original_virtual`` -> ``new_physical``
    """

    def run(self, dag):
        """Run the ApplyLayout pass on ``dag``.

        Args:
            dag (DAGCircuit): DAG to map.

        Returns:
            DAGCircuit: A mapped DAG (with physical qubits).

        Raises:
            TranspilerError: if no layout is found in ``property_set`` or no full physical qubits.
        """
        if not self.property_set["layout"]:
            raise TranspilerError(
                "No 'layout' is found in property_set. Please run a Layout pass in advance."
            )
        layout = self.property_set["layout"].get_virtual_bits()
        layout_array = [layout.get(virt, idx) for idx, virt in enumerate(dag.qubits)]
        post_layout_array = None
        print("qubits")
        print(dag.qubits)
        print("layout")
        print(layout)
        print("Layout array")
        print(layout_array)
        if self.property_set["post_layout"] is not None:
            print("post_layout")
            post_layout = self.property_set["post_layout"].get_virtual_bits()
            print(post_layout)
            print("post_layout array")
            post_layout_array = [
                post_layout.get(virt, idx) for idx, virt in enumerate(dag.qubits)
            ]
            print(post_layout_array)
        else:
            self.property_set["original_qubit_indices"] = {
                bit: index for index, bit in enumerate(dag.qubits)
            }

        final_layout_array = None
        print("finale_layout")
        print(self.property_set["final_layout"])
        if self.property_set["final_layout"] is not None:
            final_layout = self.property_set["final_layout"].get_virtual_bits()
            final_layout_array = [
                final_layout.get(virt, idx) for idx, virt in enumerate(dag.qubits)
            ]
            print("final layout array")
            print(final_layout_array)

        out_dag, full_layout, new_final_layout = apply_layout(
            dag, layout_array, post_layout_array, final_layout_array
        )
        print("RESULTS")
        print("full")
        print(full_layout)
        print("new_final")
        print(new_final_layout)

        phys_map = list(range(len(out_dag.qubits)))
        if full_layout is not None:
            new_layout = Layout()
            original_layout = self.property_set["layout"]
            old_phys_to_virt = original_layout.get_physical_bits()
            new_virtual_to_physical = self.property_set["post_layout"].get_virtual_bits()
            reverse_layout = {phys: virt for virt, phys in enumerate(full_layout)}
            for new_virt, new_phys in new_virtual_to_physical.items():
                old_phys = dag.find_bit(new_virt).index
                old_virt = old_phys_to_virt[old_phys]
                phys_map[old_phys] = new_phys
                new_layout.add(old_virt, reverse_layout[new_phys])
            for reg in original_layout.get_registers():
                new_layout.add_register(reg)
            self.property_set["layout"] = new_layout
        if new_final_layout is not None:
            self.property_set["final_layout"] = Layout(
                {
                    out_dag.qubits[phys_map[dag.find_bit(old_virt).index]]: phys_map[old_phys]
                    for old_virt, old_phys in self.property_set["final_layout"].get_virtual_bits().items()
                }
            )
        return out_dag
