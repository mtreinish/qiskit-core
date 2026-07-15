// This code is part of Qiskit.
//
// (C) Copyright IBM 2025.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#include "common.h"
#include <qiskit.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int test_call_store_add_pipeline(void) {
    int result = Ok;
    QkQuantumProgram *prog = qk_quantum_program_new();
    QkStore *s1 = qk_program_node_make_store_double(3.0);
    QkStore *s2 = qk_program_node_make_store_double(5.0);
    qk_quantum_program_add_node(prog, QkQuantumProgramNodeType_Store, s1, "s1");
    qk_quantum_program_add_node(prog, QkQuantumProgramNodeType_Store, s2, "s2");
    qk_quantum_program_add_node(prog, QkQuantumProgramNodeType_Add, NULL, "add");

    QkOwnedPath *source_s1 = qk_owned_path_new(1);
    QkPort *from_s1 = qk_quantum_program_port_new("s1", source_s1);
    QkOwnedPath *target_s1 = qk_owned_path_new(1);
    qk_owned_path_append_key(target_s1, "x");
    QkPort *to_s1 = qk_quantum_program_port_new("add", target_s1);
    qk_quantum_program_add_edge(prog, from_s1, to_s1);

    QkOwnedPath *source_s2 = qk_owned_path_new(1);
    QkPort *from_s2 = qk_quantum_program_port_new("s2", source_s2);
    QkOwnedPath *target_s2 = qk_owned_path_new(1);
    qk_owned_path_append_key(target_s2, "y");
    QkPort *to_s2 = qk_quantum_program_port_new("add", target_s2);
    qk_quantum_program_add_edge(prog, from_s2, to_s2);

    QkOwnedPath *out = qk_owned_path_new(1);
    QkPort *out_port = qk_quantum_program_port_new("add", out);
    qk_quantum_program_set_output(prog, "result", out_port);
    return result;
}

int test_quantum_program(void) {
    int num_failed = 0;

    num_failed += RUN_TEST(test_call_store_add_pipeline);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
