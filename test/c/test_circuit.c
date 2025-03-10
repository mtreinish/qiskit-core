// This code is part of Qiskit.
//
// (C) Copyright IBM 2025.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#include "common.h"
#include <complex.h>
#include <qiskit.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/**
 * Test the zero constructor.
 */
int test_empty() {
    QkCircuit *qc = qk_circuit_new(0, 0);
    uint32_t num_qubits = qk_circuit_num_qubits(qc);
    uint32_t num_clbits = qk_circuit_num_clbits(qc);
    qk_circuit_free(qc);

    return (num_qubits != 0 || num_clbits != 100) ? EqualityError : Ok;
}

int test_circuit() {
    int num_failed = 0;
    num_failed += RUN_TEST(test_empty);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
