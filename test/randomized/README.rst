Randomized Tranpiler Tests
==========================

This directory contains randomized testing for the qiskit transpiler. It works
by generating a random circuit over all the standard gates in qiskit using Aer
(or BaiscAer if Aer can't be found) to generate a unitary for that circuit,

Running
-------

Standard python unittest is used. So you can use the stdlib unittest runner
with::

   $ python -m unittest test_random.py

assuming your current working directory is this test directory.

or any other unittest compatible test runner will also work. Just be aware
that since each test is using aer to simulate the circuit the memory
requirements will be high depending on the number of configured qubits.


Configuration
-------------

There are 3 environment variables which can be used to configure the execution
of the randomized. These can be used to specify the shape of circuits to
generate or the random seed to use .

* ``QISKIT_RANDOM_QUBITS`` - Set to an integer for the number of qubits to use.
  Defaults to 5.
* ``QISKIT_RANDOM_DEPTH`` - Set to an integer for the number of gates to use
  in the circuit. Defaults to 42.
* ``QISKIT_RANDOM_QASMDIR`` - Set to a path to store generated qasm files for
  failed test runs. If one is not set use default TEMPDIR.

You can run a test using these by just specifying the env variables when running
the tests, for example::

    $ QISKIT_RANDOM_QUBITS=10 QISKIT_RANDOM_DEPTH=100 python -m unittest test_random.py
