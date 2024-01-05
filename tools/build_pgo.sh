#!/usr/bin/env bash

# Create venv for instrumented build and test
python -m venv build_pgo
# Build with instrumentation
RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data -Ctarget-cpu=x86-64-v3" build_pgo/bin/pip install -c constraints.txt .
build_pgo/bin/pip install -c constraints.txt -r requirements-dev.txt
# Run profile data generation

stestr run
