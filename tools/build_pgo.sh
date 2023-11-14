#!/usr/bin/env bash

set -xe

# Create venv for instrumented build and test
python -m venv build_pgo

python -c 'import sys;assert sys.platform == "win32"'
win32=$?
if win32 == 0; then
    source build_pgo/Scripts/activate
else
    source build_pgo/bin/activate
fi

# Build with instrumentation
RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" pip install -c constraints.txt -e .
RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" python setup.py build_rust --release --inplace
pip install -c constraints.txt -r requirements-dev.txt
# Run profile data generation

stestr run

deactivate
