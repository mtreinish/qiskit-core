#!/usr/bin/env bash

set -e

python -c 'import sys;assert sys.platform == "win32"'
is_win=$?
python -c 'import platform;assert "64" in platform.architecture()'
is_64bit=$?

set -x
# Create venv for instrumented build and test
python -m venv build_pgo

if [[ $is_win -eq 0 ]]; then
    if [[ $is_64bit -ne 0 ]];
        exit 0
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
