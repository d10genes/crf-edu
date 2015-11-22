#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

python utils.py
py.test test.py
# python test.py
mypy utils.py

echo "=> LOOKS GOOD!"
