#!/usr/bin/env bash
## NOTE: this is the preferred way to setup the environment for
# this project. The reason is that pip and conda do not seem to play
# nicely for installing JAX on macOS, which is required.

set -euo pipefail

conda env create --file=environment.yaml || conda env update --file=environment.yaml --prune
