#!/usr/bin/env bash
# Configure, build and test the C++ library inside the devtools container.
set -euo pipefail
cd "$(dirname "$0")/.."
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD":/work -w /work devtools:latest bash -c \
  "cmake -S cpp -B cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build cpp/build && cd cpp/build && ctest --output-on-failure"
