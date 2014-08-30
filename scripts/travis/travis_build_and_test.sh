#!/bin/bash
# Script called by Travis to do a CPU-only build of and test Caffe.

# if $BUILD_DOCS is set, just build the docs
if [[ ! -z "$BUILD_DOCS" ]]; then
    make docs
    python -m SimpleHTTPServer &
    sleep 600  # sleep for 10 minutes while the user reads the served documentation
    exit 0
fi

if $WITH_CMAKE; then
  mkdir build
  cd build
  cmake -DBUILD_PYTHON=ON -DBUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release -DCPU_ONLY=ON ..
  make --keep-going
  make runtest
  make lint
  make clean
  cd -
else
  export CPU_ONLY=1
  make --keep-going all test pycaffe warn lint
  make runtest
  make all
  make test
  make pycaffe
  make warn
  make lint
fi
