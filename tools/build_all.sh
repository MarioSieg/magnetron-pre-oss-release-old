#!/usr/bin/env bash

BUILD_MODE=release
MAGNETRON_ROOT=$(pwd)
MAGNETRON_BUILD=$MAGNETRON_ROOT/bin/$BUILD_MODE
MAGNETRON_SRC=$MAGNETRON_ROOT

echo "magnetron root: $MAGNETRON_ROOT"
echo "magnetron build: $MAGNETRON_BUILD"
echo "magnetron source: $MAGNETRON_SRC"
echo "Building magnetron in $BUILD_MODE mode"

cpu_count() { # Query CPUs used to compile concurrently
    OS="$(uname -s)"
    if [ "$OS" = "Linux" ]; then
        echo "$(nproc --all)"
    elif [ "$OS" = "Darwin" ] || \
         [ "$(echo "$OS" | grep -q BSD)" = "BSD" ]; then
        echo "$(sysctl -n hw.ncpu)"
    else
        echo "$(getconf _NPROCESSORS_ONLN)"
    fi
}

shared_lib_prefix() {
    OS="$(uname -s)"
    if [ "$OS" = "Linux" ] || [ "$(echo "$OS" | grep -q BSD)" = "BSD" ]; then
        echo "lib"
    elif [ "$OS" = "Darwin" ]; then
        echo "lib"
    else
        echo ""
    fi
}

shared_lib_extension() {
    OS="$(uname -s)"
    if [ "$OS" = "Linux" ] || [ "$(echo "$OS" | grep -q BSD)" = "BSD" ]; then
        echo "so"
    elif [ "$OS" = "Darwin" ]; then
        echo "dylib"
    else
        echo "dll"
    fi
}

NUM_CPUS=$(cpu_count)
if [ -z "$NUM_CPUS" ] || [ "$NUM_CPUS" -eq 0 ]; then
    NUM_CPUS=1
fi
echo "Using $NUM_CPUS CPUs to compile"

echo "Configuring magnetron SDK..."
cmake -S "$MAGNETRON_SRC" -B "$MAGNETRON_BUILD" -DCMAKE_BUILD_TYPE=$BUILD_MODE # Configure

echo "Building magnetron..."
cmake --build "$MAGNETRON_BUILD" --target all -j "$NUM_CPUS" # Build

SO_BIN="$MAGNETRON_BUILD/$(shared_lib_prefix)magnetron.$(shared_lib_extension)"
if [ -f "$SO_BIN" ]; then
    echo "magnetron shared library built at $SO_BIN"
else
    echo "! magnetron shared library not found at $SO_BIN"
fi

TEST_BIN="$MAGNETRON_BUILD/magnetron_test"
if [ -f "$TEST_BIN" ]; then
    echo "magnetron test binary built at $TEST_BIN"
else
    echo "! magnetron test binary not found at $TEST_BIN"
fi

BENCH_BIN="$MAGNETRON_BUILD/magnetron_benchmark"
if [ -f "$BENCH_BIN" ]; then
    echo "magnetron benchmark binary built at $BENCH_BIN"
else
    echo "! magnetron benchmark binary not found at $BENCH_BIN"
fi
