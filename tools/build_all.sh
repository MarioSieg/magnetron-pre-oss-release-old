#!/usr/bin/env bash

BUILD_MODE=release
WAVELET_ROOT=$(pwd)
WAVELET_BUILD=$WAVELET_ROOT/bin/$BUILD_MODE
WAVELET_SRC=$WAVELET_ROOT

echo "Wavelet root: $WAVELET_ROOT"
echo "Wavelet build: $WAVELET_BUILD"
echo "Wavelet source: $WAVELET_SRC"
echo "Building Wavelet in $BUILD_MODE mode"

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

echo "Configuring wavelet SDK..."
cmake -S "$WAVELET_SRC" -B "$WAVELET_BUILD" -DCMAKE_BUILD_TYPE=$BUILD_MODE # Configure

echo "Building wavelet..."
cmake --build "$WAVELET_BUILD" --target all -j "$NUM_CPUS" # Build

SO_BIN="$WAVELET_BUILD/$(shared_lib_prefix)wavelet.$(shared_lib_extension)"
if [ -f "$SO_BIN" ]; then
    echo "Wavelet shared library built at $SO_BIN"
else
    echo "! Wavelet shared library not found at $SO_BIN"
fi

TEST_BIN="$WAVELET_BUILD/wavelet_test"
if [ -f "$TEST_BIN" ]; then
    echo "Wavelet test binary built at $TEST_BIN"
else
    echo "! Wavelet test binary not found at $TEST_BIN"
fi

BENCH_BIN="$WAVELET_BUILD/wavelet_benchmark"
if [ -f "$BENCH_BIN" ]; then
    echo "Wavelet benchmark binary built at $BENCH_BIN"
else
    echo "! Wavelet benchmark binary not found at $BENCH_BIN"
fi
