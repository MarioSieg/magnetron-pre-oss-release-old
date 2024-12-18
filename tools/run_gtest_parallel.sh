#!/usr/bin/env bash
# This script is used to run the gtest in parallel
# Must be executed in MSML repo root directory!
mkdir .tmp
cp bin/release/test/msml_test ./.tmp/msml_test_release
cp bin/debug/test/msml_test ./.tmp/msml_test_debug
python3 tools/gtest_parallel.py ./.tmp/msml_test_release
python3 tools/gtest_parallel.py ./.tmp/msml_test_debug
rm .tmp/msml_test_release
rm .tmp/msml_test_debug
