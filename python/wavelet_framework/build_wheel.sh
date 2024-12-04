#!/usr/bin/env bash

rm -rf ./build
rm -rf ./dist
rm -rf ./wavelet.egg-info
pip3 wheel -w dist .
