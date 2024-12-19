@echo off
call build_wheel.bat
pip3 install dist\*.whl --force-reinstall
