# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

from msml._ffi_cdecl import __MSML_CDECLS
from ctypes.util import find_library
from os.path import isfile

from cffi import FFI


def load_native_msml_lib(path: str | None = None, build_dir: str = 'debug') -> (FFI, object):
    lib_path: str | None = path

    if lib_path is None or not isfile(lib_path):
        msml_lib_locations: list[str] = [
            f'../bin/{build_dir}/msml.dll',
            f'../bin/{build_dir}/libmsml.so',
            f'../bin/{build_dir}/libmsml.dylib',
        ]
        for loc in msml_lib_locations:  # Try to find the shared library in the locations (mostly used for debug builds)
            if isfile(loc):
                lib_path = loc
                break
            elif isfile(f'../{loc}'):  # Try to find the shared library in the parent directory
                lib_path = f'../{loc}'
                break
        if lib_path is None:  # If not found, try to find the shared library in the system paths
            lib_path = find_library('msml')
        assert lib_path is not None and isfile(lib_path), f'MSML shared library not found: {lib_path}'

    # Now load shared library
    ffi = FFI()
    ffi.dlopen('m')
    ffi.cdef(__MSML_CDECLS)  # Define the C declarations
    lib = ffi.dlopen(lib_path)  # Load the shared library
    assert lib is not None, f'Failed to load MSML shared library: {lib_path}'
    return ffi, lib
