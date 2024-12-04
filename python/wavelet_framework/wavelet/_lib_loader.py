from pathlib import Path
from wavelet._ffi_cdecl_generated import __WL_CDECLS
import sys

WL_LIBS = [
    ('win32', 'wavelet.dll'),
    ('linux', 'libwavelet.so'),
    ('darwin', 'libwavelet.dylib'),
]

def load_native_module():
    platform = sys.platform
    lib_name = next((lib for os, lib in WL_LIBS if platform.startswith(os)), None)
    assert lib_name, f"Unsupported platform: {platform}"

    # Locate the library in the package directory
    pkg_path = Path(__file__).parent
    lib_path = pkg_path / lib_name
    assert lib_path.exists(), f"WAVELET shared library not found: {lib_path}"

    # Load the library using cffi
    from cffi import FFI
    ffi = FFI()
    ffi.dlopen("m")  # Math library
    ffi.cdef(__WL_CDECLS)  # Define the C declarations
    lib = ffi.dlopen(str(lib_path))  # Load the shared library
    return ffi, lib