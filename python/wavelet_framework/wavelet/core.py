# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
# Implements core functionality: Context, Tensors and Operations.

# To debug Python to C FFI calls:
# $ cp examples/perceptron.py tmp.py && gdb -ex r --args python3 tmp.py
# See also https://wiki.python.org/moin/DebuggingWithGdb

import random
import faulthandler
import weakref
from dataclasses import dataclass
from os import getenv
from os.path import isfile
from wavelet._lib_loader import load_native_module
from enum import Enum, auto

# Enable faulthandler for debugging
faulthandler.enable()

ffi, C = load_native_module()  # Load the native WAVELET shared library

# Common constants
MAX_DIMS: int = 6
MAX_ARG_TENSORS: int = 2
WL_MAX_OP_PARAMS: int = 6
DIM_MAX: int = ((1 << 64) - 1) >> 1

def set_log_mode(enable_log: bool) -> None:
    """
    Set logging mode for the WAVELET backend.

    Parameters
    ----------
    enable_log : bool
        If True, enables logging of operations and internal states.
    """
    C.wl_set_set_log_mode(enable_log)

def pack_color(r: int, g: int, b: int) -> int:
    """
    Packs three 8-bit color components (R, G, B) into a single 24-bit integer.

    Parameters
    ----------
    r : int
        Red component [0-255].
    g : int
        Green component [0-255].
    b : int
        Blue component [0-255].

    Returns
    -------
    int
        A single integer representing the packed RGB color.
    """
    return C.wl_pack_color_u8(r, g, b)

class ComputeDevice(Enum):
    """
    Compute devices available for parallel computations.
    """
    CPU = 0
    CUDA = auto()

class PRNGAlgorithm(Enum):
    """
    Pseudorandom number generator algorithms.
    """
    MERSENNE_TWISTER = 0  # Default - Mersenne Twister
    PCG = auto()  # Permuted Congruential Generator

class DType(Enum):
    """
    Supported data types for tensors.
    """
    F32 = 0

class ColorChannels(Enum):
    """
    Desired color channels when loading images.
    """
    AUTO = 0   # Automatically determine the number of color channels
    GRAY = auto()   # Grayscale F32
    GRAY_A = auto() # Grayscale F32 + Alpha
    RGB = auto()    # R32G32B32
    RGBA = auto()   # R32G32B32A32

class Operator(Enum):
    """
    All supported tensor operators used in computations.
    """
    NOP = 0
    CLONE = auto()
    VIEW = auto()
    TRANSPOSE = auto()
    PERMUTE = auto()
    MEAN = auto()
    MIN = auto()
    MAX = auto()
    SUM = auto()
    ABS = auto()
    NEG = auto()
    LOG = auto()
    SQR = auto()
    SQRT = auto()
    SIN = auto()
    COS = auto()
    STEP = auto()
    SOFTMAX = auto()
    SOFTMAX_DV = auto()
    SIGMOID = auto()
    SIGMOID_DV = auto()
    HARD_SIGMOID = auto()
    SILU = auto()
    SILU_DV = auto()
    TANH = auto()
    TANH_DV = auto()
    RELU = auto()
    RELU_DV = auto()
    GELU = auto()
    GELU_DV = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    ADDS = auto()
    SUBS = auto()
    MULS = auto()
    DIVS = auto()
    MATMUL = auto()
    _COUNT = auto()

    @property
    def name(self) -> str:
        """Returns the operator name as a string."""
        assert self.value < self._COUNT.value
        return ffi.string(C.wl_op_get_name(self.value)).decode('utf-8')

    @property
    def mnemonic(self) -> str:
        """Returns a short mnemonic name for the operator."""
        assert self.value < self._COUNT.value
        return ffi.string(C.wl_op_get_mnemonic(self.value)).decode('utf-8')

    @property
    def argument_count(self) -> int:
        """Returns the number of tensor arguments required by this operator."""
        assert self.value < self._COUNT.value
        return C.wl_op_get_argcount(self.value)

    @property
    def supports_inplace(self) -> bool:
        """Checks if the operator supports in-place modifications."""
        return C.wl_op_supports_inplace(self.value)

    @property
    def is_unary(self) -> bool:
        """Checks if the operator is a unary operator."""
        return self.argument_count == 1

    @property
    def is_binary(self) -> bool:
        """Checks if the operator is a binary operator."""
        return self.argument_count == 2

class OpParam:
    """
    Represents an operation parameter to be passed to a tensor operator.
    """

    def __init__(self, value: int) -> None:
        """
        Internal constructor.

        Parameters
        ----------
        value : int
            Internal integer representation of the parameter.
        """
        self.value = value

    @staticmethod
    def new_int(x: int) -> 'OpParam':
        """
        Creates an integer operation parameter.

        Parameters
        ----------
        x : int
            The integer value to store.

        Returns
        -------
        OpParam
            A new integer operation parameter.
        """
        return OpParam(C.wl_op_param_int(x))

    @staticmethod
    def new_float(x: float) -> 'OpParam':
        """
        Creates a float operation parameter.

        Parameters
        ----------
        x : float
            The float value to store.

        Returns
        -------
        OpParam
            A new float operation parameter.
        """
        return OpParam(C.wl_op_param_float(x))

    @property
    def is_int(self) -> bool:
        """Checks if the parameter is an integer type."""
        return C.wl_op_param_is_int(self.value)

    @property
    def is_float(self) -> bool:
        """Checks if the parameter is a float type."""
        return C.wl_op_param_is_float(self.value)

    @property
    def unpack_int(self) -> int:
        """
        Returns the stored integer value.

        Returns
        -------
        int
            The integer value of the parameter.
        """
        assert self.is_int
        return C.wl_op_param_unpack_int(self.value)

    @property
    def unpack_float(self) -> float:
        """
        Returns the stored float value.

        Returns
        -------
        float
            The float value of the parameter.
        """
        assert self.is_float
        return C.wl_op_param_unpack_float(self.value)

class GraphEvalOrder(Enum):
    """
    Order in which the computation graph should be evaluated.
    Applies to deferred execution mode only.
    """
    FORWARD = 0   # Evaluate graph left-to-right
    REVERSE = 1   # Evaluate graph right-to-left

class ExecutionMode(Enum):
    """
    Execution modes for the WAVELET context.
    """
    EAGER = 0     # Execute operations immediately (dynamic graph)
    DEFERRED = 1  # Build computation graph and execute later (static graph)

@dataclass
class GlobalConfig:
    verbose: bool = (getenv('WL_VERBOSE', '0') == '1')
    compute_device: ComputeDevice = ComputeDevice.CUDA if getenv('WL_COMPUTE_DEVICE') == 'CUDA' else ComputeDevice.CPU

class Context:
    """
    Manages the WAVELET context and tensor lifecycles, including device selection,
    memory allocation, and execution mode.

    A global active context is created automatically when needed.
    """

    _active: 'Context' = None  # Global context

    @staticmethod
    def active() -> 'Context':
        """
        Returns the active global context, creating one if it does not exist.

        Returns
        -------
        Context
            The currently active context.
        """
        if Context._active is None:
            set_log_mode(GlobalConfig.verbose)
            Context._active = Context(GlobalConfig.compute_device)
        return Context._active

    def __init__(self, device: ComputeDevice, *, execution_mode: ExecutionMode = ExecutionMode.EAGER):
        """
        Initializes a new WAVELET context.

        Parameters
        ----------
        device : ComputeDevice
            The compute device (CPU or CUDA).
        execution_mode : ExecutionMode, optional
            The execution mode (eager or deferred), by default EAGER.
        """
        self.ctx = C.wl_ctx_create2(device.value)
        self.execution_mode = execution_mode

    @property
    def compute_device(self) -> ComputeDevice:
        """
        Returns the compute device used by this context.

        Returns
        -------
        ComputeDevice
            The compute device (CPU or CUDA).
        """
        return ComputeDevice(C.wl_ctx_get_compute_device_type(self.ctx))

    @property
    def compute_device_name(self) -> str:
        """
        Returns the name of the active compute device.

        Returns
        -------
        str
            Name of the device, e.g. "CPU" or "NVIDIA GPU".
        """
        return ffi.string(C.wl_ctx_get_compute_device_name(self.ctx)).decode('utf-8')

    @property
    def execution_mode(self) -> ExecutionMode:
        """
        Returns the execution mode of the context.

        Returns
        -------
        ExecutionMode
            The current execution mode (EAGER or DEFERRED).
        """
        return ExecutionMode(C.wl_ctx_get_exec_mode(self.ctx))

    @execution_mode.setter
    def execution_mode(self, mode: ExecutionMode):
        """
        Sets the execution mode of the context.

        Parameters
        ----------
        mode : ExecutionMode
            Desired mode (EAGER or DEFERRED).
        """
        C.wl_ctx_set_exec_mode(self.ctx, mode.value)

    @property
    def prng_algorithm(self) -> PRNGAlgorithm:
        """
        Returns the PRNG algorithm currently used.

        Returns
        -------
        PRNGAlgorithm
            The PRNG algorithm in use.
        """
        return PRNGAlgorithm(C.wl_ctx_get_prng_algorithm(self.ctx))

    @prng_algorithm.setter
    def prng_algorithm(self, algorithm: PRNGAlgorithm):
        """
        Sets the PRNG algorithm and re-seeds it.

        Parameters
        ----------
        algorithm : PRNGAlgorithm
            The desired PRNG algorithm.
        """
        C.wl_ctx_set_prng_algorithm(self.ctx, algorithm.value, random.randint(0, 1 << 63))

    @property
    def os_name(self) -> str:
        """
        Returns the operating system name.

        Returns
        -------
        str
            The OS name (e.g., "Linux").
        """
        return ffi.string(C.wl_ctx_get_os_name(self.ctx)).decode('utf-8')

    @property
    def cpu_name(self) -> str:
        """
        Returns the CPU name/brand string.

        Returns
        -------
        str
            The CPU name.
        """
        return ffi.string(C.wl_ctx_get_cpu_name(self.ctx)).decode('utf-8')

    @property
    def cpu_virtual_cores(self) -> int:
        """
        Returns the number of virtual CPU cores (logical processors).

        Returns
        -------
        int
            Count of virtual CPU cores.
        """
        return C.wl_ctx_get_cpu_virtual_cores(self.ctx)

    @property
    def cpu_physical_cores(self) -> int:
        """
        Returns the number of physical CPU cores.

        Returns
        -------
        int
            Count of physical CPU cores.
        """
        return C.wl_ctx_get_cpu_physical_cores(self.ctx)

    @property
    def cpu_sockets(self) -> int:
        """
        Returns the number of CPU sockets present on the system.

        Returns
        -------
        int
            CPU socket count.
        """
        return C.wl_ctx_get_cpu_sockets(self.ctx)

    @property
    def physical_memory_total(self) -> int:
        """
        Returns the total physical memory (RAM) in bytes.

        Returns
        -------
        int
            Total system memory in bytes.
        """
        return C.wl_ctx_get_physical_memory_total(self.ctx)

    @property
    def physical_memory_free(self) -> int:
        """
        Returns the amount of free physical memory (RAM) in bytes.

        Returns
        -------
        int
            Free memory in bytes.
        """
        return C.wl_ctx_get_physical_memory_free(self.ctx)

    @property
    def physical_memory_used(self) -> int:
        """
        Returns the amount of used physical memory (RAM) in bytes.

        Returns
        -------
        int
            Used memory in bytes.
        """
        return abs(self.physical_memory_total - self.physical_memory_free)

    @property
    def is_numa_system(self) -> bool:
        """
        Checks if the system uses Non-Uniform Memory Access (NUMA).

        Returns
        -------
        bool
            True if NUMA is supported, False otherwise.
        """
        return C.wl_ctx_is_numa_system(self.ctx)

    @property
    def total_allocated_pool_memory(self) -> int:
        """
        Returns the total allocated memory (pool) by this context.

        Returns
        -------
        int
            Total allocated memory in bytes.
        """
        return C.wl_ctx_total_allocated_pool_memory(self.ctx)

    @property
    def total_tensors_created(self) -> int:
        """
        Returns the total number of tensors created (including views).

        Returns
        -------
        int
            Count of all tensors created.
        """
        return C.wl_ctx_get_total_tensors_created(self.ctx)

    @property
    def total_tensors_allocated(self) -> int:
        """
        Returns the total number of allocated tensors (not including views).

        Returns
        -------
        int
            Count of allocated tensors.
        """
        return C.wl_ctx_get_total_tensors_allocated(self.ctx)

    def start_profiler(self) -> None:
        """
        Starts recording profiling information for operations.
        Profiling must be stopped to produce a report. Slightly reduces performance.
        """
        C.wl_ctx_profile_start_recording(self.ctx)

    def stop_profiler(self, export_csv_file: str | None = None) -> None:
        """
        Stops recording profiling information and generates a report.

        Parameters
        ----------
        export_csv_file : str, optional
            Path to export a CSV profiling report. If None, only an internal report is generated.
        """
        csv_file = ffi.NULL if export_csv_file is None else bytes(export_csv_file, 'utf-8')
        C.wl_ctx_profile_stop_recording(self.ctx, csv_file)

    def __del__(self):
        """
        Destructor that releases context resources.
        """
        C.wl_ctx_destroy(self.ctx)
        self.ctx = ffi.NULL

class Tensor:
    """
    Represents a tensor in the WAVELET library. Supports various operations and transformations.
    """

    def __init__(self, internal_instance: ffi.CData | None = None) -> None:
        """
        Internal constructor. Do not call directly. Use static methods or operators.

        Parameters
        ----------
        internal_instance : ffi.CData or None
            Internal tensor instance (C data pointer).
        """
        self.context_ref = None
        self.tensor = internal_instance

    def __del__(self) -> None:
        """Releases tensor resources upon object destruction."""
        C.wl_tensor_decref(self.tensor)
        self.tensor = ffi.NULL

    _DISPATCH = {
        1: C.wl_tensor_create_1d,
        2: C.wl_tensor_create_2d,
        3: C.wl_tensor_create_3d,
        4: C.wl_tensor_create_4d,
        5: C.wl_tensor_create_5d,
        6: C.wl_tensor_create_6d
    }
    assert len(_DISPATCH) == MAX_DIMS

    def _new(self, ctx: Context, *, shape: tuple[int, ...], dtype: DType = DType.F32,
             name: str | None = None) -> None:
        """
        Internal helper to create a new tensor instance.

        Parameters
        ----------
        ctx : Context
            The WAVELET context where this tensor belongs.
        shape : tuple[int, ...]
            Dimensions of the tensor.
        dtype : DType, optional
            Data type of the tensor, by default DType.F32.
        name : str or None, optional
            A friendly name for the tensor, by default None.
        """
        assert 0 < len(shape) <= MAX_DIMS, f'Invalid number of dimensions: {len(shape)}'
        assert all(0 < dim <= DIM_MAX for dim in shape), 'Invalid dimension size'
        self.context_ref = weakref.ref(ctx)
        self.tensor = self._DISPATCH[len(shape)](ctx.ctx, dtype.value, *shape)
        self.name = f'Tensor {self.shape}' if name is None else name

    @staticmethod
    def operator(op: Operator, inplace: bool = False, params: list[OpParam] | None = None, *args) -> 'Tensor':
        """
        Applies an operator to one or more tensors.

        Parameters
        ----------
        op : Operator
            The operator to apply.
        inplace : bool, optional
            If True and operator supports it, modifies the tensor in-place, by default False.
        params : list[OpParam] or None, optional
            Additional operation parameters, by default None.
        *args : Tensor
            Tensors to be used as input arguments.

        Returns
        -------
        Tensor
            The resulting tensor after applying the operation.
        """
        c_para_ptr = ffi.NULL
        if params is not None:
            assert 0 < len(params) <= WL_MAX_OP_PARAMS, f'Invalid number of operation parameters: {len(params)}'
            param_vals = [param.value & ((1 << 64) - 1) for param in params] + [0] * (WL_MAX_OP_PARAMS - len(params))
            c_para = ffi.new(f'wl_op_param_t[{WL_MAX_OP_PARAMS}]', param_vals)
            c_para_ptr = ffi.new(f'wl_op_param_t(*)[{WL_MAX_OP_PARAMS}]', c_para)
        assert len(args) == op.argument_count, f'{len(args)} != {op.argument_count} for {op}'
        tensors: ffi.CData = ffi.new(f'wl_tensor_t*[{len(args)}]', [arg.tensor for arg in args])
        ctx: ffi.CData = C.wl_tensor_get_ctx(args[0].tensor)
        instance: ffi.CData = C.wl_tensor_operator(ctx, op.value, inplace, tensors, len(args), c_para_ptr)
        assert instance != ffi.NULL, 'Operation invalid'
        return Tensor(instance)

    @staticmethod
    def empty(shape: tuple[int, ...], *, dtype: DType = DType.F32, name: str | None = None) -> 'Tensor':
        """
        Creates an empty tensor with uninitialized data.

        Parameters
        ----------
        shape : tuple[int, ...]
            The shape of the tensor.
        dtype : DType, optional
            Data type of the tensor, by default F32.
        name : str or None, optional
            A friendly name for the tensor, by default None.

        Returns
        -------
        Tensor
            The newly created empty tensor.
        """
        tensor = Tensor(None)
        tensor._new(Context.active(), shape=shape, dtype=dtype, name=name)
        return tensor

    @staticmethod
    def full(shape: tuple[int, ...], *, fill_value: float, dtype: DType = DType.F32,
             name: str | None = None) -> 'Tensor':
        """
        Creates a tensor filled with a given constant value.

        Parameters
        ----------
        shape : tuple[int, ...]
            The shape of the tensor.
        fill_value : float
            The constant value to fill.
        dtype : DType, optional
            Data type, by default F32.
        name : str or None, optional
            A friendly name for the tensor, by default None.

        Returns
        -------
        Tensor
            The filled tensor.
        """
        tensor = Tensor(None)
        tensor._new(Context.active(), shape=shape, dtype=dtype, name=name)
        C.wl_tensor_fill(tensor.tensor, fill_value)
        return tensor

    @staticmethod
    def const(data, *, dtype: DType = DType.F32,
              name: str | None = None) -> 'Tensor':
        """
        Creates a tensor from a nested Python list or a single scalar.

        Parameters
        ----------
        data : scalar or nested list
            Data to create the tensor from.
        dtype : DType, optional
            Data type, by default F32.
        name : str or None, optional
            A friendly name for the tensor, by default None.

        Returns
        -------
        Tensor
            The constructed tensor.
        """
        def flatten_nested_lists(nested):
            if not isinstance(nested, list):
                return (), [nested]
            elif len(nested) == 0:
                return (0,), []
            else:
                shapes = []
                flattened = []
                for item in nested:
                    shape_lst, flat = flatten_nested_lists(item)
                    shapes.append(shape_lst)
                    flattened.extend(flat)
                first_shape = shapes[0]
                for s in shapes:
                    assert s == first_shape, "All sub-lists must have the same shape"
                return (len(nested),) + first_shape, flattened

        shape, flattened_data = flatten_nested_lists(data)
        tensor = Tensor(None)
        tensor._new(Context.active(), shape=tuple(shape), dtype=dtype, name=name)
        size: int = len(flattened_data) * ffi.sizeof('float')
        C.wl_tensor_copy_buffer_from(tensor.tensor, ffi.new(f'float[{len(flattened_data)}]', flattened_data), size)
        return tensor

    @staticmethod
    def zeros(shape: tuple[int, ...], *, dtype: DType = DType.F32,
              name: str | None = None) -> 'Tensor':
        """
        Creates a tensor filled with zeros.

        Parameters
        ----------
        shape : tuple[int, ...]
            The shape of the tensor.
        dtype : DType, optional
            Data type, by default F32.
        name : str or None, optional
            A friendly name for the tensor, by default None.

        Returns
        -------
        Tensor
            The zero-filled tensor.
        """
        return Tensor.full(shape, fill_value=0.0, dtype=dtype, name=name)

    @staticmethod
    def rand(shape: tuple[int, ...], *, interval: (float, float) = (-1.0, 1.0), dtype: DType = DType.F32,
             name: str | None = None) -> 'Tensor':
        """
        Creates a tensor filled with random uniform values within a given interval.

        Parameters
        ----------
        shape : tuple[int, ...]
            The shape of the tensor.
        interval : (float, float), optional
            Min and max values for uniform distribution, by default (-1.0, 1.0).
        dtype : DType, optional
            Data type, by default F32.
        name : str or None, optional
            A friendly name for the tensor, by default None.

        Returns
        -------
        Tensor
            The tensor filled with random values.
        """
        tensor = Tensor(None)
        tensor._new(Context.active(), shape=shape, dtype=dtype, name=name)
        if interval[1] < interval[0]:
            interval = (interval[1], interval[0])
        C.wl_tensor_fill_random_uniform(tensor.tensor, interval[0], interval[1])
        return tensor

    @staticmethod
    def normal(shape: tuple[int, ...], *, mean: float, stddev: float) -> 'Tensor':
        """
        Creates a tensor filled with random values from a normal distribution.

        Parameters
        ----------
        shape : tuple[int, ...]
            The shape of the tensor.
        mean : float
            Mean of the normal distribution.
        stddev : float
            Standard deviation of the normal distribution.

        Returns
        -------
        Tensor
            The tensor filled with normally distributed values.
        """
        tensor = Tensor(None)
        tensor._new(Context.active(), shape=shape, dtype=DType.F32)
        C.wl_tensor_fill_random_normal(tensor.tensor, mean, stddev)
        return tensor

    @staticmethod
    def load(file_path: str) -> 'Tensor':
        """
        Loads a tensor from a binary WAVELET file.

        Parameters
        ----------
        file_path : str
            Path to the .wavelet file.

        Returns
        -------
        Tensor
            The loaded tensor.
        """
        assert file_path.endswith('.wavelet'), 'File must be a WAVELET file'
        instance = C.wl_tensor_load(Context.active().ctx, bytes(file_path, 'utf-8'))
        return Tensor(internal_instance=instance)

    @staticmethod
    def load_image(file_path: str, *,
                   name: str | None = None,
                   channels=ColorChannels.AUTO,
                   resize_to: (int, int) = (0, 0)) -> 'Tensor':
        """
        Loads an image from a file and creates a tensor.

        Parameters
        ----------
        file_path : str
            Path to the image file.
        name : str or None, optional
            A friendly name for the tensor, by default None.
        channels : ColorChannels, optional
            Desired color channels to load, by default AUTO.
        resize_to : (int, int), optional
            If not (0,0), resize the image to (width, height), by default (0,0).

        Returns
        -------
        Tensor
            The created image tensor.
        """
        assert isfile(file_path), f'File not found: {file_path}'
        instance = C.wl_tensor_load_image(Context.active().ctx, bytes(file_path, 'utf-8'), channels.value,
                                          resize_to[0], resize_to[1])
        tensor = Tensor(internal_instance=instance)
        if name is not None:
            tensor.name = name
        return tensor

    def print(self, print_header: bool = False, print_data: bool = True) -> None:
        """
        Prints the tensor metadata and optionally its data.

        Parameters
        ----------
        print_header : bool, optional
            If True, prints a header line, by default False.
        print_data : bool, optional
            If True, prints the tensor data, by default True.
        """
        C.wl_tensor_print(self.tensor, print_header, print_data)

    @property
    def name(self) -> str:
        """
        Returns the name of the tensor.

        Returns
        -------
        str
            The tensor's name.
        """
        return ffi.string(C.wl_tensor_get_name(self.tensor)).decode('utf-8')

    @name.setter
    def name(self, name: str) -> None:
        """
        Sets a name for the tensor.

        Parameters
        ----------
        name : str
            The new name for the tensor.
        """
        C.wl_tensor_set_name(self.tensor, bytes(name, 'utf-8'))

    @property
    def rank(self) -> int:
        """
        Returns the rank (number of dimensions) of the tensor.

        Returns
        -------
        int
            Number of dimensions.
        """
        return C.wl_tensor_rank(self.tensor)

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Returns the shape (dimensions) of the tensor.

        Returns
        -------
        tuple[int, ...]
            The dimensions of the tensor.
        """
        return tuple(ffi.unpack(C.wl_tensor_shape(self.tensor), self.rank))

    @property
    def strides(self) -> tuple[int, ...]:
        """
        Returns the strides of the tensor.

        Returns
        -------
        tuple[int, ...]
            The strides for each dimension.
        """
        return tuple(ffi.unpack(C.wl_tensor_strides(self.tensor), self.rank))

    @property
    def dtype(self) -> DType:
        """
        Returns the data type of the tensor.

        Returns
        -------
        DType
            The data type, e.g. DType.F32.
        """
        return DType(C.wl_tensor_dtype(self.tensor))

    @property
    def data_ptr(self) -> int:
        """
        Returns the pointer to the tensor's data buffer.

        Returns
        -------
        int
            Memory address of the tensor data.
        """
        return int(ffi.cast('uintptr_t', C.wl_tensor_data_ptr(self.tensor)))

    def to_list(self) -> list[float]:
        """
        Returns the tensor data as a Python list of floats.

        Returns
        -------
        list[float]
            A flat list containing all tensor elements.
        """
        assert self.dtype == DType.F32, 'Invalid data type'
        return ffi.unpack(ffi.cast('float*', C.wl_tensor_data_ptr(self.tensor)), self.numel)

    def scalar(self) -> float:
        """
        Returns the scalar value of a 0D tensor.

        Returns
        -------
        float
            The single scalar value.
        """
        assert self.dtype == DType.F32, 'Invalid data type'
        return ffi.unpack(ffi.cast('float*', C.wl_tensor_data_ptr(self.tensor)), 1)[0]

    @property
    def data_size(self) -> int:
        """
        Returns the size of the tensor buffer in bytes.

        Returns
        -------
        int
            Data size in bytes.
        """
        return C.wl_tensor_data_size(self.tensor)

    @property
    def numel(self) -> int:
        """
        Returns the number of elements in the tensor.

        Returns
        -------
        int
            Number of elements.
        """
        return C.wl_tensor_numel(self.tensor)

    @property
    def num_rows(self) -> int:
        """
        Returns the number of rows in a 2D tensor (matrix).

        Returns
        -------
        int
            Number of rows.
        """
        return C.wl_tensor_num_rows(self.tensor)

    @property
    def num_cols(self) -> int:
        """
        Returns the number of columns in a 2D tensor (matrix).

        Returns
        -------
        int
            Number of columns.
        """
        return C.wl_tensor_num_cols(self.tensor)

    @property
    def is_scalar(self) -> bool:
        """
        Checks if the tensor is a scalar (0D).

        Returns
        -------
        bool
            True if scalar, otherwise False.
        """
        return C.wl_tensor_is_scalar(self.tensor)

    @property
    def is_vector(self) -> bool:
        """
        Checks if the tensor is a vector (1D).

        Returns
        -------
        bool
            True if vector, otherwise False.
        """
        return C.wl_tensor_is_vector(self.tensor)

    @property
    def is_matrix(self) -> bool:
        """
        Checks if the tensor is a matrix (2D).

        Returns
        -------
        bool
            True if matrix, otherwise False.
        """
        return C.wl_tensor_is_matrix(self.tensor)

    @property
    def is_volume(self) -> bool:
        """
        Checks if the tensor is a volume (3D or higher).

        Returns
        -------
        bool
            True if volume, otherwise False.
        """
        return C.wl_tensor_is_volume(self.tensor)

    @property
    def is_transposed(self) -> bool:
        """
        Checks if the tensor is transposed.

        Returns
        -------
        bool
            True if transposed, otherwise False.
        """
        return C.wl_tensor_is_transposed(self.tensor)

    @property
    def is_permuted(self) -> bool:
        """
        Checks if the tensor is permuted.

        Returns
        -------
        bool
            True if permuted, otherwise False.
        """
        return C.wl_tensor_is_permuted(self.tensor)

    def is_shape_eq(self, other: 'Tensor') -> bool:
        """
        Checks if two tensors have the same shape.

        Parameters
        ----------
        other : Tensor
            Another tensor to compare shape.

        Returns
        -------
        bool
            True if shapes match, otherwise False.
        """
        return C.wl_tensor_is_shape_eq(self.tensor, other.tensor)

    def are_strides_eq(self, other: 'Tensor') -> bool:
        """
        Checks if two tensors have identical strides.

        Parameters
        ----------
        other : Tensor
            Another tensor to compare strides.

        Returns
        -------
        bool
            True if strides match, otherwise False.
        """
        return C.wl_tensor_are_strides_eq(self.tensor, other.tensor)

    def can_broadcast(self, other: 'Tensor') -> bool:
        """
        Checks if `other` tensor can be broadcasted to the shape of this tensor.

        Parameters
        ----------
        other : Tensor
            Another tensor.

        Returns
        -------
        bool
            True if broadcastable, otherwise False.
        """
        return C.wl_tensor_can_broadcast(self.tensor, other.tensor)

    @property
    def width(self) -> int:
        """
        Returns the width of an image tensor (assumes layout: CxHxW).

        Returns
        -------
        int
            Width dimension.
        """
        return self.shape[2]

    @property
    def height(self) -> int:
        """
        Returns the height of an image tensor (assumes layout: CxHxW).

        Returns
        -------
        int
            Height dimension.
        """
        return self.shape[1]

    @property
    def channels(self) -> int:
        """
        Returns the number of channels in an image tensor (assumes layout: CxHxW).

        Returns
        -------
        int
            Number of channels.
        """
        return self.shape[0]

    @property
    def is_contiguous(self) -> bool:
        """
        Checks if the tensor is contiguous in memory.

        Returns
        -------
        bool
            True if contiguous, otherwise False.
        """
        return C.wl_tensor_is_contiguous(self.tensor)

    def is_close(self, other: 'Tensor', eps: float = -1.0, print_eq_percent: bool = False) -> (bool, float):
        """
        Checks if the tensor is close to another tensor within a given epsilon.

        Parameters
        ----------
        other : Tensor
            Another tensor to compare to.
        eps : float, optional
            Epsilon tolerance. If -1.0, uses a default internal value.
        print_eq_percent : bool, optional
            If True, prints the percentage of equal elements.

        Returns
        -------
        (bool, float)
            A tuple (close_status, eq_percent).
            close_status is True if the tensors are close.
            eq_percent is the percentage of approximately equal elements.
        """
        percent_eq = ffi.new('double[1]')
        is_eq = C.wl_tensor_is_close(self.tensor, other.tensor, eps, percent_eq)
        if print_eq_percent:
            print(f'Tensors are close: {is_eq}, Percent equal: {percent_eq[0]:.2f}%')
        return is_eq, percent_eq[0]

    def draw_box(self, p1: (int, int), p2: (int, int), width: int = 2, rgb: int = 0xffffff):
        """
        Draws a rectangular box on an image tensor.

        Parameters
        ----------
        p1 : (int, int)
            Top-left corner coordinates (x, y).
        p2 : (int, int)
            Bottom-right corner coordinates (x, y).
        width : int, optional
            Line width of the box, by default 2.
        rgb : int, optional
            24-bit RGB color (e.g. 0xFFFFFF for white), by default 0xffffff.
        """
        assert p2[0] > p1[0] and p2[1] > p1[1] and width > 0
        C.wl_tensor_img_draw_box(self.tensor, p1[0], p1[1], p2[0], p2[1], width, rgb & 0xffffff)

    def draw_text(self, p: (int, int), size: int, txt: str, rgb: int = 0xffffff):
        """
        Draws text on an image tensor.

        Parameters
        ----------
        p : (int, int)
            Position to draw text (x, y).
        size : int
            Font size.
        txt : str
            The text to draw.
        rgb : int, optional
            24-bit RGB color, by default white (0xffffff).
        """
        C.wl_tensor_img_draw_text(self.tensor, p[0], p[1], size, rgb & 0xffffff, bytes(txt, 'utf-8'))

    def save(self, file_path: str) -> None:
        """
        Saves the tensor to a binary WAVELET file.

        Parameters
        ----------
        file_path : str
            File path to save the tensor. Appends '.wavelet' if not present.
        """
        if not file_path.endswith('.wavelet'):
            file_path += '.wavelet'
        C.wl_tensor_save(self.tensor, bytes(file_path, 'utf-8'))

    def save_image(self, file_path: str) -> None:
        """
        Saves a 3D image tensor as a JPG image file.

        Parameters
        ----------
        file_path : str
            File path to save the image.

        Raises
        ------
        AssertionError
            If tensor is not 3D or channel count is not supported.
        """
        assert self.rank == 3, 'Tensor must be a 3D image tensor'
        assert self.channels in (1, 3, 4), 'Invalid number of color channels'
        C.wl_tensor_save_image(self.tensor, bytes(file_path, 'utf-8'))

    def clone(self) -> 'Tensor':
        """
        Creates a new tensor with the same data as this one (deep copy).

        Returns
        -------
        Tensor
            A cloned tensor.
        """
        return self.operator(Operator.CLONE, False, None, self)

    def view(self) -> 'Tensor':
        """
        Creates a view of the tensor that shares underlying data (shallow copy).

        Returns
        -------
        Tensor
            A view tensor.
        """
        return self.operator(Operator.VIEW, False, None, self)

    def transpose(self) -> 'Tensor':
        """
        Transposes the tensor (swaps the last two dimensions).

        Returns
        -------
        Tensor
            A transposed tensor.
        """
        return self.operator(Operator.TRANSPOSE, False, None, self)

    def permute(self, axes: tuple[int, ...]) -> 'Tensor':
        """
        Permutes the dimensions of the tensor.

        Parameters
        ----------
        axes : tuple[int, ...]
            A tuple specifying the permutation of axes.

        Returns
        -------
        Tensor
            A tensor with permuted dimensions.
        """
        assert len(axes) == MAX_DIMS, f'Invalid number of axes: {axes}'
        for i in range(MAX_DIMS):
            assert 0 <= axes[i] < MAX_DIMS
            for j in range(i + 1, MAX_DIMS):
                assert axes[i] != axes[j], f'Duplicate axis: {axes[i]}'
        return self.operator(Operator.PERMUTE, False, [OpParam.new_int(axis) for axis in axes], self)

    def mean(self) -> 'Tensor':
        """Computes the mean of all elements in the tensor."""
        return self.operator(Operator.MEAN, False, None, self)

    def min(self) -> 'Tensor':
        """Computes the minimum value in the tensor."""
        return self.operator(Operator.MIN, False, None, self)

    def max(self) -> 'Tensor':
        """Computes the maximum value in the tensor."""
        return self.operator(Operator.MAX, False, None, self)

    def sum(self) -> 'Tensor':
        """Computes the sum of all elements in the tensor."""
        return self.operator(Operator.SUM, False, None, self)

    def abs(self) -> 'Tensor':
        """Computes element-wise absolute value."""
        return self.operator(Operator.ABS, False, None, self)

    def abs_(self) -> 'Tensor':
        """In-place element-wise absolute value."""
        return self.operator(Operator.ABS, True, None, self)

    def neg(self) -> 'Tensor':
        """Computes element-wise negation."""
        return self.operator(Operator.NEG, False, None, self)

    def neg_(self) -> 'Tensor':
        """In-place element-wise negation."""
        return self.operator(Operator.NEG, True, None, self)

    def __neg__(self) -> 'Tensor':
        """Overloads unary negation: -X."""
        return self.neg()

    def log(self) -> 'Tensor':
        """Computes element-wise natural logarithm."""
        return self.operator(Operator.LOG, False, None, self)

    def log_(self) -> 'Tensor':
        """In-place element-wise natural logarithm."""
        return self.operator(Operator.LOG, True, None, self)

    def sqr(self) -> 'Tensor':
        """Computes element-wise square of values."""
        return self.operator(Operator.SQR, False, None, self)

    def sqr_(self) -> 'Tensor':
        """In-place element-wise square of values."""
        return self.operator(Operator.SQR, True, None, self)

    def sqrt(self) -> 'Tensor':
        """Computes element-wise square root."""
        return self.operator(Operator.SQRT, False, None, self)

    def sqrt_(self) -> 'Tensor':
        """In-place element-wise square root."""
        return self.operator(Operator.SQRT, True, None, self)

    def sin(self) -> 'Tensor':
        """Computes element-wise sine."""
        return self.operator(Operator.SIN, False, None, self)

    def sin_(self) -> 'Tensor':
        """In-place element-wise sine."""
        return self.operator(Operator.SIN, True, None, self)

    def cos(self) -> 'Tensor':
        """Computes element-wise cosine."""
        return self.operator(Operator.COS, False, None, self)

    def cos_(self) -> 'Tensor':
        """In-place element-wise cosine."""
        return self.operator(Operator.COS, True, None, self)

    def heaviside_step(self) -> 'Tensor':
        """Computes element-wise Heaviside step function."""
        return self.operator(Operator.STEP, False, None, self)

    def heaviside_step_(self) -> 'Tensor':
        """In-place element-wise Heaviside step function."""
        return self.operator(Operator.STEP, True, None, self)

    def softmax(self, derivative: bool = False) -> 'Tensor':
        """
        Applies softmax or its derivative on the tensor.

        Parameters
        ----------
        derivative : bool, optional
            If True, computes softmax derivative instead of softmax, by default False.

        Returns
        -------
        Tensor
            The transformed tensor.
        """
        return self.operator(Operator.SOFTMAX_DV if derivative else Operator.SOFTMAX, False, None, self)

    def softmax_(self, derivative: bool = False) -> 'Tensor':
        """In-place softmax or softmax derivative."""
        return self.operator(Operator.SOFTMAX_DV if derivative else Operator.SOFTMAX, True, None, self)

    def sigmoid(self, derivative: bool = False) -> 'Tensor':
        """
        Applies sigmoid or its derivative on the tensor.

        Parameters
        ----------
        derivative : bool, optional
            If True, computes sigmoid derivative, by default False.

        Returns
        -------
        Tensor
            The transformed tensor.
        """
        return self.operator(Operator.SIGMOID_DV if derivative else Operator.SIGMOID, False, None, self)

    def sigmoid_(self, derivative: bool = False) -> 'Tensor':
        """In-place sigmoid or sigmoid derivative."""
        return self.operator(Operator.SIGMOID_DV if derivative else Operator.SIGMOID, True, None, self)

    def hard_sigmoid(self) -> 'Tensor':
        """Applies hard sigmoid to the tensor."""
        return self.operator(Operator.HARD_SIGMOID, False, None, self)

    def hard_sigmoid_(self) -> 'Tensor':
        """In-place hard sigmoid."""
        return self.operator(Operator.HARD_SIGMOID, True, None, self)

    def silu(self, derivative: bool = False) -> 'Tensor':
        """
        Applies SiLU (Sigmoid-Weighted Linear Unit) or its derivative.

        Parameters
        ----------
        derivative : bool, optional
            If True, applies SiLU derivative, by default False.

        Returns
        -------
        Tensor
            The transformed tensor.
        """
        return self.operator(Operator.SILU_DV if derivative else Operator.SILU, False, None, self)

    def silu_(self, derivative: bool = False) -> 'Tensor':
        """In-place SiLU or SiLU derivative."""
        return self.operator(Operator.SILU_DV if derivative else Operator.SILU, True, None, self)

    def tanh(self, derivative: bool = False) -> 'Tensor':
        """
        Applies hyperbolic tangent or its derivative.

        Parameters
        ----------
        derivative : bool, optional
            If True, computes tanh derivative, by default False.

        Returns
        -------
        Tensor
            The transformed tensor.
        """
        return self.operator(Operator.TANH_DV if derivative else Operator.TANH, False, None, self)

    def tanh_(self, derivative: bool = False) -> 'Tensor':
        """In-place tanh or tanh derivative."""
        return self.operator(Operator.TANH_DV if derivative else Operator.TANH, True, None, self)

    def relu(self, derivative: bool = False) -> 'Tensor':
        """
        Applies ReLU (Rectified Linear Unit) or its derivative.

        Parameters
        ----------
        derivative : bool, optional
            If True, computes ReLU derivative, by default False.

        Returns
        -------
        Tensor
            The transformed tensor.
        """
        return self.operator(Operator.RELU_DV if derivative else Operator.RELU, False, None, self)

    def relu_(self, derivative: bool = False) -> 'Tensor':
        """In-place ReLU or ReLU derivative."""
        return self.operator(Operator.RELU_DV if derivative else Operator.RELU, True, None, self)

    def gelu(self, derivative: bool = False) -> 'Tensor':
        """
        Applies GELU (Gaussian Error Linear Unit) or its derivative.

        Parameters
        ----------
        derivative : bool, optional
            If True, computes GELU derivative, by default False.

        Returns
        -------
        Tensor
            The transformed tensor.
        """
        return self.operator(Operator.GELU_DV if derivative else Operator.GELU, False, None, self)

    def gelu_(self, derivative: bool = False) -> 'Tensor':
        """In-place GELU or GELU derivative."""
        return self.operator(Operator.GELU_DV if derivative else Operator.GELU, True, None, self)

    def __add__(self, other: object | int | float) -> 'Tensor':
        """Element-wise addition with another tensor or scalar."""
        if isinstance(other, Tensor):
            return self.operator(Operator.ADD, False, None, self, other)
        else:
            return self.operator(Operator.ADDS, False, [OpParam.new_float(float(other))], self)

    def __iadd__(self, other: object | int | float) -> 'Tensor':
        """In-place element-wise addition."""
        if isinstance(other, Tensor):
            return self.operator(Operator.ADD, True, None, self, other)
        else:
            return self.operator(Operator.ADDS, True, [OpParam.new_float(float(other))], self)

    def __sub__(self, other: object | int | float) -> 'Tensor':
        """Element-wise subtraction with another tensor or scalar."""
        if isinstance(other, Tensor):
            return self.operator(Operator.SUB, False, None, self, other)
        else:
            return self.operator(Operator.SUBS, False, [OpParam.new_float(float(other))], self)

    def __isub__(self, other: object | int | float) -> 'Tensor':
        """In-place element-wise subtraction."""
        if isinstance(other, Tensor):
            return self.operator(Operator.SUB, True, None, self, other)
        else:
            return self.operator(Operator.SUBS, True, [OpParam.new_float(float(other))], self)

    def __mul__(self, other: object | int | float) -> 'Tensor':
        """Element-wise multiplication with another tensor or scalar."""
        if isinstance(other, Tensor):
            return self.operator(Operator.MUL, False, None, self, other)
        else:
            return self.operator(Operator.MULS, False, [OpParam.new_float(float(other))], self)

    def __imul__(self, other: object | int | float) -> 'Tensor':
        """In-place element-wise multiplication."""
        if isinstance(other, Tensor):
            return self.operator(Operator.MUL, True, None, self, other)
        else:
            return self.operator(Operator.MULS, True, [OpParam.new_float(float(other))], self)

    def __truediv__(self, other: object | int | float) -> 'Tensor':
        """Element-wise division with another tensor or scalar."""
        if isinstance(other, Tensor):
            return self.operator(Operator.DIV, False, None, self, other)
        else:
            return self.operator(Operator.DIVS, False, [OpParam.new_float(float(other))], self)

    def __itruediv__(self, other: object | int | float) -> 'Tensor':
        """In-place element-wise division."""
        if isinstance(other, Tensor):
            return self.operator(Operator.DIV, True, None, self, other)
        else:
            return self.operator(Operator.DIVS, True, [OpParam.new_float(float(other))], self)

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication with another tensor: A @ B."""
        return self.operator(Operator.MATMUL, False, None, self, other)

    def __imatmul__(self, other: 'Tensor') -> 'Tensor':
        """In-place matrix multiplication: A @= B."""
        return self.operator(Operator.MATMUL, True, None, self, other)

    def __eq__(self, other: 'Tensor') -> bool:
        """
        Checks if two tensors have identical data and shape.

        Parameters
        ----------
        other : Tensor
            Another tensor to compare.

        Returns
        -------
        bool
            True if equal, otherwise False.
        """
        return C.wl_tensor_eq(self.tensor, other.tensor)

    def __str__(self) -> str:
        """String representation prints the tensor header and data."""
        self.print(True, True)
        return ''
