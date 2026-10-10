//! CUDA driver API types and entry points as NVIDIA documents them (CUDA 12.0+ ABI), declared by hand.

pub const Result = c_int;
pub const Device = c_int;
pub const DevicePtr = u64;

pub const Context = ?*opaque {};
pub const Module = ?*opaque {};
pub const Function = ?*opaque {};
pub const Stream = ?*opaque {};
pub const Event = ?*opaque {};
pub const Graph = ?*opaque {};
pub const GraphNode = ?*opaque {};
pub const GraphExec = ?*opaque {};
pub const Kernel = ?*opaque {};

pub const success: Result = 0;
pub const error_not_found: Result = 500;
pub const error_not_ready: Result = 600;

pub const stream_non_blocking: c_uint = 1;
pub const event_disable_timing: c_uint = 2;
pub const host_alloc_portable: c_uint = 1;
pub const host_alloc_devicemap: c_uint = 2;
pub const host_register_devicemap: c_uint = 2;
pub const host_register_iomemory: c_uint = 4; // the range is device or I/O memory, not RAM

pub const CaptureMode = enum(c_int) { global = 0, thread_local = 1, relaxed = 2 };
pub const CaptureStatus = enum(c_int) { none = 0, active = 1, invalidated = 2, _ };

pub const DeviceAttribute = enum(c_int) {
    max_threads_per_block = 1,
    warp_size = 10,
    clock_rate = 13,
    multiprocessor_count = 16,
    integrated = 18,
    l2_cache_size = 38,
    compute_capability_major = 75,
    compute_capability_minor = 76,
    max_shared_memory_per_multiprocessor = 81,
    max_shared_memory_per_block_optin = 97,
};

pub const FunctionAttribute = enum(c_int) {
    max_threads_per_block = 0,
    shared_size_bytes = 1,
    const_size_bytes = 2,
    local_size_bytes = 3,
    num_regs = 4,
    ptx_version = 5,
    binary_version = 6,
    max_dynamic_shared_size_bytes = 8,
    non_portable_cluster_size_allowed = 14,
};

pub const func_cache_prefer_shared: c_int = 1;

pub const LaunchAttributeId = enum(c_uint) {
    cooperative = 2,
    cluster_dimension = 4,
    cluster_scheduling_policy_preference = 5,
    programmatic_stream_serialization = 6,
    priority = 8,
};

pub const cluster_scheduling_spread: c_int = 1;

pub const Dim3 = extern struct { x: c_uint = 1, y: c_uint = 1, z: c_uint = 1 };

/// CUlaunchAttributeValue: a 64-byte union aligned for its pointer members.
pub const LaunchAttributeValue = extern union {
    pad: [64]u8,
    align8: ?*anyopaque,
    int: c_int,
    cluster_dim: extern struct { x: c_uint, y: c_uint, z: c_uint },
};

pub const LaunchAttribute = extern struct {
    id: LaunchAttributeId,
    pad: [4]u8 = @splat(0),
    value: LaunchAttributeValue,
};

pub const LaunchConfig = extern struct {
    grid_x: c_uint,
    grid_y: c_uint,
    grid_z: c_uint,
    block_x: c_uint,
    block_y: c_uint,
    block_z: c_uint,
    shared_bytes: c_uint,
    stream: Stream,
    attrs: ?[*]LaunchAttribute,
    num_attrs: c_uint,
};

/// CUDA_KERNEL_NODE_PARAMS_v2.
pub const KernelNodeParams = extern struct {
    func: Function,
    grid_x: c_uint,
    grid_y: c_uint,
    grid_z: c_uint,
    block_x: c_uint,
    block_y: c_uint,
    block_z: c_uint,
    shared_bytes: c_uint,
    params: ?[*]?*anyopaque,
    extra: ?[*]?*anyopaque,
    kern: Kernel = null,
    ctx: Context = null,
};

pub const ExecUpdateResult = enum(c_uint) {
    success = 0,
    @"error" = 1,
    topology_changed = 2,
    node_type_changed = 3,
    function_changed = 4,
    parameters_changed = 5,
    not_supported = 6,
    unsupported_function_change = 7,
    attributes_changed = 8,
    _,
};

pub const ExecUpdateResultInfo = extern struct {
    result: ExecUpdateResult,
    error_node: GraphNode,
    error_from_node: GraphNode,
};

pub const jit_info_log_buffer: c_int = 3;
pub const jit_info_log_buffer_size: c_int = 4;
pub const jit_error_log_buffer: c_int = 5;
pub const jit_error_log_buffer_size: c_int = 6;

const R = Result;
const Ptr = ?*anyopaque;
const CPtr = ?*const anyopaque;
const Params = ?[*]?*anyopaque;

/// Each field is the exact exported symbol, so the ABI is the one that symbol version documents.
pub const Api = struct {
    cuInit: *const fn (c_uint) callconv(.c) R,
    cuDriverGetVersion: *const fn (*c_int) callconv(.c) R,
    cuDeviceGet: *const fn (*Device, c_int) callconv(.c) R,
    cuDeviceGetCount: *const fn (*c_int) callconv(.c) R,
    cuDeviceGetName: *const fn ([*]u8, c_int, Device) callconv(.c) R,
    cuDeviceGetAttribute: *const fn (*c_int, DeviceAttribute, Device) callconv(.c) R,
    cuDeviceTotalMem_v2: *const fn (*usize, Device) callconv(.c) R,
    cuDevicePrimaryCtxRetain: *const fn (*Context, Device) callconv(.c) R,
    cuDevicePrimaryCtxRelease_v2: *const fn (Device) callconv(.c) R,
    cuCtxSetCurrent: *const fn (Context) callconv(.c) R,
    cuCtxGetCurrent: *const fn (*Context) callconv(.c) R,
    cuCtxSynchronize: *const fn () callconv(.c) R,
    cuMemGetInfo_v2: *const fn (*usize, *usize) callconv(.c) R,
    cuMemAlloc_v2: *const fn (*DevicePtr, usize) callconv(.c) R,
    cuMemFree_v2: *const fn (DevicePtr) callconv(.c) R,
    cuMemHostAlloc: *const fn (*Ptr, usize, c_uint) callconv(.c) R,
    cuMemHostGetDevicePointer_v2: *const fn (*DevicePtr, Ptr, c_uint) callconv(.c) R,
    cuMemFreeHost: *const fn (Ptr) callconv(.c) R,
    cuMemHostRegister_v2: *const fn (Ptr, usize, c_uint) callconv(.c) R,
    cuMemHostUnregister: *const fn (Ptr) callconv(.c) R,
    cuMemcpyHtoD_v2: *const fn (DevicePtr, CPtr, usize) callconv(.c) R,
    cuMemcpyDtoH_v2: *const fn (Ptr, DevicePtr, usize) callconv(.c) R,
    cuMemcpyDtoD_v2: *const fn (DevicePtr, DevicePtr, usize) callconv(.c) R,
    cuMemcpyHtoDAsync_v2: *const fn (DevicePtr, CPtr, usize, Stream) callconv(.c) R,
    cuMemcpyDtoHAsync_v2: *const fn (Ptr, DevicePtr, usize, Stream) callconv(.c) R,
    cuMemcpyDtoDAsync_v2: *const fn (DevicePtr, DevicePtr, usize, Stream) callconv(.c) R,
    cuMemsetD8_v2: *const fn (DevicePtr, u8, usize) callconv(.c) R,
    cuMemsetD32_v2: *const fn (DevicePtr, c_uint, usize) callconv(.c) R,
    cuMemsetD8Async: *const fn (DevicePtr, u8, usize, Stream) callconv(.c) R,
    cuMemsetD32Async: *const fn (DevicePtr, c_uint, usize, Stream) callconv(.c) R,
    cuStreamCreate: *const fn (*Stream, c_uint) callconv(.c) R,
    cuStreamDestroy_v2: *const fn (Stream) callconv(.c) R,
    cuStreamSynchronize: *const fn (Stream) callconv(.c) R,
    cuStreamWaitEvent: *const fn (Stream, Event, c_uint) callconv(.c) R,
    cuStreamQuery: *const fn (Stream) callconv(.c) R,
    cuEventCreate: *const fn (*Event, c_uint) callconv(.c) R,
    cuEventDestroy_v2: *const fn (Event) callconv(.c) R,
    cuEventRecord: *const fn (Event, Stream) callconv(.c) R,
    cuEventSynchronize: *const fn (Event) callconv(.c) R,
    cuEventQuery: *const fn (Event) callconv(.c) R,
    cuEventElapsedTime: *const fn (*f32, Event, Event) callconv(.c) R,
    cuModuleLoadData: *const fn (*Module, CPtr) callconv(.c) R,
    cuModuleLoadDataEx: *const fn (*Module, CPtr, c_uint, ?[*]c_int, ?[*]Ptr) callconv(.c) R,
    cuModuleUnload: *const fn (Module) callconv(.c) R,
    cuModuleGetFunction: *const fn (*Function, Module, [*:0]const u8) callconv(.c) R,
    cuModuleGetGlobal_v2: *const fn (*DevicePtr, *usize, Module, [*:0]const u8) callconv(.c) R,
    cuFuncGetAttribute: *const fn (*c_int, FunctionAttribute, Function) callconv(.c) R,
    cuFuncSetAttribute: *const fn (Function, FunctionAttribute, c_int) callconv(.c) R,
    cuFuncSetCacheConfig: *const fn (Function, c_int) callconv(.c) R,
    cuOccupancyMaxActiveBlocksPerMultiprocessor: *const fn (*c_int, Function, c_int, usize) callconv(.c) R,
    cuLaunchKernel: *const fn (Function, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, Stream, Params, Params) callconv(.c) R,
    cuLaunchKernelEx: *const fn (*const LaunchConfig, Function, Params, Params) callconv(.c) R,
    cuStreamBeginCapture_v2: *const fn (Stream, CaptureMode) callconv(.c) R,
    cuStreamEndCapture: *const fn (Stream, *Graph) callconv(.c) R,
    cuStreamIsCapturing: *const fn (Stream, *CaptureStatus) callconv(.c) R,
    cuGraphCreate: *const fn (*Graph, c_uint) callconv(.c) R,
    cuGraphDestroy: *const fn (Graph) callconv(.c) R,
    cuGraphAddKernelNode_v2: *const fn (*GraphNode, Graph, ?[*]const GraphNode, usize, *const KernelNodeParams) callconv(.c) R,
    cuGraphKernelNodeGetParams_v2: *const fn (GraphNode, *KernelNodeParams) callconv(.c) R,
    cuGraphKernelNodeSetParams_v2: *const fn (GraphNode, *const KernelNodeParams) callconv(.c) R,
    cuGraphAddDependencies: *const fn (Graph, [*]const GraphNode, [*]const GraphNode, usize) callconv(.c) R,
    cuGraphGetNodes: *const fn (Graph, ?[*]GraphNode, *usize) callconv(.c) R,
    cuGraphInstantiateWithFlags: *const fn (*GraphExec, Graph, u64) callconv(.c) R,
    cuGraphUpload: *const fn (GraphExec, Stream) callconv(.c) R,
    cuGraphLaunch: *const fn (GraphExec, Stream) callconv(.c) R,
    cuGraphExecDestroy: *const fn (GraphExec) callconv(.c) R,
    cuGraphExecUpdate_v2: *const fn (GraphExec, Graph, *ExecUpdateResultInfo) callconv(.c) R,
    cuGraphExecKernelNodeSetParams_v2: *const fn (GraphExec, GraphNode, *const KernelNodeParams) callconv(.c) R,
    cuGetErrorName: *const fn (R, *?[*:0]const u8) callconv(.c) R,
    cuGetErrorString: *const fn (R, *?[*:0]const u8) callconv(.c) R,
};

comptime {
    const std = @import("std");
    std.debug.assert(@sizeOf(LaunchAttribute) == 72 and @offsetOf(LaunchAttribute, "value") == 8);
    std.debug.assert(@sizeOf(LaunchConfig) == 56 and @offsetOf(LaunchConfig, "stream") == 32);
    std.debug.assert(@sizeOf(KernelNodeParams) == 72 and @offsetOf(KernelNodeParams, "params") == 40);
    std.debug.assert(@sizeOf(ExecUpdateResultInfo) == 24);
}
