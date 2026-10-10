//! HIP 7 runtime/driver entry points; HIP device pointers are pointers, not CUDA integer addresses.
const std = @import("std");

pub const Result = c_int;
pub const Device = c_int;
pub const DevicePtr = ?*anyopaque;
pub const Context = ?*opaque {};
pub const Stream = ?*opaque {};
pub const Module = ?*opaque {};
pub const Function = ?*opaque {};
pub const Graph = ?*opaque {};
pub const GraphNode = ?*opaque {};
pub const GraphExec = ?*opaque {};

pub const host_malloc_portable: c_uint = 1;
pub const host_malloc_mapped: c_uint = 2;

pub const CaptureMode = enum(c_int) { global = 0, thread_local = 1, relaxed = 2 };
pub const CaptureStatus = enum(c_int) { none = 0, active = 1, invalidated = 2, _ };

/// hipDeviceAttribute_t: HIP's own numbering, not CUDA's.
pub const DeviceAttribute = enum(c_int) {
    clock_rate = 5,
    cooperative_launch = 10,
    integrated = 16,
    l2_cache_size = 19,
    max_threads_per_block = 56,
    multiprocessor_count = 63,
    max_shared_memory_per_block = 74,
    shared_mem_per_block_optin = 75,
    warp_size = 87,
    max_shared_memory_per_multiprocessor = 10002,
};

pub const FunctionAttribute = enum(c_int) {
    max_threads_per_block = 0,
    shared_size_bytes = 1,
    const_size_bytes = 2,
    local_size_bytes = 3,
    num_regs = 4,
    max_dynamic_shared_size_bytes = 8,
};

pub const Dim3 = extern struct { x: c_uint = 1, y: c_uint = 1, z: c_uint = 1 };

/// hipKernelNodeParams: its own field order (block first), 64 bytes.
pub const KernelNodeParams = extern struct {
    block: Dim3,
    extra: ?[*]?*anyopaque,
    func: Function,
    grid: Dim3,
    params: ?[*]?*anyopaque,
    shared_bytes: c_uint,
};

pub const ExecUpdateResult = enum(c_int) {
    success = 0,
    @"error" = 1,
    topology_changed = 2,
    node_type_changed = 3,
    function_changed = 4,
    parameters_changed = 5,
    not_supported = 6,
    unsupported_function_change = 7,
    _,
};

pub const Api = struct {
    hipInit: *const fn (c_uint) callconv(.c) Result,
    hipGetDeviceCount: *const fn (*c_int) callconv(.c) Result,
    hipDeviceGet: *const fn (*Device, c_int) callconv(.c) Result,
    hipDevicePrimaryCtxRetain: *const fn (*Context, Device) callconv(.c) Result,
    hipDevicePrimaryCtxRelease: *const fn (Device) callconv(.c) Result,
    hipCtxSetCurrent: *const fn (Context) callconv(.c) Result,
    hipDeviceSynchronize: *const fn () callconv(.c) Result,
    hipMalloc: *const fn (*DevicePtr, usize) callconv(.c) Result,
    hipFree: *const fn (DevicePtr) callconv(.c) Result,
    hipHostMalloc: *const fn (*DevicePtr, usize, c_uint) callconv(.c) Result,
    hipHostFree: *const fn (DevicePtr) callconv(.c) Result,
    hipMemcpyHtoDAsync: *const fn (DevicePtr, [*]const u8, usize, Stream) callconv(.c) Result,
    hipMemcpyDtoHAsync: *const fn ([*]u8, DevicePtr, usize, Stream) callconv(.c) Result,
    hipMemcpyHtoD: *const fn (DevicePtr, [*]const u8, usize) callconv(.c) Result,
    hipMemcpyDtoH: *const fn ([*]u8, DevicePtr, usize) callconv(.c) Result,
    hipMemset: *const fn (DevicePtr, c_int, usize) callconv(.c) Result,
    hipMemsetAsync: *const fn (DevicePtr, c_int, usize, Stream) callconv(.c) Result,
    hipStreamCreateWithFlags: *const fn (*Stream, c_uint) callconv(.c) Result,
    hipStreamDestroy: *const fn (Stream) callconv(.c) Result,
    hipStreamSynchronize: *const fn (Stream) callconv(.c) Result,
    hipModuleLoadData: *const fn (*Module, [*]const u8) callconv(.c) Result,
    hipModuleUnload: *const fn (Module) callconv(.c) Result,
    hipModuleGetFunction: *const fn (*Function, Module, [*:0]const u8) callconv(.c) Result,
    hipModuleLaunchKernel: *const fn (Function, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, Stream, ?[*]?*anyopaque, ?[*]?*anyopaque) callconv(.c) Result,
    hipGetErrorName: *const fn (Result) callconv(.c) ?[*:0]const u8,
    hipRuntimeGetVersion: *const fn (*c_int) callconv(.c) Result,
    hipDeviceGetName: *const fn ([*]u8, c_int, Device) callconv(.c) Result,
    hipDeviceGetAttribute: *const fn (*c_int, DeviceAttribute, c_int) callconv(.c) Result,
    hipHostGetDevicePointer: *const fn (*DevicePtr, ?*anyopaque, c_uint) callconv(.c) Result,
    hipMemcpyDtoD: *const fn (DevicePtr, DevicePtr, usize) callconv(.c) Result,
    hipMemcpyDtoDAsync: *const fn (DevicePtr, DevicePtr, usize, Stream) callconv(.c) Result,
    hipMemsetD32: *const fn (DevicePtr, c_int, usize) callconv(.c) Result,
    hipMemsetD32Async: *const fn (DevicePtr, c_int, usize, Stream) callconv(.c) Result,
    hipModuleGetGlobal: *const fn (*DevicePtr, *usize, Module, [*:0]const u8) callconv(.c) Result,
    hipFuncGetAttribute: *const fn (*c_int, FunctionAttribute, Function) callconv(.c) Result,
    hipModuleOccupancyMaxActiveBlocksPerMultiprocessor: *const fn (*c_int, Function, c_int, usize) callconv(.c) Result,
    hipModuleLaunchCooperativeKernel: *const fn (Function, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, Stream, ?[*]?*anyopaque) callconv(.c) Result,
    hipStreamBeginCapture: *const fn (Stream, CaptureMode) callconv(.c) Result,
    hipStreamEndCapture: *const fn (Stream, *Graph) callconv(.c) Result,
    hipStreamIsCapturing: *const fn (Stream, *CaptureStatus) callconv(.c) Result,
    hipGraphCreate: *const fn (*Graph, c_uint) callconv(.c) Result,
    hipGraphDestroy: *const fn (Graph) callconv(.c) Result,
    hipGraphAddKernelNode: *const fn (*GraphNode, Graph, ?[*]const GraphNode, usize, *const KernelNodeParams) callconv(.c) Result,
    hipGraphKernelNodeSetParams: *const fn (GraphNode, *const KernelNodeParams) callconv(.c) Result,
    hipGraphAddDependencies: *const fn (Graph, [*]const GraphNode, [*]const GraphNode, usize) callconv(.c) Result,
    hipGraphGetNodes: *const fn (Graph, ?[*]GraphNode, *usize) callconv(.c) Result,
    hipGraphInstantiateWithFlags: *const fn (*GraphExec, Graph, c_ulonglong) callconv(.c) Result,
    hipGraphUpload: *const fn (GraphExec, Stream) callconv(.c) Result,
    hipGraphLaunch: *const fn (GraphExec, Stream) callconv(.c) Result,
    hipGraphExecDestroy: *const fn (GraphExec) callconv(.c) Result,
    hipGraphExecUpdate: *const fn (GraphExec, Graph, *GraphNode, *ExecUpdateResult) callconv(.c) Result,
    hipGraphExecKernelNodeSetParams: *const fn (GraphExec, GraphNode, *const KernelNodeParams) callconv(.c) Result,
};

test "HIP ABI handles and scalar types have C widths" {
    try std.testing.expectEqual(@sizeOf(c_int), @sizeOf(Device));
    try std.testing.expectEqual(@sizeOf(*anyopaque), @sizeOf(DevicePtr));
    try std.testing.expectEqual(@sizeOf(*anyopaque), @sizeOf(Context));
    try std.testing.expectEqual(@sizeOf(*anyopaque), @sizeOf(Stream));
    try std.testing.expectEqual(@sizeOf(*anyopaque), @sizeOf(Module));
    try std.testing.expectEqual(@sizeOf(*anyopaque), @sizeOf(Function));
    try std.testing.expectEqual(@sizeOf(*anyopaque), @sizeOf(Graph));
    try std.testing.expectEqual(@sizeOf(c_int), @sizeOf(DeviceAttribute));
}

test "hipKernelNodeParams keeps HIP's layout" {
    try std.testing.expectEqual(64, @sizeOf(KernelNodeParams));
    try std.testing.expectEqual(0, @offsetOf(KernelNodeParams, "block"));
    try std.testing.expectEqual(16, @offsetOf(KernelNodeParams, "extra"));
    try std.testing.expectEqual(24, @offsetOf(KernelNodeParams, "func"));
    try std.testing.expectEqual(32, @offsetOf(KernelNodeParams, "grid"));
    try std.testing.expectEqual(48, @offsetOf(KernelNodeParams, "params"));
    try std.testing.expectEqual(56, @offsetOf(KernelNodeParams, "shared_bytes"));
}
