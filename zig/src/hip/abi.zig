//! HIP 7 runtime/driver entry points; HIP device pointers are pointers, not CUDA integer addresses.
const std = @import("std");

pub const Result = c_int;
pub const Device = c_int;
pub const DevicePtr = ?*anyopaque;
pub const Context = ?*opaque {};
pub const Stream = ?*opaque {};
pub const Module = ?*opaque {};
pub const Function = ?*opaque {};

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
};

test "HIP ABI handles and scalar types have C widths" {
    try std.testing.expectEqual(@sizeOf(c_int), @sizeOf(Device));
    try std.testing.expectEqual(@sizeOf(*anyopaque), @sizeOf(DevicePtr));
    try std.testing.expectEqual(@sizeOf(*anyopaque), @sizeOf(Context));
    try std.testing.expectEqual(@sizeOf(*anyopaque), @sizeOf(Stream));
    try std.testing.expectEqual(@sizeOf(*anyopaque), @sizeOf(Module));
    try std.testing.expectEqual(@sizeOf(*anyopaque), @sizeOf(Function));
}
