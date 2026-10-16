//! Loaded GPU code: cubin, fatbin or PTX images, their kernels by symbol name and their device globals.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;

pub const Module = struct {
    d: *const Driver,
    handle: abi.Module,

    /// A cubin or fatbin image (8-byte aligned); the driver picks the SASS for this device and copies what it needs.
    pub fn load(d: *const Driver, image: []const u8) Error!Module {
        if (image.len == 0) {
            std.log.err("empty GPU image: this binary was built without kernels (-Dnvcc or -Dfatbins)", .{});
            return error.Invalid;
        }
        if (@intFromPtr(image.ptr) % 8 != 0) return error.Invalid;
        var m: abi.Module = null;
        try d.check(d.api.cuModuleLoadData(&m, image.ptr), "cuModuleLoadData");
        return .{ .d = d, .handle = m };
    }

    /// PTX text JIT-compiled by the driver; its error log is written into `log` on failure.
    pub fn loadPtx(d: *const Driver, ptx: [:0]const u8, log: []u8) Error!Module {
        @memset(log, 0);
        var opts = [_]c_int{ abi.jit_error_log_buffer, abi.jit_error_log_buffer_size };
        var vals = [_]?*anyopaque{ log.ptr, @ptrFromInt(log.len) };
        var m: abi.Module = null;
        try d.check(d.api.cuModuleLoadDataEx(&m, ptx.ptr, opts.len, &opts, &vals), "cuModuleLoadDataEx");
        return .{ .d = d, .handle = m };
    }

    pub fn unload(self: *Module) void {
        _ = self.d.api.cuModuleUnload(self.handle);
        self.* = undefined;
    }

    /// A kernel by its exact (mangled or extern "C") symbol; the function lives as long as the module.
    pub fn function(self: Module, name: [:0]const u8) Error!Function {
        var f: abi.Function = null;
        self.d.check(self.d.api.cuModuleGetFunction(&f, self.handle, name.ptr), "cuModuleGetFunction") catch |e| {
            std.log.err("kernel symbol not in module: {s}", .{name});
            return e;
        };
        return .{ .d = self.d, .handle = f };
    }

    pub const Global = struct { ptr: abi.DevicePtr, len: usize };

    pub fn global(self: Module, name: [:0]const u8) Error!Global {
        var g: Global = .{ .ptr = 0, .len = 0 };
        try self.d.check(self.d.api.cuModuleGetGlobal_v2(&g.ptr, &g.len, self.handle, name.ptr), "cuModuleGetGlobal");
        return g;
    }

    /// Whether the loaded image defines device global `name`; any driver failure answers false, never an error.
    pub fn holds(self: Module, name: [:0]const u8) bool {
        var g: Global = .{ .ptr = 0, .len = 0 };
        return self.d.api.cuModuleGetGlobal_v2(&g.ptr, &g.len, self.handle, name.ptr) == abi.success;
    }
};

pub const Function = struct {
    d: *const Driver,
    handle: abi.Function,

    pub fn attribute(self: Function, a: abi.FunctionAttribute) Error!c_int {
        var v: c_int = 0;
        try self.d.check(self.d.api.cuFuncGetAttribute(&v, a, self.handle), "cuFuncGetAttribute");
        return v;
    }

    pub fn setAttribute(self: Function, a: abi.FunctionAttribute, value: c_int) Error!void {
        try self.d.check(self.d.api.cuFuncSetAttribute(self.handle, a, value), "cuFuncSetAttribute");
    }

    /// Resident blocks an SM holds of this kernel at `threads` a block and `shared` dynamic bytes.
    pub fn occupancy(self: Function, threads: u32, shared: usize) Error!u32 {
        var n: c_int = 0;
        try self.d.check(self.d.api.cuOccupancyMaxActiveBlocksPerMultiprocessor(&n, self.handle, @intCast(threads), shared), "cuOccupancyMaxActiveBlocksPerMultiprocessor");
        return @intCast(@max(n, 0));
    }

    /// Dynamic shared memory above 48 KiB must be opted into per kernel before launch.
    pub fn allowDynamicShared(self: Function, bytes: u32) Error!void {
        try self.setAttribute(.max_dynamic_shared_size_bytes, @intCast(bytes));
    }
};

test "holds answers whether the image defines a global, and false on any driver failure" {
    const Fake = struct {
        fn getGlobal(ptr: *abi.DevicePtr, len: *usize, _: abi.Module, name: [*:0]const u8) callconv(.c) abi.Result {
            const n = std.mem.span(name);
            if (std.mem.eql(u8, n, "marker")) {
                ptr.* = 0x1000;
                len.* = 4;
                return abi.success;
            }
            // CUDA_ERROR_INVALID_CONTEXT: a failure that is not a missing name
            return if (std.mem.eql(u8, n, "lost")) 201 else abi.error_not_found;
        }
    };
    var d: Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuModuleGetGlobal_v2 = Fake.getGlobal;
    const m: Module = .{ .d = &d, .handle = null };
    try std.testing.expect(m.holds("marker"));
    try std.testing.expect(!m.holds("absent"));
    try std.testing.expect(!m.holds("lost"));
}
