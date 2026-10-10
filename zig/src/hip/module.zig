//! Build-time HIP code objects, not runtime compilation or CUDA image aliases.
const abi = @import("abi.zig");
const runtime = @import("runtime.zig");
const code_object = @import("code_object.zig");

pub const Module = struct {
    r: *const runtime.Runtime,
    handle: abi.Module,

    pub fn loadForArchitecture(r: *const runtime.Runtime, images: []const code_object.Image, arch: []const u8) (runtime.Error || code_object.Error)!Module {
        const image = try code_object.select(images, arch);
        return load(r, image);
    }

    pub fn load(r: *const runtime.Runtime, image: []const u8) runtime.Error!Module {
        if (image.len == 0 or @intFromPtr(image.ptr) % 8 != 0) return error.Invalid;
        var handle: abi.Module = null;
        try runtime.check(r.api.hipModuleLoadData(&handle, image.ptr));
        if (handle == null) return error.Invalid;
        return .{ .r = r, .handle = handle };
    }

    pub fn unload(self: *Module) void {
        _ = self.r.api.hipModuleUnload(self.handle);
        self.* = undefined;
    }

    pub fn global(self: Module, name: [:0]const u8) runtime.Error!Global {
        var ptr: abi.DevicePtr = null;
        var len: usize = 0;
        try runtime.check(self.r.api.hipModuleGetGlobal(&ptr, &len, self.handle, name.ptr));
        if (ptr == null or len == 0) return error.Invalid;
        return .{ .address = @intFromPtr(ptr), .len = len };
    }

    pub fn function(self: Module, name: [:0]const u8) runtime.Error!Function {
        var handle: abi.Function = null;
        try runtime.check(self.r.api.hipModuleGetFunction(&handle, self.handle, name.ptr));
        if (handle == null) return error.Invalid;
        return .{ .r = self.r, .handle = handle };
    }
};

pub const Function = struct {
    r: *const runtime.Runtime,
    handle: abi.Function,

    pub fn attribute(self: Function, a: abi.FunctionAttribute) runtime.Error!c_int {
        var v: c_int = 0;
        try runtime.check(self.r.api.hipFuncGetAttribute(&v, a, self.handle));
        return v;
    }

    /// Blocks of `threads` threads and `shared` dynamic bytes one compute unit holds at once.
    pub fn occupancy(self: Function, threads: u32, shared: usize) runtime.Error!u32 {
        var n: c_int = 0;
        try runtime.check(self.r.api.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&n, self.handle, @intCast(threads), shared));
        return @intCast(@max(n, 0));
    }
};

/// A module-scope `__device__` variable: its device address and size.
pub const Global = struct { address: u64, len: usize };

test "unknown architecture is refused before invoking HIP" {
    const std = @import("std");
    const r: runtime.Runtime = undefined;
    const bytes align(8) = [_]u8{1};
    const images = [_]code_object.Image{.{ .arch = "gfx1151", .bytes = &bytes }};
    try std.testing.expectError(error.UnsupportedArchitecture, Module.loadForArchitecture(&r, &images, "gfx1201"));
}

test "selected code object reaches HIP and preserves driver failure" {
    const std = @import("std");
    const Mock = struct {
        fn load(_: *abi.Module, image: [*]const u8) callconv(.c) abi.Result {
            return if (image[0] == 2) 1 else 0;
        }
    };
    var r: runtime.Runtime = undefined;
    r.api.hipModuleLoadData = Mock.load;
    const first align(8) = [_]u8{1};
    const selected align(8) = [_]u8{2};
    const images = [_]code_object.Image{ .{ .arch = "gfx1150", .bytes = &first }, .{ .arch = "gfx1151", .bytes = &selected } };
    try std.testing.expectError(error.HipFailed, Module.loadForArchitecture(&r, &images, "gfx1151"));
}
