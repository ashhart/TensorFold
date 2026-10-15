//! Loaded GPU code: SPIR-V modules the driver builds for this device, and their kernels by name.

const std = @import("std");
const ze = @import("ze");
const Context = @import("context.zig").Context;
const Error = @import("context.zig").Error;

pub const Module = struct {
    ctx: *const Context,
    handle: ze.H,

    /// Builds a SPIR-V image; a refused build logs the driver's text.
    pub fn load(ctx: *const Context, spirv: []const u8) Error!Module {
        if (spirv.len == 0) {
            std.log.err("empty GPU image: this binary was built without kernels (-Dxpu)", .{});
            return error.Invalid;
        }
        const d = ctx.d;
        var m: ze.H = null;
        var log: ze.H = null;
        const desc: ze.ModuleDesc = .{ .format = 0, .inputSize = spirv.len, .pInputModule = spirv.ptr }; // SPIR-V
        const res = d.api.zeModuleCreate(ctx.gpu.context, ctx.gpu.device, &desc, &m, &log);
        if (res != 0) {
            var sz: usize = 0;
            _ = d.x.zeModuleBuildLogGetString(log, &sz, null);
            const buf = std.heap.page_allocator.alloc(u8, sz) catch return error.OutOfMemory;
            defer std.heap.page_allocator.free(buf);
            _ = d.x.zeModuleBuildLogGetString(log, &sz, buf.ptr);
            std.log.err("module build: {s}", .{buf});
            _ = d.x.zeModuleBuildLogDestroy(log);
            return error.BuildFailed;
        }
        if (log != null) _ = d.x.zeModuleBuildLogDestroy(log);
        return .{ .ctx = ctx, .handle = m };
    }

    pub fn unload(self: *Module) void {
        _ = self.ctx.d.x.zeModuleDestroy(self.handle);
        self.* = undefined;
    }

    /// A kernel by its exact name with its work-group size; it lives until `deinit` and before the module's unload.
    pub fn kernel(self: Module, name: [:0]const u8, local: [3]u32) Error!Kernel {
        const d = self.ctx.d;
        var k: ze.H = null;
        d.check(d.api.zeKernelCreate(self.handle, &.{ .pKernelName = name.ptr }, &k), "zeKernelCreate") catch |e| {
            std.log.err("kernel not in module: {s}", .{name});
            return e;
        };
        try d.check(d.api.zeKernelSetGroupSize(k, local[0], local[1], local[2]), "zeKernelSetGroupSize");
        return .{ .ctx = self.ctx, .handle = k };
    }
};

pub const Kernel = struct {
    ctx: *const Context,
    handle: ze.H,

    pub fn deinit(self: *Kernel) void {
        _ = self.ctx.d.x.zeKernelDestroy(self.handle);
        self.* = undefined;
    }

    /// Argument `index` as raw bytes (a pointer, u32 or f32 as the OpenCL C parameter types it).
    pub fn setBytes(self: Kernel, index: u32, bytes: []const u8) Error!void {
        try self.ctx.d.check(self.ctx.d.api.zeKernelSetArgumentValue(self.handle, index, bytes.len, bytes.ptr), "zeKernelSetArgumentValue");
    }

    pub fn setPtr(self: Kernel, index: u32, p: ?*anyopaque) Error!void {
        try self.setBytes(index, std.mem.asBytes(&p));
    }

    pub fn setU32(self: Kernel, index: u32, v: u32) Error!void {
        try self.setBytes(index, std.mem.asBytes(&v));
    }

    pub fn setF32(self: Kernel, index: u32, v: f32) Error!void {
        try self.setBytes(index, std.mem.asBytes(&v));
    }
};
