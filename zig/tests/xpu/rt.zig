//! The op tests' view of the runtime: one heap-held device, stream and module/kernel helpers, picked by the runner.

const std = @import("std");
const xpu = @import("xpu");

pub const Error = xpu.Error;

/// The device ordinal the runner picked (null: the first Arc card).
pub var ordinal: ?u32 = null;

pub const Runtime = struct {
    drv: xpu.Driver,
    ctx: xpu.Context,
    stream: xpu.Stream,

    pub fn init() Error!*Runtime {
        const self = std.heap.page_allocator.create(Runtime) catch return error.OutOfMemory;
        errdefer std.heap.page_allocator.destroy(self);
        self.drv = try xpu.Driver.open();
        errdefer self.drv.close();
        self.ctx = try xpu.Context.init(&self.drv, ordinal);
        errdefer self.ctx.deinit();
        self.stream = try xpu.Stream.init(&self.ctx);
        return self;
    }

    pub fn deinit(self: *Runtime) void {
        self.stream.deinit();
        self.ctx.deinit();
        self.drv.close();
        std.heap.page_allocator.destroy(self);
    }

    pub fn alloc(self: *Runtime, bytes: usize) Error!Buffer {
        return .{ .rt = self, .buf = try xpu.DeviceBuffer.alloc(&self.ctx, bytes) };
    }

    pub fn upload(self: *Runtime, dst: Buffer, src: []const u8) Error!void {
        try dst.buf.uploadAsync(self.stream, 0, src);
    }

    pub fn download(self: *Runtime, dst: []u8, src: Buffer) Error!void {
        try src.buf.downloadAsync(self.stream, 0, dst);
    }

    pub fn sync(self: *Runtime) Error!void {
        try self.stream.synchronize();
    }

    pub fn module(self: *Runtime, spv: []const u8) Error!Module {
        return .{ .rt = self, .m = try xpu.Module.load(&self.ctx, spv) };
    }
};

pub const Buffer = struct {
    rt: *Runtime,
    buf: xpu.DeviceBuffer,

    pub fn free(self: *Buffer) void {
        self.rt.stream.synchronize() catch {};
        self.buf.free();
    }

    /// A non-owning view `off` bytes in (not to be freed).
    pub fn at(self: Buffer, off: usize) Buffer {
        const p = self.buf.at(off) catch unreachable;
        return .{ .rt = self.rt, .buf = .{ .ctx = self.buf.ctx, .ptr = p, .len = self.buf.len - off } };
    }
};

pub const Module = struct {
    rt: *Runtime,
    m: xpu.Module,

    pub fn deinit(self: *Module) void {
        self.m.unload();
    }

    pub fn kernel(self: *Module, name: [*:0]const u8, local: [3]u32) Error!Kernel {
        return .{ .rt = self.rt, .k = try self.m.kernel(std.mem.span(name), local) };
    }
};

pub const Kernel = struct {
    rt: *Runtime,
    k: xpu.Kernel,

    pub fn deinit(self: *Kernel) void {
        self.k.deinit();
    }

    pub fn setBuffer(self: *Kernel, index: u32, b: Buffer) Error!void {
        try self.k.setPtr(index, b.buf.ptr);
    }

    pub fn setU32(self: *Kernel, index: u32, v: u32) Error!void {
        try self.k.setU32(index, v);
    }

    pub fn setF32(self: *Kernel, index: u32, v: f32) Error!void {
        try self.k.setF32(index, v);
    }

    pub fn launch(self: *Kernel, groups: [3]u32) Error!void {
        try xpu.launch.launch(self.k, self.rt.stream, groups);
    }
};
