//! Real HC adapters under recording device boundaries; source arithmetic remains a separate contract gate.
const std = @import("std");
const cuda = @import("cuda");
const family = @import("qwen4").family;
const h = family.hc_layout;
const C = family.cuda_hc.Kernels(cuda);
const T = std.testing;
const Trace = struct {
    var live: usize = 0;
    var calls: usize = 0;
    var fail_at: ?usize = null;
    var fail_drain = false;
    var dense_bound = false;
    fn load(out: *cuda.abi.Module, _: ?*const anyopaque) callconv(.c) cuda.abi.Result {
        live += 1;
        out.* = @ptrFromInt(0x100);
        return 0;
    }
    fn unload(_: cuda.abi.Module) callconv(.c) cuda.abi.Result {
        live -= 1;
        return 0;
    }
    fn function(out: *cuda.abi.Function, _: cuda.abi.Module, name: [*:0]const u8) callconv(.c) cuda.abi.Result {
        for (C.names, 0..) |n, i| if (std.mem.eql(u8, n, std.mem.span(name))) {
            out.* = @ptrFromInt(i + 1);
            return 0;
        };
        return cuda.abi.error_not_found;
    }
    fn attribute(out: *c_int, _: cuda.abi.FunctionAttribute, f: cuda.abi.Function) callconv(.c) cuda.abi.Result {
        out.* = if (fail_at == @intFromPtr(f) - 1) 128 else 256;
        return 0;
    }
    fn drain(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return if (fail_drain) cuda.abi.error_not_ready else 0;
    }
    fn arg(comptime V: type, args: ?[*]?*anyopaque, i: usize) V {
        return @as(*const V, @ptrCast(@alignCast(args.?[i].?))).*;
    }
    fn launch(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        const op: h.Op = @fromBackingInt(@as(u2, @intCast(@intFromPtr(f) - 1)));
        std.debug.assert(gx == 3 and bx == 256 and by == 1 and bz == 1 and shared == 0);
        const expected_y: usize = switch (op) {
            .sum => 20,
            .norm => 4,
            .mix => 20,
            .act => 56,
        };
        std.debug.assert(gy == expected_y and gz == @as(usize, if (op == .norm) 20 else 1));
        switch (op) {
            .sum => std.debug.assert(arg(c_int, args, 2) == 5120 and arg(c_int, args, 3) == 4),
            .norm => std.debug.assert(arg(f32, args, 4) == @as(f32, 1e-6) and arg(c_int, args, 5) == 5120 and arg(c_int, args, 6) == 4),
            .mix => std.debug.assert(arg(c_int, args, 4) == 5120 and arg(c_int, args, 5) == 4),
            .act => {
                std.debug.assert(arg(c_int, args, 3) == 14336);
                if (dense_bound) std.debug.assert(arg(u32, args, 4) == 7);
            },
        }
        calls += 1;
        return 0;
    }
};
fn driver() cuda.Driver {
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuModuleLoadData = Trace.load;
    d.api.cuModuleUnload = Trace.unload;
    d.api.cuModuleGetFunction = Trace.function;
    d.api.cuFuncGetAttribute = Trace.attribute;
    d.api.cuStreamSynchronize = Trace.drain;
    d.api.cuLaunchKernel = Trace.launch;
    return d;
}
test "Qwen4 HC ops: real CUDA ABI for sums norm mix and dense activation" {
    var d = driver();
    const s = cuda.Stream.borrowed(&d, null);
    const image: [8]u8 align(8) = @splat(1);
    const g: h.Geometry = .{ .hidden = 5120, .streams = 4 };
    var k = try C.load(&d, &image, g);
    const p: h.Call = .{ .rows = 3, .width = 14336 };
    for (std.enums.values(h.Op)) |op| {
        const sizes = try h.sizes(g, p, op);
        const count: usize = if (op == .sum) 2 else if (op == .act) 3 else 4;
        var buffers: [4]cuda.DeviceBuffer = undefined;
        var views: [4]C.View = undefined;
        for (buffers[0..count], views[0..count], sizes[0..count], 0..) |*b, *v, n, i| {
            b.* = .{ .d = &d, .ptr = (i + 1) * 0x1000000, .len = n };
            v.* = .{ .buffer = b, .len = n };
        }
        try k.encode(s, op, p, views[0..count]);
        if (op == .act) {
            var other = driver();
            try T.expectError(error.Qwen4HcContext, k.bindDense(.{ .d = &other, .handle = k.f[3].handle }));
            Trace.fail_at = 3;
            try T.expectError(error.Qwen4HcThreads, k.bindDense(k.f[3]));
            try T.expect(!k.dense_bound);
            Trace.fail_at = null;
            try k.bindDense(k.f[3]);
            k.dense_repeat = 7;
            Trace.dense_bound = true;
            try k.encode(s, op, p, views[0..count]);
            Trace.dense_bound = false;
        }
        const before = Trace.calls;
        var wrong = p;
        wrong.rows = 129;
        try T.expectError(error.Qwen4HcShape, k.encode(s, op, wrong, views[0..count]));
        const output: usize = if (op == .sum) 1 else if (op == .norm) 3 else 2;
        const saved = views[output];
        views[output] = views[0];
        const err = if (saved.len > views[0].len) error.Qwen4HcBuffer else error.Qwen4HcAlias;
        try T.expectError(err, k.encode(s, op, p, views[0..count]));
        views[output] = saved;
        try T.expectEqual(before, Trace.calls);
    }
    Trace.fail_drain = true;
    try T.expectError(error.NotReady, k.close(s));
    try T.expectEqual(@as(usize, 1), Trace.live);
    Trace.fail_drain = false;
    try k.close(s);
    for (0..4) |i| {
        Trace.fail_at = i;
        try T.expectError(error.Qwen4HcThreads, C.load(&d, &image, g));
        try T.expectEqual(@as(usize, 0), Trace.live);
    }
    Trace.fail_at = null;
}
const Recording = struct {
    pub const Size = struct {
        width: usize,
        height: usize,
        depth: usize,
        pub fn of(x: usize, y: usize, z: usize) @This() {
            return .{ .width = x, .height = y, .depth = z };
        }
    };
    pub const Device = struct { id: usize = 1 };
    pub const Buffer = struct {
        address: u64,
        size: usize,
        owner: usize = 1,
        pub fn deviceId(b: Buffer) usize {
            return b.owner;
        }
        pub fn length(b: Buffer) usize {
            return b.size;
        }
        pub fn gpuAddress(b: Buffer) u64 {
            return b.address;
        }
    };
    pub const CompileOptions = struct {
        pub fn mlx() @This() {
            return .{};
        }
    };
    pub var live: usize = 0;
    pub var calls: usize = 0;
    pub var fail_at: ?usize = null;
    pub var selected: usize = 0;
    pub var grid: Size = undefined;
    pub var grouped = false;
    pub var bindings: [5]?u64 = @splat(null);
    pub const Library = struct {
        pub fn fromSource(_: Device, text: []const u8, _: CompileOptions) !Library {
            std.debug.assert(std.mem.indexOf(u8, text, "D = 5120") != null);
            return .{};
        }
        pub fn deinit(_: Library) void {}
    };
    pub const Pipeline = struct {
        index: usize,
        pub fn init(_: Device, _: Library, name: []const u8, _: bool) !Pipeline {
            const names = family.metal_hc.Kernels(Recording).names;
            for (names, 0..) |n, i| if (std.mem.eql(u8, n, name)) {
                if (fail_at == i) return error.PlantedPipeline;
                live += 1;
                return .{ .index = i };
            };
            return error.OwnedSymbol;
        }
        pub fn maxThreads(_: Pipeline) usize {
            return 256;
        }
        pub fn simdWidth(_: Pipeline) usize {
            return 32;
        }
        pub fn deinit(_: Pipeline) void {
            live -= 1;
        }
    };
    pub const ComputeEncoder = struct {
        owner: usize = 1,
        pub fn deviceId(e: ComputeEncoder) usize {
            return e.owner;
        }
        pub fn setPipeline(_: ComputeEncoder, p: Pipeline) void {
            selected = p.index;
            calls += 1;
            bindings = @splat(null);
        }
        pub fn setBuffer(_: ComputeEncoder, b: Buffer, offset: usize, index: usize) void {
            bindings[index] = b.address + offset;
            calls += 1;
        }
        pub fn setBytes(_: ComputeEncoder, bytes: []const u8, index: usize) void {
            std.debug.assert(index == 3 and std.mem.bytesToValue(f32, bytes[0..4]) == @as(f32, 1e-6));
            calls += 1;
        }
        pub fn dispatchGroups(_: ComputeEncoder, g: Size, t: Size) void {
            std.debug.assert(t.width == 256);
            grid = g;
            grouped = true;
            calls += 1;
        }
        pub fn dispatchThreads(_: ComputeEncoder, g: Size, t: Size) void {
            std.debug.assert(t.width == 256);
            grid = g;
            grouped = false;
            calls += 1;
        }
        pub fn barrier(_: ComputeEncoder) void {
            calls += 1;
        }
    };
};
test "Qwen4 HC ops: real Metal bindings and exact grid dimensions with prefix cleanup" {
    const M = family.metal_hc.Kernels(Recording);
    const g: h.Geometry = .{ .hidden = 5120, .streams = 4 };
    const p: h.Call = .{ .rows = 3, .width = 14336 };
    var k = try M.load(T.allocator, .{}, g);
    for (std.enums.values(h.Op)) |op| {
        const sizes = try h.sizes(g, p, op);
        const count: usize = if (op == .sum) 2 else if (op == .norm) 4 else 3;
        var views: [4]M.View = undefined;
        for (views[0..count], sizes[0..count], 0..) |*view, n, i| view.* = .{ .buffer = .{ .address = (i + 1) * 0x1000000, .size = n }, .len = n };
        try k.encode(.{}, op, p, views[0..count]);
        try T.expectEqual(op == .sum, Recording.grouped);
        const grid: Recording.Size = switch (op) {
            .sum => .of(20, 3, 1),
            .norm => .of(20480, 3, 1),
            .mix => .of(5120, 3, 1),
            .act => .of(43008, 1, 1),
        };
        try T.expectEqualDeep(grid, Recording.grid);
        if (op == .sum) try T.expectEqual(Recording.bindings[0], Recording.bindings[1]);
        if (op == .norm) try T.expectEqual(views[3].buffer.address, Recording.bindings[4].?);
        const before = Recording.calls;
        try T.expectError(error.Qwen4HcContext, k.encode(.{ .owner = 2 }, op, p, views[0..count]));
        try T.expectEqual(before, Recording.calls);
    }
    k.deinit();
    try T.expectEqual(@as(usize, 0), Recording.live);
    for (0..4) |i| {
        Recording.fail_at = i;
        try T.expectError(error.PlantedPipeline, M.load(T.allocator, .{}, g));
        try T.expectEqual(@as(usize, 0), Recording.live);
    }
    Recording.fail_at = null;
}
