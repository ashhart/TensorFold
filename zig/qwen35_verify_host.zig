//! The real 4-bit verify projection dispatch through a recording CUDA driver.
const std = @import("std");
const cuda = @import("cuda");
const q = @import("qwen35");
const t = std.testing;
pub const Trace = struct {
    const Call = struct { id: usize, x: u64, xs: u64, out: u64, rows: c_int, sk: c_int, row_tiles: c_int, grid: u32, shared: u32 };
    pub var calls: [128]Call = undefined;
    pub var count: usize = 0;
    pub var prompts: usize = 0;
    fn record(f: cuda.abi.Function, gx: u32, shared: u32, args: ?[*]?*anyopaque) cuda.abi.Result {
        const id = @intFromPtr(f);
        if (id == 4) {
            prompts += 1;
            return 0;
        }
        if (id == 1) return 0; // the input-sum producer
        const a = args.?;
        const x = @as(*const u64, @ptrCast(@alignCast(a[0].?))).*;
        const xs = @as(*const u64, @ptrCast(@alignCast(a[1].?))).*;
        const rows = @as(*const c_int, @ptrCast(@alignCast(a[3].?))).*;
        var out: u64 = 0;
        var sk: c_int = 0;
        var tiles: c_int = 1;
        if (id == 8) {
            const segs = @as(*const cuda.qlinear.Segs, @ptrCast(@alignCast(a[2].?))).*;
            for (segs.s[0..@intCast(segs.count)]) |seg| {
                calls[count] = .{ .id = id, .x = x, .xs = xs, .out = seg.p.out, .rows = rows, .sk = seg.p.sk, .row_tiles = 1, .grid = gx, .shared = shared };
                count += 1;
            }
            return 0;
        }
        if (id == 2) {
            const p = @as(*const cuda.qlinear.GemvPart, @ptrCast(@alignCast(a[2].?))).*;
            out = p.out;
            sk = p.sk;
        } else if (id == 3 or id == 5 or id == 6 or id == 7) {
            const p = @as(*const cuda.qlinear.Parts, @ptrCast(@alignCast(a[2].?))).*;
            out = p.p[0].out;
            sk = p.p[0].sk;
            tiles = @as(*const c_int, @ptrCast(@alignCast(a[6].?))).*;
        } else return 0;
        calls[count] = .{ .id = id, .x = x, .xs = xs, .out = out, .rows = rows, .sk = sk, .row_tiles = tiles, .grid = gx, .shared = shared };
        count += 1;
        return 0;
    }
    fn launch(f: cuda.abi.Function, gx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, shared: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        return record(f, gx, shared, args);
    }
    fn launchEx(c: *const cuda.abi.LaunchConfig, f: cuda.abi.Function, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        return record(f, c.grid_x, c.shared_bytes, args);
    }
};
fn function(d: *const cuda.Driver, id: usize) cuda.Function {
    return .{ .d = d, .handle = @ptrFromInt(id) };
}
pub fn engine(d: *cuda.Driver, gb10: bool) q.engine.Engine {
    return device(d, gb10, false);
}
fn device(d: *cuda.Driver, gb10: bool, wide: bool) q.engine.Engine {
    d.api.cuLaunchKernel = Trace.launch;
    d.api.cuLaunchKernelEx = Trace.launchEx;
    var e: q.engine.Engine = undefined;
    e.int8 = .{};
    e.kit = null;
    e.d = d;
    e.stream = .{ .d = d, .handle = null };
    e.fns.sums = function(d, 1);
    e.affine.gemv_fn = function(d, 2);
    e.affine.group_fn = function(d, 3);
    e.affine.prefill_fn = function(d, 4);
    e.affine.prefill22_fn = function(d, 4);
    e.affine.half_prompt = false;
    if (@hasField(cuda.qlinear.Affine4Kernels, "verify_fn")) e.affine.verify_fn = function(d, 5);
    if (@hasField(cuda.qlinear.Affine4Kernels, "verify64_fn")) e.affine.verify64_fn = function(d, 6);
    if (@hasField(cuda.qlinear.Affine4Kernels, "verify128_fn")) e.affine.verify128_fn = function(d, 7);
    e.affine.gemv_blocks = 1;
    e.affine.split = null;
    e.affine.pdl = gb10;
    e.affine.wide_verify = wide;
    return e;
}
fn project(e: *q.engine.Engine, rows: usize, n: usize) !void {
    Trace.count = 0;
    const p: q.engine.Proj = .{ .a4 = .{ .w = 10, .s = 20, .b = 30, .n = n, .k = 8192, .npad = n } };
    try e.projSums(p, 0x10000, 0x20000, 0x30000, rows, false, 0);
}

test "verify matmuls: up to 16 rows already share one weight pass" {
    var d: cuda.Driver = undefined;
    var e = engine(&d, true);
    for ([_]usize{ 1, 2, 8, 15, 16 }) |rows| {
        try project(&e, rows, 4096);
        try t.expectEqual(@as(usize, 1), Trace.count);
        try t.expectEqual(rows, @as(usize, @intCast(Trace.calls[0].rows)));
    }
}

test "verify matmuls: 17 through 32 GB10 rows share one weight pass and keep the original K slices" {
    var d: cuda.Driver = undefined;
    var e = engine(&d, true);
    for (17..33) |rows| {
        try project(&e, rows, 4096);
        try t.expectEqual(@as(usize, 1), Trace.count);
        const c = Trace.calls[0];
        try t.expectEqual(@as(usize, 5), c.id);
        try t.expectEqual(rows, @as(usize, @intCast(c.rows)));
        try t.expectEqual(@as(c_int, @intCast(cuda.qlinear.splitK(4096, 8192))), c.sk);
        try t.expectEqual(@as(c_int, 1), c.row_tiles);
        try t.expectEqual(@as(u32, 32640), c.shared);
        try t.expectEqual(@as(u32, @intCast(64 * c.sk)), c.grid);
    }
}

test "verify matmuls: 33 through 128 GB10 rows share one weight pass in four or eight row tiles" {
    var d: cuda.Driver = undefined;
    var e = engine(&d, true);
    for (33..129) |rows| {
        try project(&e, rows, 4096);
        try t.expectEqual(@as(usize, 1), Trace.count);
        const c = Trace.calls[0];
        try t.expectEqual(@as(usize, if (rows <= 64) 6 else 7), c.id);
        try t.expectEqual(rows, @as(usize, @intCast(c.rows)));
        try t.expectEqual(@as(u64, 0x10000), c.x);
        try t.expectEqual(@as(u64, 0x30000), c.out);
        try t.expectEqual(@as(c_int, @intCast(cuda.qlinear.splitK(4096, 8192))), c.sk);
        try t.expectEqual(@as(u32, if (rows <= 64) 43008 else 57600), c.shared);
        try t.expectEqual(@as(u32, @intCast(64 * c.sk)), c.grid);
    }
}

test "verify matmuls: graph row buckets and segment counts do not multiply projection launches" {
    var d: cuda.Driver = undefined;
    var e = engine(&d, true);
    for ([_]usize{ 1, 2, 17 }) |segments| {
        const shape = q.engine.Engine.Shape.init(17, segments, true, false, 1);
        try project(&e, shape.total, 4096);
        try t.expectEqual(@as(usize, 1), Trace.count);
        try t.expectEqual(@as(c_int, 32), Trace.calls[0].rows);
    }
    // Large output matrices keep splitK=1 and still read one panel for all 32 rows.
    try project(&e, 32, 17408);
    try t.expectEqual(@as(usize, 1), Trace.count);
    try t.expectEqual(@as(c_int, 1), Trace.calls[0].sk);
}

test "verify matmuls: sm_120 takes the 32-row tile without programmatic dependent launch" {
    var d: cuda.Driver = undefined;
    var e = device(&d, false, true);
    for (17..33) |rows| {
        try project(&e, rows, 4096);
        try t.expectEqual(@as(usize, 1), Trace.count);
        try t.expectEqual(@as(usize, 5), Trace.calls[0].id);
        try t.expectEqual(rows, @as(usize, @intCast(Trace.calls[0].rows)));
    }
    try project(&e, 16, 4096);
    try t.expectEqual(@as(usize, 2), Trace.calls[0].id);
}

test "verify matmuls: other devices retain the original sixteen-row dispatch" {
    var d: cuda.Driver = undefined;
    var e = engine(&d, false);
    try project(&e, 32, 4096);
    try t.expectEqual(@as(usize, 2), Trace.count);
    try t.expectEqual(@as(usize, 2), Trace.calls[0].id);
    try t.expectEqual(@as(u64, 0x30000 + 16 * 4096 * 2), Trace.calls[1].out);
}

const Resolve = struct {
    var requested = false;
    var shared: [3]c_int = @splat(0); // verify32, verify64, verify128
    fn get(out: *cuda.abi.Function, _: cuda.abi.Module, name: [*:0]const u8) callconv(.c) cuda.abi.Result {
        const s = cuda.qlinear.symbols;
        const n = std.mem.span(name);
        const id: usize = if (std.mem.eql(u8, n, s.verify32)) 5 else if (std.mem.eql(u8, n, s.verify64)) 6 else if (std.mem.eql(u8, n, s.verify128)) 7 else 2;
        out.* = @ptrFromInt(id);
        requested = requested or @intFromPtr(out.*) == 5;
        return 0;
    }
    fn attribute(f: cuda.abi.Function, what: cuda.abi.FunctionAttribute, value: c_int) callconv(.c) cuda.abi.Result {
        const id = @intFromPtr(f);
        if (id >= 5 and id <= 7 and what == .max_dynamic_shared_size_bytes) shared[id - 5] = value;
        return 0;
    }
    fn occupancy(out: *c_int, _: cuda.abi.Function, _: c_int, _: usize) callconv(.c) cuda.abi.Result {
        out.* = 2;
        return 0;
    }
};

test "verify matmuls: real module resolution sets the new function's shared-memory allowance" {
    var d: cuda.Driver = undefined;
    d.api.cuModuleGetFunction = Resolve.get;
    d.api.cuFuncSetAttribute = Resolve.attribute;
    d.api.cuOccupancyMaxActiveBlocksPerMultiprocessor = Resolve.occupancy;
    const m: cuda.Module = .{ .d = &d, .handle = null };
    Resolve.requested = false;
    Resolve.shared = @splat(0);
    var a = try cuda.qlinear.Affine4Kernels.resolve(&d, m, m, m, m, 48, true);
    defer a.deinit();
    try t.expect(Resolve.requested);
    try t.expectEqual([3]c_int{ 32640, 43008, 57600 }, Resolve.shared);
    try t.expectEqual(@as(usize, 96), a.gemv_blocks);
    try t.expectEqual(@as(usize, 128), a.verifyRows());
}
