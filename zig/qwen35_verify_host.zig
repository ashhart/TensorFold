//! The real 4-bit verify projection dispatch through a recording CUDA driver.
const std = @import("std");
const cuda = @import("cuda");
const q = @import("qwen35");
const t = std.testing;
const Trace = struct {
    const Call = struct { id: usize, x: u64, xs: u64, out: u64, rows: c_int, sk: c_int, row_tiles: c_int, grid: u32, shared: u32 };
    var calls: [16]Call = undefined;
    var count: usize = 0;
    fn record(f: cuda.abi.Function, gx: u32, shared: u32, args: ?[*]?*anyopaque) cuda.abi.Result {
        const id = @intFromPtr(f);
        if (id == 1) return 0; // the input-sum producer
        const a = args.?;
        const x = @as(*const u64, @ptrCast(@alignCast(a[0].?))).*;
        const xs = @as(*const u64, @ptrCast(@alignCast(a[1].?))).*;
        const rows = @as(*const c_int, @ptrCast(@alignCast(a[3].?))).*;
        var out: u64 = 0;
        var sk: c_int = 0;
        var tiles: c_int = 1;
        if (id == 2) {
            const p = @as(*const cuda.qlinear.GemvPart, @ptrCast(@alignCast(a[2].?))).*;
            out = p.out;
            sk = p.sk;
        } else if (id == 3 or id == 5) {
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
fn engine(d: *cuda.Driver, gb10: bool) q.engine.Engine {
    d.api.cuLaunchKernel = Trace.launch;
    d.api.cuLaunchKernelEx = Trace.launchEx;
    var e: q.engine.Engine = undefined;
    e.d = d;
    e.stream = .{ .d = d, .handle = null };
    e.fns.sums = function(d, 1);
    e.affine.gemv_fn = function(d, 2);
    e.affine.group_fn = function(d, 3);
    e.affine.prefill_fn = function(d, 4);
    if (@hasField(cuda.qlinear.Affine4Kernels, "verify_fn")) e.affine.verify_fn = function(d, 5);
    e.affine.gemv_blocks = 1;
    e.affine.split = null;
    e.affine.pdl = gb10;
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

test "verify matmuls: wider windows advance inputs sums and outputs by 32 rows" {
    var d: cuda.Driver = undefined;
    var e = engine(&d, true);
    for ([_]usize{ 33, 47, 48, 63, 64 }) |rows| {
        try project(&e, rows, 4096);
        try t.expectEqual(@as(usize, 2), Trace.count);
        try t.expectEqual(@as(u64, 0x10000 + 32 * 8192 * 2), Trace.calls[1].x);
        try t.expectEqual(@as(u64, 0x20000 + 32 * (8192 / 64) * 4), Trace.calls[1].xs);
        try t.expectEqual(@as(u64, 0x30000 + 32 * 4096 * 2), Trace.calls[1].out);
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
    var shared: c_int = 0;
    fn get(out: *cuda.abi.Function, _: cuda.abi.Module, name: [*:0]const u8) callconv(.c) cuda.abi.Result {
        if (std.mem.eql(u8, std.mem.span(name), cuda.qlinear.symbols.verify32)) {
            requested = true;
            out.* = @ptrFromInt(5);
        } else out.* = @ptrFromInt(2);
        return 0;
    }
    fn attribute(f: cuda.abi.Function, what: cuda.abi.FunctionAttribute, value: c_int) callconv(.c) cuda.abi.Result {
        if (@intFromPtr(f) == 5 and what == .max_dynamic_shared_size_bytes) shared = value;
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
    Resolve.shared = 0;
    var a = try cuda.qlinear.Affine4Kernels.resolve(&d, m, m, m, m, 48, true);
    defer a.deinit();
    try t.expect(Resolve.requested);
    try t.expectEqual(@as(c_int, 32640), Resolve.shared);
    try t.expectEqual(@as(usize, 96), a.gemv_blocks);
    try t.expectEqual(@as(usize, 32), a.verifyRows());
}

test "verify matmuls: coalesced 128-row calls keep every 32-row slice at its own buffer offset" {
    var d: cuda.Driver = undefined;
    var e = engine(&d, true);
    for ([_]usize{ 65, 79, 96, 97, 127, 128 }) |rows| {
        try project(&e, rows, 4096);
        try t.expectEqual((rows + 31) / 32, Trace.count);
        for (Trace.calls[0..Trace.count], 0..) |call, index| {
            try t.expectEqual(@as(u64, 0x10000 + index * 32 * 8192 * 2), call.x);
            try t.expectEqual(@as(u64, 0x20000 + index * 32 * (8192 / 64) * 4), call.xs);
            try t.expectEqual(@as(u64, 0x30000 + index * 32 * 4096 * 2), call.out);
        }
    }
}
