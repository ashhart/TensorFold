//! Production ModelOpt wiring and lane launches with a recording-only CUDA boundary.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const kern = @import("cuda_kernels.zig");
const state = @import("cuda_state.zig");
const Engine = @import("cuda_engine.zig").Engine;
const Forward = @import("cuda_forward.zig").Forward;
const segments = @import("cuda_segments.zig");
const qmmf = cuda.qmmf;
const t = std.testing;
const Fake = struct {
    const Record = struct { fun: usize, stream: usize, grid: [3]u32, block: [3]u32, shared: u32, args: [13]u64, extended: bool, cluster: [3]u32 = .{ 0, 0, 0 } };
    var records: [8]Record = undefined;
    var count: usize = 0;
    var next: u64 = 0x100000;
    var major: c_int = 8;
    fn value(comptime T: type, p: ?*anyopaque) T {
        return @as(*align(1) const T, @ptrCast(p.?)).*;
    }
    fn record(f: cuda.abi.Function, gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32, shared: u32, s: cuda.abi.Stream, args: ?[*]?*anyopaque, extended: bool) c_int {
        var rec = Record{ .fun = @intFromPtr(f), .stream = @intFromPtr(s), .grid = .{ gx, gy, gz }, .block = .{ bx, by, bz }, .shared = shared, .args = undefined, .extended = extended };
        const sizes = [13]usize{ 8, 8, 8, 4, 8, 8, 4, 4, 4, 4, 4, 4, 4 };
        const n: usize = if (rec.fun >= 90) 5 else 13;
        for (0..n) |i| rec.args[i] = if ((if (rec.fun >= 90 and i == 4) @as(usize, 4) else sizes[i]) == 8) value(u64, args.?[i]) else value(u32, args.?[i]);
        records[count] = rec;
        count += 1;
        return 0;
    }
    fn launch(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, s: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        return record(f, gx, gy, gz, bx, by, bz, shared, s, args, false);
    }
    fn launchEx(c: *const cuda.abi.LaunchConfig, f: cuda.abi.Function, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        const result = record(f, c.grid_x, c.grid_y, c.grid_z, c.block_x, c.block_y, c.block_z, c.shared_bytes, c.stream, args, true);
        for (c.attrs.?[0..c.num_attrs]) |attr| if (attr.id == .cluster_dimension) {
            records[count - 1].cluster = .{ attr.value.cluster_dim.x, attr.value.cluster_dim.y, attr.value.cluster_dim.z };
        };
        return result;
    }
    fn alloc(out: *u64, n: usize) callconv(.c) c_int {
        out.* = next;
        next += std.mem.alignForward(usize, n, 256) + 4096;
        return 0;
    }
    fn free(_: u64) callconv(.c) c_int {
        return 0;
    }
    fn attribute(out: *c_int, _: cuda.abi.DeviceAttribute, _: cuda.abi.Device) callconv(.c) c_int {
        out.* = major;
        return 0;
    }
    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuLaunchKernel = launch;
        d.api.cuLaunchKernelEx = launchEx;
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuDeviceGetAttribute = attribute;
        return d;
    }
    fn lane(d: *const cuda.Driver, cluster: bool) qmmf.Lane {
        var l: qmmf.Lane = undefined;
        l.clusters = cluster;
        l.mod = .{ .d = d, .handle = @ptrFromInt(1) };
        for (&l.fns, 0..) |*set, mi| for (set, 0..) |*f, i| {
            f.* = .{ .d = d, .handle = @ptrFromInt(10 + mi * 16 + i) };
        };
        for (&l.f32_fns, 0..) |*set, mi| for (set, 0..) |*f, i| {
            f.* = .{ .d = d, .handle = @ptrFromInt(17 + mi * 16 + i) };
        };
        for (&l.reduce, 0..) |*f, i| f.* = .{ .d = d, .handle = @ptrFromInt(90 + i) };
        return l;
    }
    fn kernels(d: *const cuda.Driver, cluster: bool) kern.Kernels {
        var k: kern.Kernels = undefined;
        k.d = d;
        k.triton = null;
        k.ck = null;
        k.modelopt = .{ .lane = lane(d, cluster), .gemm = undefined, .experts = undefined, .pack = undefined };
        return k;
    }
};
fn config() !cfg.Config {
    var why: cfg.Why = .{};
    var c = try cfg.parse(t.allocator, cfg.lightning, &why);
    c.format = .modelopt;
    return c;
}
fn weight(mode: qmmf.Mode) qmmf.Weight {
    return .{ .mode = mode, .codes = 0x200, .scales = 0x300, .scale = 0.5, .n = 4608, .k = 2688, .npad = 4608 };
}
test "modelopt host: missing reduction storage is refused before any kernel launch" {
    const d = Fake.driver();
    const l = Fake.lane(&d, false);
    const s: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(7) };
    for ([_]qmmf.Mode{ .fp4, .fp8, .fp8g, .bf16 }) |mode| for ([_]usize{ 1, 15, 255 }) |rows| {
        Fake.count = 0;
        try t.expectError(error.MissingLanePart, l.matmul(s, 0x100, 2688, rows, weight(mode), 0x400, 0));
        try t.expectEqual(@as(usize, 0), Fake.count);
    };
    Fake.count = 0;
    try t.expectError(error.MissingLanePart, l.matmulOut(s, 0x100, 2688, 1, weight(.fp4), 0x400, .f32, 0));
    try t.expectEqual(@as(usize, 0), Fake.count);
}
const bf16_fns = 10 + 4 * 16; // Fake.lane's handles for bf16, the fifth of qmmf's modes
test "modelopt host: split reduction and fused or clustered controls keep released launch parameters" {
    const d = Fake.driver();
    var l = Fake.lane(&d, false);
    const s: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(7) };
    for ([_]usize{ 1, 15, 255 }) |rows| {
        Fake.count = 0;
        try l.matmul(s, 0x100, 3000, rows, weight(.bf16), 0x400, 0x500);
        try t.expectEqual(@as(usize, 2), Fake.count);
        const first = Fake.records[0];
        const reduce = Fake.records[1];
        try t.expectEqual(@as(usize, if (rows <= 16) bf16_fns else bf16_fns + 4), first.fun);
        try t.expectEqual([3]u32{ if (rows <= 16) 72 else 288, 1, 2 }, first.grid);
        try t.expectEqual([3]u32{ 128, 1, 1 }, first.block);
        try t.expectEqual(@as(u32, if (rows <= 16) 40960 else 65536), first.shared);
        try t.expectEqual(@as(u64, 0x500), first.args[5]);
        try t.expectEqual(@as(u64, 2), first.args[9]);
        try t.expectEqual(@as(u64, if (rows == 1) 2688 else 3000), first.args[11]);
        try t.expectEqualSlices(u64, &.{ 0x100, 0x200, 0x300, 0x3f000000, 0x400, 0x500, rows, 4608, 2688, 2, 4608, if (rows == 1) 2688 else 3000, if (rows <= 16) 1 else 4 }, &first.args);
        try t.expect(!first.extended);
        try t.expectEqual(@as(usize, 7), first.stream);
        try t.expectEqual(@as(usize, 90), reduce.fun);
        try t.expectEqual(@as(usize, 7), reduce.stream);
        try t.expectEqual([3]u32{ @intCast((rows * 4608 + 255) / 256), 1, 1 }, reduce.grid);
        try t.expectEqual([3]u32{ 256, 1, 1 }, reduce.block);
        try t.expectEqual(@as(u64, 0x500), reduce.args[0]);
        try t.expectEqualSlices(u64, &.{ 0x500, 0x400, rows * 4608, 2, 0x3f000000 }, reduce.args[0..5]);
    }
    Fake.count = 0;
    try l.matmul(s, 0x100, 2688, 256, weight(.bf16), 0x400, 0);
    try t.expectEqual(@as(usize, 1), Fake.count);
    try t.expectEqual(@as(usize, bf16_fns + 6), Fake.records[0].fun);
    try t.expectEqual([3]u32{ 288, 1, 1 }, Fake.records[0].grid);
    l.clusters = true;
    Fake.count = 0;
    try l.matmul(s, 0x100, 2688, 1, weight(.bf16), 0x400, 0);
    try t.expectEqual(@as(usize, 1), Fake.count);
    try t.expect(Fake.records[0].extended);
    try t.expectEqual(@as(usize, bf16_fns + 1), Fake.records[0].fun);
    try t.expectEqual(@as(u64, 0), Fake.records[0].args[5]);
    try t.expectEqual([3]u32{ 1, 1, 2 }, Fake.records[0].cluster);
}
test "modelopt host: loader and each bound forward select initialized scratch on its own stream" {
    const d = Fake.driver();
    var e: Engine = undefined;
    e.k = Fake.kernels(&d, false);
    e.stream = .{ .d = &d, .handle = @ptrFromInt(7) };
    e.b = undefined;
    e.b.lane_part = 0xdead;
    try t.expectEqual(@as(u64, 0), e.loadOps().part);
    try t.expectEqual(@as(u64, 0), e.loadOps().ck8);
    e.b.lane_part = 0x1000;
    e.b.ck8 = 0x9000;
    try t.expectEqual(@as(u64, 0x1000), e.ops().part);
    Fake.count = 0;
    try e.ops().dense(0x100, 0x777, .{ .bf16 = weight(.bf16) }, 0x400, 1);
    try t.expectEqual(@as(u64, 0x1000), Fake.records[0].args[5]);
    var sib = e.b;
    sib.lane_part = 0x2000;
    sib.ck8 = 0xa000;
    var parts: segments.Segments = undefined;
    parts.runner.n = 2;
    parts.runner.side[0] = .{ .d = &d, .handle = @ptrFromInt(8) };
    var w: @import("cuda_weights.zig").Weights = undefined;
    const c = try config();
    for ([2]*const state.Buffers{ &e.b, &sib }, 0..) |b, i| {
        const ops = segments.segmentOps(&e, &parts, b, i);
        try t.expectEqual(@as(usize, 7 + i), @intFromPtr(ops.s.handle));
        try t.expectEqual(@as(u64, 0x1000) * (i + 1), ops.part);
        const f = Forward.init(c, &w, b, .{ .k = &e.k, .s = ops.s, .part = 0xbeef }, 4096, 8, false);
        try t.expectEqual(ops.part, f.ops.part);
        try t.expectEqual(b.ck8, f.ops.ck8);
        Fake.count = 0;
        try f.ops.prefillDense(0x100, .{ .bf16 = weight(.bf16) }, 0x400, 1);
        try t.expectEqual(ops.part, Fake.records[0].args[5]);
    }
}
test "modelopt host: real buffer and sibling allocations cover all dense partials through the fuse boundary" {
    const d = Fake.driver();
    var c = try config();
    const l = Fake.lane(&d, false);
    for ([_]usize{ 0, 1024, 12224 }) |latent| {
        c.latent = latent;
        var b = try state.Buffers.init(&d, c, 4096, 8, null);
        defer b.deinit();
        var sibling = try b.sibling(&d, c, 8);
        defer sibling.deinit();
        sibling.follow(&b);
        const bytes = b.arena.buf.ptr + b.arena.used - b.lane_part;
        try t.expectEqual(@as(u64, if (latent == 12224) 24936960 else 21934080), bytes);
        try t.expect(b.lane_part != sibling.lane_part);
        try t.expectEqual(bytes, sibling.arena.buf.ptr + sibling.arena.used - sibling.lane_part);
        const shapes = [_][2]usize{ .{ c.projDim(), c.hidden }, .{ c.hidden, c.inner() }, .{ c.qkvDim(), c.hidden }, .{ c.hidden, c.heads * c.head_dim }, .{ c.vocab, c.hidden }, .{ c.hidden, 2 * c.hidden }, .{ c.latent, c.hidden }, .{ c.hidden, c.latent } };
        for (shapes) |s| for (1..257) |rows| {
            const q = qmmf.Weight{ .mode = .bf16, .codes = 1, .scales = 0, .n = @intCast(s[0]), .k = @intCast(s[1]), .npad = @intCast(s[0]) };
            try t.expect(l.partBytes(rows, q) <= bytes);
        };
    }
    const call = cuda.segments.chunked(2049, state.prefill_rows, 2);
    try t.expectEqual(@as(usize, 2), call.parts);
    try t.expectEqual(@as(usize, 1), cuda.segments.chunkRows(call.rows, state.prefill_rows, 1));
    try t.expectEqual(@as(usize, 36864), l.partBytes(1, weight(.bf16)));
}
test "modelopt host: unsafe floors and checkpoint precision defaults stay refused where required" {
    for ([_]u32{ 70, 75 }) |cap| try t.expect(kern.modeloptRefusal(cap) != null);
    for ([_]u32{ 80, 86, 89, 90, 120, 121 }) |cap| try t.expect(kern.modeloptRefusal(cap) == null);
    const d = Fake.driver();
    const ctx: cuda.Context = .{ .d = &d, .device = 0, .handle = @ptrFromInt(1) };
    var c = try config();
    c.latent = 1024;
    const wanted = @import("cuda_latent_ck.zig").wanted;
    Fake.major = 8;
    try t.expect(!try wanted(&ctx, c, null));
    try t.expectError(error.CheckpointMathNeedsBlackwell, wanted(&ctx, c, .checkpoint));
    Fake.major = 12;
    try t.expect(try wanted(&ctx, c, null));
    try t.expect(!try wanted(&ctx, c, .full));
}
