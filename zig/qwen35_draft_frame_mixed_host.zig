//! Host launch tests for mixed Qwen3.5 draft head widths.
const std = @import("std");
const q = @import("qwen35");
const cuda = @import("cuda");

fn config() q.draft.Config {
    return .{ .hidden = 256, .intermediate = 256, .layers = 1, .vocab = 256, .heads = 2, .kv_heads = 1, .head_dim = 128, .block = 8, .mask = 255, .window = 2048, .group = 16, .rank = 2, .topk = 2, .eps = 1e-6, .theta = 10000000, .taps = @splat(0), .n_taps = 1 };
}
fn function(d: *const cuda.Driver, id: usize) cuda.Function {
    return .{ .d = d, .handle = @ptrFromInt(id) };
}
fn weight() cuda.qlinear.Affine4 {
    return .{ .w = 1, .s = 2, .b = 3, .n = 256, .k = 256, .npad = 256 };
}

test "Draft frames: mixed affine-4 head rebuilds sums after row compaction" {
    const F = q.draft.frame;
    const Trace = struct {
        const Event = enum { compact, sums, project };
        const Copy = struct { dst: u64, src: u64, bytes: usize };
        const Sum = struct { x: u64, xs: u64, k: c_int, rows: usize, grid: [3]usize };
        const Projection = struct { x: u64, xs: u64, rows: usize, members: usize, sums: bool };
        var events: [6]Event = undefined;
        var event_count: usize = 0;
        var copies: [2]Copy = undefined;
        var copy_count: usize = 0;
        var sums: [2]Sum = undefined;
        var sum_count: usize = 0;
        var projections: [2]Projection = undefined;
        var projection_count: usize = 0;

        fn record(event: Event) void {
            events[event_count] = event;
            event_count += 1;
        }

        fn copyDevice(dst: u64, src: u64, bytes: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            copies[copy_count] = .{ .dst = dst, .src = src, .bytes = bytes };
            copy_count += 1;
            record(.compact);
            return cuda.abi.success;
        }

        fn copyHost(_: ?*anyopaque, _: u64, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            return cuda.abi.success;
        }

        fn copy2d(_: *const cuda.abi.Memcpy2D, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            return cuda.abi.success;
        }
    };
    const Draws = struct {
        pub fn topkRows(_: *@This(), _: cuda.Stream, _: u64, _: usize, _: u8, _: u64, _: u64, _: usize) !void {}
    };
    const E = struct {
        d: *cuda.Driver,
        stream: cuda.Stream,
        head: q.engine.Proj,
        affine: struct { pdl: bool },
        fns: struct { sums: cuda.Function },
        draws: Draws,

        pub fn run(_: *@This(), f: cuda.Function, grid: [3]usize, _: [3]usize, args: anytype) !void {
            if (f.handle) |handle| {
                if (@intFromPtr(handle) == 1) {
                    Trace.sums[Trace.sum_count] = .{ .x = args[0], .xs = args[1], .k = @intCast(args[2]), .rows = grid[0], .grid = grid };
                    Trace.sum_count += 1;
                    Trace.record(.sums);
                }
            }
        }

        pub fn projSums(_: *@This(), _: q.engine.Proj, x: u64, xs: u64, _: u64, rows: usize, _: bool, _: u64) !void {
            Trace.projections[Trace.projection_count] = .{ .x = x, .xs = xs, .rows = rows, .members = 1, .sums = false };
            Trace.projection_count += 1;
            Trace.record(.project);
        }

        pub fn projManySums(_: *@This(), ls: []const q.engine.Proj, x: u64, xs: u64, _: []const u64, rows: usize, sums: bool, _: u64) !void {
            Trace.projections[Trace.projection_count] = .{ .x = x, .xs = xs, .rows = rows, .members = ls.len, .sums = sums };
            Trace.projection_count += 1;
            Trace.record(.project);
        }
    };
    const C = struct { block: usize, hidden: usize, vocab: usize, topk: usize };
    const D = struct {
        e: *E,
        c: C,
        x: u64,
        h2: u64,
        logits: u64,
        xs: u64,
        parts: u64,
        words: u64,
    };

    var driver: cuda.Driver = undefined;
    driver.api.cuMemcpyDtoDAsync_v2 = Trace.copyDevice;
    driver.api.cuMemcpyDtoHAsync_v2 = Trace.copyHost;
    driver.api.cuMemcpy2DAsync_v2 = Trace.copy2d;
    const c: C = .{ .block = 8, .hidden = 64, .vocab = 248320, .topk = 2 };
    const affine_head: cuda.qlinear.Affine4 = .{ .w = 0x8000, .s = 0x9000, .b = 0xa000, .n = c.vocab, .k = c.hidden, .npad = c.vocab };
    var e = E{
        .d = &driver,
        .stream = .{ .d = &driver, .handle = null },
        .head = .{ .a4 = affine_head },
        .affine = .{ .pdl = true },
        .fns = .{ .sums = .{ .d = &driver, .handle = @ptrFromInt(1) } },
        .draws = .{},
    };
    const d = D{ .e = &e, .c = c, .x = 0x100000, .h2 = 0x200000, .logits = 0x300000, .xs = 0x400000, .parts = 0x500000, .words = 0x600000 };
    var contexts: [2]q.draft.Ctx = .{ .{ .ring = undefined, .wide = true }, .{ .ring = undefined } };
    const refs = [_]*q.draft.Ctx{ &contexts[0], &contexts[1] };
    var storage: [F.stagedLen(2)]u8 align(8) = undefined;
    Trace.event_count = 0;
    Trace.copy_count = 0;
    Trace.sum_count = 0;
    Trace.projection_count = 0;

    try F.mixedHead(&d, &refs, &storage);

    const rows = c.block - 1;
    const input_bytes = rows * c.hidden * 2;
    try std.testing.expectEqual(@as(usize, 2), Trace.copy_count);
    try std.testing.expectEqual(d.h2, Trace.copies[0].dst);
    try std.testing.expectEqual(d.x + input_bytes, Trace.copies[0].src);
    try std.testing.expectEqual(input_bytes, Trace.copies[0].bytes);
    try std.testing.expectEqual(d.h2, Trace.copies[1].dst);
    try std.testing.expectEqual(d.x, Trace.copies[1].src);
    try std.testing.expectEqual(input_bytes, Trace.copies[1].bytes);
    try std.testing.expectEqual(@as(usize, 2), Trace.sum_count);
    for (Trace.sums) |sum| {
        try std.testing.expectEqual(d.h2, sum.x);
        try std.testing.expectEqual(d.xs, sum.xs);
        try std.testing.expectEqual(@as(c_int, @intCast(c.hidden)), sum.k);
        try std.testing.expectEqual(rows, sum.rows);
        try std.testing.expectEqual([_]usize{ rows, 1, 1 }, sum.grid);
    }
    try std.testing.expectEqual(@as(usize, 2), Trace.projection_count);
    for (Trace.projections) |projection| {
        try std.testing.expectEqual(d.h2, projection.x);
        try std.testing.expectEqual(d.xs, projection.xs);
        try std.testing.expectEqual(rows, projection.rows);
    }
    try std.testing.expectEqual(@as(usize, 2), Trace.projections[0].members);
    try std.testing.expect(!Trace.projections[0].sums);
    try std.testing.expectEqual(@as(usize, 1), Trace.projections[1].members);
    try std.testing.expectEqual([_]Trace.Event{ .compact, .sums, .project, .compact, .sums, .project }, Trace.events);
}

test "Qwen CUDA contracts: mixed-width heads launch by class and restore stream rows" {
    for ([_]usize{ 8, 16 }) |block| {
        try mixedHeadContract(block, false);
        try mixedHeadContract(block, true);
    }
}

fn mixedHeadContract(block: usize, wide_even: bool) !void {
    const Trace = struct {
        const Copy = struct { dst: u64, src: u64, bytes: usize };
        var gathered: [5]Copy = undefined;
        var fetched: [5]Copy = undefined;
        var np: usize = 0;
        var nf: usize = 0;
        var heads: [2]struct { columns: u32, rows: u32 } = undefined;
        var nh: usize = 0;
        var copies2d: usize = 0;
        fn launch(f: cuda.abi.Function, _: c_uint, gy: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            if (@intFromPtr(f) == 1) {
                heads[nh] = .{ .columns = @as(*const u32, @ptrCast(@alignCast(args.?[1].?))).*, .rows = gy };
                nh += 1;
            }
            return cuda.abi.success;
        }
        fn copyDevice(dst: u64, src: u64, bytes: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            gathered[np] = .{ .dst = dst, .src = src, .bytes = bytes };
            np += 1;
            return cuda.abi.success;
        }
        fn copyHost(dst: ?*anyopaque, src: u64, bytes: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            fetched[nf] = .{ .dst = @intFromPtr(dst.?), .src = src, .bytes = bytes };
            nf += 1;
            return cuda.abi.success;
        }
        fn copy2d(_: *const cuda.abi.Memcpy2D, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            copies2d += 1;
            return cuda.abi.success;
        }
    };
    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = Trace.launch;
    driver.api.cuMemcpyDtoDAsync_v2 = Trace.copyDevice;
    driver.api.cuMemcpyDtoHAsync_v2 = Trace.copyHost;
    driver.api.cuMemcpy2DAsync_v2 = Trace.copy2d;
    var e: q.engine.Engine = undefined;
    e.d = &driver;
    e.stream = .{ .d = &driver, .handle = null };
    e.kit = null;
    e.affine.split = null;
    e.affine.mlp128 = false;
    e.affine.pdl = false;
    e.affine.wide_verify = false;
    e.affine.group_fn = function(&driver, 4);
    e.fns.sums = function(&driver, 3);
    e.draws.topk_slices_fn = function(&driver, 1);
    e.draws.topk_merge_fn = function(&driver, 2);
    var head = weight();
    head.n = 248320;
    head.npad = head.n;
    e.head = .{ .a4 = head };
    var d: q.draft.Draft = undefined;
    d.e = &e;
    d.c = config();
    d.c.block = block;
    d.c.vocab = 248320;
    d.x = 0x100000;
    d.h2 = 0x200000;
    d.logits = 0x300000;
    d.xs = 0x400000;
    d.words = 0x500000;
    d.parts = 0x600000;
    var contexts: [5]q.draft.Ctx = undefined;
    var refs: [5]*q.draft.Ctx = undefined;
    for (&contexts, &refs, 0..) |*ctx, *ref, i| {
        ctx.* = .{ .ring = undefined, .wide = if (wide_even) i % 2 == 0 else i % 2 == 1 };
        ref.* = ctx;
    }
    var storage: [q.draft.frame.stagedLen(2)]u8 align(8) = undefined;
    Trace.np = 0;
    Trace.nf = 0;
    Trace.nh = 0;
    Trace.copies2d = 0;
    try std.testing.expect(q.draft.frame.mixedHeadEligible(e.head, &refs));
    const all_rows = contexts.len * (d.c.block - 1);
    const baseline = try q.draft.frame.logitsOf(&d, &refs, d.x, d.logits, all_rows);
    try std.testing.expect(baseline == null);
    try e.draws.topkRows(e.stream, baseline orelse d.logits, q.draft.frame.columns(baseline != null, d.c.vocab), @intCast(d.c.topk), d.parts, d.words, all_rows);
    try std.testing.expectEqual(@as(usize, 1), Trace.nh);
    try std.testing.expectEqual(@as(u32, 248320), Trace.heads[0].columns);
    try std.testing.expectEqual(@as(u32, @intCast(all_rows)), Trace.heads[0].rows);
    Trace.nh = 0;
    try q.draft.frame.mixedHead(&d, &refs, &storage);
    try std.testing.expectEqual(@as(usize, 5), Trace.np);
    try std.testing.expectEqual(@as(usize, 5), Trace.nf);
    try std.testing.expectEqual(@as(usize, 2), Trace.nh);
    try std.testing.expectEqual(@as(usize, 2), Trace.copies2d);
    const narrow_count: usize = if (wide_even) 2 else 3;
    const wide_count = contexts.len - narrow_count;
    const per = d.c.block - 1;
    try std.testing.expectEqual(@as(u32, 98592), Trace.heads[0].columns);
    try std.testing.expectEqual(@as(u32, @intCast(narrow_count * per)), Trace.heads[0].rows);
    try std.testing.expectEqual(@as(u32, 248320), Trace.heads[1].columns);
    try std.testing.expectEqual(@as(u32, @intCast(wide_count * per)), Trace.heads[1].rows);
    const input_bytes = per * 256 * 2;
    const word_bytes = per * 43 * 4;
    const order = if (wide_even) [_]usize{ 1, 3, 0, 2, 4 } else [_]usize{ 0, 2, 4, 1, 3 };
    for (order, 0..) |original, i| {
        const group_index = if (i < narrow_count) i else i - narrow_count;
        try std.testing.expectEqual(d.x + original * input_bytes, Trace.gathered[i].src);
        try std.testing.expectEqual(d.h2 + group_index * input_bytes, Trace.gathered[i].dst);
        try std.testing.expectEqual(@as(usize, input_bytes), Trace.gathered[i].bytes);
        try std.testing.expectEqual(d.words + group_index * word_bytes, Trace.fetched[i].src);
        try std.testing.expectEqual(@intFromPtr(&storage) + q.draft.frame.words_at + original * word_bytes, Trace.fetched[i].dst);
        try std.testing.expectEqual(@as(usize, word_bytes), Trace.fetched[i].bytes);
    }
    try std.testing.expectEqual(@as(usize, 98592), q.draft.frame.columns(true, 248320));
    try std.testing.expectEqual(@as(usize, 248320), q.draft.frame.columns(false, 248320));
    try std.testing.expectEqual(@as(u32, 248032), q.draft.frame.columnId(98304));
    try std.testing.expectEqual(@as(u32, 248319), q.draft.frame.columnId(98591));
}

test "Qwen CUDA contracts: mixed-width eligibility covers A4 and EXL3, not homogeneous batches" {
    var narrow: q.draft.Ctx = .{ .ring = undefined };
    var wide: q.draft.Ctx = .{ .ring = undefined, .wide = true };
    var refs = [_]*q.draft.Ctx{ &narrow, &wide };
    var e: q.engine.Engine = undefined;
    var a4 = weight();
    a4.n = 248320;
    a4.npad = a4.n;
    e.head = .{ .a4 = a4 };
    try std.testing.expect(q.draft.frame.mixedHeadEligible(e.head, &refs));
    wide.wide = false;
    try std.testing.expect(!q.draft.frame.mixedHeadEligible(e.head, &refs));
    narrow.wide = true;
    wide.wide = true;
    try std.testing.expect(!q.draft.frame.mixedHeadEligible(e.head, &refs));

    narrow.wide = false;
    wide.wide = true;
    const x3 = cuda.exl3;
    const st = x3.strides(5120, 248320, 12, false);
    const x3_head: x3.Weight = .{ .trellis = 0x1000, .suh = 0x2000, .svh = 0x3000, .k = 5120, .n = 248320, .halves = 12, .codebook = .mul1, .stride_k = st.k, .stride_nb = st.nb };
    e.head = .{ .x3 = x3_head };
    try std.testing.expect(q.draft.frame.mixedHeadEligible(e.head, &refs));
    wide.wide = false;
    try std.testing.expect(!q.draft.frame.mixedHeadEligible(e.head, &refs));
    narrow.wide = true;
    wide.wide = true;
    try std.testing.expect(!q.draft.frame.mixedHeadEligible(e.head, &refs));
}
