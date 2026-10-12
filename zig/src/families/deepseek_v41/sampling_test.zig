//! The sampling pipeline on in-process ranks against the serving goldens: each row split over ranks and picked.

const std = @import("std");
const testing = std.testing;
const tp = @import("tp");
const json = @import("json");
const serve = @import("dsv41_serve");
const pipe = @import("sampling_pipe.zig");

const Case = struct {
    logits: []const f32,
    world: u32,
    rows: usize,
    sampling: pipe.Sampling,
    positions: []const u64,
    nucleus: u32 = serve.sampling.nucleus_count,
    got: []u32,
    whole: []usize,
    /// rank 0: vs_choose's mirror over the gathered candidates (sampling_pipe.chooseMirror), per row; null: not run
    mirror: ?[]i64 = null,
};

/// The download hook on rank 0: the mirror over the gathered candidates, then the pipe's own download (false).
const Mirror = struct {
    c: *Case,
    fn run(ctx: *anyopaque, gathered: u64, n: usize, k: usize, count: usize, host: []u8) anyerror!bool {
        _ = host;
        const m: *Mirror = @ptrCast(@alignCast(ctx));
        const W: usize = m.c.world;
        const g: [*]const f32 = @ptrFromInt(gathered);
        const a = testing.allocator;
        const keys = try a.alloc(u64, W * k);
        defer a.free(keys);
        const work = try a.alloc(f64, 1024);
        defer a.free(work);
        pipe.chooseMirror(keys, work, g[0 .. W * n * 2 * k], W, n, k, count, m.c.sampling, m.c.positions[0], m.c.mirror.?);
        return false;
    }
};

/// Rank `me` of a window: its vocabulary slice of each row (contiguous [rows, V]), the pipeline's tokens on rank 0.
fn rankBody(comm: tp.collective.Collective, me: u32, c: *Case) anyerror!void {
    const a = testing.allocator;
    const vocab = c.logits.len / c.rows;
    const V = vocab / c.world;
    const mine = try a.alloc(f32, c.rows * V);
    defer a.free(mine);
    for (0..c.rows) |r| @memcpy(mine[r * V ..][0..V], c.logits[r * vocab + me * V ..][0..V]);
    var dev: pipe.HostDevice = .{ .gpa = a };
    defer dev.deinit();
    var p = try pipe.Pipe.init(a, comm, @intCast(V), c.nucleus);
    defer p.deinit();
    var mh: Mirror = .{ .c = c };
    if (me == 0 and c.mirror != null) p.after_gather = .{ .ctx = &mh, .run = Mirror.run };
    const out = try a.alloc(u32, c.rows);
    defer a.free(out);
    try p.choose(dev.device(), @intFromPtr(mine.ptr), c.rows, c.sampling, if (me == 0) c.positions else null, out);
    if (me == 0) {
        @memcpy(c.got, out);
        c.whole[0] = p.whole_rows;
    }
}

fn run(c: *Case) !void {
    try tp.host.run(c.world, c, rankBody);
}

fn parse(x: std.mem.Allocator, line: []const u8) !struct { logits: []f32, world: u32, s: ?pipe.Sampling, position: u64, full: bool, token: u32 } {
    const c = (try json.parseText(x, line)).ok;
    const b64 = c.get("logits").?.string;
    const raw = try x.alloc(u8, try std.base64.standard.Decoder.calcSizeForSlice(b64));
    try std.base64.standard.Decoder.decode(raw, b64);
    const logits = try x.alloc(f32, raw.len / 4);
    for (logits, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, raw[4 * i ..][0..4], .little));
    const sv = c.get("sampling").?;
    const s: ?pipe.Sampling = if (sv == .array) .{
        .seed = @truncate(@as(u128, @bitCast(@as(i128, try std.fmt.parseInt(i128, sv.array[0].int, 10))))),
        .temperature = sv.array[1].float64().?,
        .top_k = @intCast(sv.array[2].int64().?),
        .top_p = sv.array[3].float64().?,
        .min_p = sv.array[4].float64().?,
    } else null;
    return .{ .logits = logits, .world = @intCast(c.get("world").?.int64().?), .s = s, .position = @intCast(c.get("position").?.int64().?), .full = c.get("full").?.bool, .token = @intCast(c.get("token").?.int64().?) };
}

test "sampled rows on two ranks pick the Python engine's tokens (the committed serving goldens)" {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var it = std.mem.splitScalar(u8, serve.fixtures.sampling, '\n');
    var rows: usize = 0;
    var bad: usize = 0;
    var whole: usize = 0;
    var fallbacks: usize = 0;
    while (it.next()) |line| {
        if (line.len == 0) continue;
        _ = arena.reset(.retain_capacity);
        const g = try parse(arena.allocator(), line);
        const s = g.s orelse continue; // greedy rows: forward.greedy's path
        if (!pipe.sampled(s) or g.logits.len % g.world != 0) continue;
        var got: [1]u32 = .{0};
        var w: [1]usize = .{0};
        var c: Case = .{ .logits = g.logits, .world = g.world, .rows = 1, .sampling = s, .positions = &.{g.position}, .got = &got, .whole = &w };
        try run(&c);
        rows += 1;
        whole += w[0];
        fallbacks += @intFromBool(g.full);
        if (got[0] != g.token) {
            bad += 1;
            if (bad <= 5) std.debug.print("row {d}: want {d} got {d} (vocab {d}, world {d}, sampling {any})\n", .{ rows, g.token, got[0], g.logits.len, g.world, s });
        }
    }
    std.debug.print("sampling pipeline: {d}/{d} sampled rows as Python picks them ({d} taken whole, {d} nucleus fallbacks in the goldens)\n", .{ rows - bad, rows, whole, fallbacks });
    try testing.expect(rows > 100 and fallbacks > 0);
    try testing.expectEqual(@as(usize, 0), bad);
}

test "a window's rows pick as each row does alone (one call, several positions, every path)" {
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(41);
    const rnd = prng.random();
    const vocab = 4096;
    const n = 6;
    const logits = try a.alloc(f32, n * vocab);
    defer a.free(logits);
    for (logits, 0..) |*v, i| {
        const x: f32 = @floatCast(rnd.floatNorm(f64) * 3.0 + @as(f64, if (i % vocab == 77) 6 else 0));
        v.* = @bitCast(@as(u32, @bitCast(x)) & 0xFFFF_0000); // bf16-valued, as kit_logits leaves them
    }
    const positions = [_]u64{ 100, 101, 102, 103, 104, 105 };
    const cases = [_]pipe.Sampling{
        .{ .seed = 7, .temperature = 0.7, .top_k = 0, .top_p = 0.9 }, // nucleus
        .{ .seed = 9, .temperature = 1.0, .top_k = 40, .top_p = 0.95, .min_p = 0.05 }, // top-k
        .{ .seed = 3, .temperature = 1.3, .top_k = 0, .top_p = 1.0 }, // whole rows (k > max_k)
        .{ .seed = 1 << 63 | 5, .temperature = 0.6, .top_k = 0, .top_p = 0.8 },
    };
    for (cases) |s| for ([_]u32{ 1, 2 }) |world| {
        var all: [n]u32 = undefined;
        var w: [1]usize = .{0};
        var c: Case = .{ .logits = logits, .world = world, .rows = n, .sampling = s, .positions = &positions, .got = &all, .whole = &w };
        try run(&c);
        for (0..n) |r| {
            var one: [1]u32 = undefined;
            var w1: [1]usize = .{0};
            var c1: Case = .{ .logits = logits[r * vocab ..][0..vocab], .world = world, .rows = 1, .sampling = s, .positions = positions[r..][0..1], .got = &one, .whole = &w1 };
            try run(&c1);
            try testing.expectEqual(one[0], all[r]);
        }
        // the keyed choice is the host rule's over the whole row (lanes.sampling.choose = exact_sampling.choose)
        for (0..n) |r| {
            const row = logits[r * vocab ..][0..vocab];
            const vals = try a.alloc(f64, vocab);
            defer a.free(vals);
            const ids = try a.alloc(u64, vocab);
            defer a.free(ids);
            for (vals, ids, row, 0..) |*v, *id, x, i| {
                v.* = x;
                id.* = i;
            }
            const want = try @import("lanes").sampling.choose(a, vals, ids, positions[r], s);
            try testing.expectEqual(@as(u32, @intCast(want)), all[r]);
        }
    };
}

test "vs_choose's arithmetic (its host mirror) picks the pipeline's tokens on the serving goldens' keyed rows" {
    // the device choice only decides where the speculative pass starts; its host mirror must equal the host pick
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var it = std.mem.splitScalar(u8, serve.fixtures.sampling, '\n');
    var rows: usize = 0;
    var bad: usize = 0;
    while (it.next()) |line| {
        if (line.len == 0) continue;
        _ = arena.reset(.retain_capacity);
        const g = try parse(arena.allocator(), line);
        const s = g.s orelse continue;
        if (!pipe.sampled(s) or g.logits.len % g.world != 0) continue;
        if (s.top_k == 0 and s.top_p > 0 and s.top_p < 1) continue; // nucleus rows: the host's statistics
        var got: [1]u32 = .{0};
        var w: [1]usize = .{0};
        var mirror: [1]i64 = .{-2};
        var c: Case = .{ .logits = g.logits, .world = g.world, .rows = 1, .sampling = s, .positions = &.{g.position}, .got = &got, .whole = &w, .mirror = &mirror };
        try run(&c);
        if (mirror[0] == -2) continue; // taken whole (k past the device bound): no gathered candidates
        rows += 1;
        if (mirror[0] != got[0]) {
            bad += 1;
            if (bad <= 5) std.debug.print("row {d}: mirror {d}, pipeline {d} (sampling {any})\n", .{ rows, mirror[0], got[0], s });
        }
    }
    std.debug.print("vs_choose mirror: {d}/{d} keyed rows as the pipeline picks them\n", .{ rows - bad, rows });
    try testing.expect(rows > 20);
    try testing.expectEqual(@as(usize, 0), bad);
}

test "vs_choose's mirror == the pipeline's picks over top_k x top_p x min_p x temperature, one and two ranks" {
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(4101);
    const rnd = prng.random();
    const vocab = 4096;
    const n = 6;
    const logits = try a.alloc(f32, n * vocab);
    defer a.free(logits);
    const positions = [_]u64{ 300, 301, 302, 303, 304, 305 };
    var rows: usize = 0;
    for (0..3) |round| {
        // flat-ish rows (the top-p cut lands inside the top k), peaked rows, and ties (bf16-valued)
        for (logits, 0..) |*v, i| {
            const spread: f64 = if (round == 0) 0.6 else 2.5;
            const x: f32 = @floatCast(rnd.floatNorm(f64) * spread + @as(f64, if (i % vocab < 3) 4 else 0));
            v.* = if (round == 2) @bitCast(@as(u32, @bitCast(x)) & 0xFFFF_0000) else x;
        }
        for ([_]u32{ 20, 40, 100 }) |top_k| for ([_]f64{ 0.5, 0.8, 0.95, 1.0 }) |top_p| for ([_]f64{ 0, 0.05 }) |min_p| for ([_]f64{ 0.6, 1.3 }) |t| for ([_]u32{ 1, 2 }) |world| {
            const s: pipe.Sampling = .{ .seed = rnd.int(u64) >> 1, .temperature = t, .top_k = top_k, .top_p = top_p, .min_p = min_p };
            var got: [n]u32 = undefined;
            var w: [1]usize = .{0};
            var mirror: [n]i64 = @splat(-2);
            var c: Case = .{ .logits = logits, .world = world, .rows = n, .sampling = s, .positions = &positions, .got = &got, .whole = &w, .mirror = &mirror };
            try run(&c);
            for (got, mirror) |g, m| {
                try testing.expectEqual(@as(i64, g), m);
                rows += 1;
            }
        };
    }
    try testing.expectEqual(@as(usize, 3 * 3 * 4 * 2 * 2 * 2 * n), rows);
}

const SegSpec = struct { row0: u32, n: u32, s: pipe.Sampling, positions: []const u64 };

/// Several segments of one forward's rows (rows not in a segment are another window's, never picked).
const Multi = struct {
    logits: []const f32,
    world: u32,
    rows: usize,
    segs: []const SegSpec,
    batched: bool,
    got: []u32,
    whole: usize = 0,
    /// rank 0, batched: vs_choose's mirror of each top_k segment over the one gather (row0-shifted draws), its rows
    mirror: []i64 = &.{},
};

/// The batched call's hook on rank 0: each top_k segment's device choice from the shared gather, then download.
const SegMirror = struct {
    c: *Multi,
    fn run(ctx: *anyopaque, gathered: u64, n: usize, k: usize, segs: []const pipe.Pipe.Seg, host: []u8) anyerror!bool {
        _ = host;
        const m: *SegMirror = @ptrCast(@alignCast(ctx));
        const W: usize = m.c.world;
        const g: [*]const f32 = @ptrFromInt(gathered);
        const a = testing.allocator;
        const keys = try a.alloc(u64, W * k);
        defer a.free(keys);
        const work = try a.alloc(f64, 1024);
        defer a.free(work);
        const out = try a.alloc(i64, n);
        defer a.free(out);
        const r0 = segs[0].row0;
        for (segs) |sg| {
            if (serve.sampling.isNucleus(sg.s)) continue;
            const off = sg.row0 - r0;
            const count = pipe.countOf(sg.s, @intCast(m.c.logits.len / m.c.rows), serve.sampling.nucleus_count);
            pipe.chooseMirror(keys, work, g[0 .. W * n * 2 * k], W, n, k, count, sg.s, sg.positions.?[0] -% off, out);
            @memcpy(m.c.mirror[sg.row0..][0..sg.n], out[off..][0..sg.n]);
        }
        return false;
    }
};

fn multiBody(comm: tp.collective.Collective, me: u32, c: *Multi) anyerror!void {
    const a = testing.allocator;
    const vocab = c.logits.len / c.rows;
    const V = vocab / c.world;
    const mine = try a.alloc(f32, c.rows * V);
    defer a.free(mine);
    for (0..c.rows) |r| @memcpy(mine[r * V ..][0..V], c.logits[r * vocab + me * V ..][0..V]);
    var dev: pipe.HostDevice = .{ .gpa = a };
    defer dev.deinit();
    var p = try pipe.Pipe.init(a, comm, @intCast(V), serve.sampling.nucleus_count);
    defer p.deinit();
    const lg: u64 = @intFromPtr(mine.ptr);
    const out = try a.alloc(u32, c.rows);
    defer a.free(out);
    @memset(out, 0);
    var whole: usize = 0;
    if (c.batched) {
        var segs: [8]pipe.Pipe.Seg = undefined;
        for (c.segs, segs[0..c.segs.len]) |g, *x| {
            try testing.expect(p.batchable(g.s));
            x.* = .{ .row0 = g.row0, .n = g.n, .s = g.s, .positions = if (me == 0) g.positions else null, .out = out[g.row0..][0..g.n] };
        }
        var mh: SegMirror = .{ .c = c };
        try p.chooseSegments(dev.device(), lg, segs[0..c.segs.len], if (me == 0) .{ .ctx = &mh, .run = SegMirror.run } else null);
        whole = p.whole_rows;
    } else for (c.segs) |g| {
        try p.choose(dev.device(), lg + 4 * @as(u64, g.row0) * V, g.n, g.s, if (me == 0) g.positions else null, out[g.row0..][0..g.n]);
        whole += p.whole_rows;
    }
    if (me == 0) {
        @memcpy(c.got, out);
        c.whole = whole;
    }
}

test "a forward's keyed segments in one call (TF_DSV41_SAMP_BATCH) pick the tokens and keyed draws of one call a segment" {
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(4113);
    const rnd = prng.random();
    const vocab = 4096;
    const rows = 12;
    const logits = try a.alloc(f32, rows * vocab);
    defer a.free(logits);
    for (0..rows) |r| for (0..vocab) |c| {
        // rows 3 and 9 flat (their top-p mass needs more than the nucleus candidates: the whole-row fallback)
        const x: f64 = if (r == 3 or r == 9) (if (c % 2 == 0) 0.25 else 0.0) else rnd.floatNorm(f64) * 3.0 + @as(f64, if (c == 13 * r + 5) 7 else 0);
        logits[r * vocab + c] = @bitCast(@as(u32, @bitCast(@as(f32, @floatCast(x)))) & 0xFFFF_0000);
    };
    const T = pipe.Sampling;
    // row 5 belongs to no segment (another window's row); rows 3 and 9 (flat) take the nucleus fallback
    const nucleus_segs = [_]SegSpec{
        .{ .row0 = 0, .n = 5, .s = T{ .seed = 7, .temperature = 0.7, .top_k = 0, .top_p = 0.95 }, .positions = &.{ 40, 41, 42, 43, 44 } },
        .{ .row0 = 6, .n = 2, .s = T{ .seed = 1 << 63 | 9, .temperature = 1.1, .top_k = 0, .top_p = 0.8, .min_p = 0.02 }, .positions = &.{ 900, 901 } },
        .{ .row0 = 8, .n = 4, .s = T{ .seed = 7, .temperature = 0.5, .top_k = 0, .top_p = 0.9 }, .positions = &.{ 3, 4, 5, 6 } },
    };
    // the served default (top_k 20 under top_p 0.95), other cuts, a wrapping draw start, and a nucleus segment
    const topk_segs = [_]SegSpec{
        .{ .row0 = 0, .n = 5, .s = T{ .seed = 1234, .temperature = 0.7, .top_k = 20, .top_p = 0.95 }, .positions = &.{ 40, 41, 42, 43, 44 } },
        .{ .row0 = 6, .n = 2, .s = T{ .seed = 1 << 63 | 9, .temperature = 1.1, .top_k = 5, .top_p = 1.0, .min_p = 0.05 }, .positions = &.{ 2, 3 } },
        .{ .row0 = 8, .n = 3, .s = T{ .seed = 1234, .temperature = 0.7, .top_k = 0, .top_p = 0.9 }, .positions = &.{ 70, 71, 72 } },
        .{ .row0 = 11, .n = 1, .s = T{ .seed = 3, .temperature = 0.4, .top_k = 300, .top_p = 0.99 }, .positions = &.{500} },
    };
    for ([_][]const SegSpec{ &nucleus_segs, &topk_segs }) |segs| for ([_]u32{ 1, 2 }) |world| {
        var got: [2][rows]u32 = undefined;
        var whole: [2]usize = undefined;
        var mirror: [rows]i64 = @splat(-2);
        for ([_]bool{ false, true }, 0..) |batched, i| {
            var c: Multi = .{ .logits = logits, .world = world, .rows = rows, .segs = segs, .batched = batched, .got = &got[i], .mirror = &mirror };
            try tp.host.run(world, &c, multiBody);
            whole[i] = c.whole;
        }
        try testing.expectEqualSlices(u32, &got[0], &got[1]);
        try testing.expectEqual(whole[0], whole[1]);
        if (segs.ptr == &nucleus_segs) try testing.expect(whole[1] >= 2); // the flat rows took the fallback
        // each top_k segment's device choice from the shared gather == its host pick
        var mirrored: usize = 0;
        for (segs) |g| if (!serve.sampling.isNucleus(g.s)) for (0..g.n) |j| {
            try testing.expectEqual(@as(i64, got[1][g.row0 + j]), mirror[g.row0 + j]);
            mirrored += 1;
        };
        if (segs.ptr == &topk_segs) try testing.expect(mirrored == 8);
        // each row's token is the keyed rule over its whole row (exact_sampling.choose at its own position)
        for (segs) |g| for (0..g.n) |j| {
            const r = g.row0 + j;
            const vals = try a.alloc(f64, vocab);
            defer a.free(vals);
            const ids = try a.alloc(u64, vocab);
            defer a.free(ids);
            for (vals, ids, logits[r * vocab ..][0..vocab], 0..) |*v, *id, x, c| {
                v.* = x;
                id.* = c;
            }
            const want = try @import("lanes").sampling.choose(a, vals, ids, g.positions[j], g.s);
            try testing.expectEqual(@as(u32, @intCast(want)), got[1][r]);
        };
    };
}
