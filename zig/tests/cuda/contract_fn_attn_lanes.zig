//! flashnext_attn_lanes.cuh: a shared round's attention in one launch a kernel, bit for bit the per-lane launches.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const gates = core.contract.gates;
const c = core.contract.spec;
const table = flash.batch_lanes;

const q_heads = 24;
const kv_heads = 2;
const head_dim = 256;
const index_heads = 4;
const index_dim = 128;
const half = 32;
const proj_n = 13952;
const out_k = q_heads * head_dim;
const ratio = 4;
const top = 512;
const idw = top * ratio + ratio;
const dense_keys = (top + 1) * ratio - 1;
const nch = 5;
const eps: f32 = 1e-6;
const row_kv = kv_heads * head_dim;
const max_lanes = 32;

/// Each lane's cached positions and the rows the round adds to it.
pub const Lane = struct { len: usize, rows: usize };
const sixteen: [16]Lane = @splat(.{ .len = 3000, .rows = 16 });
const widest: [max_lanes]Lane = @splat(.{ .len = 700, .rows = 8 });
const cases = [_][]const Lane{
    &.{ .{ .len = 0, .rows = 1 }, .{ .len = 3, .rows = 1 } }, // first keys
    &.{ .{ .len = 127, .rows = 16 }, .{ .len = 128, .rows = 16 }, .{ .len = 2040, .rows = 7 } }, // pooled blocks at edges
    &.{ .{ .len = 2040, .rows = 16 }, .{ .len = 2100, .rows = 1 }, .{ .len = 9000, .rows = 4 }, .{ .len = 300, .rows = 16 } }, // dense beside sparse
    &.{.{ .len = 2040, .rows = 5 }}, // one stream
    &.{ .{ .len = 100, .rows = 4 }, .{ .len = 2040, .rows = 0 }, .{ .len = 9000, .rows = 3 } }, // a stream with no rows this round
    &sixteen, // sixteen streams, all sparse
    &widest, // the table's lanes
};

/// Lanes for a gate's `rows`: up to 16 rows each on `base`'s positions, two lanes at least.
fn split(a: std.mem.Allocator, base: []const Lane, rows: usize) ![]Lane {
    const n = @max(2, (rows + 15) / 16);
    const out = try a.alloc(Lane, n);
    var left = rows;
    for (out, 0..) |*l, i| {
        const take = if (i + 1 == n) left else @min(left - (n - 1 - i), 16);
        l.* = .{ .len = base[i % base.len].len, .rows = take };
        left -= take;
    }
    return out;
}

const Fixture = struct {
    lanes: []const Lane,
    caches: [][4][]u16, // [lane][k, v, indexer k, pooled]: written rows random, the rest NaN
    proj: []u16, // [rows][proj_n]
    qw: []f32,
    kw: []f32,
    iw: []f32,
    ikw: []f32,
    inv: []f32,

    fn rows(f: Fixture) usize {
        var n: usize = 0;
        for (f.lanes) |l| n += l.rows;
        return n;
    }

    fn cap(f: Fixture) usize {
        var n: usize = 0;
        for (f.lanes) |l| n = @max(n, l.len + l.rows);
        return (n + ratio - 1) / ratio * ratio;
    }

    fn make(a: std.mem.Allocator, lanes: []const Lane, seed: u64) !Fixture {
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        var f: Fixture = .{ .lanes = lanes, .caches = try a.alloc([4][]u16, lanes.len), .proj = undefined, .qw = try weights(a, rng, head_dim), .kw = try weights(a, rng, head_dim), .iw = try weights(a, rng, index_dim), .ikw = try weights(a, rng, index_dim), .inv = try a.alloc(f32, half) };
        for (lanes, 0..) |l, i| for ([_]usize{ row_kv, row_kv, index_dim, index_dim }, 0..) |width, k| {
            const n = if (k == 3) f.cap() / ratio else f.cap();
            const written = if (k == 3) l.len / ratio else l.len;
            const x = try a.alloc(u16, n * width);
            for (x[0 .. written * width]) |*v| v.* = bf16(rng.float(f32) * 2 - 1);
            @memset(x[written * width ..], nan);
            f.caches[i][k] = x;
        };
        f.proj = try a.alloc(u16, f.rows() * proj_n);
        for (f.proj) |*v| v.* = bf16((rng.float(f32) * 2 - 1) * 2);
        for (f.inv, 0..) |*o, i| o.* = @floatCast(std.math.pow(f64, 10_000_000.0, -@as(f64, @floatFromInt(i)) / half));
        return f;
    }
};

const nan: u16 = 0x7fc1; // a quiet NaN no kernel's arithmetic makes

fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn weights(a: std.mem.Allocator, rng: std.Random, n: usize) ![]f32 {
    const out = try a.alloc(f32, n);
    for (out) |*x| x.* = 0.5 + rng.float(f32);
    return out;
}

fn cdiv(x: usize, y: usize) usize {
    return (x + y - 1) / y;
}

fn cint(v: usize) c_int {
    return @intCast(v);
}

const Fns = struct { g: *const flash.Glue, select: cuda.Function, select_lanes: cuda.Function };
const Way = enum { lanes, one_launch, swapped_table };

/// The round through `way`, its outputs and every cache: `.swapped_table` crosses two lanes' caches.
fn launch(gpu: check.Gpu, s: cuda.Stream, k: Fns, f: Fixture, way: Way, a: std.mem.Allocator) ![]const u8 {
    var rig = rig_.Rig{ .gpu = gpu, .s = s };
    defer {
        cuda.completion.drain(gpu.d, s, "the lane contract did not drain");
        rig.deinit();
    }
    const n = f.rows();
    const nb = flash.batch_state.scoreStride(f.cap());
    const kv = try a.alloc([4]u64, f.lanes.len);
    for (f.caches, kv) |cs, *o| for (cs, o) |x, *p| {
        const bytes = std.mem.sliceAsBytes(x);
        p.* = try rig.scratch(bytes.len);
        try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = p.*, .len = bytes.len }).uploadAsync(0, bytes, s.handle);
    };
    const proj = try rig.input(std.mem.sliceAsBytes(f.proj));
    const qw = try rig.input(std.mem.sliceAsBytes(f.qw));
    const kw = try rig.input(std.mem.sliceAsBytes(f.kw));
    const iw = try rig.input(std.mem.sliceAsBytes(f.iw));
    const ikw = try rig.input(std.mem.sliceAsBytes(f.ikw));
    const inv = try rig.input(std.mem.sliceAsBytes(f.inv));
    const q = try rig.output(n * q_heads * head_dim * 2);
    const iq = try rig.output(n * index_heads * index_dim * 2);
    const ao = try rig.output(n * out_k * 2);
    const sc = try rig.output(n * nb * 4);
    const ids = try rig.output(n * idw * 4);
    const nk = try rig.output(n * 4);
    const spr = try rig.output(n * 4);
    const po = try rig.scratch(n * nch * q_heads * head_dim * 4);
    const pm = try rig.scratch(n * nch * q_heads * 4);
    const pl = try rig.scratch(n * nch * q_heads * 4);
    var inputs: []const u8 = &.{};
    if (way == .lanes) {
        var row0: usize = 0;
        for (f.lanes, kv) |l, cs| {
            if (l.rows == 0) continue; // nothing of a lane without rows is launched
            const pos_host: [4]i32 = .{ @intCast(l.len), 0, 0, 0 };
            const pos = try rig.input(std.mem.sliceAsBytes(&pos_host));
            const at = row0 * proj_n * 2;
            const qr = q + row0 * q_heads * head_dim * 2;
            const iqr = iq + row0 * index_heads * index_dim * 2;
            const scr = sc + row0 * nb * 4;
            const ends = l.len + l.rows;
            try k.g.attnPrep(s, proj + at, pos, qw, kw, iw, inv, qr, cs[0], cs[1], iqr, cs[2], eps, l.rows, .{});
            try k.g.qsaPool(s, cs[2], cs[3], pos, ikw, inv, eps, l.rows, .{});
            if (ends / ratio > top) try k.g.qsaScores(s, iqr, cs[3], pos, scr, nb, l.rows, @min(nb, cdiv(ends, ratio)), .{});
            var args: cuda.Args = .{};
            inline for (.{ scr, pos, ids + row0 * idw * 4, nk + row0 * 4, spr + row0 * 4, cint(nb), cint(idw), cint(top), cint(ratio) }) |x| args.add(x);
            try cuda.launch.launch(k.select, .{ .grid = .{ .x = @intCast(l.rows) }, .block = .{ .x = 1024 } }, s, &args);
            const chunks = cdiv(@min(ends, dense_keys), 512);
            try k.g.attnChunks(s, qr, cs[0], cs[1], pos, po, pm, pl, ids + row0 * idw * 4, nk + row0 * 4, spr + row0 * 4, l.rows, kv_heads, chunks, true, .{});
            try k.g.attnMerge(s, po, pm, pl, pos, ao + row0 * out_k * 2, nk + row0 * 4, spr + row0 * 4, l.rows, kv_heads, true);
            row0 += l.rows;
        }
    } else {
        const tab = try a.alignedAlloc(u8, .@"16", flash.batch_state.lanes_bytes);
        @memset(tab, 0);
        var row0: usize = 0;
        var max_rows: usize = 0;
        var max_blocks: usize = 0;
        var chunks: usize = 0;
        for (f.lanes, 0..) |l, i| {
            const cs = kv[if (way == .swapped_table) (i + 1) % kv.len else i];
            table.putCaches(tab, 0, i, cs[0], cs[1], cs[2], cs[3]);
            table.putLane(tab, i, row0, l.len, l.rows);
            row0 += l.rows;
            if (l.rows == 0) continue;
            const ends = l.len + l.rows;
            max_rows = @max(max_rows, l.rows);
            if (ends / ratio > top) max_blocks = @max(max_blocks, @min(nb, cdiv(ends, ratio)));
            chunks = @max(chunks, cdiv(@min(ends, dense_keys), 512));
        }
        const lp = try rig.input(tab);
        const li = lp + flash.batch_state.lanes_ptrs;
        inputs = try a.dupe(u8, try rig.inputBytes());
        try k.g.attnPrepLanes(s, proj, lp, li, qw, kw, iw, inv, q, iq, eps, n);
        try k.g.qsaPoolLanes(s, lp, li, ikw, inv, eps, f.lanes.len, max_rows);
        if (max_blocks > 0) try k.g.qsaScoresLanes(s, iq, lp, li, sc, nb, f.lanes.len, max_rows, max_blocks);
        var args: cuda.Args = .{};
        inline for (.{ sc, li, ids, nk, spr, cint(nb), cint(idw), cint(top), cint(ratio) }) |x| args.add(x);
        try cuda.launch.launch(k.select_lanes, .{ .grid = .{ .x = @intCast(n) }, .block = .{ .x = 1024 } }, s, &args);
        try k.g.attnChunksLanes(s, q, lp, li, po, pm, pl, ids, nk, spr, n, kv_heads, chunks);
        try k.g.attnMergeLanes(s, po, pm, pl, li, ao, nk, spr, n, kv_heads);
    }
    var out: std.ArrayList(u8) = .empty;
    for ([_][2]u64{ .{ q, n * q_heads * head_dim * 2 }, .{ iq, n * index_heads * index_dim * 2 }, .{ ao, n * out_k * 2 }, .{ sc, n * nb * 4 }, .{ ids, n * idw * 4 }, .{ nk, n * 4 }, .{ spr, n * 4 } }) |o| try out.appendSlice(a, try rig.bytes(o[0], o[1]));
    for (f.caches, kv) |xs, ps| for (xs, ps) |x, p| try out.appendSlice(a, try rig.bytes(p, x.len * 2));
    if (way != .lanes) try check.expect(std.mem.eql(u8, inputs, try rig.inputBytes()), "flashnext attention lanes: a launch changed an input", .{});
    try check.expect(try rig.intact() and !rig.overlap(), "flashnext attention lanes: a launch broke a guard or overlapped", .{});
    return out.items;
}

const State = struct {
    gpu: check.Gpu,
    stream: cuda.Stream,
    k: Fns,
    base: []const Lane,
    gate: gates.Gate = .shapes,
    inputs_arena: std.heap.ArenaAllocator,
    outputs_arena: std.heap.ArenaAllocator,
    fixture: ?Fixture = null,
    one: []const u8 = &.{},
    per_lane: []const u8 = &.{},
    checked: bool = false,

    fn of(p: *anyopaque) *State {
        return @ptrCast(@alignCast(p));
    }
    fn prepare(p: *anyopaque, shape: c.Shape, fill: gates.Fill, seed: u64) !bool {
        const s = of(p);
        s.inputs_arena.deinit();
        s.inputs_arena = .init(s.gpu.gpa);
        const a = s.inputs_arena.allocator();
        const lanes = if (s.gate == .shapes and shape.rows >= 2) try split(a, s.base, shape.rows) else s.base;
        s.fixture = try Fixture.make(a, lanes, seed);
        if (fill == .nonfinite) {
            const f = &s.fixture.?;
            f.proj[0] = c.toBf16(std.math.nan(f32));
            f.proj[1] = c.toBf16(std.math.inf(f32));
            f.proj[2] = c.toBf16(-std.math.inf(f32));
            if (lanes[0].len > 0) f.caches[0][1][0] = c.toBf16(std.math.nan(f32));
        }
        s.checked = false;
        return true;
    }
    fn runOne(p: *anyopaque) !void {
        const s = of(p);
        s.outputs_arena.deinit();
        s.outputs_arena = .init(s.gpu.gpa);
        const a = s.outputs_arena.allocator();
        s.per_lane = try launch(s.gpu, s.stream, s.k, s.fixture.?, .lanes, a);
        s.one = try launch(s.gpu, s.stream, s.k, s.fixture.?, .one_launch, a);
        s.checked = true;
    }
    fn output(p: *anyopaque) ![]const u8 {
        return of(p).one;
    }
    fn replaced(p: *anyopaque) ![]const u8 {
        return of(p).per_lane;
    }
    fn inputs(p: *anyopaque) ![]const u8 {
        const s = of(p);
        const f = s.fixture.?;
        var out: std.ArrayList(u8) = .empty;
        const a = s.outputs_arena.allocator();
        try out.appendSlice(a, std.mem.sliceAsBytes(f.proj));
        for (f.caches) |lane| for (lane) |x| try out.appendSlice(a, std.mem.sliceAsBytes(x));
        inline for (.{ "qw", "kw", "iw", "ikw", "inv" }) |field| try out.appendSlice(a, std.mem.sliceAsBytes(@field(f, field)));
        return out.items;
    }
    fn intact(p: *anyopaque) !bool {
        return of(p).checked; // launch checks every device input, guard and overlap before it returns
    }
    fn overlap(_: *anyopaque) bool {
        return false;
    }
    fn count(p: *anyopaque) usize {
        return of(p).one.len;
    }
    fn reference(_: *anyopaque, _: []f64, _: []f64) !void {
        return error.ReplacementUsesPerLaneBytes;
    }
    fn own(p: *anyopaque, _: gates.Gate) !?[]const u8 {
        const s = of(p);
        return if (s.checked and std.mem.eql(u8, s.one, s.per_lane)) null else "one-launch bytes differ from the per-lane kernels";
    }
    const vt: gates.VTable = .{ .prepare = prepare, .run = runOne, .output = output, .inputs = inputs, .intact = intact, .overlap = overlap, .count = count, .reference = reference, .check = own, .replaced = replaced };
};

fn depth(_: c.Shape) usize {
    return 0;
}

pub fn run(gpu: check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var g = try flash.Glue.load(gpu.d);
    defer g.deinit();
    var batch = try cuda.Module.load(gpu.d, cuda.kernels.flashnext_batch);
    defer batch.unload();
    const k: Fns = .{ .g = &g, .select = try batch.function("fn_qsa_select"), .select_lanes = try batch.function("fn_qsa_select_lanes") };
    var red = false;
    for (cases, 0..) |lanes, ci| {
        var arena = std.heap.ArenaAllocator.init(gpu.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const f = try Fixture.make(a, lanes, 0xa77e + ci);
        const per_lane = try launch(gpu, s, k, f, .lanes, a);
        try check.sameBytes("flashnext attention lanes: one launch against per-lane", try launch(gpu, s, k, f, .one_launch, a), per_lane);
        red = red or !std.mem.eql(u8, try launch(gpu, s, k, f, .swapped_table, a), per_lane);
        var st: State = .{ .gpu = gpu, .stream = s, .k = k, .base = lanes, .inputs_arena = .init(gpu.gpa), .outputs_arena = .init(gpu.gpa) };
        defer st.inputs_arena.deinit();
        defer st.outputs_arena.deinit();
        const served = [_]c.Shape{.{ .rows = f.rows(), .n = lanes.len, .k = 1 }};
        const kernel: gates.Kernel = .{ .ctx = &st, .vt = &State.vt, .spec = .{
            .name = "cuda Flash Next attention, a shared round's lanes in one launch a kernel",
            .accum = .none,
            .depth = depth,
            .out = .raw,
            .replaces = "the same programs launched a lane at a time",
            .nan = .{ .stated = "all outputs and caches match the per-lane kernels, including NaN and both infinities" },
            .served = &served,
            .tiles = .{ .{ .size = 16, .step = 1, .min = 2 }, .{}, .{} },
        } };
        for ([_]gates.Gate{ .nonfinite, .repeat, .shapes }) |gate| {
            st.gate = gate;
            try check.expect(try gates.one(gpu.gpa, gate, kernel, 0xa700 + ci) == .pass, "Flash Next attention lanes gate {t} failed for case {d}", .{ gate, ci });
        }
        std.debug.print("PASS flashnext attention lanes: {d} lanes, {d} rows: one launch == per-lane (queries, rows, scores, selection, caches)\n", .{ lanes.len, f.rows() });
    }
    try check.expect(red, "flashnext attention lanes: a crossed lane table went unseen (control missed)", .{});
    std.debug.print("PASS flashnext attention lanes: control RED as planted (crossed lane table)\n", .{});
}

test "gate shapes keep two lanes and at most sixteen rows a lane" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    for ([_]usize{ 2, 15, 16, 17, 32, 33, 64, 256 }) |rows| {
        const lanes = try split(arena.allocator(), &.{.{ .len = 5, .rows = 1 }}, rows);
        var total: usize = 0;
        try std.testing.expect(lanes.len >= 2);
        for (lanes) |l| {
            try std.testing.expect(l.rows >= 1 and l.rows <= 16);
            total += l.rows;
        }
        try std.testing.expectEqual(rows, total);
    }
}

test "the lane table holds each row's lane and each lane's caches at their kernel offsets" {
    const tab = try std.testing.allocator.alignedAlloc(u8, .@"16", flash.batch_state.lanes_bytes);
    defer std.testing.allocator.free(tab);
    @memset(tab, 0);
    table.putCaches(tab, 1, 2, 11, 22, 33, 44);
    table.putLane(tab, 2, 5, 100, 3);
    const ptrs: []const u64 = @alignCast(std.mem.bytesAsSlice(u64, tab[0..flash.batch_state.lanes_ptrs]));
    const ints: []const i32 = @alignCast(std.mem.bytesAsSlice(i32, tab[flash.batch_state.lanes_ptrs..]));
    try std.testing.expectEqualSlices(u64, &.{ 11, 22, 33, 44 }, ptrs[(max_lanes + 2) * 4 ..][0..4]);
    try std.testing.expectEqual(@as(i32, 95), ints[2]);
    try std.testing.expectEqual(@as(i32, 100), ints[max_lanes + 2]);
    try std.testing.expectEqual(@as(i32, 3), ints[2 * max_lanes + 2]);
    try std.testing.expectEqual(@as(i32, 5), ints[3 * max_lanes + 2]);
    for (5..8) |r| try std.testing.expectEqual(@as(i32, 2), ints[4 * max_lanes + r]);
}
