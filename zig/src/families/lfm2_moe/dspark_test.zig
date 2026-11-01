//! The DSpark drafter's kernels on the GPU against host references: taps, the Markov step, half-split RoPE.
const std = @import("std");
const mtl = @import("metal");
const c = @import("config.zig");
const kernels = @import("kernels.zig");
const quant = @import("dspark_quant.zig");
const block = @import("dspark_block.zig");
const Pick = @import("tree_round_gpu").contract.Pick;
const ds = @import("dspark.zig");
const dc = @import("dspark_config.zig");
const lanes = @import("lanes");

const Model = @import("model.zig").Model;

const shared = mtl.ResourceOptions.shared;

pub const Rig = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    k: kernels.Kernels,

    pub fn init() !Rig {
        const device = mtl.Device.init() catch return error.SkipZigTest;
        errdefer device.deinit();
        const queue = try device.queue();
        errdefer queue.deinit();
        return .{ .device = device, .queue = queue, .k = try kernels.load(device) };
    }

    pub fn deinit(r: *Rig) void {
        r.k.deinit();
        r.queue.deinit();
        r.device.deinit();
    }

    pub fn buffer(r: *Rig, comptime T: type, n: usize) !mtl.Buffer {
        const b = try r.device.buffer(n * @sizeOf(T), shared);
        @memset(b.slice(u8, n * @sizeOf(T)), 0);
        return b;
    }

    fn run(r: *Rig, comptime key: []const u8, bufs: []const mtl.Buffer, value: anytype, at: usize, grid: [2]usize) !void {
        const cb = r.queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(r.k.get(key));
        for (bufs, 0..) |b, i| e.setBuffer(b, 0, if (i >= at) i + 1 else i);
        e.setValue(value, at);
        e.dispatchThreads(mtl.Size.of(grid[0], grid[1], 1), mtl.Size.of(@min(grid[0], 256), 1, 1));
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |_| return error.GpuFailed;
    }
};

pub fn noise(seed: u64, out: []u16, scale: f32) void {
    var prng = std.Random.DefaultPrng.init(seed);
    for (out) |*v| v.* = quant.toBf16((prng.random().float(f32) * 2 - 1) * scale);
}

test "lfm2 dspark: the tap kernel copies each row's hidden values into its tap's columns, the rest untouched" {
    var r = try Rig.init();
    defer r.deinit();
    const rows = 5;
    const taps = 3;
    const h = try r.buffer(u16, rows * c.hidden);
    defer h.deinit();
    const out = try r.buffer(u16, rows * taps * c.hidden);
    defer out.deinit();
    noise(7, h.slice(u16, rows * c.hidden), 4);
    try r.run("lfm2_tap", &.{ h, out }, [2]u32{ taps * c.hidden, c.hidden }, 2, .{ c.hidden / 8, rows });
    const want = h.slice(u16, rows * c.hidden);
    const got = out.slice(u16, rows * taps * c.hidden);
    for (0..rows) |row| {
        try std.testing.expectEqualSlices(u16, want[row * c.hidden ..][0..c.hidden], got[(row * taps + 1) * c.hidden ..][0..c.hidden]);
        for (got[row * taps * c.hidden ..][0..c.hidden]) |v| try std.testing.expectEqual(@as(u16, 0), v);
        for (got[(row * taps + 2) * c.hidden ..][0..c.hidden]) |v| try std.testing.expectEqual(@as(u16, 0), v);
    }
}

test "lfm2 dspark: a Markov step picks the argmax of bf16(base + bf16(W2 . W1[prev])) over every threadgroup" {
    var r = try Rig.init();
    defer r.deinit();
    const rank = 256;
    const vocab = 1300; // three scan threadgroups, the last one short
    const streams = 4;
    const groups = (vocab + 511) / 512;
    const w1 = try r.buffer(u16, vocab * rank);
    defer w1.deinit();
    noise(11, w1.slice(u16, vocab * rank), 1);
    const codes = try r.buffer(u8, vocab * rank);
    defer codes.deinit();
    for (codes.slice(u8, vocab * rank), 0..) |*q, i| q.* = @truncate(i *% 2654435761 >> 7);
    const sc = try r.buffer(u16, vocab * rank / 64);
    defer sc.deinit();
    const bi = try r.buffer(u16, vocab * rank / 64);
    defer bi.deinit();
    noise(12, sc.slice(u16, vocab * rank / 64), 0.01);
    noise(13, bi.slice(u16, vocab * rank / 64), 0.5);
    // a token every second word (the picks' layout): inside W1, its last row, past it (a zero row), and a NaN row
    const ids = try r.buffer(u32, 2 * streams);
    defer ids.deinit();
    @memcpy(ids.slice(u32, 2 * streams), &[_]u32{ 17, 9, vocab - 1, 9, vocab + 5, 0, 3, 3 });
    const logits = try r.buffer(u16, streams * 2 * vocab);
    defer logits.deinit();
    noise(14, logits.slice(u16, streams * 2 * vocab), 4);
    logits.slice(u16, streams * 2 * vocab)[3 * 2 * vocab + 700] = 0x7fc0;
    const best = try r.buffer(f32, streams * groups);
    defer best.deinit();
    const at = try r.buffer(u32, streams * groups);
    defer at.deinit();
    const picks = try r.buffer(u32, 2 * streams);
    defer picks.deinit();
    const cb = r.queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(r.k.get("lfm2_markov_scan"));
    for ([_]mtl.Buffer{ logits, w1, codes, ids }, 0..) |b, i| e.setBuffer(b, 0, i);
    e.setValue([4]u32{ 2 * vocab, rank, vocab, 2 }, 4);
    for ([_]mtl.Buffer{ best, at, sc, bi }, 5..) |b, i| e.setBuffer(b, 0, i);
    e.dispatchGroups(mtl.Size.of(groups, streams, 1), mtl.Size.of(256, 1, 1));
    e.setPipeline(r.k.get("lfm2_markov_pick"));
    e.setBuffer(best, 0, 0);
    e.setBuffer(at, 0, 1);
    e.setValue(@as(u32, groups), 2);
    e.setBuffer(picks, 0, 3);
    e.dispatchGroups(mtl.Size.of(streams, 1, 1), mtl.Size.of(256, 1, 1));
    e.end();
    cb.commit();
    cb.wait();
    const got = picks.slice(u32, 2 * streams);
    const base = logits.slice(u16, streams * 2 * vocab);
    const id_list = ids.slice(u32, 2 * streams);
    for (0..streams) |s| {
        if (s == 3) { // the planted NaN: flagged
            try std.testing.expectEqual(@as(u32, 1), got[2 * s + 1]);
            continue;
        }
        try std.testing.expectEqual(@as(u32, 0), got[2 * s + 1]);
        const id = id_list[2 * s];
        var top: f64 = -std.math.inf(f64);
        var mine: f64 = 0;
        for (0..vocab) |v| {
            var dot: f64 = 0;
            for (0..rank) |i| {
                const g = (v * rank + i) / 64;
                const w = @as(f64, quant.toF32(sc.slice(u16, vocab * rank / 64)[g])) * @as(f64, @floatFromInt(codes.slice(u8, vocab * rank)[v * rank + i])) + quant.toF32(bi.slice(u16, vocab * rank / 64)[g]);
                dot += w * (if (id < vocab) quant.toF32(w1.slice(u16, vocab * rank)[id * rank + i]) else 0);
            }
            const score: f64 = quant.toF32(quant.toBf16(quant.toF32(base[s * 2 * vocab + v]) + quant.toF32(quant.toBf16(@floatCast(dot)))));
            top = @max(top, score);
            if (v == got[2 * s]) mine = score;
        }
        // the pick is the best score, or within two bf16 steps of it where fp32 sums round apart
        try std.testing.expect(got[2 * s] < vocab and mine >= top - @abs(top) * 0x1p-7);
    }
}

test "lfm2 dspark: lfm2_qk on half-split rows and gains gives interleaved RoPE's keys, reordered" {
    var r = try Rig.init();
    defer r.deinit();
    const hd = c.head_dim;
    const pos: u32 = 777;
    const theta: f32 = 1e7;
    var k_il: [c.kv_heads * hd]u16 = undefined;
    var gain_il: [hd]u16 = undefined;
    noise(21, &k_il, 2);
    noise(22, &gain_il, 1.5);
    const q = try r.buffer(u16, c.query_heads * hd);
    defer q.deinit();
    const k = try r.buffer(u16, c.kv_heads * hd);
    defer k.deinit();
    const v = try r.buffer(u16, c.kv_heads * hd);
    defer v.deinit();
    const qn = try r.buffer(u16, hd);
    defer qn.deinit();
    const kn = try r.buffer(u16, hd);
    defer kn.deinit();
    for (k.slice(u16, c.kv_heads * hd), 0..) |*x, i| x.* = k_il[quant.halfSplit(i, hd)];
    for (kn.slice(u16, hd), 0..) |*x, i| x.* = gain_il[quant.halfSplit(i, hd)];
    @memset(qn.slice(u16, hd), quant.toBf16(1));
    const cap = 1024;
    const queries = try r.buffer(u16, c.query_heads * hd);
    defer queries.deinit();
    const keys = try r.buffer(u16, c.kv_heads * cap * hd);
    defer keys.deinit();
    const values = try r.buffer(u16, c.kv_heads * cap * hd);
    defer values.deinit();
    const Args = extern struct { len: u32, cap: u32, eps: f32, log2_theta: f32 };
    const cb = r.queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(r.k.get("lfm2_qk"));
    for ([_]mtl.Buffer{ q, k, v, qn, kn }, 0..) |b, i| e.setBuffer(b, 0, i);
    const args = Args{ .len = pos, .cap = cap, .eps = c.eps, .log2_theta = std.math.log2(theta) };
    e.setBytes(std.mem.asBytes(&args), 5);
    for ([_]mtl.Buffer{ queries, keys, values }, 6..) |b, i| e.setBuffer(b, 0, i);
    e.dispatchGroups(mtl.Size.of(c.query_heads + c.kv_heads, 1, 1), mtl.Size.of(32, 1, 1));
    e.end();
    cb.commit();
    cb.wait();
    const got = keys.slice(u16, c.kv_heads * cap * hd);
    for (0..c.kv_heads) |h| {
        var ss: f32 = 0;
        for (0..hd) |i| ss += quant.toF32(k_il[h * hd + i]) * quant.toF32(k_il[h * hd + i]);
        const inv = 1 / @sqrt(ss / hd + c.eps);
        for (0..hd / 2) |i| {
            const angle = @as(f32, pos) * std.math.pow(f32, theta, -@as(f32, @floatFromInt(i)) / (hd / 2));
            const a = quant.toF32(k_il[h * hd + 2 * i]) * inv * quant.toF32(gain_il[2 * i]);
            const b = quant.toF32(k_il[h * hd + 2 * i + 1]) * inv * quant.toF32(gain_il[2 * i + 1]);
            const want = [2]f32{ a * @cos(angle) - b * @sin(angle), a * @sin(angle) + b * @cos(angle) };
            const at = (h * cap + pos) * hd;
            const have = [2]f32{ quant.toF32(got[at + i]), quant.toF32(got[at + i + hd / 2]) };
            for (want, have) |w, x| try std.testing.expect(@abs(w - x) <= 0.02 * @max(1, @abs(w)));
        }
    }
}

/// lfm2_draft_ids on one pick (token, nonfinite flag) at a stride of 4 words: the id it wrote and the bad flag.
fn draftIds(r: *Rig, pick: Pick) ![2]u32 {
    const picks = try r.buffer(u32, 4);
    defer picks.deinit();
    const ids = try r.buffer(u32, 1);
    defer ids.deinit();
    const bad = try r.buffer(u32, 1);
    defer bad.deinit();
    picks.slice(u32, 4)[0] = pick.token;
    picks.slice(u32, 4)[1] = pick.nonfinite;
    try r.run("lfm2_draft_ids", &.{ picks, ids, bad }, [4]u32{ 4, 1, c.vocab, 0 }, 1, .{ 1, 1 });
    return .{ ids.slice(u32, 1)[0], bad.slice(u32, 1)[0] };
}

test "lfm2 dspark: held ids carry a flagged or invalid pick as token 0 and the bad flag, a good pick untouched" {
    var r = try Rig.init();
    defer r.deinit();
    try std.testing.expectEqual([2]u32{ 5, 0 }, try draftIds(&r, .{ .token = 5 }));
    try std.testing.expectEqual([2]u32{ c.vocab - 1, 0 }, try draftIds(&r, .{ .token = c.vocab - 1 }));
    for ([_]u32{ 1, 2, 4, 7 }) |kind| try std.testing.expectEqual([2]u32{ 0, 1 }, try draftIds(&r, .{ .token = 9, .nonfinite = kind }));
    try std.testing.expectEqual([2]u32{ 0, 1 }, try draftIds(&r, .{ .token = std.math.maxInt(u32) }));
    try std.testing.expectEqual([2]u32{ 0, 1 }, try draftIds(&r, .{ .token = c.vocab }));
}

test "lfm2 dspark: a settled hold refuses a NaN, infinity or invalid pick as the synchronous path does" {
    var picks: [4 * ds.pass_rows]Pick = @splat(.{ .token = 3 });
    var a: [4]u32 = undefined;
    var b: [4]u32 = undefined;
    const chains = [_][]u32{ &a, &b };
    try block.copyPicks(&picks, &chains, &.{ 2, 2 });
    try std.testing.expectEqualSlices(u32, &.{ 3, 3 }, a[0..2]);
    const bad = [_]Pick{ .{ .token = 3, .nonfinite = 1 }, .{ .token = 3, .nonfinite = 2 }, .{ .token = 3, .nonfinite = 4 }, .{ .token = std.math.maxInt(u32) }, .{ .token = c.vocab } };
    for (bad) |p| {
        picks[ds.pass_rows + 1] = p; // the second block's second step
        try std.testing.expectError(error.DraftNonfinite, block.copyPicks(&picks, &chains, &.{ 2, 2 }));
        picks[ds.pass_rows + 1] = .{ .token = 3 };
    }
    picks[3 * ds.pass_rows] = .{ .nonfinite = 1 }; // a step past the block's depth is not read
    try block.copyPicks(&picks, &chains, &.{ 2, 2 });
}

/// Project pieces separated by NaN sentinels; a stray gather copy must change the output.
fn fcOnce(rig: *Rig, d: *ds.Drafter, src: mtl.Buffer, sizes: []const usize, gather: ?*usize, out: []u16) !void {
    var pieces: [40]block.Piece = undefined;
    var row: usize = 0;
    for (sizes, pieces[0..sizes.len]) |n, *pc| {
        pc.* = .{ .lane = undefined, .src = .{ .buf = src, .off = row * d.tapBytes() }, .rows = n, .p0 = 0 };
        row += n + 1;
    }
    @memset(d.s.fcx.slice(u16, out.len), 0x5a5a);
    const cb = rig.queue.commandBuffer();
    const e = cb.compute(if (gather == null) .serial else .concurrent);
    const rows = try block.fcRows(d, e, gather != null, pieces[0..sizes.len], gather);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |_| return error.GpuFailed;
    try std.testing.expectEqual(out.len / c.hidden, rows);
    @memcpy(out, d.s.fcx.slice(u16, out.len));
}

test "lfm2 dspark: small staged pieces sharing one fc call give the bytes each gives alone, at the boundaries of a call's rows" {
    var rig = try Rig.init();
    defer rig.deinit();
    var target: Model = undefined;
    target.device = rig.device;
    target.kernels = rig.k;
    var d: ds.Drafter = undefined;
    d.target = &target;
    d.config = try dc.Config.parse(std.testing.allocator, dc.stock);
    const k = d.config.tap_count * c.hidden;
    const most = 200; // source rows: 40 pieces of four rows, each followed by a NaN row
    d.s.sums = try rig.buffer(f32, most * (k / 64));
    defer d.s.sums.deinit();
    d.s.fcx = try rig.buffer(u16, most * c.hidden);
    defer d.s.fcx.deinit();
    d.s.gather = try rig.buffer(u16, ds.gather_rows * k);
    defer d.s.gather.deinit();
    const src = try rig.buffer(u16, most * k);
    defer src.deinit();
    const w16 = try rig.buffer(u16, c.hidden * k);
    defer w16.deinit();
    noise(41, w16.slice(u16, c.hidden * k), 0.5);
    const wq = try rig.buffer(u8, c.hidden * k);
    defer wq.deinit();
    var prng = std.Random.DefaultPrng.init(42);
    prng.random().bytes(wq.slice(u8, c.hidden * k));
    const scales = try rig.buffer(u16, c.hidden * (k / 64));
    defer scales.deinit();
    noise(43, scales.slice(u16, c.hidden * (k / 64)), 0.02);
    const biases = try rig.buffer(u16, c.hidden * (k / 64));
    defer biases.deinit();
    noise(44, biases.slice(u16, c.hidden * (k / 64)), 0.1);
    // the pieces' rows, and the rows of `gather` the pieces that share a call take (a call takes 32 rows, gather 128)
    const cases = [_]struct { sizes: []const usize, used: usize, scalar_used: usize }{
        .{ .sizes = &.{ 1, 1, 1, 1, 1, 1, 1, 1 }, .used = 8, .scalar_used = 8 },
        .{ .sizes = &.{ 1, 2, 8, 9, 16, 32 }, .used = 20, .scalar_used = 3 },
        .{ .sizes = &.{ 16, 16, 1 }, .used = 32, .scalar_used = 0 },
        .{ .sizes = &.{ 33, 1, 1, 40 }, .used = 2, .scalar_used = 2 },
        .{ .sizes = &.{ 32, 1 }, .used = 0, .scalar_used = 0 },
        .{ .sizes = &.{ 31, 1, 1 }, .used = 32, .scalar_used = 2 },
        .{ .sizes = &.{ 5, 5, 5, 5, 5, 5, 5, 5 }, .used = 40, .scalar_used = 0 },
        .{ .sizes = &@as([24]usize, @splat(6)), .used = 120, .scalar_used = 0 },
        .{ .sizes = &@as([40]usize, @splat(4)), .used = 128, .scalar_used = 128 },
    };
    for ([_]u8{ 0, 1, 2 }) |mode| { // bf16; 8-bit with the tiles; 8-bit with the twin
        target.kernels.twin = mode == 2;
        const scalar = mode == 2 and target.kernels.naxPipes(8) == null;
        d.w.fc = if (mode == 0)
            .{ .bf16 = .{ .w = w16, .n = c.hidden, .k = k } }
        else
            .{ .codes = .{ .w = wq, .scales = scales, .biases = biases, .n = c.hidden, .k = k, .bits = 8 } };
        for (cases) |cs| for ([_]bool{ false, true }) |bad| {
            const all = src.slice(u16, most * k);
            noise(45, all, 2);
            var first: [40]usize = undefined;
            var row: usize = 0;
            var total: usize = 0;
            for (cs.sizes, first[0..cs.sizes.len]) |n, *f| {
                f.* = row;
                @memset(all[(row + n) * k ..][0..k], 0x7fc0); // the NaN row after the piece
                row += n + 1;
                total += n;
            }
            const last = cs.sizes.len - 1;
            if (bad) { // a NaN, a -inf and a +inf among the rows
                all[first[0] * k + 3] = 0x7fc0;
                all[first[cs.sizes.len / 2] * k + 17] = 0xff80;
                all[(first[last] + cs.sizes[last] - 1) * k + k - 1] = 0x7f80;
            }
            const want = try std.testing.allocator.alloc(u16, total * c.hidden);
            defer std.testing.allocator.free(want);
            const got = try std.testing.allocator.alloc(u16, total * c.hidden);
            defer std.testing.allocator.free(got);
            try fcOnce(&rig, &d, src, cs.sizes, null, want);
            var used: usize = 0;
            try fcOnce(&rig, &d, src, cs.sizes, &used, got);
            try std.testing.expectEqual(if (scalar) cs.scalar_used else cs.used, used);
            try std.testing.expectEqualSlices(u16, want, got);
            var nonfinite = false;
            var nonzero = false;
            for (want) |v| {
                nonfinite = nonfinite or !std.math.isFinite(quant.toF32(v));
                nonzero = nonzero or v != 0;
            }
            try std.testing.expect(nonzero);
            try std.testing.expectEqual(bad, nonfinite);
        };
    }
}

test "lfm2 dspark: a completed hold moves the timed passes at its block counts a tenth of the way to its GPU ms" {
    var d: ds.Drafter = undefined;
    const grid = [_]lanes.hold_price.HoldCost{ .{ .streams = 1, .depth = 1, .ms = 1 }, .{ .streams = 4, .depth = 1, .ms = 2 }, .{ .streams = 4, .depth = 4, .ms = 4 }, .{ .streams = 8, .depth = 1, .ms = 3 } };
    @memcpy(d.holds[0..grid.len], &grid);
    d.holds_len = grid.len;
    d.priceHold(4, 1, 4); // priced 2 ms: the 4-block passes rise by a tenth of the doubling
    try std.testing.expectApproxEqAbs(@as(f64, 2.2), d.holds[1].ms, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.4), d.holds[2].ms, 1e-12);
    try std.testing.expectEqual(@as(f64, 1), d.holds[0].ms);
    try std.testing.expectEqual(@as(f64, 3), d.holds[3].ms);
    d.priceHold(6, 1, 0); // between 4 and 8 blocks, no GPU time: nothing learned
    try std.testing.expectEqual(@as(f64, 2.2), d.holds[1].ms);
    d.priceHold(6, 1, 2.6 * 2); // both neighbors move
    try std.testing.expect(d.holds[1].ms > 2.2 and d.holds[3].ms > 3 and d.holds[0].ms == 1);
    const before = d.holds[1].ms;
    d.priceHold(0, 1, 5); // a pass of no blocks prices nothing
    try std.testing.expectEqual(before, d.holds[1].ms);
    d.holds_len = 0; // no grid (the load's timing holds): nothing to move
    d.priceHold(4, 1, 100);
    try std.testing.expectEqual(before, d.holds[1].ms);
}

/// A drafter with a device and a config but no weights: absorbs that only stage never reach the GPU.
fn bare(target: *Model) !ds.Drafter {
    return .{
        .gpa = std.testing.allocator,
        .io = undefined,
        .target = target,
        .config = try dc.Config.parse(std.testing.allocator, dc.stock),
        .w = undefined,
        .options = .{ .capacity = 64 },
        .s = undefined,
    };
}

/// `rows` rows of `width` bytes that differ by position and `salt`.
fn pattern(buf: mtl.Buffer, rows: usize, width: usize, salt: u8) void {
    for (buf.slice(u8, rows * width), 0..) |*b, i| b.* = @as(u8, @truncate((i *% 2654435761) >> 7)) +% salt;
}

/// An absorb of `rows` rows of the target's taps at `row` of `src`, the context from position `start`.
fn tapped(key: *lanes.Stream, src: mtl.Buffer, width: usize, row: usize, rows: u32, start: u64, prompt: bool) lanes.drafter.Absorb {
    return .{
        .stream = key,
        .features = .{ .buffer = @intFromPtr(src.id), .offset = row * width, .rows = rows, .row_bytes = @intCast(width), .space = .device, .dtype = .bf16 },
        .start = start,
        .follow = &.{},
        .prompt = prompt,
    };
}

test "lfm2 dspark: an absorb owns the rows it stages, prompt rows too, and a refused call stages none of its items" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    var target: Model = undefined;
    target.device = device;
    var d = try bare(&target);
    defer d.lanes.deinit(d.gpa);
    const width = d.tapBytes();
    const l = try d.newLane(64); // 73 positions: at most 73 staged rows
    const stage = l.mem.contents() + l.stage_at;
    defer d.dropLane(l);
    const key: *lanes.Stream = @ptrFromInt(0x1000);
    try d.lanes.put(d.gpa, key, l);
    const src = try device.buffer(80 * width, shared);
    defer src.deinit();
    pattern(src, 80, width, 0);
    const first = try std.testing.allocator.dupe(u8, src.contents()[0 .. 80 * width]);
    defer std.testing.allocator.free(first);
    const drafter = d.drafter();
    try drafter.absorb(&.{tapped(key, src, width, 0, 10, 0, true)}); // the prompt's rows
    try drafter.absorb(&.{tapped(key, src, width, 10, 20, 10, false)}); // a round's verified rows
    try std.testing.expectEqual(@as(u32, 30), l.staged);
    try std.testing.expectEqual(@as(u64, 0), l.end);
    @memset(src.contents()[0 .. 80 * width], 0xee); // the target's next forward overwrites its taps
    try std.testing.expectEqualSlices(u8, first[0 .. 30 * width], stage[0 .. 30 * width]);
    pattern(src, 80, width, 1);
    const second = try std.testing.allocator.dupe(u8, src.contents()[0 .. 80 * width]);
    defer std.testing.allocator.free(second);
    try drafter.absorb(&.{tapped(key, src, width, 40, 4, 25, false)}); // a rewrite: rows from 25 are replaced
    try std.testing.expectEqual(@as(u32, 29), l.staged);
    try std.testing.expectEqualSlices(u8, first[0 .. 25 * width], stage[0 .. 25 * width]);
    try std.testing.expectEqualSlices(u8, second[40 * width .. 44 * width], stage[25 * width .. 29 * width]);
    const stranger: *lanes.Stream = @ptrFromInt(0x2000);
    try std.testing.expectError(error.DraftStreamUnknown, drafter.absorb(&.{ tapped(key, src, width, 0, 2, 29, false), tapped(stranger, src, width, 0, 2, 0, false) }));
    try std.testing.expectError(error.DraftContextGap, drafter.absorb(&.{ tapped(key, src, width, 0, 2, 29, false), tapped(key, src, width, 0, 60, 29, false) }));
    try std.testing.expectEqual(@as(u32, 29), l.staged);
    try std.testing.expectEqualSlices(u8, second[40 * width .. 44 * width], stage[25 * width .. 29 * width]);
    try std.testing.expectEqual(@as(u64, 0), l.end);
    try std.testing.expectEqual(@as(f64, 0), d.hold_spent_ms); // staging runs no GPU work
}

test "lfm2 dspark: packed lane accounting, alignment and a whole prompt chunk" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    var target: Model = undefined;
    target.device = device;
    var d = try bare(&target);
    defer d.lanes.deinit(d.gpa);
    const width = d.tapBytes();
    for ([_]usize{ 1, 64, 2039, 4096 }) |capacity| {
        const l = try d.newLane(capacity);
        defer d.dropLane(l);
        try std.testing.expectEqual(d.laneBytes(capacity), l.mem.length());
        try std.testing.expectEqual(@as(usize, @intCast(d.keptBytes(l.cap))), l.stage_at);
        try std.testing.expectEqual(@min(ds.stage_rows, l.cap) * width, l.mem.length() - l.stage_at);
        try std.testing.expectEqual(@as(usize, 0), l.stage_at % 64);
        for (0..d.config.layers) |i| for ([_]bool{ false, true }) |values| {
            const kv = l.kv(i, values);
            const off = (2 * i + @intFromBool(values)) * l.cap * c.kv_heads * c.head_dim * 2;
            try std.testing.expectEqual(l.mem.id, kv.buf.id);
            try std.testing.expectEqual(off, kv.off);
            try std.testing.expectEqual(@as(usize, 0), kv.off % 64);
            try std.testing.expect(kv.off + l.cap * c.kv_heads * c.head_dim * 2 <= l.stage_at);
        };
    }
    const l = try d.newLane(4096);
    defer d.dropLane(l);
    const key: *lanes.Stream = @ptrFromInt(0x1000);
    try d.lanes.put(d.gpa, key, l);
    const chunk = try device.buffer(ds.absorb_rows * width, shared);
    defer chunk.deinit();
    pattern(chunk, ds.absorb_rows, width, 7);
    try d.drafter().absorb(&.{tapped(key, chunk, width, 0, ds.absorb_rows, 0, true)});
    try std.testing.expectEqual(@as(u32, ds.absorb_rows), l.staged);
    try std.testing.expectEqualSlices(u8, chunk.contents()[0 .. ds.absorb_rows * width], (l.mem.contents() + l.stage_at)[0 .. ds.absorb_rows * width]);
}

test "lfm2 dspark: failed lane creation and map insertion clean up" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    var target: Model = undefined;
    target.device = device;
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    var d = try bare(&target);
    d.gpa = failing.allocator();
    defer d.lanes.deinit(d.gpa);
    try std.testing.expectError(error.OutOfMemory, d.newLane(64));
    try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
    failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 1 });
    var stream = try lanes.Stream.init(std.testing.allocator, .{ .id = "failed-open", .prompt = &.{1}, .max_new = 1 });
    defer stream.deinit(std.testing.allocator);
    try std.testing.expectError(error.OutOfMemory, d.drafter().open(&stream));
    try std.testing.expectEqual(@as(usize, 0), d.lanes.count());
    try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
}
