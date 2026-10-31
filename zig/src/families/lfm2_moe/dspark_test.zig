//! The DSpark drafter's kernels on the GPU against host references: taps, the Markov step, half-split RoPE.
const std = @import("std");
const mtl = @import("metal");
const c = @import("config.zig");
const kernels = @import("kernels.zig");
const quant = @import("dspark_quant.zig");
const block = @import("dspark_block.zig");
const Pick = @import("tree_round_gpu").contract.Pick;
const ds = @import("dspark.zig");
const lanes = @import("lanes");

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
test "lfm2 dspark: a settled hold moves the timed passes at its block counts a tenth of the way to its GPU ms" {
    var d: ds.Drafter = undefined;
    const grid = [_]lanes.hold_price.HoldCost{ .{ .streams = 1, .depth = 1, .ms = 1 }, .{ .streams = 4, .depth = 1, .ms = 2 }, .{ .streams = 4, .depth = 4, .ms = 4 }, .{ .streams = 8, .depth = 1, .ms = 3 } };
    @memcpy(d.holds[0..grid.len], &grid);
    d.holds_len = grid.len;
    d.pass_len = 4;
    d.depths[0..4].* = .{ 1, 1, 1, 1 };
    d.priceHold(4); // priced 2 ms: the 4-block passes rise by a tenth of the doubling
    try std.testing.expectApproxEqAbs(@as(f64, 2.2), d.holds[1].ms, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.4), d.holds[2].ms, 1e-12);
    try std.testing.expectEqual(@as(f64, 1), d.holds[0].ms);
    try std.testing.expectEqual(@as(f64, 3), d.holds[3].ms);
    d.pass_len = 6; // between 4 and 8 blocks: both move
    d.depths[0..6].* = .{ 1, 1, 1, 1, 1, 1 };
    d.priceHold(0);
    try std.testing.expectEqual(@as(f64, 2.2), d.holds[1].ms); // no GPU time: nothing learned
    d.priceHold(2.6 * 2);
    try std.testing.expect(d.holds[1].ms > 2.2 and d.holds[3].ms > 3 and d.holds[0].ms == 1);
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
