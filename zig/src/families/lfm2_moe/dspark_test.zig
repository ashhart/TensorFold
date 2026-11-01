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
const dc = @import("dspark_config.zig");
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

test "lfm2 dspark: fused normalization taps match the separate copy bit for bit" {
    var rig = try Rig.init();
    defer rig.deinit();
    for ([_]usize{ 1, 16, 32, 2048 }) |rows| for ([_]bool{ false, true }) |combine| for ([_]bool{ false, true }) |nonfinite| {
        const count = rows * c.hidden;
        const h0 = try rig.buffer(u16, count);
        defer h0.deinit();
        const h1 = try rig.buffer(u16, count);
        defer h1.deinit();
        const residual = try rig.buffer(u16, if (combine) count * c.topk else count);
        defer residual.deinit();
        const router = try rig.buffer(u16, rows * c.topk);
        defer router.deinit();
        const weight = try rig.buffer(u16, c.hidden);
        defer weight.deinit();
        const x0 = try rig.buffer(u16, count);
        defer x0.deinit();
        const x1 = try rig.buffer(u16, count);
        defer x1.deinit();
        const taps = try rig.buffer(u16, count * 3);
        defer taps.deinit();
        const reference_taps = try rig.buffer(u16, count * 3);
        defer reference_taps.deinit();
        noise(30, h0.slice(u16, count), 2);
        const initial = h0.slice(u16, count);
        if (nonfinite) {
            initial[0] = 0x7fc0;
            initial[1] = 0x7f80;
            initial[2] = 0xff80;
        }
        @memcpy(h1.slice(u16, count), initial);
        noise(31, residual.slice(u16, if (combine) count * c.topk else count), 0.2);
        noise(32, router.slice(u16, rows * c.topk), 0.2);
        noise(33, weight.slice(u16, c.hidden), 1);
        @memset(taps.slice(u16, count * 3), 0x7fc0);
        @memset(reference_taps.slice(u16, count * 3), 0x7fc0);
        for ([_]bool{ false, true }) |fused| {
            const cb = rig.queue.commandBuffer();
            const e = cb.compute(.serial);
            if (combine) {
                if (fused) e.setPipeline(rig.k.get("lfm2_combine_norm_tap")) else e.setPipeline(rig.k.get("lfm2_combine_norm"));
                e.setBuffer(residual, 0, 0);
                e.setBuffer(router, 0, 1);
                e.setBuffer(if (fused) h1 else h0, 0, 2);
                e.setBuffer(weight, 0, 3);
                e.setValue(c.eps, 4);
            } else {
                if (fused) e.setPipeline(rig.k.get("lfm2_norm_tap")) else e.setPipeline(rig.k.get("lfm2_norm"));
                e.setBuffer(if (fused) h1 else h0, 0, 0);
                e.setBuffer(residual, 0, 1);
                e.setBuffer(weight, 0, 2);
                e.setValue(c.eps, 3);
                e.setValue(@as(u32, 1), 4);
            }
            e.setBuffer(if (fused) x1 else x0, 0, 5);
            if (fused) {
                e.setBuffer(taps, 0, 6);
                e.setValue([2]u32{ 3 * c.hidden, c.hidden }, 7);
            }
            e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(c.hidden / 4, 1, 1));
            if (!fused) {
                e.setPipeline(rig.k.get("lfm2_tap"));
                e.setBuffer(h0, 0, 0);
                e.setBuffer(reference_taps, 0, 1);
                e.setValue([2]u32{ 3 * c.hidden, c.hidden }, 2);
                e.dispatchThreads(mtl.Size.of(c.hidden / 8, rows, 1), mtl.Size.of(256, 1, 1));
            }
            e.end();
            cb.commit();
            cb.wait();
            defer cb.release();
            if (cb.failure()) |_| return error.GpuFailed;
        }
        try std.testing.expectEqualSlices(u16, h0.slice(u16, count), h1.slice(u16, count));
        try std.testing.expectEqualSlices(u16, x0.slice(u16, count), x1.slice(u16, count));
        const out = taps.slice(u16, count * 3);
        try std.testing.expectEqualSlices(u16, reference_taps.slice(u16, count * 3), out);
        for (0..rows) |row| {
            try std.testing.expectEqualSlices(u16, h0.slice(u16, count)[row * c.hidden ..][0..c.hidden], out[(3 * row + 1) * c.hidden ..][0..c.hidden]);
            for (out[(3 * row) * c.hidden ..][0..c.hidden]) |v| try std.testing.expectEqual(@as(u16, 0x7fc0), v);
            for (out[(3 * row + 2) * c.hidden ..][0..c.hidden]) |v| try std.testing.expectEqual(@as(u16, 0x7fc0), v);
        }
    };
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
