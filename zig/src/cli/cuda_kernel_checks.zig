//! Our replacement kernels against the ones they replace, on the checkpoint's own weights: every byte must match.

const std = @import("std");
const nemotron = @import("nemotron");
const kern = nemotron.kernels;

/// Deterministic bf16 activations in roughly [-2, 2), with their fp32 64-group sums as a norm would write them.
fn inputs(gpa: std.mem.Allocator, rng: std.Random, rows: usize, k: usize) !struct { x: []u16, xs: []f32 } {
    const x = try gpa.alloc(u16, rows * k);
    errdefer gpa.free(x);
    const xs = try gpa.alloc(f32, rows * (k / 64));
    for (x) |*v| {
        const f: f32 = (rng.float(f32) - 0.5) * 4.0;
        v.* = @truncate(@as(u32, @bitCast(f)) >> 16);
    }
    for (0..rows) |r| for (0..k / 64) |g| {
        var s: f32 = 0;
        for (x[r * k + g * 64 ..][0..64]) |v| s += @bitCast(@as(u32, v) << 16);
        xs[r * (k / 64) + g] = s;
    };
    return .{ .x = x, .xs = xs };
}

fn sameBytes(gpa: std.mem.Allocator, e: *nemotron.Engine, a: u64, b: u64, len: usize) !bool {
    const ha = try gpa.alloc(u8, len);
    defer gpa.free(ha);
    const hb = try gpa.alloc(u8, len);
    defer gpa.free(hb);
    try e.ops().download(ha, a);
    try e.ops().download(hb, b);
    try e.stream.synchronize();
    return std.mem.eql(u8, ha, hb);
}

/// Every width equals lane_gemv's 16-row bytes, qmm_group where supported and split-K; then check forked MoE.
pub fn check(gpa: std.mem.Allocator, e: *nemotron.Engine) !u8 {
    const clusters = try e.ctx.capability() >= 90; // qmm_group's cluster sum over K slices needs sm_90
    // GB10 serves with lane_gemv, but the gate must still exercise the split-K kernel on its weights.
    var split = if (e.k.affine.split == null) try kern.Split.init(e.ctx.d) else null;
    defer if (split) |*sp| sp.deinit();
    const checked_split = e.k.affine.split orelse split.?;
    const w = &e.w;
    const Named = struct { name: []const u8, q: kern.QLinear };
    var named: std.ArrayList(Named) = .empty;
    defer named.deinit(gpa);
    for (w.blocks) |blk| switch (blk.kind) {
        .mamba => {
            try named.append(gpa, .{ .name = "in_proj", .q = blk.mamba.in_proj });
            try named.append(gpa, .{ .name = "out_proj", .q = blk.mamba.out_proj });
        },
        .attention => {
            try named.append(gpa, .{ .name = "qkv", .q = blk.attn.qkv });
            try named.append(gpa, .{ .name = "o", .q = blk.attn.o });
        },
        .moe => {},
    };
    try named.append(gpa, .{ .name = "head", .q = w.head });
    if (w.mtp) |m| {
        try named.append(gpa, .{ .name = "mtp eh_proj", .q = m.eh_proj });
        try named.append(gpa, .{ .name = "mtp qkv", .q = m.attn.qkv });
        try named.append(gpa, .{ .name = "mtp o", .q = m.attn.o });
    }
    if (w.draft_head) |h| try named.append(gpa, .{ .name = "draft head", .q = h });
    var prng = std.Random.DefaultPrng.init(0x7f4a7c15);
    const rng = prng.random();
    const b = &e.b;
    var bad: usize = 0;
    var cases: usize = 0;
    var splits: usize = 0;
    for (named.items) |nq| {
        const n = nq.q.outputs();
        const k = nq.q.inputs();
        const in = try inputs(gpa, rng, 16, k);
        defer gpa.free(in.x);
        defer gpa.free(in.xs);
        try e.ops().upload(b.emb, std.mem.sliceAsBytes(in.x));
        try e.ops().upload(b.xs, std.mem.sliceAsBytes(in.xs));
        const wide = b.ymoe + 2 * 16 * n * 2; // the 16-row result every narrower window's rows must equal
        const q = switch (nq.q) {
            .affine4 => |a| a,
            else => { // a lane format: each width's rows byte-equal to the 16-row call's
                try e.ops().dense(b.emb, b.xs, nq.q, wide, 16);
                for (1..17) |rows| {
                    const len = rows * n * 2;
                    try e.ops().fill32(b.ymoe + len, 0x7fc07fc0, len / 4);
                    try e.ops().dense(b.emb, b.xs, nq.q, b.ymoe + len, rows);
                    cases += 1;
                    if (!try sameBytes(gpa, e, b.ymoe + len, wide, len)) {
                        bad += 1;
                        std.debug.print("DIFFER {s} ({d}x{d}, {s}) at {d} rows\n", .{ nq.name, n, k, @tagName(nq.q), rows });
                    }
                }
                continue;
            },
        };
        const sk = kern.splitK(q.n, q.k);
        try e.ops().gemv(b.emb, b.xs, q, wide, 16, sk);
        for (1..17) |rows| {
            const len = rows * q.n * 2;
            try e.ops().gemv(b.emb, b.xs, q, b.ymoe + len, rows, sk);
            cases += 1;
            var same = try sameBytes(gpa, e, b.ymoe + len, wide, len);
            if (clusters or sk == 1) {
                try e.ops().cluster(b.emb, b.xs, q, b.ymoe, rows, sk);
                same = same and try sameBytes(gpa, e, b.ymoe, b.ymoe + len, len);
            }
            if (sk > 1) {
                try e.ops().fill32(b.ymoe, 0x7fc07fc0, len / 4);
                try e.ops().gemvSplit(b.emb, b.xs, q, b.ymoe, rows, sk, checked_split);
                same = same and try sameBytes(gpa, e, b.ymoe, b.ymoe + len, len);
                splits += 1;
            }
            if (!same) {
                bad += 1;
                std.debug.print("DIFFER {s} ({d}x{d}, {d} K slices) at {d} rows\n", .{ nq.name, q.n, q.k, sk, rows });
            }
        }
    }
    const against = if (clusters) "its 16-row bytes and qmm_group" else "its 16-row bytes, and qmm_group where K is one slice (no clusters before sm_90)";
    std.debug.print("{s} lane_gemv: {d} of {d} projection cases byte-equal to {s}; {d} split-K cases among them\n", .{ if (bad == 0) "PASS" else "FAIL", cases - bad, cases, against, splits });
    const mbad = try forkedMoe(gpa, e, rng);
    return if (bad == 0 and mbad == 0) 0 else 1;
}

/// The MoE with the shared halves on the side stream against the one-stream MoE: every pair's fp32 output equal.
fn forkedMoe(gpa: std.mem.Allocator, e: *nemotron.Engine, rng: std.Random) !usize {
    const b = &e.b;
    const c = e.c;
    var layers: std.ArrayList(nemotron.weights.MoE) = .empty;
    defer layers.deinit(gpa);
    for (e.w.blocks) |blk| if (blk.kind == .moe) try layers.append(gpa, blk.moe);
    if (e.w.mtp) |m| try layers.append(gpa, m.moe);
    var bad: usize = 0;
    var cases: usize = 0;
    var plain = e.forward(null);
    plain.side = null;
    const forked = e.forward(null);
    if (forked.side == null) return error.NoSideStream;
    for (layers.items) |moe| {
        for (1..17) |rows| {
            const in = try inputs(gpa, rng, rows, c.hidden);
            defer gpa.free(in.x);
            defer gpa.free(in.xs);
            try e.ops().upload(b.y, std.mem.sliceAsBytes(in.x));
            try e.ops().upload(b.xs, std.mem.sliceAsBytes(in.xs));
            const len = rows * c.slots() * c.hidden * 4;
            const keep = b.ymoe + (64 << 20);
            try plain.experts(moe, rows, false);
            try e.ops().copy(keep, b.ymoe, len);
            try e.ops().fill32(b.ymoe, 0x7fc00000, len / 4);
            try forked.experts(moe, rows, false);
            cases += 1;
            if (!try sameBytes(gpa, e, b.ymoe, keep, len)) bad += 1;
        }
    }
    std.debug.print("{s} forked MoE: {d} of {d} (layer, rows) cases byte-equal to the one-stream MoE\n", .{ if (bad == 0) "PASS" else "FAIL", cases - bad, cases });
    return bad;
}
