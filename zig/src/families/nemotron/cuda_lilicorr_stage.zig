//! LiLiCorr open-time fixture probes, ladder part 1: token projection, ladder attention and factor+edge diffs.

const std = @import("std");
const cuda = @import("cuda");
const head = @import("cuda_lilicorr_head.zig");
const nvfp4 = @import("cuda_lilicorr_nvfp4.zig");
const lc = @import("cuda_lilicorr.zig");
const stack = @import("cuda_lilicorr_stack.zig");

const readFixtureF32 = stack.readFixtureF32;
const bf16Ulp = stack.bf16Ulp;
const run = lc.run;
const Draft = lc.Draft;

/// S2: the device token projection vs the fixture's WIDENED f32 candidate embeddings, narrowed to bf16 at RNE.
pub fn deviceTokenStatesAB(gpa: std.mem.Allocator, io: std.Io, d: *Draft, fix_dir: []const u8) !void {
    const rows = head.lattice;
    const in_dim = head.model_hidden;
    const out_dim = head.hd;
    const emb = try readFixtureF32(io, gpa, fix_dir, "token_embeddings.f32", rows * in_dim);
    const gold = try readFixtureF32(io, gpa, fix_dir, "token_states.f32", rows * out_dim);
    const emb_bits = try gpa.alloc(u16, rows * in_dim);
    defer gpa.free(emb_bits);
    for (emb, 0..) |x, i| emb_bits[i] = @truncate(@as(u32, @bitCast(nvfp4.bf16(x))) >> 16);

    var in_buf = try cuda.DeviceBuffer.alloc(d.e.ctx.d, rows * in_dim * 2);
    defer in_buf.free();
    var out_buf = try cuda.DeviceBuffer.alloc(d.e.ctx.d, rows * out_dim * 4);
    defer out_buf.free();
    try in_buf.upload(0, std.mem.sliceAsBytes(emb_bits));
    try run(d.e, d.fns.token_proj, .{ (out_dim + 31) / 32, 1, 1 }, .{ 256, 1, 1 }, .{ in_buf.ptr, d.dhw.?.token_proj.w, d.dhw.?.token_proj.b, out_buf.ptr, @as(c_int, @intCast(in_dim)), @as(c_int, @intCast(out_dim)), @as(c_int, @intCast(rows)) });
    try d.e.stream.synchronize();

    const got_bytes = try gpa.alloc(u8, rows * out_dim * 4);
    defer gpa.free(got_bytes);
    try out_buf.download(0, got_bytes);
    const got = @as([*]align(1) const f32, @ptrCast(got_bytes.ptr))[0 .. rows * out_dim];
    var max_diff: f32 = 0;
    for (got, gold) |gv, want| max_diff = @max(max_diff, @abs(gv - want));
    // three-way split: device vs CHECKPOINT-host-linear vs fixture-gold — separates a kernel

    // defect (device != host) from a weights mismatch (checkpoint-host != fixture gold)
    var dev_vs_host: f32 = 0;
    var host_vs_gold: f32 = 0;
    for (0..rows) |r| {
        for (0..out_dim) |t| {
            var acc: f32 = 0;
            const wrow = d.hw.token_proj.w[t * in_dim ..][0..in_dim];
            for (0..in_dim) |i| acc += head.widenBits(emb_bits[r * in_dim + i]) * wrow[i];
            acc += d.hw.token_proj.b[t];
            dev_vs_host = @max(dev_vs_host, @abs(got[r * out_dim + t] - acc));
            host_vs_gold = @max(host_vs_gold, @abs(acc - gold[r * out_dim + t]));
        }
    }
    std.debug.print("token_proj device A/B: dev-vs-gold {e:.2} | dev-vs-host {e:.2} | host-vs-gold {e:.2} (gate 4e-3)\n", .{ max_diff, dev_vs_host, host_vs_gold });
    if (!(max_diff < 4e-3)) return error.TokenProjectionABFailed;
}

/// Lattice attention A/B: compares device kernel against host reference and fixture gold.
pub fn deviceLatticeAttentionAB(gpa: std.mem.Allocator, io: std.Io, d: *Draft, fix_dir: []const u8) !void {
    const rows = head.lattice;
    const in_dim = head.hd;
    const out_dim = head.hd;

    // Read fixture data
    const qkv1_pre = try readFixtureF32(io, gpa, fix_dir, "qkv1_pre.f32", rows * 3 * in_dim);
    const attn1_raw = try readFixtureF32(io, gpa, fix_dir, "attn1_raw.f32", rows * out_dim);

    // Build bias
    const bias = try head.buildBias(gpa, &d.hw);
    defer gpa.free(bias);

    // Upload qkv and bias to device
    var qkv_buf = try cuda.DeviceBuffer.alloc(d.e.ctx.d, rows * 3 * in_dim * 4);
    defer qkv_buf.free();
    var bias_buf = try cuda.DeviceBuffer.alloc(d.e.ctx.d, head.heads * rows * rows * 4);
    defer bias_buf.free();
    var out_buf = try cuda.DeviceBuffer.alloc(d.e.ctx.d, rows * out_dim * 4);
    defer out_buf.free();

    try qkv_buf.upload(0, std.mem.sliceAsBytes(qkv1_pre));
    try bias_buf.upload(0, std.mem.sliceAsBytes(bias));

    // Launch kernel
    try run(d.e, d.fns.lat_attn, .{ rows * 8, 1, 1 }, .{ 128, 1, 1 }, .{ qkv_buf.ptr, bias_buf.ptr, out_buf.ptr, @as(c_int, @intCast(rows)), @as(c_int, @intCast(out_dim)), @as(c_int, @intCast(head.head_dim)) });
    try d.e.stream.synchronize();

    // Download result
    const got_bytes = try gpa.alloc(u8, rows * out_dim * 4);
    defer gpa.free(got_bytes);
    try out_buf.download(0, got_bytes);
    const got = @as([*]align(1) const f32, @ptrCast(got_bytes.ptr))[0 .. rows * out_dim];

    // Compare against host reference and fixture gold
    const host_ref = try gpa.alloc(f32, rows * out_dim);
    defer gpa.free(host_ref);

    // Compute host reference
    head.latticeAttention(qkv1_pre, bias, host_ref);

    var max_diff: f32 = 0;
    var dev_vs_host: f32 = 0;
    var host_vs_gold: f32 = 0;

    for (got, attn1_raw, host_ref) |dev_val, gold_val, host_val| {
        max_diff = @max(max_diff, @abs(dev_val - gold_val));
        dev_vs_host = @max(dev_vs_host, @abs(dev_val - host_val));
        host_vs_gold = @max(host_vs_gold, @abs(host_val - gold_val));
    }

    std.debug.print("lat_attn device A/B: dev-vs-gold {e:.2} | dev-vs-host {e:.2} | host-vs-gold {e:.2} (gate 4e-3)\n", .{ max_diff, dev_vs_host, host_vs_gold });
    if (!(max_diff < 4e-3)) return error.LatticeAttentionABFailed;
}

/// Factor and edge A/B: compares device factor block and edge head against host reference and fixture gold.
pub fn deviceFactorEdgeAB(gpa: std.mem.Allocator, io: std.Io, d: *Draft, fix_dir: []const u8) !void {
    const rows = head.lattice;
    const hd = head.hd;
    const factor = head.factor;
    const topk = head.topk;
    const slots = head.slots;

    // Read fixture data
    const out_normed = try readFixtureF32(io, gpa, fix_dir, "out_normed.f32", rows * hd);
    const anchor_normed = try readFixtureF32(io, gpa, fix_dir, "anchor_normed.f32", hd);
    const factor_hidden = try readFixtureF32(io, gpa, fix_dir, "factor_hidden.f32", rows * hd);
    const edges = try readFixtureF32(io, gpa, fix_dir, "edges.f32", rows * 2 * factor);
    const start_scores = try readFixtureF32(io, gpa, fix_dir, "start_scores.f32", topk);
    const pair_scores = try readFixtureF32(io, gpa, fix_dir, "pair_scores.f32", (slots - 1) * topk * topk);

    // Upload inputs to device
    var out_normed_buf = try cuda.DeviceBuffer.alloc(d.e.ctx.d, rows * hd * 4);
    defer out_normed_buf.free();
    var anchor_normed_buf = try cuda.DeviceBuffer.alloc(d.e.ctx.d, hd * 4);
    defer anchor_normed_buf.free();
    var factor_hidden_buf = try cuda.DeviceBuffer.alloc(d.e.ctx.d, rows * hd * 4);
    defer factor_hidden_buf.free();
    var edges_buf = try cuda.DeviceBuffer.alloc(d.e.ctx.d, rows * 2 * factor * 4);
    defer edges_buf.free();
    var start_scores_buf = try cuda.DeviceBuffer.alloc(d.e.ctx.d, topk * 4);
    defer start_scores_buf.free();
    var pair_scores_buf = try cuda.DeviceBuffer.alloc(d.e.ctx.d, (slots - 1) * topk * topk * 4);
    defer pair_scores_buf.free();

    try out_normed_buf.upload(0, std.mem.sliceAsBytes(out_normed));
    try anchor_normed_buf.upload(0, std.mem.sliceAsBytes(anchor_normed));
    try factor_hidden_buf.upload(0, std.mem.sliceAsBytes(factor_hidden));
    try edges_buf.upload(0, std.mem.sliceAsBytes(edges));
    try start_scores_buf.upload(0, std.mem.sliceAsBytes(start_scores));
    try pair_scores_buf.upload(0, std.mem.sliceAsBytes(pair_scores));

    // Prepare device buffers for computation
    var pre_bits_buf = try cuda.DeviceBuffer.alloc(d.e.ctx.d, rows * hd * 2); // u16
    defer pre_bits_buf.free();

    var fused_w_buf = try cuda.DeviceBuffer.alloc(d.e.ctx.d, 2 * factor * hd * 4);
    defer fused_w_buf.free();

    var fused_b_buf = try cuda.DeviceBuffer.alloc(d.e.ctx.d, 2 * factor * 4);
    defer fused_b_buf.free();

    var start_dev_buf = try cuda.DeviceBuffer.alloc(d.e.ctx.d, topk * 4);
    defer start_dev_buf.free();

    var pair_dev_buf = try cuda.DeviceBuffer.alloc(d.e.ctx.d, (slots - 1) * topk * topk * 4);
    defer pair_dev_buf.free();

    // Build the fused weight host-side: fused_w = concat(d.hw.out_head.w, d.hw.in_head.w) [2048*1024],

    // fused_b = concat(b's) [2048]
    const fused_w = try gpa.alloc(f32, 2 * factor * hd);
    defer gpa.free(fused_w);
    const fused_b = try gpa.alloc(f32, 2 * factor);
    defer gpa.free(fused_b);

    @memcpy(fused_w[0 .. factor * hd], d.hw.out_head.w);
    @memcpy(fused_w[factor * hd ..], d.hw.in_head.w);
    @memcpy(fused_b[0..factor], d.hw.out_head.b);
    @memcpy(fused_b[factor..], d.hw.in_head.b);

    // Upload fused weights
    try fused_w_buf.upload(0, std.mem.sliceAsBytes(fused_w));
    try fused_b_buf.upload(0, std.mem.sliceAsBytes(fused_b));

    // anchor input as bf16 bits (the values are bf16-representable - the bits round-trip, S2's pattern)
    const anchor_bits = try gpa.alloc(u16, hd);
    defer gpa.free(anchor_bits);
    for (anchor_normed, 0..) |x, ix| anchor_bits[ix] = @truncate(@as(u32, @bitCast(nvfp4.bf16(x))) >> 16);
    var anchor_bits_buf = try cuda.DeviceBuffer.alloc(d.e.ctx.d, hd * 2);
    defer anchor_bits_buf.free();
    try anchor_bits_buf.upload(0, std.mem.sliceAsBytes(anchor_bits));

    // the device workspace: edges [56*2048] + the anchor_out tail [1024] so one l2norm grid(57) covers both
    const anchor_tail_at: u64 = @intCast(rows * 2 * factor * 4); // byte offset of the anchor tail
    var edges_dev_buf = try cuda.DeviceBuffer.alloc(d.e.ctx.d, rows * 2 * factor * 4 + hd * 4);
    defer edges_dev_buf.free();

    // K1 factor block -> pre bf16 bits
    try run(d.e, d.fns.factor, .{ (hd + 7) / 8, 1, 1 }, .{ 256, 1, 1 }, .{ out_normed_buf.ptr, anchor_normed_buf.ptr, d.dhw.?.factor.w, d.dhw.?.factor.b, pre_bits_buf.ptr, @as(c_int, @intCast(rows)), @as(c_int, @intCast(hd)) });
    try d.e.stream.synchronize();

    // K2 edge head (pre bits @ fused [2048,1024] -> edges) and anchor head (anchor bits -> tail)
    try run(d.e, d.fns.edge_linear, .{ (2 * factor + 31) / 32, 1, 1 }, .{ 256, 1, 1 }, .{ pre_bits_buf.ptr, fused_w_buf.ptr, fused_b_buf.ptr, edges_dev_buf.ptr, @as(c_int, @intCast(hd)), @as(c_int, @intCast(2 * factor)), @as(c_int, @intCast(rows)) });
    try run(d.e, d.fns.edge_linear, .{ (hd + 31) / 32, 1, 1 }, .{ 256, 1, 1 }, .{ anchor_bits_buf.ptr, d.dhw.?.anchor_out_head.w, d.dhw.?.anchor_out_head.b, edges_dev_buf.ptr + anchor_tail_at, @as(c_int, @intCast(hd)), @as(c_int, @intCast(hd)), @as(c_int, 1) });
    try d.e.stream.synchronize();

    // the edges golden is PRE-norm: download the diff copy before normalizing
    const edges_dev_bytes = try gpa.alloc(u8, rows * 2 * factor * 4);
    defer gpa.free(edges_dev_bytes);
    try edges_dev_buf.download(0, edges_dev_bytes);

    // K3 l2norm: grid(57) - blocks 0..55 normalize the edges' two halves, block 56 the anchor tail
    try run(d.e, d.fns.l2norm, .{ rows + 1, 1, 1 }, .{ 1024, 1, 1 }, .{ edges_dev_buf.ptr, @as(c_int, @intCast(rows)), @as(c_int, @intCast(2 * factor)), @as(c_int, @intCast(hd)), @as(f32, head.vector_eps) });
    try d.e.stream.synchronize();

    // K4 dots: anchor tail vs the in-halves (base +1024 elements), out-halves at the base
    try run(d.e, d.fns.dots, .{ 8 + (slots - 1) * topk * topk, 1, 1 }, .{ 256, 1, 1 }, .{ edges_dev_buf.ptr + anchor_tail_at, edges_dev_buf.ptr, edges_dev_buf.ptr + @as(u64, factor) * 4, start_dev_buf.ptr, pair_dev_buf.ptr, @as(c_int, @intCast(2 * factor)) });
    try d.e.stream.synchronize();

    // Download results for comparison
    const start_dev_bytes = try gpa.alloc(u8, topk * 4);
    defer gpa.free(start_dev_bytes);
    try start_dev_buf.download(0, start_dev_bytes);

    const pair_dev_bytes = try gpa.alloc(u8, (slots - 1) * topk * topk * 4);
    defer gpa.free(pair_dev_bytes);
    try pair_dev_buf.download(0, pair_dev_bytes);

    const pre_bits_bytes = try gpa.alloc(u8, rows * hd * 2);
    defer gpa.free(pre_bits_bytes);
    try pre_bits_buf.download(0, pre_bits_bytes);

    const pre_bits_u16 = std.mem.bytesAsSlice(u16, pre_bits_bytes);

    // HOST chain at the head's cast points (pre = bf16(silu(3 dots + b)); edges = fused linear of

    // the bf16 pre): dev-vs-host isolates KERNEL correctness; host-vs-gold carries the fixture's

    // raw-silu-vs-bf16 cast noise and gates only where the lane's contract gates (the scores).
    const pre_host = try gpa.alloc(f32, rows * hd);
    defer gpa.free(pre_host);
    for (0..rows) |r| {
        for (0..hd) |t| {
            var v: f32 = d.hw.factor.b[t];
            const wrow = d.hw.factor.w[t * 3 * hd ..][0 .. 3 * hd];
            for (0..hd) |ix| v += out_normed[r * hd + ix] * wrow[ix];
            for (0..hd) |ix| v += anchor_normed[ix] * wrow[hd + ix];
            for (0..hd) |ix| v += out_normed[r * hd + ix] * anchor_normed[ix] * wrow[2 * hd + ix];
            pre_host[r * hd + t] = nvfp4.bf16(v / (1.0 + @exp(-v)));
        }
    }
    // edges host recomputation consumes the DEVICE's own pre bits (widened) — the pre-cast

    // differences then cancel and dev-vs-host isolates the K2 kernel alone
    const pre_dev_f32 = try gpa.alloc(f32, rows * hd);
    defer gpa.free(pre_dev_f32);
    for (pre_bits_u16, 0..) |bits, iw| pre_dev_f32[iw] = head.widenBits(bits);
    const edges_host = try gpa.alloc(f32, rows * 2 * factor);
    defer gpa.free(edges_host);
    for (0..rows) |r| {
        for (0..2 * factor) |t| {
            var v: f32 = fused_b[t];
            const wrow = fused_w[t * hd ..][0..hd];
            for (0..hd) |ix| v += pre_dev_f32[r * hd + ix] * wrow[ix];
            edges_host[r * 2 * factor + t] = v;
        }
    }

    var pre_diff: f32 = 0; // counted in bf16 ULPs of the host value
    var pre_ulp: f32 = 0;
    for (pre_bits_u16, pre_host) |bits, want| {
        const got = head.widenBits(bits);
        const d1 = @abs(got - want);
        const u = bf16Ulp(want);
        pre_ulp = @max(pre_ulp, d1 / u);
        pre_diff = @max(pre_diff, d1);
    }

    const edges_dev_f32 = @as([*]align(1) const f32, @ptrCast(edges_dev_bytes.ptr))[0 .. rows * 2 * factor];
    var edges_diff: f32 = 0;
    var edges_host_gold: f32 = 0;
    for (edges_dev_f32, edges_host, edges) |got, host, want| {
        edges_diff = @max(edges_diff, @abs(got - host));
        edges_host_gold = @max(edges_host_gold, @abs(host - want));
    }

    const start_dev_f32 = @as([*]align(1) const f32, @ptrCast(start_dev_bytes.ptr))[0..topk];
    var start_diff: f32 = 0;
    for (start_dev_f32, start_scores) |got, want| start_diff = @max(start_diff, @abs(got - want));

    const pair_dev_f32 = @as([*]align(1) const f32, @ptrCast(pair_dev_bytes.ptr))[0 .. (slots - 1) * topk * topk];
    var pair_diff: f32 = 0;
    for (pair_dev_f32, pair_scores) |got, want| pair_diff = @max(pair_diff, @abs(got - want));

    std.debug.print("factor/edge A/B: pre {e:.2} ({d:.1} ulp) | edges dev-vs-host {e:.2} (host-vs-gold {e:.2}) | start {e:.2} | pair {e:.2} (gate 4e-3)\n", .{ pre_diff, pre_ulp, edges_diff, edges_host_gold, start_diff, pair_diff });
    if (!(pre_ulp <= 1.0 and edges_diff < 4e-3 and start_diff < 4e-3 and pair_diff < 4e-3)) return error.FactorEdgeABFailed;
}
