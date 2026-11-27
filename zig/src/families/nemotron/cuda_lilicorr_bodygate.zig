//! The LiLiCorr fixture gates on the live kernel chain: 16b body fixture replay and the S5b full-chain finisher.

const std = @import("std");
const cuda = @import("cuda");
const head = @import("cuda_lilicorr_head.zig");
const walk = @import("cuda_lilicorr_walk.zig");
const nvfp4 = @import("cuda_lilicorr_nvfp4.zig");
const lc = @import("cuda_lilicorr.zig");
const stack = @import("cuda_lilicorr_stack.zig");
const round = @import("cuda_lilicorr_round.zig");

const readFixtureF32 = stack.readFixtureF32;
const featuresBits = round.featuresBits;
const deviceScore = round.deviceScore;
const run = lc.run;
const norm = round.norm;
const projDense = round.projDense;
const ringK = lc.ringK;
const ringV = lc.ringV;
const f32acc = lc.f32acc;
const Segment = lc.Segment;
const Draft = lc.Draft;

/// 16b body-fixture gate helpers: per-row cos + L2-relative + bad-rate (law 2), never a mean over rows/elements.
pub fn bf16BitsOf(x: f32) u16 {
    return @truncate(@as(u32, @bitCast(nvfp4.bf16(x))) >> 16);
}

pub fn uploadFixtureBf16(d: *Draft, dst: u64, f: []const f32) !void {
    const bits = try d.gpa.alloc(u16, f.len);
    defer d.gpa.free(bits);
    for (f, 0..) |x, i| bits[i] = bf16BitsOf(x);
    try d.e.ctx.d.check(d.e.ctx.d.api.cuMemcpyHtoDAsync_v2(dst, bits.ptr, bits.len * 2, d.e.stream.handle), "cuMemcpyHtoDAsync fixture-upload");
    try d.e.stream.synchronize();
}

/// A downloaded bf16 stage vs the fixture's golden f32 blob, per row; PASS bar: cos > 0.99 and rel < rel_bar.
pub fn bodyGateStage(d: *Draft, io: std.Io, fix_dir: []const u8, dev: u64, rows: usize, dim: usize, rel_bar: f32, comptime fmt: []const u8, fmt_args: anytype, add_name: ?[]const u8) !bool {
    var nb: [64]u8 = undefined;
    const name = try std.fmt.bufPrint(&nb, fmt, fmt_args);
    const gold = try readFixtureF32(io, d.gpa, fix_dir, name, rows * dim);
    defer d.gpa.free(gold);
    var add: ?[]f32 = null;
    defer if (add) |a| d.gpa.free(a);
    if (add_name) |an| {
        const a = try readFixtureF32(io, d.gpa, fix_dir, an, rows * dim);
        add = a;
    }
    const bytes = try d.gpa.alloc(u8, rows * dim * 2);
    defer d.gpa.free(bytes);
    try d.e.ctx.d.check(d.e.ctx.d.api.cuMemcpyDtoHAsync_v2(bytes.ptr, dev, bytes.len, d.e.stream.handle), "cuMemcpyDtoHAsync body-gate");
    try d.e.stream.synchronize();
    var worst_cos: f32 = 2.0;
    var worst_rel: f32 = 0.0;
    var bad: usize = 0;
    const total = rows * dim;
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        var dot: f64 = 0;
        var na: f64 = 0;
        var nb2: f64 = 0;
        var d2: f64 = 0;
        var c: usize = 0;
        while (c < dim) : (c += 1) {
            const idx = r * dim + c;
            const a: f32 = head.widenBits(std.mem.bytesToValue(u16, bytes[idx * 2 ..][0..2]));
            const b: f32 = gold[idx] + (if (add) |aa| aa[idx] else 0);
            dot += @as(f64, a) * b;
            na += @as(f64, a) * a;
            nb2 += @as(f64, b) * b;
            const dv = @abs(a - b);
            d2 += dv * dv;
            if (dv > 0.03 and dv > 0.05 * @abs(b)) bad += 1;
        }
        const cosv: f32 = @floatCast(dot / (@sqrt(na) * @sqrt(nb2) + 1e-9));
        const rel: f32 = @floatCast(@sqrt(d2) / (@sqrt(nb2) + 1e-9));
        if (cosv < worst_cos) worst_cos = cosv;
        if (rel > worst_rel) worst_rel = rel;
    }
    const bad_rate = @as(f32, @floatFromInt(bad)) / @as(f32, @floatFromInt(total));
    std.debug.print("[body-gate] {s}: worst-row cos {d:.5} rel {e:.2} bad {d:.2}% (bar {e:.2})\n", .{ name, worst_cos, worst_rel, bad_rate * 100, rel_bar });
    return worst_cos > 0.99 and worst_rel < rel_bar;
}

/// 16b (TF_BODY_FIXTURE): our body chain on the fixture's inputs, the golden K/V staged; PASS=*** per-row gate.
pub fn deviceBodyFixtureAB(gpa: std.mem.Allocator, io: std.Io, d: *Draft, fix_dir: []const u8) !void {
    const c = d.c;
    const e = d.e;
    const B = c.block;
    const H = c.hidden;
    const n_ctx = 36;
    const W = c.ringRows();
    const width = c.kvWidth();
    var ok = true;

    // golden ctx K/V per layer -> dedicated ring buffer, rows 0..35
    var ring = try cuda.DeviceBuffer.alloc(e.ctx.d, d.w.layers.len * 2 * W * width * 2);
    defer ring.free();
    for (0..d.w.layers.len) |l| {
        var nb: [32]u8 = undefined;
        const kn = try std.fmt.bufPrint(&nb, "ctx_k{d}.f32", .{l});
        const k = try readFixtureF32(io, gpa, fix_dir, kn, n_ctx * width);
        defer gpa.free(k);
        var nb2: [32]u8 = undefined;
        const vn = try std.fmt.bufPrint(&nb2, "ctx_v{d}.f32", .{l});
        const v = try readFixtureF32(io, gpa, fix_dir, vn, n_ctx * width);
        defer gpa.free(v);
        try uploadFixtureBf16(d, ring.ptr + l * 2 * W * width * 2, k);
        try uploadFixtureBf16(d, ring.ptr + (l * 2 * W * width + W * width) * 2, v);
    }

    // the frame segment
    var seg_buf = try cuda.DeviceBuffer.alloc(e.ctx.d, @sizeOf(Segment));
    defer seg_buf.free();
    const seg = [_]Segment{.{ .ring = ring.ptr, .begin = 0, .position = n_ctx }};
    try seg_buf.upload(0, std.mem.sliceAsBytes(&seg));
    const saved_segments = d.segments;
    d.segments = seg_buf.ptr;
    defer d.segments = saved_segments;

    // block embeds -> d.h (the residual + L0's norm input)
    const embeds = try readFixtureF32(io, gpa, fix_dir, "block_embeds.f32", B * H);
    defer gpa.free(embeds);
    try uploadFixtureBf16(d, d.h, embeds);

    for (d.w.layers, 0..) |l, i| {
        var res_name: []const u8 = "block_embeds.f32";
        var res_buf: ?[]u8 = null;
        defer if (res_buf) |b| gpa.free(b);
        if (i != 0) {
            res_buf = try gpa.alloc(u8, 32);
            res_name = try std.fmt.bufPrint(res_buf.?, "L{d}_res_out.f32", .{i - 1});
        }
        try norm(d, d.h, l.input_norm, d.x, B);
        ok = try bodyGateStage(d, io, fix_dir, d.x, B, H, 2e-2 + 2.5e-2 * @as(f32, @floatFromInt(i)), "L{d}_postnorm_in.f32", .{i}, null) and ok;
        try projDense(d, l.attn_conv, d.x, d.dyn, B);
        _ = try bodyGateStage(d, io, fix_dir, d.dyn, B, c.dynamic(), 2e-2 + 2.5e-2 * @as(f32, @floatFromInt(i)), "L{d}_kp_attn.f32", .{i}, null);
        try run(d.e, d.fns.dyn_conv_batch, .{ B, (H + 255) / 256, 1 }, .{ 256, 1, 1 }, .{ d.x, d.dyn, l.attn_base, @as(u64, 0), d.xin, @as(c_int, @intCast(H)), @as(c_int, 16), @as(c_int, 0), @as(c_int, @intCast(B)), f32acc() });
        _ = try bodyGateStage(d, io, fix_dir, d.xin, B, H, 2e-2 + 2.5e-2 * @as(f32, @floatFromInt(i)), "L{d}_convprep.f32", .{i}, null);
        try projDense(d, l.q, d.xin, d.qraw, B);
        try projDense(d, l.k, d.xin, d.kraw, B);
        try projDense(d, l.v, d.xin, d.vb, B);
        try run(d.e, d.fns.norm_rope_batch, .{ B * c.heads, 1, 1 }, .{ 32, 1, 1 }, .{ d.qraw, l.q_norm, d.q, @as(c_int, @intCast(c.heads)), @as(c_int, @intCast(c.heads * c.head_dim)), d.segments, @as(c_int, @intCast(c.block)), c.eps, c.theta, f32acc() });
        try run(d.e, d.fns.norm_rope_batch, .{ B * c.kv_heads, 1, 1 }, .{ 32, 1, 1 }, .{ d.kraw, l.k_norm, d.kb, @as(c_int, @intCast(c.kv_heads)), @as(c_int, @intCast(c.kv_heads * c.head_dim)), d.segments, @as(c_int, @intCast(c.block)), c.eps, c.theta, f32acc() });
        _ = try bodyGateStage(d, io, fix_dir, d.q, B, c.qWidth(), 2e-2 + 2.5e-2 * @as(f32, @floatFromInt(i)), "L{d}_q_roped.f32", .{i}, null);
        _ = try bodyGateStage(d, io, fix_dir, d.kb, B, c.kvWidth(), 2e-2 + 2.5e-2 * @as(f32, @floatFromInt(i)), "L{d}_k_roped.f32", .{i}, null);
        _ = try bodyGateStage(d, io, fix_dir, d.vb, B, c.kvWidth(), 2e-2 + 2.5e-2 * @as(f32, @floatFromInt(i)), "L{d}_v.f32", .{i}, null);
        try run(d.e, d.fns.attn_partial, .{ B, c.heads, c.chunks() }, .{ 128, 1, 1 }, .{ d.q, d.kb, d.vb, d.segments, @as(c_int, @intCast(c.block)), @as(c_int, @intCast(c.heads)), @as(c_int, @intCast(c.kv_heads)), @as(c_int, @intCast(W)), @as(c_int, @intCast(c.window)), @as(c_int, @intCast(i)), @as(c_int, @intCast(c.chunks())), @as(c_int, 0), d.pm, d.pl, d.po });
        if (i == 0 and (@import("sys").environment.read(bool, "TF_BODY_GATE_DUMP", false) catch false)) {
            // one-shot L0 attention diagnostic: the partial stats + the merge output
            const stats = try gpa.alloc(u8, B * c.heads * c.chunks() * 4);
            defer gpa.free(stats);
            try e.ctx.d.check(e.ctx.d.api.cuMemcpyDtoHAsync_v2(stats.ptr, d.pm, stats.len, e.stream.handle), "DtoH pm");
            try e.stream.synchronize();
            const pm32 = std.mem.bytesAsSlice(f32, stats);
            std.debug.print("[body-gate] L0 PM row0 head0 chunks:", .{});
            for (0..c.chunks()) |ch| std.debug.print(" {e:.3}", .{pm32[(0 * c.heads + 0) * c.chunks() + ch]});
            std.debug.print("\n", .{});
            try e.ctx.d.check(e.ctx.d.api.cuMemcpyDtoHAsync_v2(stats.ptr, d.pl, stats.len, e.stream.handle), "DtoH pl");
            try e.stream.synchronize();
            const pl32 = std.mem.bytesAsSlice(f32, stats);
            std.debug.print("[body-gate] L0 PL row0 head0 chunks:", .{});
            for (0..c.chunks()) |ch| std.debug.print(" {e:.3}", .{pl32[(0 * c.heads + 0) * c.chunks() + ch]});
            std.debug.print("\n", .{});
            const sk = try gpa.alloc(u8, c.heads * 2);
            defer gpa.free(sk);
            try e.ctx.d.check(e.ctx.d.api.cuMemcpyDtoHAsync_v2(sk.ptr, l.sink, sk.len, e.stream.handle), "DtoH sink");
            try e.stream.synchronize();
            std.debug.print("[body-gate] L0 sinks:", .{});
            for (0..@min(c.heads, 8)) |h| std.debug.print(" {d:.3}", .{head.widenBits(std.mem.bytesToValue(u16, sk[h * 2 ..][0..2]))});
            std.debug.print("\n", .{});
            // the ring readback: the golden upload as the kernel sees it (rows 0..35, layer 0)
            const rb = try gpa.alloc(u8, n_ctx * width * 2);
            defer gpa.free(rb);
            try e.ctx.d.check(e.ctx.d.api.cuMemcpyDtoHAsync_v2(rb.ptr, ring.ptr, rb.len, e.stream.handle), "DtoH ringK");
            try e.stream.synchronize();
            try std.Io.Dir.cwd().writeFile(d.e.io, .{ .sub_path = "/tmp/our_ringk0.u16", .data = rb });
            try e.ctx.d.check(e.ctx.d.api.cuMemcpyDtoHAsync_v2(rb.ptr, ring.ptr + W * width * 2, rb.len, e.stream.handle), "DtoH ringV");
            try e.stream.synchronize();
            try std.Io.Dir.cwd().writeFile(d.e.io, .{ .sub_path = "/tmp/our_ringv0.u16", .data = rb });
            // the merge output
            const att = try gpa.alloc(u8, B * H * 2);
            defer gpa.free(att);
            try e.ctx.d.check(e.ctx.d.api.cuMemcpyDtoHAsync_v2(att.ptr, d.att, att.len, e.stream.handle), "DtoH att");
            try e.stream.synchronize();
            try std.Io.Dir.cwd().writeFile(d.e.io, .{ .sub_path = "/tmp/our_att0.u16", .data = att });
            std.debug.print("[body-gate] L0 diagnostics dumped: ring k/v + att\n", .{});
        }
        try run(d.e, d.fns.attn_merge, .{ B, c.heads, 1 }, .{ 32, 1, 1 }, .{ d.pm, d.pl, d.po, @as(c_int, @intCast(c.heads)), @as(c_int, @intCast(c.chunks())), l.sink, d.att });
        if (i == 0 and (@import("sys").environment.read(bool, "TF_BODY_GATE_DUMP", false) catch false)) {
            // post-merge diagnostic: d.att as the kernel produced it
            const att = try gpa.alloc(u8, B * H * 2);
            defer gpa.free(att);
            try e.ctx.d.check(e.ctx.d.api.cuMemcpyDtoHAsync_v2(att.ptr, d.att, att.len, e.stream.handle), "DtoH att");
            try e.stream.synchronize();
            try std.Io.Dir.cwd().writeFile(d.e.io, .{ .sub_path = "/tmp/our_att0.u16", .data = att });
            // the ISOLATED o_proj gate: the fixture's pre-o_proj attention through OUR dense
            const gatt = try readFixtureF32(io, gpa, fix_dir, "L0_att.f32", B * H);
            defer gpa.free(gatt);
            try uploadFixtureBf16(d, d.x, gatt);
            try projDense(d, l.o, d.x, d.o, B);
            const iso = try bodyGateStage(d, io, fix_dir, d.o, B, H, 2e-2, "L{d}_attnout.f32", .{i}, null);
            std.debug.print("[body-gate] L0 ISOLATED o_proj (golden attention in): {s}\n", .{if (iso) "PASS" else "FAIL"});
        }
        try projDense(d, l.o, d.att, d.o, B);
        _ = try bodyGateStage(d, io, fix_dir, d.o, B, H, 2e-2 + 2.5e-2 * @as(f32, @floatFromInt(i)), "L{d}_attnout.f32", .{i}, null);
        try run(d.e, d.fns.dyn_conv_batch, .{ B, (H + 255) / 256, 1 }, .{ 256, 1, 1 }, .{ d.o, d.dyn, l.attn_base, d.h, d.h2, @as(c_int, @intCast(H)), @as(c_int, 16), @as(c_int, 1), @as(c_int, @intCast(B)), f32acc() });
        try norm(d, d.h2, l.post_norm, d.x, B);
        _ = try bodyGateStage(d, io, fix_dir, d.h2, B, H, 2e-2 + 2.5e-2 * @as(f32, @floatFromInt(i)), "L{d}_convfin.f32", .{i}, res_name);
        ok = try bodyGateStage(d, io, fix_dir, d.x, B, H, 2e-2 + 2.5e-2 * @as(f32, @floatFromInt(i)), "L{d}_postnorm_out.f32", .{i}, null) and ok;
        try projDense(d, l.mlp_conv, d.x, d.dyn, B);
        _ = try bodyGateStage(d, io, fix_dir, d.dyn, B, c.dynamic(), 2e-2 + 2.5e-2 * @as(f32, @floatFromInt(i)), "L{d}_kp_mlp.f32", .{i}, null);
        try run(d.e, d.fns.dyn_conv_batch, .{ B, (H + 255) / 256, 1 }, .{ 256, 1, 1 }, .{ d.x, d.dyn, l.mlp_base, @as(u64, 0), d.xin, @as(c_int, @intCast(H)), @as(c_int, 16), @as(c_int, 0), @as(c_int, @intCast(B)), f32acc() });
        _ = try bodyGateStage(d, io, fix_dir, d.xin, B, H, 2e-2 + 2.5e-2 * @as(f32, @floatFromInt(i)), "L{d}_mlpconvprep.f32", .{i}, null);
        try projDense(d, l.gate, d.xin, d.gate, B);
        try projDense(d, l.up, d.xin, d.up, B);
        try run(d.e, d.fns.mlp_act, .{ (B * c.intermediate + 1023) / 1024, 1, 1 }, .{ 256, 1, 1 }, .{ d.gate, d.up, d.act, @as(c_longlong, @intCast(B * c.intermediate)), @as(u64, 0), @as(c_int, @intCast(c.intermediate)), d.xs });
        try projDense(d, l.down, d.act, d.o, B);
        _ = try bodyGateStage(d, io, fix_dir, d.o, B, H, 2e-2 + 2.5e-2 * @as(f32, @floatFromInt(i)), "L{d}_mlpout.f32", .{i}, null);
        try run(d.e, d.fns.dyn_conv_batch, .{ B, (H + 255) / 256, 1 }, .{ 256, 1, 1 }, .{ d.o, d.dyn, l.mlp_base, d.h2, d.h, @as(c_int, @intCast(H)), @as(c_int, 16), @as(c_int, 1), @as(c_int, @intCast(B)), f32acc() });
        _ = try bodyGateStage(d, io, fix_dir, d.h, B, H, 2e-2 + 2.5e-2 * @as(f32, @floatFromInt(i)), "L{d}_res_out.f32", .{i}, null);
    }
    try norm(d, d.h, d.w.final_norm, d.x, B);
    ok = try bodyGateStage(d, io, fix_dir, d.x, B, H, 2e-2 + 2.5e-2 * @as(f32, @floatFromInt(d.w.layers.len)), "final_hidden.f32", .{}, null) and ok;

    // the recall meter datum: our top-8 ids vs the fixture's cand_ids (report-only)
    try e.ops().dense(d.x, d.xs, e.w.head, d.logits, B);
    try e.k.draws.logprobRows(e.stream, d.logits, c.vocab, d.draft_picks, @intCast(c.topk), false, d.words, B);
    try e.stream.synchronize();
    const words = try gpa.alloc(u8, B * (3 + 2 * 20) * 4);
    defer gpa.free(words);
    try e.ctx.d.check(e.ctx.d.api.cuMemcpyDtoHAsync_v2(words.ptr, d.words, words.len, e.stream.handle), "cuMemcpyDtoHAsync body-gate words");
    try e.stream.synchronize();
    const w32 = std.mem.bytesAsSlice(u32, words);
    const gold_ids = try readFixtureF32(io, gpa, fix_dir, "cand_ids.f32", B * 64);
    defer gpa.free(gold_ids);
    var match: usize = 0;
    for (0..B) |r| {
        const row = w32[r * (3 + 2 * 20) ..][0 .. 3 + 2 * 20];
        var m: usize = 0;
        for (0..8) |k| {
            if (@as(u32, @intFromFloat(gold_ids[r * 64 + @as(usize, @intCast(k))])) == row[2 + k]) m += 1;
        }
        match += m;
        std.debug.print("[body-gate] row {d}: ids {d},{d},{d},{d},{d},{d},{d},{d} gold-match {d}/8\n", .{ r, row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], m });
    }
    std.debug.print("[body-gate] recall datum: {d}/64 of the gold top-8 reproduced\n", .{match});
    if (!ok) return error.BodyFixtureABFailed;
    std.debug.print("[body-gate] PASS all stages\n", .{});
}

/// S5b A/B: the full device chain vs the fixture's start/pair goldens PLUS identical chain indices (diff < 4e-3).
pub fn deviceFullChainAB(gpa: std.mem.Allocator, io: std.Io, d: *Draft, fix_dir: []const u8) !void {
    const rows = head.lattice;
    const topk = head.topk;
    const slots = head.slots;
    const n_pair = (slots - 1) * topk * topk;

    const token_emb = try readFixtureF32(io, gpa, fix_dir, "token_embeddings.f32", rows * head.model_hidden);
    const pass_hidden = try readFixtureF32(io, gpa, fix_dir, "pass_hidden.f32", head.model_hidden);
    const anchor_hidden = try readFixtureF32(io, gpa, fix_dir, "anchor_hidden.f32", head.model_hidden);
    const log_probs = try readFixtureF32(io, gpa, fix_dir, "candidate_log_probs.f32", rows);
    const start_gold = try readFixtureF32(io, gpa, fix_dir, "start_scores.f32", topk);
    const pair_gold = try readFixtureF32(io, gpa, fix_dir, "pair_scores.f32", n_pair);

    // inputs as bf16 bits — the serve path's dtype (S2's round-trip pattern)
    const tok_bits = try gpa.alloc(u16, rows * head.model_hidden);
    defer gpa.free(tok_bits);
    for (token_emb, 0..) |v, i| tok_bits[i] = @truncate(@as(u32, @bitCast(nvfp4.bf16(v))) >> 16);
    const pass_bits = try gpa.alloc(u16, head.model_hidden);
    defer gpa.free(pass_bits);
    for (pass_hidden, 0..) |v, i| pass_bits[i] = @truncate(@as(u32, @bitCast(nvfp4.bf16(v))) >> 16);
    const anchor_bits = try gpa.alloc(u16, head.model_hidden);
    defer gpa.free(anchor_bits);
    for (anchor_hidden, 0..) |v, i| anchor_bits[i] = @truncate(@as(u32, @bitCast(nvfp4.bf16(v))) >> 16);

    var tok_buf = try cuda.DeviceBuffer.alloc(d.e.ctx.d, tok_bits.len * 2);
    defer tok_buf.free();
    var pass_buf = try cuda.DeviceBuffer.alloc(d.e.ctx.d, pass_bits.len * 2);
    defer pass_buf.free();
    var anchor_buf = try cuda.DeviceBuffer.alloc(d.e.ctx.d, anchor_bits.len * 2);
    defer anchor_buf.free();
    var feat_buf = try cuda.DeviceBuffer.alloc(d.e.ctx.d, rows * 5 * 2);
    defer feat_buf.free();
    try tok_buf.upload(0, std.mem.sliceAsBytes(tok_bits));
    try pass_buf.upload(0, std.mem.sliceAsBytes(pass_bits));
    try anchor_buf.upload(0, std.mem.sliceAsBytes(anchor_bits));

    const fb = try gpa.alloc(u16, rows * 5);
    defer gpa.free(fb);
    try featuresBits(gpa, &d.hw, log_probs, fb);
    try feat_buf.upload(0, std.mem.sliceAsBytes(fb));

    var start_dev: [topk]f32 = undefined;
    var pair_dev: [n_pair]f32 = undefined;
    try deviceScore(d, tok_buf.ptr, pass_buf.ptr, anchor_buf.ptr, true, feat_buf.ptr, &start_dev, &pair_dev);

    var sd: f32 = 0;
    for (start_dev, start_gold) |got, want| sd = @max(sd, @abs(got - want));
    var pd: f32 = 0;
    for (pair_dev, pair_gold) |got, want| pd = @max(pd, @abs(got - want));

    // the chain identity: device scores and goldens must walk to the same indices
    var cand0: [head.slots * topk]u32 = @splat(0);
    const ch_dev = try walk.chain(&start_dev, &pair_dev, &cand0);
    const ch_gold = try walk.chain(start_gold, pair_gold, &cand0);
    var chain_ok = true;
    for (ch_dev.index, ch_gold.index) |a, b| {
        if (a != b) chain_ok = false;
    }
    std.debug.print("full chain A/B: start_max_diff {e:.2} pair_max_diff {e:.2} chain_dev {any} chain_gold {any} chain_match {} (gates: <4e-3 + identical)\n", .{ sd, pd, ch_dev.index, ch_gold.index, chain_ok });
    if (sd >= 4e-3 or pd >= 4e-3 or !chain_ok) return error.DeviceFullChainABFailed;
}
