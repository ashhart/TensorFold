//! DeepSeek-V4.1 glue launches: the window's and the prefill segment's torch glue as kernels_ops.Ops methods.

const std = @import("std");
const cuda = @import("cuda");
const ops = @import("kernels_ops.zig");

const Ops = ops.Ops;
const glue = ops.glue;
const pfglue = ops.pfglue;

pub fn glueGo(o: Ops, kind: glue.Kind, n: u64, args: *cuda.Args) !void {
    try o.go(o.f.glue[@backingInt(kind)], .{ glue.grid(n), 1, 1 }, 256, 0, null, args);
}

/// Embedding send: rows [n, D] bf16 = embed[ids - vocab_lo] when in this rank's slice [0, V), else +0.
pub fn embedSend(o: Ops, ids: u64, n: usize, embed: u64, vocab_lo: i64, V: usize, D: usize, rows: u64) !void {
    if (n == 0) return;
    var a: cuda.Args = .{};
    a.add(ids);
    a.add(@as(c_int, @intCast(n)));
    a.add(embed);
    a.add(vocab_lo);
    a.add(@as(c_int, @intCast(V)));
    a.add(@as(c_int, @intCast(D)));
    a.add(rows);
    try o.glueGo(.embed_send, @as(u64, n) * D, &a);
}

/// Embedding sum: recv [W, n, D] bf16 -> out [n, 4 D] bf16 (fp32 adds in rank order, one rounding).
pub fn embedSum(o: Ops, recv: u64, W: usize, n: usize, D: usize, out: u64) !void {
    if (n == 0) return;
    if (W < 1) return error.Shape;
    var a: cuda.Args = .{};
    a.add(recv);
    for ([_]usize{ W, n, D }) |v| a.add(@as(c_int, @intCast(v)));
    a.add(out);
    try o.glueGo(.embed_sum, @as(u64, n) * D, &a);
}

/// y bf16 = x.to(bf16) over n elements (the MoE partial into the exchange).
pub fn castBf16(o: Ops, x: u64, y: u64, n: usize) !void {
    if (n == 0) return;
    var a: cuda.Args = .{};
    a.add(x);
    a.add(y);
    a.add(@as(i64, @intCast(n)));
    try o.glueGo(.cast_bf16, n, &a);
}

/// pick.kit_weights: y = x.to(fp16).to(fp32) over n elements (y may be x).
pub fn kitWeights(o: Ops, x: u64, y: u64, n: usize) !void {
    if (n == 0) return;
    var a: cuda.Args = .{};
    a.add(x);
    a.add(y);
    a.add(@as(i64, @intCast(n)));
    try o.glueGo(.kit_weights, n, &a);
}

/// pick.kit_logits in place: x = x.to(bf16).to(fp32) over n elements.
pub fn kitLogits(o: Ops, x: u64, n: usize) !void {
    if (n == 0) return;
    var a: cuda.Args = .{};
    a.add(x);
    a.add(@as(i64, @intCast(n)));
    try o.glueGo(.kit_logits, n, &a);
}

/// Win.make's tensors: pos int32 [1] = start, pos64 int64 [n] = start + i, lo int32 [n] = 0.
pub fn positions(o: Ops, start: i32, n: usize, pos: u64, pos64: u64, lo: u64) !void {
    var a: cuda.Args = .{};
    a.add(start);
    a.add(@as(c_int, @intCast(n)));
    a.add(pos);
    a.add(pos64);
    a.add(lo);
    try o.go(o.f.glue[@backingInt(glue.Kind.positions)], .{ (@max(n, 1) + 255) / 256, 1, 1 }, 256, 0, null, &a);
}

/// DSpark's candidates (glue.cu ds_cands): each row's best k as packed [rows][2 k] fp32 (values, then ids).
pub fn dsCandidates(o: Ops, lg: u64, ld: usize, rows: usize, V: usize, k: usize, id0: u32, out: u64) !void {
    const f = o.f.ds_cands orelse return error.Unsupported;
    if (k < 1 or k > glue.ds_cands_max or V < 1) return error.Shape;
    if (rows == 0) return;
    var a: cuda.Args = .{};
    a.add(lg);
    a.add(@as(i64, @intCast(ld)));
    a.add(@as(i32, @intCast(V)));
    a.add(@as(i32, @intCast(k)));
    a.add(@as(i32, @intCast(id0)));
    a.add(out);
    try o.go(f, .{ rows, 1, 1 }, 1024, 0, null, &a);
}

/// DSpark's one-slot statics from the device pick `pick` into "w.ds.i64" / "w.ds.i32" at offsets `off`.
pub fn dsStage(o: Ops, pick: u64, nw: usize, s64: u64, s32: u64, n: usize, W: usize, ring: usize, valid: u64, noise: u32, off: [6]i64) !void {
    const f = o.f.ds_stage orelse return error.Unsupported;
    var a: cuda.Args = .{};
    a.add(pick);
    a.add(@as(i32, @intCast(nw)));
    a.add(s64);
    a.add(s32);
    a.add(@as(i32, @intCast(n)));
    a.add(@as(i32, @intCast(W)));
    a.add(@as(i32, @intCast(ring)));
    a.add(@as(i64, @intCast(valid)));
    a.add(@as(i32, @intCast(noise)));
    for (off) |x| a.add(@as(i32, @intCast(x)));
    try o.go(f, .{ 1, 1, 1 }, 256, 0, null, &a);
}

/// A row window's slots' (accepted, bonus, next position) into `acc` [k][3] from `picks`, `ids` and `segs`.
pub fn dsAcceptRows(o: Ops, picks: u64, ids: u64, segs: u64, k: usize, acc: u64) !void {
    const f = o.f.ds_accept_rows orelse return error.Unsupported;
    if (k < 1 or k > 32) return error.Shape;
    var a: cuda.Args = .{};
    a.add(picks);
    a.add(ids);
    a.add(segs);
    a.add(@as(i32, @intCast(k)));
    a.add(acc);
    try o.go(f, .{ 1, 1, 1 }, 32, 0, null, &a);
}

/// A several-slot pass's position-dependent statics for `g` members `mem` [g][3], from acc's bonus and position.
pub fn dsStageRows(o: Ops, acc: u64, mem: u64, g: usize, s64: u64, s32: u64, n: usize, W: usize, ring: usize, noise: u32, off: [6]i64) !void {
    const f = o.f.ds_stage_rows orelse return error.Unsupported;
    var a: cuda.Args = .{};
    a.add(acc);
    a.add(mem);
    a.add(@as(i32, @intCast(g)));
    a.add(s64);
    a.add(s32);
    a.add(@as(i32, @intCast(n)));
    a.add(@as(i32, @intCast(W)));
    a.add(@as(i32, @intCast(ring)));
    a.add(@as(i32, @intCast(noise)));
    for (off) |x| a.add(@as(i32, @intCast(x)));
    try o.go(f, .{ 1, 1, 1 }, 256, 0, null, &a);
}

/// vsample.choose over gathered candidates `g` for `n` rows: tokens into `out` int64 [n] (-1: a nucleus row).
pub fn vsChoose(o: Ops, g: u64, W: usize, n: usize, k: usize, count: usize, seed: u64, pos0: u64, top_k: u32, temp: f64, top_p: f64, min_p: f64, out: u64) !void {
    const f = o.f.vs_choose orelse return error.Unsupported;
    if (W * k > glue.vs_max or n == 0) return error.Shape;
    var a: cuda.Args = .{};
    a.add(g);
    a.add(@as(i32, @intCast(W)));
    a.add(@as(i32, @intCast(n)));
    a.add(@as(i32, @intCast(k)));
    a.add(@as(i32, @intCast(count)));
    a.add(seed);
    a.add(@as(i64, @bitCast(pos0))); // the kernel draws at pos0 + r in int64: a pos0 below 0 (a row-shifted start) wraps back
    a.add(@as(i32, @intCast(top_k)));
    a.add(temp);
    a.add(top_p);
    a.add(min_p);
    a.add(out);
    try o.go(f, .{ n, 1, 1 }, 256, 0, null, &a);
}

/// bf16 [n] -> fp32 [n], exact (the DSpark pass's q before _rope).
pub fn widenF32(o: Ops, src: u64, dst: u64, n: usize) !void {
    if (n == 0) return;
    var a: cuda.Args = .{};
    a.add(src);
    a.add(dst);
    a.add(@as(i64, @intCast(n)));
    try o.glueGo(.widen_f32, n, &a);
}

pub fn pfGo(o: Ops, kind: pfglue.Kind, n: u64, args: *cuda.Args) !void {
    try o.go(o.f.pfglue[@backingInt(kind)], .{ glue.grid(n), 1, 1 }, 256, 0, null, args);
}

/// Each of R rows' `count` largest int64 keys as ascending positions, -1 padded: out int32 [R, count].
pub fn topPositions(o: Ops, keys: u64, ld: usize, R: usize, n: usize, count: usize, out: u64) !void {
    if (count < 1 or count > pfglue.max_count) return error.Shape;
    if (R == 0) return;
    var a: cuda.Args = .{};
    a.add(keys);
    a.add(@as(i64, @intCast(ld)));
    a.add(@as(c_int, @intCast(n)));
    a.add(@as(c_int, @intCast(count)));
    a.add(out);
    try o.go(o.f.pfglue[@backingInt(pfglue.Kind.top_positions)], .{ R, 1, 1 }, 1024, 0, null, &a);
}

/// index.candidate_keys: blocks int32 [R, nb] (row stride ld) -> out int32 [R, nb * bs].
pub fn candKeys(o: Ops, blocks: u64, ld: usize, R: usize, nb: usize, bs: usize, out: u64) !void {
    if (R * nb == 0) return;
    var a: cuda.Args = .{};
    a.add(blocks);
    a.add(@as(i64, @intCast(ld)));
    for ([_]usize{ R, nb, bs }) |v| a.add(@as(c_int, @intCast(v)));
    a.add(out);
    try o.pfGo(.cand_keys, R * nb * bs, &a);
}

/// out int32 [R] = #(sel >= 0 and sel < (pos + 1) / ratio), sel int32 [R, K], pos int64 [R].
pub fn visibleCounts(o: Ops, sel: u64, K: usize, pos: u64, ratio: u32, R: usize, out: u64) !void {
    if (R == 0) return;
    if (ratio < 1) return error.Shape;
    var a: cuda.Args = .{};
    a.add(sel);
    a.add(@as(c_int, @intCast(K)));
    a.add(pos);
    a.add(@as(c_int, @intCast(ratio)));
    a.add(@as(c_int, @intCast(R)));
    a.add(out);
    try o.go(o.f.pfglue[@backingInt(pfglue.Kind.counts)], .{ (R + 7) / 8, 1, 1 }, 256, 0, null, &a);
}

/// Routed picks: pk int32 [n, routed] = pick[:, :routed], w6 fp32 = f32(f16(wts[:, :routed])) (shared slot after).
pub fn gmPicks(o: Ops, pick: u64, wts: u64, ld: usize, routed: usize, n: usize, pk: u64, w6: u64) !void {
    if (n * routed == 0) return;
    var a: cuda.Args = .{};
    a.add(pick);
    a.add(wts);
    for ([_]usize{ ld, routed, n }) |v| a.add(@as(c_int, @intCast(v)));
    a.add(pk);
    a.add(w6);
    try o.pfGo(.gm_picks, n * routed, &a);
}

/// One width's picks: out[p] = k2tab[pick[p]] == k2 ? pick[p] : E (the input of gmPlan).
pub fn widthMask(o: Ops, pick: u64, P: usize, k2tab: u64, E: usize, k2: u32, out: u64) !void {
    if (P == 0) return;
    var a: cuda.Args = .{};
    a.add(pick);
    a.add(@as(i64, @intCast(P)));
    a.add(k2tab);
    a.add(@as(c_int, @intCast(E)));
    a.add(@as(c_int, @intCast(k2)));
    a.add(out);
    try o.pfGo(.width_mask, P, &a);
}

/// The indexer head weights' casts: y fp32 = f32(bf16(f32(x fp64))), n elements (glue step ix_w).
pub fn f64Bf16F32(o: Ops, x: u64, y: u64, n: usize) !void {
    if (n == 0) return;
    var a: cuda.Args = .{};
    a.add(x);
    a.add(y);
    a.add(@as(i64, @intCast(n)));
    try o.pfGo(.f64_bf16_f32, n, &a);
}

/// blocks._projection's fp32: out [n, hd] = f32(kv), or (gate != 0, ratio 2) [n, 2 hd] = [f32(kv) | f32(gate)].
pub fn widenCat(o: Ops, kv: u64, gate: u64, n: usize, hd: usize, out: u64) !void {
    if (n == 0) return;
    var a: cuda.Args = .{};
    a.add(kv);
    a.add(gate);
    a.add(@as(c_int, @intCast(n)));
    a.add(@as(c_int, @intCast(hd)));
    a.add(out);
    try o.pfGo(.widen_cat, n * hd * @as(usize, if (gate != 0) 2 else 1), &a);
}

/// Stage rows of one tensor: dst row p % dst_ring = src row p % src_ring for p in [lo, hi).
pub fn ringCopy(o: Ops, src: u64, src_ring: usize, dst: u64, dst_ring: usize, row_bytes: usize, lo: u64, hi: u64) !void {
    if (hi <= lo) return;
    var a: cuda.Args = .{};
    a.add(src);
    a.add(@as(c_int, @intCast(src_ring)));
    a.add(dst);
    a.add(@as(c_int, @intCast(dst_ring)));
    a.add(@as(i64, @intCast(row_bytes)));
    a.add(@as(i64, @intCast(lo)));
    a.add(@as(i64, @intCast(hi)));
    try o.pfGo(.ring_copy, (hi - lo) * (row_bytes / 4 + 1), &a);
}

/// positions with `start` read from the device (int32 at `start_dev`): a graphed window's form, the same values.
pub fn positionsDev(o: Ops, start_dev: u64, n: usize, pos: u64, pos64: u64, lo: u64) !void {
    var a: cuda.Args = .{};
    a.add(start_dev);
    a.add(@as(c_int, @intCast(n)));
    a.add(pos);
    a.add(pos64);
    a.add(lo);
    try o.go(o.f.glue[@backingInt(glue.Kind.positions_dev)], .{ (@max(n, 1) + 255) / 256, 1, 1 }, 256, 0, null, &a);
}

/// forward.gather_cols after the all-gather: recv [W, n, c] -> out [n, W c] (2-byte elements, rank order).
pub fn gatherCols(o: Ops, recv: u64, W: usize, n: usize, c: usize, out: u64) !void {
    if (n == 0 or c == 0) return;
    var a: cuda.Args = .{};
    a.add(recv);
    for ([_]usize{ W, n, c }) |v| a.add(@as(c_int, @intCast(v)));
    a.add(out);
    try o.glueGo(.gather_cols, @as(u64, W) * n * c, &a);
}

/// glue.cu pick_pack: f64 (value, column + id0) pairs [n][2] from topk_keys' k-1 vals fp32 [n] / cols int64 [n].
pub fn pickPack(o: Ops, vals: u64, cols: u64, n: usize, id0: u64, pairs: u64) !void {
    const f = o.f.pick_pack orelse return error.NoPickKernel;
    var a: cuda.Args = .{};
    a.add(vals);
    a.add(cols);
    a.add(@as(c_int, @intCast(n)));
    a.add(@as(i64, @intCast(id0)));
    a.add(pairs);
    try o.go(f, .{ glue.grid(n), 1, 1 }, 256, 0, null, &a);
}

/// glue.cu pick_merge: out int64 [n + 3] = each row's pick over the ranks, then accepted rows, bonus, next start.
pub fn pickMerge(o: Ops, all: u64, W: usize, n: usize, ids: u64, start: i64, out: u64) !void {
    const f = o.f.pick_merge orelse return error.NoPickKernel;
    if (n == 0 or n > glue.pick_max_rows) return error.Shape;
    var a: cuda.Args = .{};
    a.add(all);
    a.add(@as(c_int, @intCast(W)));
    a.add(@as(c_int, @intCast(n)));
    a.add(ids);
    a.add(start);
    a.add(out);
    try o.go(f, .{ 1, 1, 1 }, @intCast(@max(32, std.mem.alignForward(usize, n, 32))), 0, null, &a);
}

/// The compressor's carry: carry fp32 [cols] = bf16 row of src; row *accepted (device int32) if `accepted` != 0.
pub fn carry(o: Ops, src: u64, ld: usize, accepted: u64, row: usize, cols: usize, out: u64) !void {
    var a: cuda.Args = .{};
    a.add(src);
    a.add(@as(i64, @intCast(ld)));
    a.add(accepted);
    a.add(@as(c_int, @intCast(row)));
    a.add(@as(c_int, @intCast(cols)));
    a.add(out);
    try o.glueGo(.carry, cols, &a);
}
