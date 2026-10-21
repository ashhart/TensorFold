//! An EXL3 pack's MTP head on v1.1.0's Flash Next family (cuda_mtp.zig): the draft vocabulary rows here, the rest a design.
//!
//! ## What the head computes (upstream cuda_mtp_ops.zig `forwardHead`, = tensorfold qwen4_exp mtp.py `FlashMTP`)
//!   e   = fc_embedding(norm_e(embed(token t+1)))             norm over 2,560 (tri.rmsnorm, group = dims)
//!   hs  = fc_hidden(norm_h(streams_t) as 4 rows of 2,560)     ONE RMS over the whole 10,240 row (group = wide), then
//!                                                             fc_hidden per stream: what the EXL3 reference does too
//!   x   = e + hs[s] per stream; one attention + MoE decoder layer (`mtp.layers.0`, its own KV, no PLE)
//!   out = hyper_connection_mixer readout of x, then the draft head: bf16 rows of lm_head at the 79,591 draft ids.
//! Upstream's maths already matches the EXL3 reference; only the weights' formats differ.
//!
//! ## EXL3 tensors (pack on par1-cs13, index + mtp_hyper_connection_mixer_patch.safetensors, read 2026-10-19)
//!   mtp.fc_embedding.{trellis,suh,svh,mul1}, mtp.fc_hidden.{...}       EXL3 2560 -> 2560 (mtp_bits 3)
//!   mtp.pre_fc_norm_embedding.weight [2560], mtp.pre_fc_norm_hidden.weight [10240]   gamma - 1 (store.norm_offset)
//!   mtp.layers.0.{self_attn.*, mlp.experts.N.*, mlp.shared_expert.*, mlp.gate, mlp.shared_expert_gate,
//!                 attn_hyper_connection.*, mlp_hyper_connection.*}     as a main attention layer (F16 gates and HC)
//!   mtp.hyper_connection_mixer.{hc_norm [10240], input_mix_weight_down [320,10240], _up [10240,320]}  F16, in the
//!                 patch file, which model.safetensors.index.json does not name
//!   lm_head.{trellis [160, 15520, 80], suh [2560], svh [248320], mul1}  EXL3 5-bit (head_bits 5), K = 2560, N = vocab
//!
//! ## Hook points (one line each into upstream files; the work lives in cuda_exl3_*.zig)
//!   1. cuda_mtp_load.zig `draft` (after the mode check): `if (try exl3_mtp.draft(a, d, store, ids, affine)) |q| return q;`
//!      DONE here: `draft` below dequantises the lm_head columns of the draft ids (exl3_format.dequantize on the
//!      128-column blocks that hold them) to bf16 rows [ids, 2560] and uploads them through projection.dense (or the
//!      affine4 packer under TF_FLASHNEXT_DRAFT_HEAD=affine4). The forward's `p.lane(.., h.draft, ..)` is unchanged.
//!   2. cuda_mtp.zig Head.init `fc` (line ~198): `cuda_projection_load.load` refuses EXL3 (`UnsupportedQuantization`:
//!      no `.weight`). E1's `Projection.exl3` load (plan step 4) covers it: in `load`, `if (store.has(prefix ++
//!      ".trellis")) return cuda_exl3_proj.load(...)`. `forwardHead` runs fc_e/fc_h through `Pass.lane(.., fc_sk = 4)`:
//!      an EXL3 lane must honour `sk > 1` callers that read `part` (fp32 slices) and `sumSlices`, or return sk = 1
//!      with the sum already in `out` (then the two `sumSlices` calls at cuda_mtp_ops.zig:146 and :150 must be
//!      skipped for EXL3: one branch each, `if (!h.fc_e.isExl3())`).
//!   3. cuda_mtp.zig Head.init `wt.loadHc(.., "language_model.mtp.hyper_connection_mixer", false)` (line ~179): the
//!      Store (cuda_prompt_wt.zig `resolveKey`/`shard`) must fall back to the patch file for names the index lacks
//!      (plan step 2), and `loadHc` needs the F16 dense HC path (plan step 3; `loadDenseHc` takes bf16 today, and
//!      `asF32` refuses F16, so `normScale` of an F16 hc_norm fails too).
//!   4. cuda_mtp.zig Head.init `load.loadLayerAt(.., "language_model.mtp.layers.0", false, false)` (line ~185): the
//!      alias `language_model.mtp.` -> `mtp.` already resolves; the layer needs E1's EXL3 attention projections and
//!      `Experts` (plan steps 4-5) and the F16 router/shared gate (step 3). Its experts can stay resident (one layer,
//!      513 x 3-bit experts, about 0.95 GB) or ride on the offload pool as a 49th layer.
//!   5. cuda_mtp.zig `store.detectNorms()` (line ~164) reads the 48 decoder `attn_hyper_connection.hc_norm` (F16 in
//!      this pack): same F16 `asF32` fix as 3. q_norm/k_norm of `mtp.layers.0.self_attn` average ~2.7 stored, still
//!      gamma - 1 (the branch's exl3_mtp.zig used the single convention); upstream applies one offset to all: fine.
//!   6. cuda_mtp.zig `headBytes` (admission estimate) counts affine bytes; an EXL3 head is fc 2 x 2560 x 2560 x 3/8,
//!      the layer's EXL3 bytes and the bf16 draft rows (79,591 x 2560 x 2 = 407.5 MB). A one-line branch on the format.
//!   7. Not needed: the draft chain, verify windows, rollback (cuda_mtp_draft.zig, cuda_mtp_ops.zig) are format-free.
//!
//! ## Draft rows: what is exact and what is not
//! EXL3 columns are not separable (a 128-column Hadamard mixes them), so a row is decoded from its whole column block
//! and rounded to bf16 once (round to nearest even from the float64 value). A draft logit then differs from the full
//! head's by at most 2^-9 * sum_k |x_k W[k,v]| (bf16 keeps 8 significant bits), the bound `tf-cuda-test exl3-draft`
//! checks against the full EXL3 head (upstream exl3.Dense, fp32 out) and an exact float64 product. Drafts are proposals:
//! the main head verifies every token, so acceptance (drafted == undrafted) is unaffected; only the acceptance rate is.

const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const projection = @import("cuda_projection.zig");
const fmt = core.exl3_format;

const block_n: usize = fmt.hadamard_dim;

/// lm_head as stored: trellis int16 [K/16, N/16, 8 * halves] (bytes, possibly unaligned), suh fp16 [K], svh fp16 [N].
pub const Head = struct {
    trellis: []const u8,
    suh: []const u8,
    svh: []const u8,
    k: usize,
    n: usize,
    halves: u16,

    pub fn words(h: Head) usize {
        return @as(usize, h.halves) * 8;
    }

    /// From the stored tensors; `mul1` (when present) must hold the mul1 multiplier.
    pub fn of(trellis: core.safetensors.Tensor, suh: core.safetensors.Tensor, svh: core.safetensors.Tensor, mul1: ?core.safetensors.Tensor) !Head {
        if (trellis.dtype != .i16 or trellis.rank != 3 or trellis.dim(2) % 8 != 0) return error.UnexpectedTensor;
        const h: Head = .{ .trellis = trellis.bytes, .suh = suh.bytes, .svh = svh.bytes, .k = trellis.dim(0) * 16, .n = trellis.dim(1) * 16, .halves = @intCast(trellis.dim(2) / 8) };
        if (suh.dtype != .f16 or suh.numel() != h.k or svh.dtype != .f16 or svh.numel() != h.n) return error.UnexpectedTensor;
        if (h.k % block_n != 0 or h.n % block_n != 0 or h.halves < 2 or h.halves > 16) return error.UnexpectedTensor;
        if (mul1) |m| {
            if (m.bytes.len != 4 or std.mem.readInt(u32, m.bytes[0..4], .little) != fmt.Mul1Mul) return error.UnexpectedTensor;
        }
        return h;
    }
};

/// f64 -> bf16 bits, one rounding to nearest even (no f32 step, so no double rounding).
pub fn bf16Of(v: f64) u16 {
    const near = wt.toBf16(@floatCast(v));
    if (std.math.isNan(v) or std.math.isInf(v)) return near;
    var best = near;
    var best_d = @abs(@as(f64, wt.promote(near)) - v);
    for ([_]u16{ near -% 1, near +% 1 }) |cand| {
        const cv: f64 = wt.promote(cand);
        if (std.math.isNan(cv) or std.math.isInf(cv)) continue;
        if (std.math.signbit(cv) != std.math.signbit(@as(f64, wt.promote(near))) and cv != 0) continue;
        const d = @abs(cv - v);
        if (d < best_d or (d == best_d and cand & 1 == 0)) {
            best = cand;
            best_d = d;
        }
    }
    return best;
}

const Work = struct {
    head: Head,
    ids: []const u32,
    /// Indices into `ids`, sorted by id.
    order: []const u32,
    /// Start of each block's run in `order`; `blocks` entries plus one.
    starts: []const u32,
    out: []u16,
    next: std.atomic.Value(usize) = .init(0),
    failed: ?anyerror = null,
};

fn copyF16(bytes: []const u8, out: []f16) void {
    for (out, 0..) |*o, i| o.* = @bitCast(std.mem.readInt(u16, bytes[2 * i ..][0..2], .little));
}

fn worker(w: *Work, gpa: std.mem.Allocator) void {
    work(w, gpa) catch |err| {
        w.failed = err;
    };
}

fn work(w: *Work, gpa: std.mem.Allocator) !void {
    const h = w.head;
    const tiles_k = h.k / 16;
    const tiles_n = h.n / 16;
    const per = block_n / 16;
    const words = h.words();
    const sub = try gpa.alloc(i16, tiles_k * per * words);
    defer gpa.free(sub);
    const vals = try gpa.alloc(f64, h.k * block_n);
    defer gpa.free(vals);
    const scratch = try gpa.alloc(f64, 2 * h.k * block_n);
    defer gpa.free(scratch);
    const suh = try gpa.alloc(f16, h.k);
    defer gpa.free(suh);
    copyF16(h.suh, suh);
    var svh: [block_n]f16 = undefined;
    while (true) {
        const b = w.next.fetchAdd(1, .monotonic);
        if (b + 1 >= w.starts.len) return;
        const lo = w.starts[b];
        const hi = w.starts[b + 1];
        if (lo == hi) continue;
        const nb = w.ids[w.order[lo]] / block_n;
        // the block's tiles, k-major as a [tiles_k, per] trellis of their own
        for (0..tiles_k) |tk| for (0..per) |j| {
            const src = ((tk * tiles_n) + nb * per + j) * words * 2;
            const dst = sub[(tk * per + j) * words ..][0..words];
            for (dst, 0..) |*d, i| d.* = std.mem.readInt(i16, h.trellis[src + 2 * i ..][0..2], .little);
        };
        copyF16(h.svh[nb * block_n * 2 ..][0 .. block_n * 2], &svh);
        fmt.dequantize(.{ .halves = h.halves }, .mul1, sub, tiles_k, per, suh, &svh, vals, scratch);
        for (w.order[lo..hi]) |i| {
            const col = w.ids[i] - nb * block_n;
            const row = w.out[@as(usize, i) * h.k ..][0..h.k];
            for (row, 0..) |*o, k| o.* = bf16Of(vals[k * block_n + col]);
        }
    }
}

/// Row i of `out` ([ids.len, K] bf16) is lm_head's column `ids[i]` (the weight row of token ids[i]), decoded in float64
/// by exl3_format.dequantize on the 128-column block that holds it and rounded once to bf16. Up to `threads` threads.
pub fn draftRows(gpa: std.mem.Allocator, head: Head, ids: []const u32, out: []u16, threads: usize) !void {
    if (out.len != ids.len * head.k) return error.UnexpectedTensor;
    for (ids) |id| if (id >= head.n) return error.InvalidToken;
    const order = try gpa.alloc(u32, ids.len);
    defer gpa.free(order);
    for (order, 0..) |*o, i| o.* = @intCast(i);
    std.mem.sort(u32, order, ids, struct {
        fn lt(x: []const u32, a: u32, b: u32) bool {
            return x[a] < x[b];
        }
    }.lt);
    // runs of one block each
    var starts: std.ArrayList(u32) = .empty;
    defer starts.deinit(gpa);
    var prev: ?usize = null;
    for (order, 0..) |i, at| {
        const nb = ids[i] / block_n;
        if (prev == null or prev.? != nb) try starts.append(gpa, @intCast(at));
        prev = nb;
    }
    try starts.append(gpa, @intCast(order.len));
    var w: Work = .{ .head = head, .ids = ids, .order = order, .starts = starts.items, .out = out };
    const n_threads = @max(1, @min(threads, starts.items.len - 1));
    if (n_threads == 1) {
        try work(&w, gpa);
        return;
    }
    var pool: [64]std.Thread = undefined;
    var spawned: usize = 0;
    defer for (pool[0..spawned]) |t| t.join();
    for (0..@min(n_threads, pool.len)) |_| {
        pool[spawned] = try std.Thread.spawn(.{}, worker, .{ &w, gpa });
        spawned += 1;
    }
    for (pool[0..spawned]) |t| t.join();
    spawned = 0;
    if (w.failed) |err| return err;
}

/// Threads for a load: the machine's cores, at most 32 (each holds about 8 MB of float64 scratch).
pub fn loadThreads() usize {
    return @min(32, std.Thread.getCpuCount() catch 1);
}

/// The draft head of an EXL3 pack (null for any other): bf16 rows uploaded as a dense projection, or affine4 rows.
pub fn draft(a: std.mem.Allocator, d: *cuda.Driver, store: *wt.Store, ids: []const u32, affine: bool) !?wt.Q4 {
    if (!store.has("lm_head.trellis")) return null;
    const head = try Head.of(try store.hold("lm_head.trellis"), try store.hold("lm_head.suh"), try store.hold("lm_head.svh"), if (store.has("lm_head.mul1")) try store.hold("lm_head.mul1") else null);
    if (head.k != wt.dims or head.n != wt.vocab) return error.UnexpectedTensor;
    const rows = try a.alloc(u16, ids.len * wt.dims);
    defer a.free(rows);
    try draftRows(a, head, ids, rows, loadThreads());
    const bytes = std.mem.sliceAsBytes(rows);
    if (!affine) return try projection.dense(a, d, .{ .dtype = .bf16, .rank = 2, .shape = .{ ids.len, wt.dims, 1, 1, 1 }, .bytes = bytes });
    const words = try a.alloc(u8, ids.len * wt.dims / 2);
    defer a.free(words);
    const scales = try a.alloc(u8, ids.len * wt.dims / 16);
    defer a.free(scales);
    const biases = try a.alloc(u8, scales.len);
    defer a.free(biases);
    for (0..ids.len) |i| try @import("cuda_draft_affine.zig").row(bytes[i * wt.dims * 2 ..][0 .. wt.dims * 2], words[i * wt.dims / 2 ..][0 .. wt.dims / 2], scales[i * wt.dims / 16 ..][0 .. wt.dims / 16], biases[i * wt.dims / 16 ..][0 .. wt.dims / 16]);
    return try wt.uploadQ4(a, d, words, scales, biases, ids.len, wt.dims / 8);
}

// ---- tests --------------------------------------------------------------------------------------------------------

test "exl3 draft: bf16Of rounds float64 once, to nearest even" {
    try std.testing.expectEqual(@as(u16, 0x3F80), bf16Of(1.0));
    try std.testing.expectEqual(@as(u16, 0x3F80), bf16Of(1.0 + 1.0 / 256.0)); // tie: even (down)
    try std.testing.expectEqual(@as(u16, 0x3F82), bf16Of(1.0 + 3.0 / 256.0)); // tie: even (up)
    // just above a tie after an f32 step would round to the tie: float64 sees it above
    const above = 1.0 + 1.0 / 256.0 + 1.0e-12;
    try std.testing.expectEqual(@as(u16, 0x3F81), bf16Of(above));
    try std.testing.expectEqual(@as(u16, 0xBF80), bf16Of(-1.0));
    try std.testing.expectEqual(@as(u16, 0), bf16Of(0));
}

fn synthHead(gpa: std.mem.Allocator, r: std.Random, k: usize, n: usize, halves: u16) !struct { t: []i16, su: []f16, sv: []f16, head: Head } {
    const t = try gpa.alloc(i16, (k / 16) * (n / 16) * halves * 8);
    for (t) |*x| x.* = @bitCast(r.int(u16));
    const su = try gpa.alloc(f16, k);
    for (su) |*x| x.* = @floatCast((r.float(f32) * 0.02 + 0.005) * (if (r.boolean()) @as(f32, 1) else -1));
    const sv = try gpa.alloc(f16, n);
    for (sv) |*x| x.* = @floatCast((r.float(f32) * 0.5 + 0.5) * (if (r.boolean()) @as(f32, 1) else -1));
    return .{ .t = t, .su = su, .sv = sv, .head = .{ .trellis = std.mem.sliceAsBytes(t), .suh = std.mem.sliceAsBytes(su), .svh = std.mem.sliceAsBytes(sv), .k = k, .n = n, .halves = halves } };
}

test "exl3 draft: rows equal the whole matrix's float64 decode rounded to bf16, any id order, any thread count" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xd4af);
    const r = prng.random();
    const k = 256;
    const n = 384;
    const s = try synthHead(gpa, r, k, n, 10);
    defer gpa.free(s.t);
    defer gpa.free(s.su);
    defer gpa.free(s.sv);
    const full = try gpa.alloc(f64, k * n);
    defer gpa.free(full);
    const scratch = try gpa.alloc(f64, 2 * k * n);
    defer gpa.free(scratch);
    fmt.dequantize(.{ .halves = 10 }, .mul1, s.t, k / 16, n / 16, s.su, s.sv, full, scratch);
    const ids = [_]u32{ 300, 5, 127, 128, 383, 5, 0, 200, 255, 256 };
    for ([_]usize{ 1, 3 }) |threads| {
        var rows: [ids.len * k]u16 = undefined;
        try draftRows(gpa, s.head, &ids, &rows, threads);
        for (ids, 0..) |id, i| for (0..k) |kk| {
            try std.testing.expectEqual(bf16Of(full[kk * n + id]), rows[i * k + kk]);
        };
    }
    var bad: [k]u16 = undefined;
    try std.testing.expectError(error.InvalidToken, draftRows(gpa, s.head, &.{384}, &bad, 1));
}

test "exl3 draft: a draft logit is the full head's within the bf16 weight bound" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x10917);
    const r = prng.random();
    const k = 256;
    const n = 256;
    const s = try synthHead(gpa, r, k, n, 6);
    defer gpa.free(s.t);
    defer gpa.free(s.su);
    defer gpa.free(s.sv);
    const full = try gpa.alloc(f64, k * n);
    defer gpa.free(full);
    const scratch = try gpa.alloc(f64, 2 * k * n);
    defer gpa.free(scratch);
    fmt.dequantize(.{ .halves = 6 }, .mul1, s.t, k / 16, n / 16, s.su, s.sv, full, scratch);
    var x: [k]f64 = undefined;
    for (&x) |*v| v.* = (r.float(f64) - 0.5) * 4;
    var ids: [n]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(i);
    const rows = try gpa.alloc(u16, n * k);
    defer gpa.free(rows);
    try draftRows(gpa, s.head, &ids, rows, 2);
    for (0..n) |v| {
        var exact: f64 = 0;
        var mag: f64 = 0;
        var got: f64 = 0;
        for (0..k) |kk| {
            exact += x[kk] * full[kk * n + v];
            mag += @abs(x[kk] * full[kk * n + v]);
            got += x[kk] * @as(f64, wt.promote(rows[v * k + kk]));
        }
        try std.testing.expect(@abs(got - exact) <= mag / 512.0 + 1e-12);
    }
}
