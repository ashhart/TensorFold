//! CPU oracle for one Qwen3.8-Flash-Next full-attention layer (layers 3, 7, ... 47), written against the model's own definition:
//! `qwen4_exp/model_layers.py` (`SparseAttention`, `Indexer`, `CenteredRMSNorm`) and the Python family's `glue.attn_prep`
//! and `attention._pool/_scores/_select`, read from the TensorFold Python family on par1-cs13.
//!
//! What was checked against that definition (and what each test below pins):
//!  - q_proj has 2 * 24 * 256 outputs and is reshaped [heads, 2 * 256]: per head the first 256 are the query, the last 256 the
//!    output gate (interleaved per head, not two halves). `prepare` reads q at `h * 512` and `attend` reads gate at `h * 512 + 256`.
//!  - q_norm / k_norm / indexer q_layernorm / k_layernorm: fp32 RMSNorm, eps 1e-6, scale `1 + w` for centred storage (offset from
//!    `exl3_scan.centredOffset`, the caller adds it), result rounded to bf16.
//!  - RoPE is `mx.fast.rope(traditional=False)`: rotate-half over the first `rotary_dim = 0.25 * 256 = 64` dims, pairs (i, i + 32),
//!    angle `pos * theta^(-2 i / 64)`, theta 1e7. mrope_interleaved sections [11, 11, 10] pick the t/h/w position per frequency index
//!    (`i % 3 == 1 && i < 33` is h, `i % 3 == 2 && i < 30` is w, else t); for text every axis is the same 1D position, so the angle
//!    is `pos * inv_freq[i]` (`mropeAxisCounts` below pins the 11/11/10 partition).
//!  - Output gate: `bf16(bf16(o) / (1 + exp(-gate)))` before o_proj (fp16 here: the EXL3 linear reads fp16 and bf16 values fit).
//!  - Indexer: `index_qk_proj` is 5 * 128 outputs, 4 query heads then one key. The key is stored raw (bf16). Blocks of 4 keys are
//!    pooled (fp32 mean -> bf16 -> k_layernorm -> RoPE at the block's first position 4 * b -> bf16). Queries get q_layernorm + RoPE at
//!    their own position. Score = sum over heads of relu(q . pooled) / sqrt(128). A row with at most 512 complete blocks (end <= 2051)
//!    attends densely; otherwise it attends to the 512 best blocks (ties to the lower block id) and the 0..3 unfinished tail keys.
//!  - Attention scale is `256 ** -0.5`; GQA maps q head h to kv head h / 12.
//! Arithmetic is f64 with the model's rounding points (bf16 after norm, after RoPE, after o; fp16 where a linear reads or writes).

const std = @import("std");
const shape = @import("exl3_shape.zig");

pub const q_heads: usize = shape.heads;
pub const kv_heads: usize = shape.kv_heads;
pub const head_dim: usize = shape.head_dim;
pub const rotary_dim: usize = 64;
pub const half: usize = rotary_dim / 2;
pub const index_heads: usize = 4;
pub const index_dim: usize = 128;
pub const ratio: usize = 4;
pub const top_blocks: usize = 512;
pub const theta: f64 = 1.0e7;
pub const eps: f64 = 1.0e-6;
/// Most keys one row ever reads: 512 selected blocks and a 3-key tail, or a dense row of up to 2051 keys.
pub const max_keys: usize = top_blocks * ratio + ratio;
pub const index_width: usize = (index_heads + 1) * index_dim;

pub fn roundBf16(x: f64) f64 {
    const f: f32 = @floatCast(x);
    var u: u32 = @bitCast(f);
    if (!std.math.isFinite(f)) return f;
    // round to nearest even on the low 16 bits
    const lsb = (u >> 16) & 1;
    u +%= 0x7fff + lsb;
    u &= 0xffff0000;
    return @as(f32, @bitCast(u));
}

pub fn roundF16(x: f64) f64 {
    const h: f16 = @floatCast(x);
    return h;
}

/// theta^(-2 i / rotary_dim) as the f32 table the kernels read.
pub fn invFreq() [half]f32 {
    var t: [half]f32 = undefined;
    for (&t, 0..) |*v, i| v.* = @floatCast(std.math.pow(f64, theta, -2.0 * @as(f64, @floatFromInt(i)) / @as(f64, rotary_dim)));
    return t;
}

pub const Norms = struct {
    /// Scales with the centred offset already added: `w + offset`.
    q: []const f32,
    k: []const f32,
    iq: []const f32,
    ik: []const f32,
    inv: [half]f32,
};

/// fp32 RMSNorm, bf16, rotate-half RoPE over the first 64 dims at `pos`, bf16. `x` and `out` are `width` long.
pub fn normRope(x: []const f64, scale: []const f32, pos: f64, inv: *const [half]f32, out: []f64) void {
    const width = x.len;
    var ss: f64 = 0;
    for (x) |v| ss += v * v;
    const rinv = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(width)) + eps);
    var xn: [head_dim]f64 = undefined;
    for (x, 0..) |v, i| xn[i] = roundBf16(v * rinv * @as(f64, scale[i]));
    const posf: f32 = @floatCast(pos);
    for (0..width) |d| {
        if (d < half) {
            const ang: f32 = posf * inv[d];
            out[d] = roundBf16(xn[d] * @cos(@as(f64, ang)) - xn[d + half] * @sin(@as(f64, ang)));
        } else if (d < 2 * half) {
            const ang: f32 = posf * inv[d - half];
            out[d] = roundBf16(xn[d - half] * @sin(@as(f64, ang)) + xn[d] * @cos(@as(f64, ang)));
        } else out[d] = xn[d];
    }
}

/// How many of the 32 frequency indices each mrope axis owns with `mrope_interleaved` sections [11, 11, 10].
pub fn mropeAxisCounts() [3]usize {
    var c: [3]usize = .{ 0, 0, 0 };
    for (0..half) |i| {
        if (i % 3 == 1 and i < 3 * 11) {
            c[1] += 1;
        } else if (i % 3 == 2 and i < 3 * 10) {
            c[2] += 1;
        } else c[0] += 1;
    }
    return c;
}

/// One full-attention layer's state: bf16-rounded keys and values, raw index keys, pooled blocks.
pub const Cache = struct {
    gpa: std.mem.Allocator,
    tokens: usize,
    keys: []f64, // [pos][kv head][256]
    values: []f64,
    raw: []f64, // [pos][128]
    pooled: []f64, // [pos / 4][128]

    pub fn init(gpa: std.mem.Allocator, tokens: usize) !Cache {
        const keys = try gpa.alloc(f64, tokens * kv_heads * head_dim);
        errdefer gpa.free(keys);
        const values = try gpa.alloc(f64, tokens * kv_heads * head_dim);
        errdefer gpa.free(values);
        const raw = try gpa.alloc(f64, tokens * index_dim);
        errdefer gpa.free(raw);
        const pooled = try gpa.alloc(f64, (tokens / ratio + 1) * index_dim);
        return .{ .gpa = gpa, .tokens = tokens, .keys = keys, .values = values, .raw = raw, .pooled = pooled };
    }

    pub fn deinit(c: *Cache) void {
        c.gpa.free(c.keys);
        c.gpa.free(c.values);
        c.gpa.free(c.raw);
        c.gpa.free(c.pooled);
    }
};

/// Projections of one token (what the EXL3 linears emit, fp16): q_proj [24 * 512], k_proj [512], v_proj [512], index_qk_proj [640].
pub const Proj = struct {
    qp: []const f16,
    kp: []const f16,
    vp: []const f16,
    ip: []const f16,
};

/// The normed, rotated queries (bf16 values, [24 * 256]) and indexer queries ([4 * 128]) of one token. Writes this token's
/// K, V and raw index key into the cache.
pub fn prepare(c: *Cache, pos: usize, p: Proj, n: *const Norms, q_out: []f64, iq_out: []f64) void {
    var buf: [head_dim]f64 = undefined;
    for (0..q_heads) |h| {
        for (0..head_dim) |d| buf[d] = p.qp[h * 2 * head_dim + d];
        normRope(buf[0..head_dim], n.q, @floatFromInt(pos), &n.inv, q_out[h * head_dim ..][0..head_dim]);
    }
    for (0..kv_heads) |j| {
        for (0..head_dim) |d| buf[d] = p.kp[j * head_dim + d];
        normRope(buf[0..head_dim], n.k, @floatFromInt(pos), &n.inv, c.keys[(pos * kv_heads + j) * head_dim ..][0..head_dim]);
        for (0..head_dim) |d| c.values[(pos * kv_heads + j) * head_dim + d] = roundBf16(p.vp[j * head_dim + d]);
    }
    for (0..index_heads) |h| {
        for (0..index_dim) |d| buf[d] = p.ip[h * index_dim + d];
        normRope(buf[0..index_dim], n.iq, @floatFromInt(pos), &n.inv, iq_out[h * index_dim ..][0..index_dim]);
    }
    for (0..index_dim) |d| c.raw[pos * index_dim + d] = roundBf16(p.ip[index_heads * index_dim + d]);
}

/// Pool block `b` (keys 4b .. 4b + 3 must be in `raw`).
pub fn poolBlock(c: *Cache, b: usize, n: *const Norms) void {
    var mean: [index_dim]f64 = undefined;
    for (0..index_dim) |d| {
        var acc: f32 = @floatCast(c.raw[(ratio * b) * index_dim + d]);
        for (1..ratio) |k| acc = acc + @as(f32, @floatCast(c.raw[(ratio * b + k) * index_dim + d]));
        mean[d] = roundBf16(@as(f64, acc / @as(f32, ratio)));
    }
    normRope(&mean, n.ik, @floatFromInt(ratio * b), &n.inv, c.pooled[b * index_dim ..][0..index_dim]);
}

/// Score of every complete block for one indexer query: sum_h relu(q_h . pooled_b) / sqrt(128).
pub fn blockScores(c: *const Cache, iq: []const f64, complete: usize, out: []f64) void {
    for (0..complete) |b| {
        var total: f64 = 0;
        for (0..index_heads) |h| {
            var dot: f64 = 0;
            for (0..index_dim) |d| dot += iq[h * index_dim + d] * c.pooled[b * index_dim + d];
            total += @max(dot, 0);
        }
        out[b] = total / @sqrt(@as(f64, index_dim));
    }
}

/// The `top_blocks` best blocks, ties to the lower id, in ascending block order. `scores.len` is the complete-block count.
pub fn selectBlocks(gpa: std.mem.Allocator, scores: []const f64, out: *[top_blocks]usize) !void {
    const order = try gpa.alloc(usize, scores.len);
    defer gpa.free(order);
    for (order, 0..) |*o, i| o.* = i;
    const Ctx = struct {
        s: []const f64,
        fn less(ctx: @This(), a: usize, b: usize) bool {
            if (ctx.s[a] != ctx.s[b]) return ctx.s[a] > ctx.s[b];
            return a < b;
        }
    };
    std.mem.sort(usize, order, Ctx{ .s = scores }, Ctx.less);
    @memcpy(out, order[0..top_blocks]);
    std.mem.sort(usize, out, {}, std.sort.asc(usize));
}

/// Keys a row at `end = pos + 1` reads. `blocks` is the selection for a sparse row (ignored when dense).
/// Dense rows read keys 0 .. end; sparse rows read 4 keys per selected block, then the tail. Returns the key count.
pub fn keyList(end: usize, blocks: ?*const [top_blocks]usize, out: *[max_keys]usize) usize {
    const complete = end / ratio;
    if (complete <= top_blocks) {
        for (0..end) |i| out[i] = i;
        return end;
    }
    const sel = blocks.?;
    var n: usize = 0;
    for (sel) |b| {
        for (0..ratio) |k| {
            out[n] = b * ratio + k;
            n += 1;
        }
    }
    var t = ratio * complete;
    while (t < end) : (t += 1) {
        out[n] = t;
        n += 1;
    }
    return n;
}

/// out[h * 256 + d] = fp16(bf16(bf16(o) / (1 + exp(-gate)))), o = softmax(q . K / 16) V over `keys`.
pub fn attend(c: *const Cache, q: []const f64, qp: []const f16, keys: []const usize, out: []f64) void {
    var s: [max_keys]f64 = undefined;
    const group = q_heads / kv_heads;
    for (0..q_heads) |h| {
        const kvh = h / group;
        var m: f64 = -std.math.inf(f64);
        for (keys, 0..) |key, j| {
            var dot: f64 = 0;
            for (0..head_dim) |d| dot += q[h * head_dim + d] * c.keys[(key * kv_heads + kvh) * head_dim + d];
            s[j] = dot * 0.0625;
            m = @max(m, s[j]);
        }
        var l: f64 = 0;
        for (keys, 0..) |_, j| {
            s[j] = @exp(s[j] - m);
            l += s[j];
        }
        for (0..head_dim) |d| {
            var acc: f64 = 0;
            for (keys, 0..) |key, j| acc += s[j] * c.values[(key * kv_heads + kvh) * head_dim + d];
            const o = roundBf16(acc / l);
            const g: f64 = qp[h * 2 * head_dim + head_dim + d];
            out[h * head_dim + d] = roundF16(roundBf16(o / (1.0 + @exp(-g))));
        }
    }
}

pub const RowInfo = struct {
    sparse: bool = false,
    nk: usize = 0,
    /// The selected blocks of a sparse row (undefined when dense).
    blocks: [top_blocks]usize = undefined,
};

/// The whole row: prepare, pool the blocks this token completes, select, attend. `scores` is scratch for tokens / 4 blocks.
pub fn row(
    c: *Cache,
    pos: usize,
    p: Proj,
    n: *const Norms,
    scores: []f64,
    q: []f64,
    iq: []f64,
    out: []f64,
    info: *RowInfo,
) !void {
    prepare(c, pos, p, n, q, iq);
    const end = pos + 1;
    if (end % ratio == 0) poolBlock(c, end / ratio - 1, n);
    const complete = end / ratio;
    info.sparse = complete > top_blocks;
    if (info.sparse) {
        blockScores(c, iq, complete, scores);
        try selectBlocks(c.gpa, scores[0..complete], &info.blocks);
    }
    var keys: [max_keys]usize = undefined;
    info.nk = keyList(end, if (info.sparse) &info.blocks else null, &keys);
    attend(c, q, p.qp, keys[0..info.nk], out);
}

// -- tests -----------------------------------------------------------------

test "mrope sections [11, 11, 10] interleave over 32 frequencies and need no per-axis table for text" {
    const counts = mropeAxisCounts();
    try std.testing.expectEqual([3]usize{ 11, 11, 10 }, counts);
    try std.testing.expectEqual(@as(usize, 32), counts[0] + counts[1] + counts[2]);
    try std.testing.expectEqual(half, counts[0] + counts[1] + counts[2]);
}

test "rotate-half RoPE matches an explicit rotation of the pairs (i, i + 32) and leaves dims 64.. alone" {
    const inv = invFreq();
    try std.testing.expectApproxEqRel(@as(f32, 1.0), inv[0], 1e-7);
    // theta^(-2*31/64) for the last frequency
    try std.testing.expectApproxEqRel(@as(f32, @floatCast(std.math.pow(f64, 1e7, -62.0 / 64.0))), inv[31], 1e-6);
    var x: [head_dim]f64 = undefined;
    var rng = std.Random.DefaultPrng.init(7);
    for (&x) |*v| v.* = rng.random().float(f64) * 2 - 1;
    var scale: [head_dim]f32 = @splat(1);
    scale[3] = 2.5;
    var out: [head_dim]f64 = undefined;
    const pos: usize = 1234;
    normRope(&x, &scale, @floatFromInt(pos), &inv, &out);

    var ss: f64 = 0;
    for (x) |v| ss += v * v;
    const rinv = 1.0 / @sqrt(ss / head_dim + eps);
    for (0..half) |i| {
        const a = roundBf16(x[i] * rinv * scale[i]);
        const b = roundBf16(x[i + half] * rinv * scale[i + half]);
        const ang: f32 = @as(f32, @floatFromInt(pos)) * inv[i];
        try std.testing.expectEqual(roundBf16(a * @cos(@as(f64, ang)) - b * @sin(@as(f64, ang))), out[i]);
        try std.testing.expectEqual(roundBf16(a * @sin(@as(f64, ang)) + b * @cos(@as(f64, ang))), out[i + half]);
    }
    for (rotary_dim..head_dim) |d| try std.testing.expectEqual(roundBf16(x[d] * rinv * scale[d]), out[d]);
    // position 0 is the identity rotation
    normRope(&x, &scale, 0, &inv, &out);
    for (0..head_dim) |d| try std.testing.expectEqual(roundBf16(x[d] * rinv * scale[d]), out[d]);
}

test "bf16 rounding is round-to-nearest-even" {
    try std.testing.expectEqual(@as(f64, 1.0), roundBf16(1.0 + 1.0 / 512.0)); // below half an ulp (ulp 2^-7)
    try std.testing.expectEqual(@as(f64, 1.0), roundBf16(1.0 + 1.0 / 256.0)); // tie goes to even (1.0)
    try std.testing.expectEqual(@as(f64, 1.0 + 1.0 / 64.0), roundBf16(1.0 + 3.0 / 256.0)); // tie goes up to even mantissa
    try std.testing.expectEqual(@as(f64, 1.0 + 1.0 / 128.0), roundBf16(1.0 + 1.0 / 128.0 + 1.0 / 1024.0));
}

test "selection takes the 512 best blocks, ties to the lower id, in block order" {
    const gpa = std.testing.allocator;
    const n: usize = 700;
    const scores = try gpa.alloc(f64, n);
    defer gpa.free(scores);
    var rng = std.Random.DefaultPrng.init(11);
    for (scores) |*v| v.* = @floor(rng.random().float(f64) * 40) / 4; // many exact ties
    var sel: [top_blocks]usize = undefined;
    try selectBlocks(gpa, scores, &sel);
    try std.testing.expect(std.sort.isSorted(usize, &sel, {}, std.sort.asc(usize)));
    // the cut: every unselected block scores no more than every selected one, and a tie at the cut keeps the lower id
    var chosen: [700]bool = @splat(false);
    var min_sel: f64 = std.math.inf(f64);
    for (sel) |b| {
        chosen[b] = true;
        min_sel = @min(min_sel, scores[b]);
    }
    var count: usize = 0;
    for (chosen) |c| count += @intFromBool(c);
    try std.testing.expectEqual(top_blocks, count);
    var max_out: f64 = -1;
    var lowest_out_at_cut: usize = n;
    for (0..n) |b| if (!chosen[b]) {
        max_out = @max(max_out, scores[b]);
        if (scores[b] == min_sel) lowest_out_at_cut = @min(lowest_out_at_cut, b);
    };
    try std.testing.expect(max_out <= min_sel);
    for (0..n) |b| if (chosen[b] and scores[b] == min_sel) try std.testing.expect(b < lowest_out_at_cut);
}

test "key list: dense through 2051 keys, then 2048 selected plus the unfinished tail" {
    var sel: [top_blocks]usize = undefined;
    for (&sel, 0..) |*b, i| b.* = i + 1; // blocks 1..512 of 513
    var keys: [max_keys]usize = undefined;
    try std.testing.expectEqual(@as(usize, 2051), keyList(2051, null, &keys));
    try std.testing.expectEqual(@as(usize, 2050), keys[2050]);
    // end 2052: 513 complete blocks, no tail, block 0 dropped
    try std.testing.expectEqual(@as(usize, 2048), keyList(2052, &sel, &keys));
    try std.testing.expectEqual(@as(usize, 4), keys[0]);
    try std.testing.expectEqual(@as(usize, 2051), keys[2047]);
    // end 2055: three tail keys 2052, 2053, 2054
    try std.testing.expectEqual(@as(usize, 2051), keyList(2055, &sel, &keys));
    try std.testing.expectEqual(@as(usize, 2052), keys[2048]);
    try std.testing.expectEqual(@as(usize, 2054), keys[2050]);
}

test "attend with one visible key returns that value gated" {
    const gpa = std.testing.allocator;
    var c = try Cache.init(gpa, 4);
    defer c.deinit();
    for (0..kv_heads * head_dim) |i| {
        c.keys[i] = 0.25;
        c.values[i] = if (i < head_dim) 2.0 else -3.0;
    }
    var q: [q_heads * head_dim]f64 = @splat(0.5);
    var qp: [q_heads * 2 * head_dim]f16 = @splat(0);
    for (0..q_heads) |h| for (0..head_dim) |d| {
        qp[h * 2 * head_dim + head_dim + d] = 0; // sigmoid(0) = 1/2
    };
    var out: [q_heads * head_dim]f64 = undefined;
    attend(&c, &q, &qp, &.{0}, &out);
    for (0..12) |h| try std.testing.expectEqual(@as(f64, 1.0), out[h * head_dim]);
    for (12..q_heads) |h| try std.testing.expectEqual(@as(f64, -1.5), out[h * head_dim]);
}

test "the device KV layout is exl3_kv.Table's for one layer" {
    const kv = @import("exl3_kv.zig");
    const gpa = std.testing.allocator;
    const tokens: usize = 8;
    var t = try kv.Table.init(gpa, tokens, .bf16, 0, 1);
    defer t.deinit(gpa);
    try std.testing.expectEqual(@as(usize, kv_heads), t.heads);
    const slot_bytes = 2 * head_dim * 2;
    for ([_]usize{ 0, 1, 2 }) |layer| for ([_]usize{ 0, 5, 7 }) |token| for ([_]usize{ 0, 1 }) |head| {
        var src: [slot_bytes]u8 = undefined;
        @memset(&src, @intCast(1 + layer * 40 + token * 4 + head));
        try t.write(head, layer, token, &src);
        // the kernel's byte offset: layer base + (token * kv_heads + head) * slot
        const off = layer * tokens * kv_heads * slot_bytes + (token * kv_heads + head) * slot_bytes;
        try std.testing.expectEqualSlices(u8, &src, t.bytes[off..][0..slot_bytes]);
    };
}

test "a full row past 2048 keys reads the selected blocks, and the dense row at 2051 keys does not" {
    const gpa = std.testing.allocator;
    const tokens: usize = 2100;
    var c = try Cache.init(gpa, tokens);
    defer c.deinit();
    var rng = std.Random.DefaultPrng.init(3);
    const inv = invFreq();
    const q_scale: [head_dim]f32 = @splat(1);
    const i_scale: [index_dim]f32 = @splat(1);
    const norms = Norms{ .q = &q_scale, .k = &q_scale, .iq = &i_scale, .ik = &i_scale, .inv = inv };
    const qp = try gpa.alloc(f16, q_heads * 2 * head_dim);
    defer gpa.free(qp);
    const kp = try gpa.alloc(f16, kv_heads * head_dim);
    defer gpa.free(kp);
    const vp = try gpa.alloc(f16, kv_heads * head_dim);
    defer gpa.free(vp);
    const ip = try gpa.alloc(f16, index_width);
    defer gpa.free(ip);
    const scores = try gpa.alloc(f64, tokens / ratio + 1);
    defer gpa.free(scores);
    var q: [q_heads * head_dim]f64 = undefined;
    var iq: [index_heads * index_dim]f64 = undefined;
    var out: [q_heads * head_dim]f64 = undefined;
    var info = RowInfo{};
    for (0..2056) |pos| {
        for (qp) |*v| v.* = @floatCast(rng.random().floatNorm(f32));
        for (kp) |*v| v.* = @floatCast(rng.random().floatNorm(f32));
        for (vp) |*v| v.* = @floatCast(rng.random().floatNorm(f32));
        for (ip) |*v| v.* = @floatCast(rng.random().floatNorm(f32));
        const proj = Proj{ .qp = qp, .kp = kp, .vp = vp, .ip = ip };
        switch (pos) {
            2050, 2051, 2054 => {
                try row(&c, pos, proj, &norms, scores, &q, &iq, &out, &info);
                switch (pos) {
                    2050 => try std.testing.expect(!info.sparse and info.nk == 2051),
                    2051 => try std.testing.expect(info.sparse and info.nk == 2048),
                    else => try std.testing.expect(info.sparse and info.nk == 2051),
                }
            },
            else => {
                prepare(&c, pos, proj, &norms, &q, &iq);
                if ((pos + 1) % ratio == 0) poolBlock(&c, (pos + 1) / ratio - 1, &norms);
            },
        }
    }
    for (out) |v| try std.testing.expect(std.math.isFinite(v));
}
