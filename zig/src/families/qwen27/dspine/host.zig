//! DSpine's code bridge, serial tables and greedy table on the host, rounding where Tencent's dspine_mlx rounds.
const std = @import("std");
const core = @import("core");

/// torch.finfo(torch.float32).tiny: the floor of the bridge read's mean square.
const tiny: f32 = 1.17549435e-38;

pub fn float(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

/// f32 to bf16, round to nearest even (NaN stays NaN).
pub fn round(x: f32) u16 {
    const u: u32 = @bitCast(x);
    if (std.math.isNan(x)) return @intCast((u >> 16) | 0x40);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn dot(a: []align(1) const u16, b: []align(1) const u16) f32 {
    var s: f32 = 0;
    for (a, b) |x, y| s += float(x) * float(y);
    return s;
}

fn sigmoid(x: f32) f32 {
    return 1 / (1 + @exp(-x));
}

/// The bridge's widths and its eps (the bridge's, not the norms').
pub const Dims = struct { hidden: usize, code: usize, rank: usize, vocab: usize, eps: f32 };

/// One write's weights, bf16 row-major as the checkpoint stores them.
pub const Bridge = struct {
    read: []align(1) const u16, // [code, hidden]: early_read_weights[l], or read_weight on the last layer
    codebook: []align(1) const u16, // [vocab, code]
    packed_w: []align(1) const u16, // [3 rank, code]: gate, predecessor_gate, prev
    bias: []align(1) const u16, // [rank]: gate.bias
    write: []align(1) const u16, // [hidden, rank]: feature_write_weights[l]
};

/// The last write's state before it writes: what the serial second pass redoes the write from.
pub const Stash = struct { raw: []u16, rms: []f32, packed_rows: []u16 };

/// The next norm: its weight and eps (the next layer's input_layernorm, or the final norm).
pub const Norm = struct { weight: []align(1) const u16, eps: f32 };

/// rms_norm as the bridge closes a write: y = bf16(x * rsqrt(mean x^2 + eps)), out = bf16(y * w).
pub fn normRow(x: []align(1) const u16, n: Norm, out: []u16) void {
    var ss: f32 = 0;
    for (x) |v| ss += float(v) * float(v);
    const inv = 1 / @sqrt(ss / @as(f32, @floatFromInt(x.len)) + n.eps);
    for (out, x, n.weight) |*o, v, w| o.* = round(float(round(float(v) * inv)) * float(w));
}

/// One write over a block's rows (row 0 the anchor): `out` the written stream, `normed` it through `next`.
pub fn write(gpa: std.mem.Allocator, d: Dims, b: Bridge, layer_out: []align(1) const u16, anchor: u32, next: Norm, out: []u16, normed: []u16, stash: ?Stash) !void {
    const t_rows = layer_out.len / d.hidden;
    if (t_rows < 1 or layer_out.len != t_rows * d.hidden or out.len != layer_out.len or normed.len != out.len or anchor >= d.vocab) return error.BadBridgeShape;
    const sqrt_c: f32 = @sqrt(@as(f32, @floatFromInt(d.code)));
    const rms = try gpa.alloc(f32, t_rows);
    defer gpa.free(rms);
    const xn = try gpa.alloc(u16, d.hidden);
    defer gpa.free(xn);
    const proj = try gpa.alloc(f32, d.code);
    defer gpa.free(proj);
    const codes = try gpa.alloc(u16, t_rows * d.code);
    defer gpa.free(codes);
    const packed_rows = try gpa.alloc(u16, t_rows * 3 * d.rank);
    defer gpa.free(packed_rows);
    const message = try gpa.alloc(u16, d.rank);
    defer gpa.free(message);
    for (0..t_rows) |t| {
        const row = layer_out[t * d.hidden ..][0..d.hidden];
        var ss: f32 = 0;
        for (row) |v| ss += float(v) * float(v);
        rms[t] = @sqrt(@max(ss / @as(f32, @floatFromInt(d.hidden)), tiny));
        const code = codes[t * d.code ..][0..d.code];
        if (t == 0) {
            for (code, d.code * anchor..) |*c, i| c.* = round(sqrt_c * float(b.codebook[i]));
        } else {
            for (xn, row) |*x, v| x.* = round(float(v) / (rms[t] + d.eps));
            var norm: f32 = 0;
            for (proj, 0..) |*p, c| {
                p.* = float(round(dot(xn, b.read[c * d.hidden ..][0..d.hidden])));
                norm += p.* * p.*;
            }
            for (code, proj) |*c, p| c.* = round(sqrt_c * p / (@sqrt(norm) + d.eps));
        }
        for (packed_rows[t * 3 * d.rank ..][0 .. 3 * d.rank], 0..) |*p, j| p.* = round(dot(code, b.packed_w[j * d.code ..][0..d.code]));
    }
    for (0..t_rows) |t| {
        const raw = layer_out[t * d.hidden ..][0..d.hidden];
        const o = out[t * d.hidden ..][0..d.hidden];
        if (t == 0) {
            @memcpy(o, raw);
        } else {
            const q = packed_rows[t * 3 * d.rank ..][0..d.rank];
            const sv = packed_rows[(t - 1) * 3 * d.rank ..][d.rank .. 3 * d.rank];
            for (message, 0..) |*m, i| m.* = round(sigmoid(float(q[i]) + float(sv[i]) + float(b.bias[i])) * float(sv[d.rank + i]));
            for (o, raw, 0..) |*y, x, h| y.* = round(float(x) + rms[t] * float(round(dot(message, b.write[h * d.rank ..][0..d.rank]))));
        }
        normRow(o, next, normed[t * d.hidden ..][0..d.hidden]);
    }
    if (stash) |s| {
        if (s.raw.len != layer_out.len or s.rms.len != t_rows or s.packed_rows.len != packed_rows.len) return error.BadBridgeShape;
        @memcpy(s.raw, layer_out);
        @memcpy(s.rms, rms);
        @memcpy(s.packed_rows, packed_rows);
    }
}

/// TG's row for one head row: bf16(bf16(head * norm) @ write_last) [rank].
pub fn tgRow(d: Dims, head_row: []align(1) const u16, norm_w: []align(1) const u16, write_last: []align(1) const u16, out: []u16) void {
    for (out, 0..) |*o, r| {
        var s: f32 = 0;
        for (head_row, norm_w, 0..) |x, w, h| s += float(round(float(x) * float(w))) * float(write_last[h * d.rank + r]);
        o.* = round(s);
    }
}

/// TSV's row for one id: bf16(bf16(sqrt C * codebook[id]) @ [predecessor_gate; prev].T) [2 rank].
pub fn tsvRow(d: Dims, b: Bridge, id: u32, out: []u16) void {
    const sqrt_c: f32 = @sqrt(@as(f32, @floatFromInt(d.code)));
    const e = b.codebook[d.code * id ..][0..d.code];
    for (out, 0..) |*o, j| {
        const w = b.packed_w[(d.rank + j) * d.code ..][0..d.code];
        var s: f32 = 0;
        for (e, w) |x, y| s += float(round(sqrt_c * float(x))) * float(y);
        o.* = round(s);
    }
}

/// The top `out.len` ids of a logits row, ascending by id; equal logits go to the lower id.
pub fn topk(logits: []align(1) const u16, out: []u32) void {
    var n: usize = 0;
    for (logits, 0..) |v, i| {
        const x = float(v);
        var at = n;
        while (at > 0 and x > float(logits[out[at - 1]])) at -= 1;
        if (at >= out.len) continue;
        if (n < out.len) n += 1;
        var j = n - 1;
        while (j > at) : (j -= 1) out[j] = out[j - 1];
        out[at] = @intCast(i);
    }
    std.mem.sort(u32, out, {}, std.sort.asc(u32));
}

/// The greedy second pass's inputs for a block's n = T - 1 head rows.
pub const Tail = struct {
    dims: Dims,
    topk: usize,
    stash: Stash, // the whole block's T rows; rows 1.. are scored
    anchor: u32,
    candidates: []const u32, // [n, topk], ascending per row
    head_rows: []align(1) const u16, // [n, topk, hidden]: the target head's row of each candidate, bf16
    norm_w: []align(1) const u16, // the final norm's weight
    bias: []align(1) const u16, // gate.bias
    tg: []align(1) const u16, // [vocab, rank]
    tsv: []align(1) const u16, // [vocab, 2 rank]
};

/// score[i][s][j] = raw_i * norm . head_j + rms_i * (sigmoid(q_i + G[sender] + b) * V[sender]) . TG[j], all f32.
pub fn table(x: Tail, out: []f32) !void {
    const d = x.dims;
    const k = x.topk;
    const n = x.candidates.len / k;
    if (n == 0 or x.candidates.len != n * k or out.len != n * k * k or x.stash.rms.len != n + 1) return error.BadBridgeShape;
    var base: [core.dspine.head_topk]f32 = undefined;
    var m: [4096]f32 = undefined;
    if (k > base.len or d.rank > m.len) return error.BadBridgeShape;
    for (0..n) |i| {
        const raw = x.stash.raw[(i + 1) * d.hidden ..][0..d.hidden];
        const rms = x.stash.rms[i + 1];
        const q = x.stash.packed_rows[(i + 1) * 3 * d.rank ..][0..d.rank];
        for (base[0..k], 0..) |*bv, j| {
            const hr = x.head_rows[(i * k + j) * d.hidden ..][0..d.hidden];
            var s: f32 = 0;
            for (raw, x.norm_w, hr) |r, w, h| s += float(r) * float(w) * float(h);
            bv.* = s;
        }
        for (0..k) |s| {
            const sid = if (i == 0) x.anchor else x.candidates[(i - 1) * k + s];
            const sv = x.tsv[2 * d.rank * sid ..][0 .. 2 * d.rank];
            for (m[0..d.rank], 0..) |*mv, r| mv.* = sigmoid(float(q[r]) + float(sv[r]) + float(x.bias[r])) * float(sv[d.rank + r]);
            for (0..k) |j| {
                const tg = x.tg[d.rank * x.candidates[i * k + j] ..][0..d.rank];
                var s2: f32 = 0;
                for (m[0..d.rank], tg) |mv, g| s2 += mv * float(g);
                out[(i * k + s) * k + j] = base[j] + rms * s2;
            }
        }
    }
}

test {
    _ = @import("host_test.zig");
}
