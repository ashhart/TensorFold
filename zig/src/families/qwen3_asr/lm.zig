//! The Qwen3 language model's weights by name and its greedy pass for every backend; `r` binds the kernels.
const std = @import("std");
const cfg = @import("config.zig");
const names = @import("names.zig");
const tower = @import("tower.zig");

/// A layer's tensors: q, k and v as one [(h + 2 hk) d, hidden] matrix, gate and up as one [2 i, hidden].
pub const Layer = struct { in_norm: u64, post_norm: u64, qkv: u64, o: u64, q_norm: u64, k_norm: u64, gate_up: u64, down: u64 };

pub const Weights = struct {
    embed: u64, // also the head: the stored lm_head is checked bit-equal to it
    norm: u64,
    layers: []Layer,

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        gpa.free(w.layers);
    }

    pub fn load(gpa: std.mem.Allocator, t: cfg.Text, src: anytype, up: anytype) !Weights {
        const embed = try src.expect(names.lm ++ "embed_tokens.weight", &.{ t.vocab, t.hidden });
        if (!std.mem.eql(u8, embed, try src.expect(names.head, &.{ t.vocab, t.hidden }))) return error.UntiedHead;
        var w: Weights = .{ .embed = try up.put(embed), .norm = try up.put(try src.expect(names.lm ++ "norm.weight", &.{t.hidden})), .layers = &.{} };
        w.layers = try gpa.alloc(Layer, t.layers);
        errdefer gpa.free(w.layers);
        const qd = t.heads * t.head_dim;
        const kvd = t.kv_heads * t.head_dim;
        var nb: [3][160]u8 = undefined;
        for (w.layers, 0..) |*l, i| {
            const p = struct {
                fn f(b: *[160]u8, li: usize, n: []const u8) ![]const u8 {
                    return std.fmt.bufPrint(b, names.lm ++ "layers.{d}.{s}", .{ li, n });
                }
            }.f;
            const qkv = [3][]const u8{ try src.expect(try p(&nb[0], i, "self_attn.q_proj.weight"), &.{ qd, t.hidden }), try src.expect(try p(&nb[1], i, "self_attn.k_proj.weight"), &.{ kvd, t.hidden }), try src.expect(try p(&nb[2], i, "self_attn.v_proj.weight"), &.{ kvd, t.hidden }) };
            l.qkv = try tower.joined(gpa, up, &qkv);
            const gu = [2][]const u8{ try src.expect(try p(&nb[0], i, "mlp.gate_proj.weight"), &.{ t.intermediate, t.hidden }), try src.expect(try p(&nb[1], i, "mlp.up_proj.weight"), &.{ t.intermediate, t.hidden }) };
            l.gate_up = try tower.joined(gpa, up, &gu);
            const one = .{
                .{ &l.in_norm, "input_layernorm.weight", &[_]usize{t.hidden} },   .{ &l.post_norm, "post_attention_layernorm.weight", &[_]usize{t.hidden} },
                .{ &l.o, "self_attn.o_proj.weight", &[_]usize{ t.hidden, qd } },  .{ &l.q_norm, "self_attn.q_norm.weight", &[_]usize{t.head_dim} },
                .{ &l.k_norm, "self_attn.k_norm.weight", &[_]usize{t.head_dim} }, .{ &l.down, "mlp.down_proj.weight", &[_]usize{ t.hidden, t.intermediate } },
            };
            inline for (one) |x| x[0].* = try up.put(try src.expect(try p(&nb[2], i, x[1]), x[2]));
        }
        return w;
    }
};

/// Audio rows spliced into a pass: `n` rows of the model's width at `ptr`, standing at positions [at, at + n).
pub const Splice = struct { ptr: u64, at: usize, n: usize };

/// A pass's bf16 rows, each from the backend's `al.alloc(bytes)`; the KV caches are the binding's.
pub const Buffers = struct {
    e: u64,
    h: [2]u64,
    y: u64,
    qkv: u64,
    att: u64,
    o: u64,
    gu: u64,
    act: u64,
    dn: u64,
    max_rows: usize,

    pub fn alloc(t: cfg.Text, max_rows: usize, al: anytype) !Buffers {
        const r = max_rows;
        const qd = t.heads * t.head_dim;
        const kvd = t.kv_heads * t.head_dim;
        return .{
            .e = try al.alloc(2 * r * t.hidden),
            .h = .{ try al.alloc(2 * r * t.hidden), try al.alloc(2 * r * t.hidden) },
            .y = try al.alloc(2 * r * t.hidden),
            .qkv = try al.alloc(2 * r * (qd + 2 * kvd)),
            .att = try al.alloc(2 * r * qd),
            .o = try al.alloc(2 * r * t.hidden),
            .gu = try al.alloc(2 * r * 2 * t.intermediate),
            .act = try al.alloc(2 * r * t.intermediate),
            .dn = try al.alloc(2 * r * t.hidden),
            .max_rows = max_rows,
        };
    }
};

/// One pass over `ids` at positions [pos, pos + ids.len), the audio rows spliced in: the last row's greedy pick.
pub fn forward(r: anytype, t: cfg.Text, w: *const Weights, b: Buffers, ids: []const u32, pos: usize, audio: ?Splice) !u32 {
    const rows = ids.len;
    if (rows == 0 or rows > b.max_rows) return error.PassRows;
    try r.positions(ids, pos);
    try r.embed(w.embed, b.e, rows, t.hidden);
    if (audio) |sp| {
        if (sp.at < pos or sp.at + sp.n > pos + rows) return error.SpliceOutsidePass;
        try r.copy(b.e + (sp.at - pos) * t.hidden * 2, sp.ptr, sp.n * t.hidden * 2);
    }
    try r.dump("lm_embed", b.e, rows, t.hidden);
    const qd = t.heads * t.head_dim;
    const kvd = t.kv_heads * t.head_dim;
    var x = b.e;
    var res: ?u64 = null;
    var cur: usize = 0;
    var name: [24]u8 = undefined;
    for (w.layers, 0..) |l, i| {
        try r.addNorm(x, res, l.in_norm, b.h[cur], b.y, rows);
        if (i > 0) try r.dump(try layerName(&name, i - 1), b.h[cur], rows, t.hidden); // h: the layer before's output
        if (i == 0) try r.dump("lm0_norm", b.y, rows, t.hidden); // layer 0's steps, as the case taps them
        try r.gemm(b.y, l.qkv, b.qkv, rows, qd + 2 * kvd, t.hidden);
        try r.qkRope(b.qkv, l.q_norm, l.k_norm, i, rows);
        try r.attention(i, b.att, rows);
        if (i == 0) try r.dump("lm0_attn", b.att, rows, qd);
        try r.gemm(b.att, l.o, b.o, rows, t.hidden, qd);
        if (i == 0) try r.dump("lm0_o", b.o, rows, t.hidden);
        const stream = if (res == null) x else b.h[cur]; // with no residual the norm leaves h unwritten: x is the stream
        try r.addNorm(b.o, stream, l.post_norm, b.h[1 - cur], b.y, rows);
        if (i == 0) try r.dump("lm0_post", b.y, rows, t.hidden);
        cur = 1 - cur;
        try r.gemm(b.y, l.gate_up, b.gu, rows, 2 * t.intermediate, t.hidden);
        try r.swiglu(b.gu, b.act, rows, t.intermediate);
        try r.gemm(b.act, l.down, b.dn, rows, t.hidden, t.intermediate);
        if (i == 0) try r.dump("lm0_mlp", b.dn, rows, t.hidden);
        x = b.dn;
        res = b.h[cur];
        cur = 1 - cur;
    }
    try r.addNorm(x, res, w.norm, b.h[cur], b.y, rows);
    try r.dump(try layerName(&name, w.layers.len - 1), b.h[cur], rows, t.hidden);
    try r.dump("lm_norm", b.y, rows, t.hidden);
    return r.greedy(b.y + (rows - 1) * t.hidden * 2, w.embed);
}

fn layerName(buf: *[24]u8, i: usize) ![]const u8 {
    return std.fmt.bufPrint(buf, "lm_layer{d}", .{i});
}

/// The first largest bf16 logit (torch.argmax's tie rule); a NaN logit wins, as torch's max does.
pub fn argmax(logits: []const u16) u32 {
    var best: usize = 0;
    var top: f32 = -std.math.inf(f32);
    for (logits, 0..) |bits, i| {
        const v: f32 = @bitCast(@as(u32, bits) << 16);
        if (std.math.isNan(v)) return @intCast(i);
        if (v > top) {
            top = v;
            best = i;
        }
    }
    return @intCast(best);
}

/// The pick's lead over the best other logit (a greedy difference is listed with it).
pub fn margin(logits: []const u16, best: u32) f32 {
    var other: f32 = -std.math.inf(f32);
    for (logits, 0..) |bits, i| if (i != best) {
        other = @max(other, @as(f32, @bitCast(@as(u32, bits) << 16)));
    };
    return @as(f32, @bitCast(@as(u32, logits[best]) << 16)) - other;
}

/// RoPE's cos and sin at positions [0, n) as transformers stores them: fp32 angles, the values cast to bf16.
pub fn ropeTables(t: cfg.Text, cs: []f32, sn: []f32, n: usize) void {
    const dim = t.head_dim;
    for (0..dim / 2) |i| {
        const e: f32 = @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(dim));
        const inv: f32 = 1.0 / @as(f32, @floatCast(std.math.pow(f64, t.theta, e)));
        for (0..n) |p| {
            const ang: f32 = @as(f32, @floatFromInt(p)) * inv;
            for ([2]usize{ i, i + dim / 2 }) |col| {
                cs[p * dim + col] = @bitCast(@as(u32, tower.bf16Bits(@cos(ang))) << 16);
                sn[p * dim + col] = @bitCast(@as(u32, tower.bf16Bits(@sin(ang))) << 16);
            }
        }
    }
}

test "argmax: the first of equal logits, a NaN first" {
    try std.testing.expectEqual(@as(u32, 1), argmax(&.{ 0x3f80, 0x4000, 0x4000 }));
    try std.testing.expectEqual(@as(u32, 2), argmax(&.{ 0x3f80, 0x4000, 0x7fc0 }));
}

test "margin: the lead over the runner-up" {
    try std.testing.expectEqual(@as(f32, 1), margin(&.{ 0x3f80, 0x4000, 0x3f80 }, 1));
}

test "ropeTables: position 0 is cos 1, sin 0; the halves repeat" {
    const t: cfg.Text = .{ .hidden = 8, .layers = 1, .heads = 1, .kv_heads = 1, .head_dim = 4, .intermediate = 8, .vocab = 4, .eps = 1e-6, .theta = 1e6, .context = 8 };
    var cs: [8]f32 = undefined;
    var sn: [8]f32 = undefined;
    ropeTables(t, &cs, &sn, 2);
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1, 1 }, cs[0..4]);
    try std.testing.expectEqual(@as(f32, 0), sn[0]);
    try std.testing.expectEqual(cs[4], cs[6]);
}
