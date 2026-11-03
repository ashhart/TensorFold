//! MOSS-Image's prompt encoder on Metal: the chat-formatted prompt through Qwen3.5's text model to its final-norm rows.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("encoder_config.zig");
const wts = @import("encoder_weights.zig");
const ks = @import("kernels.zig");
const bf16_round = @import("bf16_round");
const rope_freq = @import("../../core/rope_freq.zig");
const Ref = ks.Ref;

const H = cfg.hidden;
const F = cfg.ffn;
const D = cfg.head_dim;

/// The pipeline's chat-formatted prompt: one user turn, then the assistant's turn opened with thinking on.
pub fn format(a: std.mem.Allocator, prompt: []const u8) ![]u8 {
    return std.mem.concat(a, u8, &.{ "<|im_start|>user\n", prompt, "<|im_end|>\n<|im_start|>assistant\n<think>\n" });
}

/// The prompt's ids by `tok` (the checkpoint's tokenizer.json) as the pipeline makes them, cut at 4,096; caller frees.
pub fn ids(tok: anytype, a: std.mem.Allocator, prompt: []const u8) ![]u32 {
    const text = try format(a, prompt);
    defer a.free(text);
    const all = try tok.encode(a, text);
    if (all.len <= cfg.max_rows) return all;
    defer a.free(all);
    return a.dupe(u32, all[0..cfg.max_rows]);
}

/// The rotary tables [rows, 32] as the text model computes them: fp32 angles, their cos and sin, rounded to bf16.
pub fn rotary(cos: []f32, sin: []f32, rows: usize) void {
    const half = cfg.rotary / 2;
    var freqs: [half]f32 = undefined;
    rope_freq.inverseFrequencies(cfg.theta, cfg.rotary, &freqs);
    for (freqs, 0..) |freq, i| {
        for (0..rows) |r| {
            const angle: f32 = @as(f32, @floatFromInt(r)) * freq;
            cos[r * half + i] = bf(@floatCast(@cos(@as(f64, angle))));
            sin[r * half + i] = bf(@floatCast(@sin(@as(f64, angle))));
        }
    }
}

const bf = bf16_round.round32;

pub const Encoder = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    k: ks.Kernels,
    w: wts.Weights,

    /// The checkpoint folder `dir`: its text_encoder/.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8) !*Encoder {
        if (!ks.available(device)) {
            std.log.err("moss image encoder: {s} has no tensor units; this family runs on the M5's alone, the only chip it was checked on", .{device.name()});
            return error.TensorUnitsRequired;
        }
        const t = try gpa.create(Encoder);
        errdefer gpa.destroy(t);
        t.gpa = gpa;
        t.device = device;
        t.queue = try device.queue();
        errdefer t.queue.deinit();
        t.k = try ks.Kernels.load(gpa, device);
        errdefer t.k.deinit();
        const text = try std.fs.path.join(gpa, &.{ dir, "text_encoder" });
        defer gpa.free(text);
        t.w = try wts.load(gpa, io, device, text);
        return t;
    }

    pub fn deinit(t: *Encoder) void {
        t.w.deinit();
        t.k.deinit();
        t.queue.deinit();
        t.gpa.destroy(t);
    }

    /// The final-norm rows [tokens, 4096] bf16 into `out`; with `kept`, each layer's rows [layers, tokens, 4096].
    pub fn rows(t: *Encoder, tokens: []const u32, out: []u16, kept: ?[]u16) !void {
        const T = tokens.len;
        if (T == 0 or T > cfg.max_rows) return error.PromptRows;
        if (out.len != T * H) return error.PromptRows;
        if (kept) |k| if (k.len != cfg.layers * T * H) return error.PromptRows;
        const s = try scratch(t.device, T);
        defer s.buf.deinit();
        const x = s.at(.x);
        for (tokens, 0..) |id, r| {
            if (id >= cfg.vocab) return error.TokenOutOfRange;
            @memcpy(halves(x.at(r * H * 2), H), t.w.embed[id * H ..][0..H]);
        }
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        var cb = t.queue.commandBuffer();
        var e = cb.compute(.serial);
        for (t.w.layers, 0..) |l, i| {
            t.layer(e, s, l, T);
            if (kept) |k| {
                try finish(cb, e);
                @memcpy(k[i * T * H ..][0 .. T * H], halves(x, T * H));
                cb = t.queue.commandBuffer();
                e = cb.compute(.serial);
            }
        }
        ks.encoderNorm(e, t.k.get("me_norm"), x, t.w.norm, s.at(.out), T, H, H, H, cfg.eps);
        try finish(cb, e);
        @memcpy(out, halves(s.at(.out), T * H));
    }

    /// For checks, layer i on block i of `inputs` [layers + 1, rows, 4096], the final norm on the last.
    pub fn steps(t: *Encoder, T: usize, inputs: []const u16, out: []u16) !void {
        const n = (cfg.layers + 1) * T * H;
        if (T == 0 or T > cfg.max_rows) return error.PromptRows;
        if (inputs.len != n or out.len != n) return error.PromptRows;
        const s = try scratch(t.device, T);
        defer s.buf.deinit();
        const x = s.at(.x);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        for (0..cfg.layers + 1) |i| {
            @memcpy(halves(x, T * H), inputs[i * T * H ..][0 .. T * H]);
            const cb = t.queue.commandBuffer();
            const e = cb.compute(.serial);
            const last = i == cfg.layers;
            if (last) ks.encoderNorm(e, t.k.get("me_norm"), x, t.w.norm, s.at(.out), T, H, H, H, cfg.eps) else t.layer(e, s, t.w.layers[i], T);
            try finish(cb, e);
            @memcpy(out[i * T * H ..][0 .. T * H], halves(if (last) s.at(.out) else x, T * H));
        }
    }

    fn layer(t: *const Encoder, e: mtl.ComputeEncoder, s: Scratch, l: wts.Layer, T: usize) void {
        const x = s.at(.x);
        const xn = s.at(.xn);
        const y = s.at(.y);
        ks.encoderNorm(e, t.k.get("me_norm"), x, l.norm1, xn, T, H, H, H, cfg.eps);
        switch (l.mixer) {
            .delta => |m| t.delta(e, s, m, T),
            .attention => |m| t.attention(e, s, m, T),
        }
        ks.gateAdd(e, t.k.get("mi_gate_add"), x, y, s.at(.ones), T, H);
        ks.encoderNorm(e, t.k.get("me_norm"), x, l.norm2, xn, T, H, H, H, cfg.eps);
        t.k.linear(e, xn, l.gate, s.at(.gate), T, F, H);
        t.k.linear(e, xn, l.up, s.at(.up), T, F, H);
        ks.encoderRows(e, t.k.get("me_silu_mul"), s.at(.gate), s.at(.up), s.at(.act), T, F);
        t.k.linear(e, s.at(.act), l.down, y, T, H, F);
        ks.gateAdd(e, t.k.get("mi_gate_add"), x, y, s.at(.ones), T, H);
    }

    fn delta(t: *const Encoder, e: mtl.ComputeEncoder, s: Scratch, m: wts.Delta, T: usize) void {
        const xn = s.at(.xn);
        t.k.linear(e, xn, m.qkv, s.at(.mix), T, cfg.mix, H);
        t.k.linear(e, xn, m.z, s.at(.z), T, H, H);
        t.k.linear(e, xn, m.b, s.at(.b), T, cfg.v_heads, H);
        t.k.linear(e, xn, m.a, s.at(.a), T, cfg.v_heads, H);
        ks.encoderRows(e, t.k.get("me_conv"), s.at(.mix), m.conv, s.at(.conv), T, cfg.mix);
        const g: ks.Gdn = .{ .rows = @intCast(T), .kheads = cfg.k_heads, .vheads = cfg.v_heads, .mix_stride = cfg.mix, .scale = 1.0 / @sqrt(@as(f32, cfg.k_dim)) };
        ks.encoderGdnPrep(e, t.k.get("me_gdn_prep"), s.at(.conv), s.at(.a), s.at(.b), m.a_log, m.dt_bias, s.at(.qn), s.at(.kn), s.at(.g), s.at(.beta), g);
        ks.encoderGdn(e, t.k.get("me_gdn"), s.at(.qn), s.at(.kn), s.at(.conv), s.at(.g), s.at(.beta), s.at(.core), g);
        ks.encoderGatedNorm(e, t.k.get("me_gated_norm"), s.at(.core), s.at(.z), m.norm, s.at(.mixed), T, H, cfg.eps);
        t.k.linear(e, s.at(.mixed), m.out, s.at(.y), T, H, H);
    }

    fn attention(t: *const Encoder, e: mtl.ComputeEncoder, s: Scratch, m: wts.Attention, T: usize) void {
        const xn = s.at(.xn);
        const kv = cfg.kv_heads * D;
        t.k.linear(e, xn, m.q, s.at(.mix), T, 2 * cfg.heads * D, H);
        t.k.linear(e, xn, m.k, s.at(.kp), T, kv, H);
        t.k.linear(e, xn, m.v, s.at(.vp), T, kv, H);
        const heads = t.k.get("me_head");
        ks.encoderHeads(e, heads, s.at(.mix), m.q_norm, s.at(.cos), s.at(.sin), s.at(.q), T, cfg.heads, 2 * cfg.heads * D, 2 * D, cfg.eps, true);
        ks.encoderHeads(e, heads, s.at(.kp), m.k_norm, s.at(.cos), s.at(.sin), s.at(.k), T, cfg.kv_heads, kv, D, cfg.eps, true);
        ks.encoderHeads(e, heads, s.at(.vp), m.k_norm, s.at(.cos), s.at(.sin), s.at(.v), T, cfg.kv_heads, kv, D, cfg.eps, false);
        ks.encoderAttention(e, t.k.get("me_attention"), s.at(.q), s.at(.k), s.at(.v), s.at(.att), T, cfg.heads, cfg.kv_heads, 1.0 / @sqrt(@as(f32, D)));
        ks.encoderRows(e, t.k.get("me_gate"), s.at(.att), s.at(.mix), s.at(.mixed), T, H);
        t.k.linear(e, s.at(.mixed), m.o, s.at(.y), T, H, H);
    }
};

fn finish(cb: mtl.CommandBuffer, e: mtl.ComputeEncoder) !void {
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("moss image encoder: a pass failed on the GPU: {s}", .{msg});
        return error.GpuFailure;
    }
}

/// A prompt of `T` rows' scratch with its rotary tables and unit gate filled.
fn scratch(device: mtl.Device, T: usize) !Scratch {
    const s = try Scratch.make(device, T);
    rotary(floats(s.at(.cos), T * 32), floats(s.at(.sin), T * 32), T);
    @memset(halves(s.at(.ones), H), 0x3f80);
    return s;
}

fn floats(r: Ref, n: usize) []f32 {
    return @as([*]f32, @ptrCast(@alignCast(r.addr())))[0..n];
}

fn halves(r: Ref, n: usize) []u16 {
    return @as([*]u16, @ptrCast(@alignCast(r.addr())))[0..n];
}

/// One buffer's parts for a prompt of `rows` rows, each 256-aligned.
const Scratch = struct {
    const Part = enum { x, xn, y, mix, conv, z, a, b, qn, kn, g, beta, core, mixed, kp, vp, q, k, v, att, gate, up, act, cos, sin, ones, out };
    buf: mtl.Buffer,
    offs: [std.enums.values(Part).len]usize,

    fn bytes(p: Part, T: usize) usize {
        return switch (p) {
            .x, .xn, .y, .z, .core, .mixed, .q, .att, .out => T * H * 2,
            .mix, .conv => T * cfg.mix * 2,
            .a, .b => T * cfg.v_heads * 2,
            .qn, .kn => T * cfg.k_heads * cfg.k_dim * 4,
            .g, .beta => T * cfg.v_heads * 4,
            .kp, .vp, .k, .v => T * cfg.kv_heads * D * 2,
            .gate, .up, .act => T * F * 2,
            .cos, .sin => T * 32 * 4,
            .ones => H * 2,
        };
    }

    fn make(device: mtl.Device, T: usize) !Scratch {
        var s: Scratch = undefined;
        var total: usize = 0;
        for (std.enums.values(Part), 0..) |p, i| {
            s.offs[i] = total;
            total += std.mem.alignForward(usize, bytes(p, T), 256);
        }
        s.buf = try device.buffer(total, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        return s;
    }

    fn at(s: Scratch, p: Part) Ref {
        return .{ .buf = s.buf, .off = s.offs[@backingInt(p)] };
    }
};

test "the pipeline's chat format" {
    const a = std.testing.allocator;
    const text = try format(a, "A red fox");
    defer a.free(text);
    try std.testing.expectEqualStrings("<|im_start|>user\nA red fox<|im_end|>\n<|im_start|>assistant\n<think>\n", text);
}

test "rotary bf16: today's bits on every zero, subnormal, infinity and tie; a NaN stays NaN" {
    const old = struct {
        /// bf as it was spelled before it was core's round32(): the oracle for every non-NaN input.
        fn spelled(v: f32) f32 {
            const u: u32 = @bitCast(v);
            return @bitCast(((u +% 0x7fff +% ((u >> 16) & 1)) >> 16) << 16);
        }

        fn kept(u: u32) !void {
            const v: f32 = @bitCast(u);
            const h: u32 = @bitCast(bf(v));
            if (!std.math.isNan(v)) return std.testing.expectEqual(@as(u32, @bitCast(spelled(v))), h);
            try std.testing.expect(std.math.isNan(@as(f32, @bitCast(h))));
            try std.testing.expectEqual(u >> 31, h >> 31);
        }
    };
    for (0..1 << 25) |i| { // sign, exponent 0 or 255, every mantissa
        const top: u32 = if (i & 1 << 23 != 0) 0x7f80_0000 else 0;
        try old.kept(@as(u32, @intCast(i >> 24)) << 31 | top | @as(u32, @intCast(i & 0x7f_ffff)));
    }
    for (0..1 << 16) |hi| for ([_]u32{ 0, 1, 0x7fff, 0x8000, 0x8001, 0xffff }) |lo| try old.kept(@as(u32, @intCast(hi)) << 16 | lo);
    var prng = std.Random.DefaultPrng.init(1863);
    for (0..1 << 22) |_| try old.kept(prng.random().int(u32));
    try std.testing.expect(std.math.isInf(old.spelled(@bitCast(@as(u32, 0x7f800001))))); // the old copy's +Inf
    try std.testing.expectEqual(@as(u32, 0x80000000), @as(u32, @bitCast(old.spelled(@bitCast(@as(u32, 0x7fff8001)))))); // its -0
    for ([_]u32{ 0x7f800001, 0x7fff8001, 0xffff8001, 0x7fc00000, 0xffffffff, 0x7fbfffff, 0xffa00000 }) |u| try old.kept(u);
}

/// The replaced frequency (rotary before #1852), kept as a reference: Zig's f32 pow of an f32 theta, fp32 reciprocal.
fn replacedFrequency(i: usize) f32 {
    const exponent: f32 = @as(f32, @floatFromInt(2 * i)) / @as(f32, cfg.rotary);
    return 1.0 / std.math.pow(f32, @floatCast(cfg.theta), exponent);
}

/// The replaced rotary: its frequencies through the same fp32 angles and bf16 cos and sin.
fn replacedRotary(cos: []f32, sin: []f32, rows: usize) void {
    const half = cfg.rotary / 2;
    for (0..half) |i| {
        const freq = replacedFrequency(i);
        for (0..rows) |r| {
            const angle: f32 = @as(f32, @floatFromInt(r)) * freq;
            cos[r * half + i] = bf(@floatCast(@cos(@as(f64, angle))));
            sin[r * half + i] = bf(@floatCast(@sin(@as(f64, angle))));
        }
    }
}

const Diff = struct { differ: usize, ulps: u32 };

/// Prints both tables' words and values side by side, then the largest absolute and relative difference.
fn sideBySide(label: []const u8, old: []const f32, new: []const f32) Diff {
    var r: Diff = .{ .differ = 0, .ulps = 0 };
    var abs: f64 = 0;
    var rel: f64 = 0;
    std.debug.print("{s}\n   i  old                         new                         ulps\n", .{label});
    for (old, new, 0..) |o, n, i| {
        const a: u32 = @bitCast(o);
        const b: u32 = @bitCast(n);
        const u = if (a > b) a - b else b - a;
        const d = @abs(@as(f64, n) - o);
        if (u != 0) r.differ += 1;
        r.ulps = @max(r.ulps, u);
        abs = @max(abs, d);
        rel = @max(rel, d / @abs(@as(f64, o)));
        const sign = if (b > a) "+" else if (b < a) "-" else " ";
        std.debug.print("{d:>4}  0x{x:0>8} {e:<16}  0x{x:0>8} {e:<16}  {s}{d}\n", .{ i, a, o, b, n, sign, u });
    }
    std.debug.print("{s}: {d} of {d} words differ, largest {d} ulp, abs {e:.3}, rel {e:.3}\n", .{
        label, r.differ, old.len, r.ulps, abs, rel,
    });
    return r;
}

const TableDiff = struct { differ: usize, first: ?usize };

/// The entries of two bf16-valued tables that differ, printed with the largest difference and the first row hit.
fn tableDiff(label: []const u8, old: []const f32, new: []const f32, width: usize) TableDiff {
    var r: TableDiff = .{ .differ = 0, .first = null };
    var most: f32 = 0;
    for (old, new, 0..) |o, n, i| if (o != n) {
        r.differ += 1;
        most = @max(most, @abs(n - o));
        if (r.first == null) r.first = i / width;
    };
    std.debug.print("{s}: {d} of {d} entries differ, largest {e:.3}, first at row {?d}\n", .{
        label, r.differ, old.len, most, r.first,
    });
    return r;
}

test "rotary: transformers' frequencies (theta 1e7 over 64), printed beside the fp32-pow form they replace" {
    // transformers 5.22.1's words for this config, as rope_freq.zig's first test has them
    const torch = [32]u32{
        0x3f800000, 0x3f1ab32b, 0x3ebaf81b, 0x3e61f835, 0x3e088d77, 0x3da50956, 0x3d47763f, 0x3cf11177,
        0x3c91ad39, 0x3c301052, 0x3bd4ca15, 0x3b80967d, 0x3b1b690d, 0x3abbd3ed, 0x3a6301e2, 0x3a092e02,
        0x39a5cb60, 0x394860c1, 0x38f22ce2, 0x3892587e, 0x3830df52, 0x37d5c441, 0x37812dab, 0x371c1fc4,
        0x36bcb0c1, 0x36640cc6, 0x3609cf4a, 0x35a68e4c, 0x35494c57, 0x34f3499d, 0x3493048e, 0x3431af44,
    };
    const half = cfg.rotary / 2;
    var new: [half]f32 = undefined;
    rope_freq.inverseFrequencies(cfg.theta, cfg.rotary, &new);
    for (torch, new) |w, f| try std.testing.expectEqual(w, @as(u32, @bitCast(f)));
    var old: [half]f32 = undefined;
    for (&old, 0..) |*f, i| f.* = replacedFrequency(i);
    const got = sideBySide("MOSS-Image encoder (theta 1e7, rotary 64)", &old, &new);
    try std.testing.expectEqual(Diff{ .differ = 15, .ulps = 5 }, got);
    const a = std.testing.allocator;
    const n = cfg.max_rows * half;
    const t = try a.alloc(f32, 4 * n);
    defer a.free(t);
    rotary(t[0..n], t[n..][0..n], cfg.max_rows);
    replacedRotary(t[2 * n ..][0..n], t[3 * n ..][0..n], cfg.max_rows);
    const cos = tableDiff("MOSS-Image encoder bf16 cos over 4096 rows", t[2 * n ..][0..n], t[0..n], half);
    const sin = tableDiff("MOSS-Image encoder bf16 sin over 4096 rows", t[3 * n ..][0..n], t[n..][0..n], half);
    // a prompt of up to 565 rows (its formatted ids) reads the same tables as before
    try std.testing.expectEqual(TableDiff{ .differ = 76, .first = 565 }, cos);
    try std.testing.expectEqual(TableDiff{ .differ = 65, .first = 860 }, sin);
}
