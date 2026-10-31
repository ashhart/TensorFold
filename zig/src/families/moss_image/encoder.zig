//! MOSS-Image's prompt encoder on Metal: the chat-formatted prompt through Qwen3.5's text model to its final-norm rows.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("encoder_config.zig");
const wts = @import("encoder_weights.zig");
const ks = @import("kernels.zig");
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
    for (0..half) |i| {
        const exponent: f32 = @as(f32, @floatFromInt(2 * i)) / @as(f32, cfg.rotary);
        const freq: f32 = 1.0 / std.math.pow(f32, @floatCast(cfg.theta), exponent);
        for (0..rows) |r| {
            const angle: f32 = @as(f32, @floatFromInt(r)) * freq;
            cos[r * half + i] = bf(@floatCast(@cos(@as(f64, angle))));
            sin[r * half + i] = bf(@floatCast(@sin(@as(f64, angle))));
        }
    }
}

fn bf(v: f32) f32 {
    const u: u32 = @bitCast(v);
    return @bitCast(((u +% 0x7fff +% ((u >> 16) & 1)) >> 16) << 16);
}

pub const Encoder = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    k: ks.Kernels,
    w: wts.Weights,

    /// The checkpoint folder `dir`: its text_encoder/.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8) !*Encoder {
        if (!device.tensorUnits()) {
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
        const s = try Scratch.make(t.device, T);
        defer s.buf.deinit();
        const x = s.at(.x);
        for (tokens, 0..) |id, r| {
            if (id >= cfg.vocab) return error.TokenOutOfRange;
            @memcpy(halves(x.at(r * H * 2), H), t.w.embed[id * H ..][0..H]);
        }
        rotary(floats(s.at(.cos), T * 32), floats(s.at(.sin), T * 32), T);
        @memset(halves(s.at(.ones), H), 0x3f80);
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
