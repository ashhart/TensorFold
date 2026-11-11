//! tf_idol_attention under the contract: every row over every key, head width 64, fp16 in and out, head by head.
const std = @import("std");
const cuda = @import("cuda");
const idol = @import("idol_cuda");
const rig_ = @import("contract_rig.zig");
const attn = @import("contract_attn.zig");
const r = @import("contract_idol_ref.zig");

const c = r.c;
const g = r.g;

const why = "P enters the P x V mma as fp16: gates 1 and 3 hold each output to attention_ref's float64 bound (bf16 weights)";

/// A row's P x V chain: every key, a 64-key tile at a time, then the 4 lanes' fold.
fn keyChain(s: c.Shape) usize {
    return s.k + 64 + 2;
}

/// Heads (rows) over tokens (n, each a query and key): served token counts at one head, the head strides at 16.
const served = [_]c.Shape{ .{ .rows = 1, .n = r.cond, .k = r.cond }, .{ .rows = 1, .n = r.latent, .k = r.latent }, .{ .rows = 1, .n = r.joint, .k = r.joint }, .{ .rows = r.heads, .n = 300, .k = 300 } };

pub fn spec(name: []const u8) c.Spec {
    return .{ .name = name, .accum = .f32, .depth = keyChain, .ops = 6, .out = .f16, .tiles = .{ .{}, .{ .size = 16 * idol.attention_warps }, .{} }, .served = &served, .sums = .{ .why = why } };
}

fn half(x: f64) u16 {
    return @bitCast(@as(f16, @floatCast(@as(f32, @floatCast(x)))));
}

fn value(b: u16) f64 {
    return @as(f16, @bitCast(b));
}

pub const Attn = struct {
    rig: rig_.Rig,
    k: *const idol.Kernels,
    name: []const u8,
    ref: ?attn.Ref = null,
    heads: usize = 0,
    tokens: usize = 0,
    qkv: std.ArrayList(u16) = .empty,
    /// The key head whose K and V rows the reference holds now.
    held: ?usize = null,
    dev: u64 = 0,
    out: u64 = 0,

    pub fn deinit(t: *Attn) void {
        if (t.ref) |*x| x.deinit();
        t.qkv.deinit(t.rig.gpu.gpa);
        t.rig.deinit();
    }

    pub fn kernel(t: *Attn) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(Attn);

    fn stride(t: *const Attn) usize {
        return 3 * t.heads * r.D;
    }

    pub fn prepare(t: *Attn, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        t.rig.reset();
        const a = t.rig.gpu.gpa;
        t.heads = shape.rows;
        t.tokens = shape.n;
        t.held = null;
        try t.qkv.resize(a, t.tokens * t.stride());
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (t.qkv.items) |*e| e.* = half(rng.floatNorm(f64));
        if (t.ref) |*x| x.deinit();
        t.ref = null;
        t.ref = try attn.Ref.init(a, r.D, t.tokens);
        t.ref.?.out = .f16;
        switch (fill) {
            .random => {},
            .nonfinite => t.plant(),
            .stress => t.stress(rng),
        }
        t.dev = try t.rig.input(std.mem.sliceAsBytes(t.qkv.items));
        t.out = try t.rig.output(t.count() * 2);
        return true;
    }

    fn at(t: *Attn, row: usize, part: usize, head: usize, e: usize) *u16 {
        return &t.qkv.items[row * t.stride() + (part * t.heads + head) * r.D + e];
    }

    /// NaN in row 0's query (head 0); +inf and -inf in one value column of the last head, at two keys.
    fn plant(t: *Attn) void {
        const last = t.heads - 1;
        t.at(0, 0, 0, 3).* = half(r.nan);
        t.at(t.tokens / 2, 2, last, 7).* = half(r.inf);
        t.at(t.tokens - 1, 2, last, 7).* = half(-r.inf);
        t.at(t.tokens / 3, 2, last, 9).* = half(r.inf);
    }

    /// Gate 4: every key of a head alike (each weight 1), values small odd integers mirrored over the keys.
    fn stress(t: *Attn, rng: std.Random) void {
        const n = t.tokens;
        for (0..t.heads) |hd| for (0..r.D) |e| {
            const key = t.at(0, 1, hd, e).*;
            for (0..n) |row| t.at(row, 1, hd, e).* = key;
            for (0..n / 2) |row| {
                const v: f64 = @floatFromInt(1 + 2 * rng.uintLessThan(u32, 8));
                t.at(row, 2, hd, e).* = half(v);
                t.at(n / 2 + row, 2, hd, e).* = half((if (row < 8) @as(f64, 1) else 0) - v);
            }
            if (n % 2 == 1) t.at(n - 1, 2, hd, e).* = half(0);
        };
    }

    pub fn run(t: *Attn) !void {
        try idol.attention(t.k, t.rig.s, t.dev, t.out, t.tokens, t.heads, 1.0 / 8.0);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Attn) usize {
        return t.tokens * t.heads * r.D;
    }

    /// Output head i (token i / heads, head i % heads): its query and the head's every key and value.
    pub fn load(t: *Attn, i: usize, hd: *attn.Head) !void {
        const row = i / t.heads;
        const head = i % t.heads;
        for (hd.q, 0..) |*x, e| x.* = value(t.at(row, 0, head, e).*);
        hd.n = t.tokens;
        if (t.held == head) return;
        for (0..t.tokens) |j| for (0..r.D) |e| {
            hd.k[j * r.D + e] = value(t.at(j, 1, head, e).*);
            hd.v[j * r.D + e] = value(t.at(j, 2, head, e).*);
        };
        t.held = head;
    }

    fn outputs(t: *Attn) usize {
        return t.tokens * t.heads;
    }

    pub fn check(t: *Attn, gate: g.Gate) !?[]const u8 {
        return t.ref.?.check(t, t.outputs(), 1.0 / 8.0, try t.rig.outputBytes(), gate);
    }

    pub fn reference(t: *Attn, want: []f64, mag: []f64) !void {
        return t.ref.?.reference(t, t.outputs(), 1.0 / 8.0, want, mag);
    }

    pub fn simulate(t: *Attn, acc: c.Accum, order: g.Order, out: []f64) !void {
        return t.ref.?.simulate(t, t.outputs(), 1.0 / 8.0, acc, order, out);
    }
};
