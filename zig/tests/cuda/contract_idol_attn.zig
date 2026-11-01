//! tf_idol_attention under the contract, every row over every key at head 64; `Attention(E)` over another type.
const std = @import("std");
const cuda = @import("cuda");
const idol = @import("idol_cuda");
const rig_ = @import("contract_rig.zig");
const attn = @import("contract_attn.zig");
const r = @import("contract_idol_ref.zig");

const c = r.c;
const g = r.g;

/// A row's P x V chain: every key, a 64-key tile at a time, then the 4 lanes' fold.
fn keyChain(s: c.Shape) usize {
    return s.k + 64 + 2;
}

/// The fp16 instance: heads (rows) over tokens (n, each a query and key), served counts at one head, strides at 16.
pub const Fp16 = struct {
    pub const Kernels = idol.Kernels;
    pub const elem: c.Elem = .f16;
    pub const why = "P enters the P x V mma as fp16: gates 1 and 3 hold each output to attention_ref's float64 bound (bf16 weights)";
    pub const served = [_]c.Shape{ .{ .rows = 1, .n = r.cond, .k = r.cond }, .{ .rows = 1, .n = r.latent, .k = r.latent }, .{ .rows = 1, .n = r.joint, .k = r.joint }, .{ .rows = r.heads, .n = 300, .k = 300 } };
    pub const warps = idol.attention_warps;

    pub fn bits(x: f64) u16 {
        return @bitCast(@as(f16, @floatCast(@as(f32, @floatCast(x)))));
    }

    pub fn value(b: u16) f64 {
        return @as(f16, @bitCast(b));
    }

    pub fn launch(k: *const Kernels, s: cuda.Stream, qkv: u64, out: u64, tokens: usize, heads: usize, scale: f32) !void {
        try idol.attention(k, s, qkv, out, tokens, heads, scale);
    }
};

pub const Attn = Attention(Fp16);

/// The test over element type `E`: its Kernels, elem, why, served, warps, bits, value and launch.
pub fn Attention(comptime E: type) type {
    return struct {
        rig: rig_.Rig,
        k: *const E.Kernels,
        name: []const u8,
        /// The calls it is registered at: E's (the denoiser's), or the Gaussian decoder's self attention over anchors.
        served: []const c.Shape = &E.served,
        ref: ?attn.Ref = null,
        heads: usize = 0,
        tokens: usize = 0,
        qkv: std.ArrayList(u16) = .empty,
        /// The key head whose K and V rows the reference holds now.
        held: ?usize = null,
        dev: u64 = 0,
        out: u64 = 0,

        const Self = @This();

        pub fn spec(name: []const u8, at: []const c.Shape) c.Spec {
            return .{ .name = name, .accum = .f32, .depth = keyChain, .ops = 6, .out = E.elem, .tiles = .{ .{}, .{ .size = 16 * E.warps }, .{} }, .served = at, .sums = .{ .why = E.why } };
        }

        pub fn deinit(t: *Self) void {
            if (t.ref) |*x| x.deinit();
            t.qkv.deinit(t.rig.gpu.gpa);
            t.rig.deinit();
        }

        pub fn kernel(t: *Self) g.Kernel {
            return .{ .spec = spec(t.name, t.served), .ctx = t, .vt = &vt };
        }
        const vt = g.hooks(Self);

        fn stride(t: *const Self) usize {
            return 3 * t.heads * r.D;
        }

        pub fn prepare(t: *Self, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
            if (shape.rows == 0 or shape.n == 0) return false;
            t.rig.reset();
            const a = t.rig.gpu.gpa;
            t.heads = shape.rows;
            t.tokens = shape.n;
            t.held = null;
            try t.qkv.resize(a, t.tokens * t.stride());
            var prng = std.Random.DefaultPrng.init(seed);
            const rng = prng.random();
            for (t.qkv.items) |*e| e.* = E.bits(rng.floatNorm(f64));
            if (t.ref) |*x| x.deinit();
            t.ref = null;
            t.ref = try attn.Ref.init(a, r.D, t.tokens);
            t.ref.?.out = E.elem;
            switch (fill) {
                .random => {},
                .nonfinite => t.plant(),
                .stress => t.stress(rng),
            }
            t.dev = try t.rig.input(std.mem.sliceAsBytes(t.qkv.items));
            t.out = try t.rig.output(t.count() * 2);
            return true;
        }

        fn at(t: *Self, row: usize, part: usize, head: usize, e: usize) *u16 {
            return &t.qkv.items[row * t.stride() + (part * t.heads + head) * r.D + e];
        }

        /// NaN in row 0's query (head 0); +inf and -inf in one value column of the last head, at two keys.
        fn plant(t: *Self) void {
            const last = t.heads - 1;
            t.at(0, 0, 0, 3).* = E.bits(r.nan);
            t.at(t.tokens / 2, 2, last, 7).* = E.bits(r.inf);
            t.at(t.tokens - 1, 2, last, 7).* = E.bits(-r.inf);
            t.at(t.tokens / 3, 2, last, 9).* = E.bits(r.inf);
        }

        /// Gate 4: every key of a head alike (each weight 1), values small odd integers mirrored over the keys.
        fn stress(t: *Self, rng: std.Random) void {
            const n = t.tokens;
            for (0..t.heads) |hd| for (0..r.D) |e| {
                const key = t.at(0, 1, hd, e).*;
                for (0..n) |row| t.at(row, 1, hd, e).* = key;
                for (0..n / 2) |row| {
                    const v: f64 = @floatFromInt(1 + 2 * rng.uintLessThan(u32, 8));
                    t.at(row, 2, hd, e).* = E.bits(v);
                    t.at(n / 2 + row, 2, hd, e).* = E.bits((if (row < 8) @as(f64, 1) else 0) - v);
                }
                if (n % 2 == 1) t.at(n - 1, 2, hd, e).* = E.bits(0);
            };
        }

        pub fn run(t: *Self) !void {
            try E.launch(t.k, t.rig.s, t.dev, t.out, t.tokens, t.heads, 1.0 / 8.0);
            try t.rig.s.synchronize();
        }

        pub fn count(t: *Self) usize {
            return t.tokens * t.heads * r.D;
        }

        /// Output head i (token i / heads, head i % heads): its query and the head's every key and value.
        pub fn load(t: *Self, i: usize, hd: *attn.Head) !void {
            const row = i / t.heads;
            const head = i % t.heads;
            for (hd.q, 0..) |*x, e| x.* = E.value(t.at(row, 0, head, e).*);
            hd.n = t.tokens;
            if (t.held == head) return;
            for (0..t.tokens) |j| for (0..r.D) |e| {
                hd.k[j * r.D + e] = E.value(t.at(j, 1, head, e).*);
                hd.v[j * r.D + e] = E.value(t.at(j, 2, head, e).*);
            };
            t.held = head;
        }

        fn outputs(t: *Self) usize {
            return t.tokens * t.heads;
        }

        pub fn check(t: *Self, gate: g.Gate) !?[]const u8 {
            return t.ref.?.check(t, t.outputs(), 1.0 / 8.0, try t.rig.outputBytes(), gate);
        }

        pub fn reference(t: *Self, want: []f64, mag: []f64) !void {
            return t.ref.?.reference(t, t.outputs(), 1.0 / 8.0, want, mag);
        }

        pub fn simulate(t: *Self, acc: c.Accum, order: g.Order, out: []f64) !void {
            return t.ref.?.simulate(t, t.outputs(), 1.0 / 8.0, acc, order, out);
        }
    };
}
