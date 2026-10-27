//! Q3 of the Gemma 4 E4B port: every projection class against an FP64 reference of the same 4-bit codes (--gemm-test).
const std = @import("std");
const dev = @import("xpu_dev.zig");
const fwd = @import("xpu_forward.zig");
const load_mod = @import("xpu_load.zig");
const plan = @import("cuda_plan.zig");
const alloc = dev.alloc;
const Proj = load_mod.Proj;

fn bf(u: u16) f64 {
    return @as(f32, @bitCast(@as(u32, u) << 16));
}

/// The dequantized weights of the stacked parts, f64 [N][K].
fn weights(r: *fwd.Runner, prefix: []const u8, parts: []const []const u8, n_total: usize, k: usize) ![]f64 {
    const out = try alloc.alloc(f64, n_total * k);
    var n0: usize = 0;
    for (parts) |p| {
        const w = try r.ckpt.get(try std.fmt.allocPrint(alloc, "{s}{s}.weight", .{ prefix, p }));
        const s = try r.ckpt.get(try std.fmt.allocPrint(alloc, "{s}{s}.scales", .{ prefix, p }));
        const b = try r.ckpt.get(try std.fmt.allocPrint(alloc, "{s}{s}.biases", .{ prefix, p }));
        const n: usize = @intCast(w.shape[0]);
        for (0..n) |row| for (0..k) |col| {
            const word = std.mem.readInt(u32, w.bytes[(row * (k / 8) + col / 8) * 4 ..][0..4], .little);
            const q: f64 = @floatFromInt((word >> @intCast(4 * (col % 8))) & 0xF);
            const g = row * (k / 64) + col / 64;
            const sc = bf(std.mem.readInt(u16, s.bytes[g * 2 ..][0..2], .little));
            const bi = bf(std.mem.readInt(u16, b.bytes[g * 2 ..][0..2], .little));
            out[(n0 + row) * k + col] = q * sc + bi;
        };
        n0 += n;
    }
    return out;
}

const Case = struct { name: []const u8, prefix: []const u8, parts: []const []const u8, proj: Proj, amp: f32 };

fn run(r: *fwd.Runner, c: Case, rows: u32, rng: *std.Random) !void {
    const n = c.proj.n;
    const k = c.proj.k;
    const w = try weights(r, c.prefix, c.parts, n, k);
    defer alloc.free(w);
    const h = try alloc.alloc(f16, rows * k);
    defer alloc.free(h);
    for (h) |*x| x.* = @floatCast(rng.floatNorm(f32) * c.amp);
    const g = k / 64;
    const hi = try alloc.alloc(f16, rows * g);
    defer alloc.free(hi);
    const lo = try alloc.alloc(f16, rows * g);
    defer alloc.free(lo);
    for (0..rows) |m| for (0..g) |gi| {
        var s: f32 = 0;
        for (0..64) |j| s += @as(f32, h[m * k + gi * 64 + j]);
        s *= 1.0 / 64.0;
        const hh: f16 = @floatCast(s * 256.0);
        hi[m * g + gi] = hh;
        lo[m * g + gi] = @floatCast(s * 256.0 - @as(f32, hh));
    };
    const a = &r.a;
    try r.d.copyIn(a.h, std.mem.sliceAsBytes(h));
    try r.d.copyIn(a.ahi, std.mem.sliceAsBytes(hi));
    try r.d.copyIn(a.alo, std.mem.sliceAsBytes(lo));
    try r.gemm(a.h, a.ahi, a.alo, c.proj, a.p, rows);
    const out = try alloc.alloc(f16, rows * n);
    defer alloc.free(out);
    try r.d.read(std.mem.sliceAsBytes(out), a.p);
    var err2: f64 = 0;
    var ref2: f64 = 0;
    var worst: f64 = 0;
    var refmax: f64 = 0;
    for (0..rows) |m| for (0..n) |col| {
        var acc: f64 = 0;
        for (0..k) |j| acc += @as(f64, h[m * k + j]) * w[col * k + j];
        const d = @as(f64, out[m * n + col]) - acc;
        err2 += d * d;
        ref2 += acc * acc;
        worst = @max(worst, @abs(d));
        refmax = @max(refmax, @abs(acc));
    };
    std.debug.print("  {s:<22} {d:2} rows  N {d:6} K {d:6}  rel L2 {e:.2}  worst |d| / max|ref| {e:.2}\n", .{ c.name, rows, n, k, @sqrt(err2 / ref2), worst / refmax });
}

pub fn gemmTest(r: *fwd.Runner) !void {
    var prng = std.Random.DefaultPrng.init(7);
    var rng = prng.random();
    const lm = plan.prefix;
    const sa0 = lm ++ "layers.0.self_attn.";
    const sa5 = lm ++ "layers.5.self_attn.";
    const sa30 = lm ++ "layers.30.self_attn.";
    const sa35 = lm ++ "layers.35.self_attn.";
    const L = r.m.layers;
    const cases = [_]Case{
        .{ .name = "qkv sliding (L0)", .prefix = sa0, .parts = &.{ "q_proj", "k_proj", "v_proj" }, .proj = L[0].qkv, .amp = 2 },
        .{ .name = "qkv global (L5)", .prefix = sa5, .parts = &.{ "q_proj", "k_proj", "v_proj" }, .proj = L[5].qkv, .amp = 2 },
        .{ .name = "q sliding (L30)", .prefix = sa30, .parts = &.{"q_proj"}, .proj = L[30].qkv, .amp = 2 },
        .{ .name = "q global (L35)", .prefix = sa35, .parts = &.{"q_proj"}, .proj = L[35].qkv, .amp = 2 },
        .{ .name = "o sliding (L0)", .prefix = sa0, .parts = &.{"o_proj"}, .proj = L[0].o, .amp = 1 },
        .{ .name = "o global (L5)", .prefix = sa5, .parts = &.{"o_proj"}, .proj = L[5].o, .amp = 1 },
        .{ .name = "gate|up (L3)", .prefix = lm ++ "layers.3.mlp.", .parts = &.{ "gate_proj", "up_proj" }, .proj = L[3].gu, .amp = 2 },
        .{ .name = "down (L3)", .prefix = lm ++ "layers.3.mlp.", .parts = &.{"down_proj"}, .proj = L[3].down, .amp = 1 },
        .{ .name = "ple gate (L3)", .prefix = lm ++ "layers.3.", .parts = &.{"per_layer_input_gate"}, .proj = L[3].ple_gate, .amp = 8 },
        .{ .name = "ple proj (L3)", .prefix = lm ++ "layers.3.", .parts = &.{"per_layer_projection"}, .proj = L[3].ple_proj, .amp = 1 },
        .{ .name = "ple model proj", .prefix = lm, .parts = &.{"per_layer_model_projection"}, .proj = r.m.ple_proj, .amp = 8 },
        .{ .name = "head (tied embed)", .prefix = lm ++ "embed_tokens", .parts = &.{""}, .proj = r.m.head, .amp = 4 },
    };
    for (cases) |c| {
        if (c.proj.n > 100000 or c.proj.dense) continue; // the head (see headTest), BF16 linears (see --ple-check)
        for ([_]u32{ 1, 8, 16 }) |rows| try run(r, c, rows, &rng);
    }
}
