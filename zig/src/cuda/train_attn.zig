//! train_attn.cu's gated attention input gradients over 4-bit projections, with train.cu's causal key and value part.
const std = @import("std");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const kernels = @import("kernels.zig");
const q4 = @import("train_q4.zig");

const Driver = @import("driver.zig").Driver;
const Module = module.Module;
const Function = module.Function;
const Args = launch_.Args;

/// Attention's shape (train.cu's Heads): rows, heads, kv heads, head dim, the q|k|v row width, scale, prefix keys.
pub const Heads = extern struct { rows: c_int, heads: c_int, kv_heads: c_int, dim: c_int, qkv: c_int, scale: f32, prefix: c_int = 0 };

/// Keys tf_gattn_q reads at most (its shared score rows).
pub const max_keys = 2048;

pub const Attn = struct {
    q: q4.Q4, // q_proj [heads 2 dim, d]: each head's q then its gate
    k: q4.Q4, // k_proj [kv dim, d]
    v: q4.Q4, // v_proj [kv dim, d]
    o: q4.Q4, // o_proj [d, heads dim]
    qscale: u64, // q_norm's fp32 scale [dim]
    kscale: u64, // k_norm's fp32 scale [dim]
    inv: u64, // RoPE's fp32 inverse frequencies [half]
    heads: usize,
    kv_heads: usize,
    dim: usize,
    half: usize,
    eps: f32,
};

/// What attention kept, bf16: pq [rows, heads 2 dim], pk and pv [rows, kv dim], o_proj's input gated [rows, q dim].
pub const Kept = struct { pq: u64, pk: u64, pv: u64, gated: u64, pos0: usize };

/// Scratch: qkv bf16 [rows, (heads + 2 kv) dim], the rest fp32; probs and dscores [heads, rows, rows].
pub const Scratch = struct { qkv: u64, dg: u64, dout: u64, dpq: u64, dq: u64, dk: u64, dv: u64, dpk: u64, probs: u64, dscores: u64 };

const names = [_][:0]const u8{ "tf_gattn_prep", "tf_gattn_q", "tf_gattn_gate_back", "tf_gattn_prep_back" };

fn index(comptime name: []const u8) usize {
    inline for (names, 0..) |n, i| if (comptime std.mem.eql(u8, n, name)) return i;
    @compileError("no kernel " ++ name);
}

pub const Kernels = struct {
    attn: Module,
    train: Module,
    fns: [names.len]Function,
    kv: Function,

    pub fn load(d: *const Driver) !Kernels {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = .{ .attn = try Module.load(d, kernels.train_attn), .train = undefined, .fns = undefined, .kv = undefined };
        errdefer k.attn.unload();
        k.train = try Module.load(d, kernels.train);
        errdefer k.train.unload();
        for (names, &k.fns) |n, *f| f.* = try k.attn.function(n);
        k.kv = try k.train.function("tf_train_attn_kv");
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.attn.unload();
        k.train.unload();
    }
};

fn int(x: usize) c_int {
    return @intCast(x);
}

fn u(x: usize) u32 {
    return @intCast(x);
}

fn up(n: usize, by: usize) u32 {
    return @intCast((n + by - 1) / by);
}

/// Launches on the products' stream.
pub const Ops = struct {
    k: *const Kernels,
    q: q4.Ops,

    fn go(o: Ops, f: Function, grid: [3]u32, block: u32, args: *Args) !void {
        try launch_.launch(f, .{ .grid = .{ .x = grid[0], .y = grid[1], .z = grid[2] }, .block = .{ .x = block } }, o.q.s, args);
    }

    /// The prepared heads (normed, rotated q and k; v) into qkv, as the forward made them.
    pub fn prep(o: Ops, at: Attn, kept: Kept, qkv: u64, rows: usize) !void {
        var a: Args = .{};
        for ([_]u64{ kept.pq, kept.pk, kept.pv, at.qscale, at.kscale, at.inv }) |v| a.add(v);
        a.add(int(kept.pos0));
        a.add(qkv);
        for ([_]usize{ at.heads, at.kv_heads, at.dim, at.half }) |v| a.add(int(v));
        a.add(at.eps);
        try o.go(o.k.fns[index("tf_gattn_prep")], .{ u(at.heads + 2 * at.kv_heads), u(rows), 1 }, u(at.dim), &a);
    }

    /// Causal attention over `rows` rows undone: dx [rows, d] (+)= its input gradient from g, its output's [rows, d].
    pub fn back(o: Ops, at: Attn, kept: Kept, g: u64, dx: u64, t: Scratch, rows: usize, accumulate: bool) !void {
        if (rows > max_keys or at.dim > 256 or at.dim % 32 != 0) return error.UnsupportedShape;
        const qw = at.heads * at.dim;
        const kw = at.kv_heads * at.dim;
        const d = at.o.n;
        try o.prep(at, kept, t.qkv, rows);
        try o.q.product(g, false, d, at.o, t.dg, qw, rows, false, 0);
        var a: Args = .{};
        for ([_]u64{ t.dg, kept.gated, kept.pq, t.dout, t.dpq }) |v| a.add(v);
        a.add(int(at.heads));
        a.add(int(at.dim));
        try o.go(o.k.fns[index("tf_gattn_gate_back")], .{ up(qw, 256), u(rows), 1 }, 256, &a);
        const h: Heads = .{ .rows = int(rows), .heads = int(at.heads), .kv_heads = int(at.kv_heads), .dim = int(at.dim), .qkv = int(qw + 2 * kw), .scale = @floatCast(1 / @sqrt(@as(f64, @floatFromInt(at.dim)))) };
        var b: Args = .{};
        for ([_]u64{ t.qkv, t.dout, t.probs, t.dscores, t.dq }) |v| b.add(v);
        b.add(h);
        b.add(t.qkv);
        b.add(t.qkv);
        try o.go(o.k.fns[index("tf_gattn_q")], .{ u(at.heads), u(rows), 1 }, 256, &b);
        var c: Args = .{};
        for ([_]u64{ t.qkv, t.dout, t.probs, t.dscores, t.dk, t.dv }) |v| c.add(v);
        c.add(h);
        try o.go(o.k.kv, .{ u(at.kv_heads), u(rows), 1 }, u(at.dim), &c);
        var p: Args = .{};
        for ([_]u64{ kept.pq, kept.pk, at.qscale, at.kscale, at.inv }) |v| p.add(v);
        p.add(int(kept.pos0));
        for ([_]u64{ t.dq, t.dk, t.dpq, t.dpk }) |v| p.add(v);
        for ([_]usize{ at.heads, at.kv_heads, at.dim, at.half }) |v| p.add(int(v));
        p.add(at.eps);
        try o.go(o.k.fns[index("tf_gattn_prep_back")], .{ u(at.heads + at.kv_heads), u(rows), 1 }, u(at.dim), &p);
        try o.q.product(t.dpq, false, 2 * qw, at.q, dx, d, rows, accumulate, 0);
        try o.q.product(t.dpk, false, kw, at.k, dx, d, rows, true, 0);
        try o.q.product(t.dv, false, kw, at.v, dx, d, rows, true, 0);
    }
};

test "the shape the kernels read is train.cu's" {
    try std.testing.expectEqual(@as(usize, 28), @sizeOf(Heads));
}
