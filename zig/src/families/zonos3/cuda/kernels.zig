//! ZONOS3 Mini's launches: zonos3.cu by symbol. Every row kernel takes prompt rows and the decode row alike.
const std = @import("std");
const cuda = @import("cuda");

const names = .{
    "z3_embed",      "z3_rmsnorm",   "z3_attnres",   "z3_linear",   "z3_conv_silu", "z3_gdn_prep", "z3_gdn",
    "z3_gated_norm", "z3_attn_prep", "z3_attention", "z3_mul_silu", "z3_route",     "z3_combine",  "z3_softcap",
};

pub const F = enum(u8) { embed, rmsnorm, attnres, linear, conv_silu, gdn_prep, gdn, gated_norm, attn_prep, attention, mul_silu, route, combine, softcap };

const n_fns = @typeInfo(F).@"enum".field_names.len;

pub const Kernels = struct {
    mod: cuda.Module,
    fns: [n_fns]cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = undefined;
        k.mod = try cuda.Module.load(d, cuda.kernels.zonos3);
        errdefer k.mod.unload();
        inline for (names, 0..) |name, j| k.fns[j] = try k.mod.function(name);
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
    }

    fn go(k: *const Kernels, which: F, s: cuda.Stream, gx: usize, gy: usize, block: usize, shared: usize, a: *cuda.Args) !void {
        try cuda.launch.launch(k.fns[@backingInt(which)], .{ .grid = .{ .x = @intCast(gx), .y = @intCast(gy), .z = 1 }, .block = .{ .x = @intCast(block) }, .shared = @intCast(shared) }, s, a);
    }

    fn i(x: usize) i32 {
        return @intCast(x);
    }

    pub fn embed(k: *const Kernels, s: cuda.Stream, rows: u64, tables: u64, audio_rows: usize, out: u64, d: usize, t: usize) !void {
        var a: cuda.Args = .{};
        a.add(rows);
        a.add(tables);
        a.add(i(audio_rows));
        a.add(out);
        a.add(i(d));
        try k.go(.embed, s, t, 1, 256, 0, &a);
    }

    /// w == 0: the weightless norm.
    pub fn rmsnorm(k: *const Kernels, s: cuda.Stream, x: u64, xs: usize, w: u64, y: u64, ys: usize, d: usize, eps: f32, t: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(i(xs));
        a.add(w);
        a.add(y);
        a.add(i(ys));
        a.add(i(d));
        a.add(eps);
        try k.go(.rmsnorm, s, t, 1, 256, 0, &a);
    }

    pub fn attnres(k: *const Kernels, s: cuda.Stream, bank: u64, n_src: usize, cap: usize, query: u64, out: u64, d: usize, eps: f32, t: usize) !void {
        var a: cuda.Args = .{};
        a.add(bank);
        a.add(i(n_src));
        a.add(i(cap));
        a.add(query);
        a.add(out);
        a.add(i(d));
        a.add(eps);
        try k.go(.attnres, s, t, 1, 256, 0, &a);
    }

    pub const Linear = struct { bias: u64 = 0, picks: u64 = 0, pick_stride: usize = 0, xrep: usize = 1 };

    /// y [rows, n] = x [rows / xrep, kdim] W^T: bf16 weights, fp32 rows xs / ys elements apart.
    pub fn linear(k: *const Kernels, s: cuda.Stream, w: u64, x: u64, xs: usize, kdim: usize, n: usize, y: u64, ys: usize, rows: usize, o: Linear) !void {
        var a: cuda.Args = .{};
        a.add(w);
        a.add(x);
        a.add(i(xs));
        a.add(i(o.xrep));
        a.add(i(kdim));
        a.add(i(n));
        a.add(y);
        a.add(i(ys));
        a.add(o.bias);
        a.add(o.picks);
        a.add(@as(i64, @intCast(o.pick_stride)));
        try k.go(.linear, s, (n + 7) / 8, rows, 256, kdim * 4, &a);
    }

    pub fn convSilu(k: *const Kernels, s: cuda.Stream, proj: u64, ps: usize, t: usize, w: u64, state: u64, out: u64, c: usize) !void {
        var a: cuda.Args = .{};
        a.add(proj);
        a.add(i(ps));
        a.add(i(t));
        a.add(w);
        a.add(state);
        a.add(out);
        a.add(i(c));
        try k.go(.conv_silu, s, (c + 255) / 256, 1, 256, 0, &a);
    }

    pub fn gdnPrep(k: *const Kernels, s: cuda.Stream, conv: u64, proj: u64, ps: usize, a_off: usize, a_log: u64, dt_bias: u64, gates: u64, t: usize) !void {
        var a: cuda.Args = .{};
        a.add(conv);
        a.add(proj);
        a.add(i(ps));
        a.add(i(a_off));
        a.add(a_log);
        a.add(dt_bias);
        a.add(gates);
        try k.go(.gdn_prep, s, t, 4, 128, 0, &a);
    }

    pub fn gdn(k: *const Kernels, s: cuda.Stream, conv: u64, gates: u64, t: usize, state: u64, o: u64) !void {
        var a: cuda.Args = .{};
        a.add(conv);
        a.add(gates);
        a.add(i(t));
        a.add(state);
        a.add(o);
        try k.go(.gdn, s, 4, 1, 128, 0, &a);
    }

    pub fn gatedNorm(k: *const Kernels, s: cuda.Stream, o: u64, proj: u64, ps: usize, z_off: usize, w: u64, y: u64, t: usize) !void {
        var a: cuda.Args = .{};
        a.add(o);
        a.add(proj);
        a.add(i(ps));
        a.add(i(z_off));
        a.add(w);
        a.add(y);
        try k.go(.gated_norm, s, t, 4, 128, 0, &a);
    }

    pub fn attnPrep(k: *const Kernels, s: cuda.Stream, q: u64, kv: u64, temp: u64, pos0: usize, kc: u64, vc: u64, theta: f32, t: usize) !void {
        var a: cuda.Args = .{};
        a.add(q);
        a.add(kv);
        a.add(temp);
        a.add(i(pos0));
        a.add(kc);
        a.add(vc);
        a.add(theta);
        try k.go(.attn_prep, s, t, 1, 512, 0, &a);
    }

    pub fn attention(k: *const Kernels, s: cuda.Stream, q: u64, kc: u64, vc: u64, gate: u64, pos0: usize, out: u64, t: usize) !void {
        var a: cuda.Args = .{};
        a.add(q);
        a.add(kc);
        a.add(vc);
        a.add(gate);
        a.add(i(pos0));
        a.add(out);
        try k.go(.attention, s, t, 16, 256, 0, &a);
    }

    /// out [rows, n] = in[a_off ..] * silu(in[b_off ..]).
    pub fn mulSilu(k: *const Kernels, s: cuda.Stream, in: u64, stride: usize, a_off: usize, b_off: usize, n: usize, out: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(in);
        a.add(i(stride));
        a.add(i(a_off));
        a.add(i(b_off));
        a.add(i(n));
        a.add(out);
        try k.go(.mul_silu, s, (n + 255) / 256, rows, 256, 0, &a);
    }

    pub fn route(k: *const Kernels, s: cuda.Stream, logits: u64, bias: u64, e: usize, picks: u64, w: u64, t: usize) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(bias);
        a.add(i(e));
        a.add(picks);
        a.add(w);
        try k.go(.route, s, t, 1, 32, 0, &a);
    }

    pub fn combine(k: *const Kernels, s: cuda.Stream, down: u64, w: u64, n: usize, out: u64, t: usize) !void {
        var a: cuda.Args = .{};
        a.add(down);
        a.add(w);
        a.add(i(n));
        a.add(out);
        try k.go(.combine, s, (n + 255) / 256, t, 256, 0, &a);
    }

    pub fn softcap(k: *const Kernels, s: cuda.Stream, x: u64, n: usize, cap: f32) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@as(i64, @intCast(n)));
        a.add(cap);
        try k.go(.softcap, s, (n + 255) / 256, 1, 256, 0, &a);
    }
};
