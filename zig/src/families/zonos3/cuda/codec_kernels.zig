//! The codec decoder's launches: zonos3_codec.cu by symbol.
const std = @import("std");
const cuda = @import("cuda");
const Blas = @import("blas.zig").Blas;

const names = .{
    "zc_linear", "zc_layernorm", "zc_attn_prep", "zc_attention", "zc_add_scaled", "zc_gelu", "zc_patch_up",
    "zc_dwconv", "zc_gather",    "zc_latent",    "zc_spec",      "zc_ola",        "zc_mix",  "zc_resample",
};

pub const F = enum(u8) { linear, layernorm, attn_prep, attention, add_scaled, gelu, patch_up, dwconv, gather, latent, spec, ola, mix, resample };

const n_fns = @typeInfo(F).@"enum".field_names.len;

pub const Kernels = struct {
    mod: cuda.Module,
    fns: [n_fns]cuda.Function,
    /// Set by the decoder: `linear` is then cuBLAS's GEMM, and `ones` holds a 1.0 for each row of a pass.
    blas: ?*const Blas,
    ones: u64,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = undefined;
        k.mod = try cuda.Module.load(d, cuda.kernels.zonos3_codec);
        errdefer k.mod.unload();
        k.blas = null;
        k.ones = 0;
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

    fn l(x: usize) i64 {
        return @intCast(x);
    }

    /// y [rows, n] = x [rows, kdim] W^T (+ bias when not 0), all fp32.
    pub fn linear(k: *const Kernels, s: cuda.Stream, w: u64, bias: u64, x: u64, xs: usize, kdim: usize, n: usize, y: u64, ys: usize, rows: usize) !void {
        if (k.blas) |b| {
            // column-major: y^T [n, rows] = (W^T [kdim, n])^T x^T [kdim, rows], then + bias [n, 1] ones [1, rows]
            try b.gemm(s, .t, .n, n, rows, kdim, 1.0, w, kdim, 0, x, xs, 0, 0.0, y, ys, 0, 1, .f32, .f32);
            if (bias != 0) try b.gemm(s, .n, .n, n, rows, 1, 1.0, bias, n, 0, k.ones, 1, 0, 1.0, y, ys, 0, 1, .f32, .f32);
            return;
        }
        var a: cuda.Args = .{};
        a.add(w);
        a.add(x);
        a.add(i(xs));
        a.add(i(kdim));
        a.add(i(n));
        a.add(y);
        a.add(i(ys));
        a.add(bias);
        try k.go(.linear, s, (n + 7) / 8, rows, 256, kdim * 4, &a);
    }

    pub fn layernorm(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, b: u64, y: u64, d: usize, eps: f32, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(w);
        a.add(b);
        a.add(y);
        a.add(i(d));
        a.add(eps);
        try k.go(.layernorm, s, rows, 1, 256, 0, &a);
    }

    pub fn attnPrep(k: *const Kernels, s: cuda.Stream, qkv: u64, d: usize, pos0: usize, kc: u64, vc: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(qkv);
        a.add(i(d));
        a.add(i(pos0));
        a.add(kc);
        a.add(vc);
        try k.go(.attn_prep, s, rows, d / 64, 32, 0, &a);
    }

    pub fn attention(k: *const Kernels, s: cuda.Stream, qkv: u64, d: usize, kc: u64, vc: u64, pos0: usize, ctx: usize, out: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(qkv);
        a.add(i(d));
        a.add(kc);
        a.add(vc);
        a.add(i(pos0));
        a.add(i(ctx));
        a.add(out);
        try k.go(.attention, s, rows, d / 64, 128, 0, &a);
    }

    /// x += scale * y per channel; scale == 0 is a plain add.
    pub fn addScaled(k: *const Kernels, s: cuda.Stream, x: u64, y: u64, scale: u64, d: usize, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(y);
        a.add(scale);
        a.add(i(d));
        try k.go(.add_scaled, s, (d + 255) / 256, rows, 256, 0, &a);
    }

    pub fn gelu(k: *const Kernels, s: cuda.Stream, x: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(l(n));
        try k.go(.gelu, s, (n + 255) / 256, 1, 256, 0, &a);
    }

    /// in [rows / 2][c] to out [rows][c / 2].
    pub fn patchUp(k: *const Kernels, s: cuda.Stream, in: u64, out: u64, c: usize, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(in);
        a.add(out);
        a.add(i(c));
        try k.go(.patch_up, s, (c / 2 + 255) / 256, rows, 256, 0, &a);
    }

    pub fn dwconv(k: *const Kernels, s: cuda.Stream, x: u64, t: usize, w: u64, b: u64, hist: u64, out: u64, c: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(i(t));
        a.add(w);
        a.add(b);
        a.add(hist);
        a.add(out);
        a.add(i(c));
        try k.go(.dwconv, s, (c + 255) / 256, 1, 256, 0, &a);
    }

    pub fn gather(k: *const Kernels, s: cuda.Stream, codes: u64, col: usize, table: u64, out: u64, d: usize, t: usize) !void {
        var a: cuda.Args = .{};
        a.add(codes);
        a.add(i(col));
        a.add(table);
        a.add(out);
        a.add(i(d));
        try k.go(.gather, s, (d + 255) / 256, t, 256, 0, &a);
    }

    pub fn latent(k: *const Kernels, s: cuda.Stream, codes: u64, tables: u64, z: u64, d: usize, t: usize) !void {
        var a: cuda.Args = .{};
        a.add(codes);
        a.add(tables);
        a.add(z);
        a.add(i(d));
        try k.go(.latent, s, (d + 255) / 256, t, 256, 0, &a);
    }

    pub fn spec(k: *const Kernels, s: cuda.Stream, y: u64, half: usize, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(y);
        a.add(i(half));
        try k.go(.spec, s, (half + 255) / 256, rows, 256, 0, &a);
    }

    pub fn ola(k: *const Kernels, s: cuda.Stream, frames: u64, win2: u64, f: usize, n: usize, hop: usize, n0: usize, count: usize, out: u64) !void {
        var a: cuda.Args = .{};
        a.add(frames);
        a.add(win2);
        a.add(i(f));
        a.add(i(n));
        a.add(i(hop));
        a.add(l(n0));
        a.add(l(count));
        a.add(out);
        try k.go(.ola, s, (count + 255) / 256, 1, 256, 0, &a);
    }

    /// Branch rows `stride` samples apart, read as zero outside [0, total).
    pub fn mix(k: *const Kernels, s: cuda.Stream, branches: u64, stride: usize, total: usize, w: u64, n0: usize, count: usize, out: u64) !void {
        var a: cuda.Args = .{};
        a.add(branches);
        a.add(l(stride));
        a.add(l(total));
        a.add(w);
        a.add(l(n0));
        a.add(l(count));
        a.add(out);
        try k.go(.mix, s, (count + 255) / 256, 1, 256, 0, &a);
    }

    pub fn resample(k: *const Kernels, s: cuda.Stream, x: u64, total: usize, kernel: u64, m0: usize, count: usize, out: u64) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(l(total));
        a.add(kernel);
        a.add(l(m0));
        a.add(l(count));
        a.add(out);
        try k.go(.resample, s, (count + 255) / 256, 1, 256, 0, &a);
    }
};
