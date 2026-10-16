//! lfm2.cu's glue kernels, one call each on a stream: bf16 activations, torch's bf16 rounding, fixed-order fp32 sums.
const cuda = @import("cuda");

const threads = 256; // lfm2.cu's NT
const max_width = 16 * threads; // NT * MAXE
const lse_threads = 1024; // LT
const max_dim = 64; // 2 * HALF

pub const Kernels = struct {
    mod: cuda.Module,
    add_rms_fn: cuda.Function,
    short_conv_fn: cuda.Function,
    qk_norm_rope_fn: cuda.Function,
    silu_mul_fn: cuda.Function,
    label_logits_fn: cuda.Function,
    logsumexp_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.lfm2);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .add_rms_fn = try mod.function("tf_lfm2_add_rms"),
            .short_conv_fn = try mod.function("tf_lfm2_short_conv"),
            .qk_norm_rope_fn = try mod.function("tf_lfm2_qk_norm_rope"),
            .silu_mul_fn = try mod.function("tf_lfm2_silu_mul"),
            .label_logits_fn = try mod.function("tf_lfm2_label_logits"),
            .logsumexp_fn = try mod.function("tf_lfm2_logsumexp"),
        };
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
    }

    /// h += y in place (y null: h unchanged), then x = rms(h) * w; rows up to 4096 wide.
    pub fn addRms(k: *const Kernels, s: cuda.Stream, h: u64, y: ?u64, w: u64, x: u64, rows: usize, width: usize, eps: f32) !void {
        if (width > max_width) return error.RowShape;
        if (rows == 0) return;
        var a: cuda.Args = .{};
        for ([_]u64{ h, y orelse 0, w, x }) |v| a.add(v);
        a.add(int(width));
        a.add(eps);
        try go(k.add_rms_fn, s, rows, 1, threads, &a);
    }

    /// out [rows, C] = C * conv(B * x), proj [rows, 3C] = B | C | x, prev [rows, taps - 1] newest first (-1: zero).
    pub fn shortConv(k: *const Kernels, s: cuda.Stream, proj: u64, weight: u64, prev: u64, out: u64, rows: usize, channels: usize, taps: usize) !void {
        if (rows == 0) return;
        var a: cuda.Args = .{};
        for ([_]u64{ proj, weight, prev, out }) |v| a.add(v);
        a.add(int(channels));
        a.add(int(taps));
        try go(k.short_conv_fn, s, rows, blocks(channels, threads), threads, &a);
    }

    /// qkv [rows, (H + 2KV) D] -> q [rows, H, D], k [rows, KV, D] normed and rotated at positions, v [rows, KV, D].
    pub fn qkNormRope(k: *const Kernels, s: cuda.Stream, qkv: u64, qw: u64, kw: u64, positions: u64, q: u64, kout: u64, v: u64, rows: usize, heads: usize, kv_heads: usize, dim: usize, theta: f32, eps: f32) !void {
        if (dim % 2 != 0 or dim > max_dim) return error.RowShape;
        if (rows == 0) return;
        var a: cuda.Args = .{};
        for ([_]u64{ qkv, qw, kw, positions, q, kout, v }) |p| a.add(p);
        for ([_]usize{ heads, kv_heads, dim }) |n| a.add(int(n));
        a.add(theta);
        a.add(eps);
        try go(k.qk_norm_rope_fn, s, rows, 1, threads, &a);
    }

    /// out [rows, F] = silu(gu[:, :F]) * gu[:, F:].
    pub fn siluMul(k: *const Kernels, s: cuda.Stream, gu: u64, out: u64, rows: usize, ffn: usize) !void {
        if (rows == 0) return;
        var a: cuda.Args = .{};
        a.add(gu);
        a.add(out);
        a.add(int(ffn));
        try go(k.silu_mul_fn, s, rows, blocks(ffn, threads), threads, &a);
    }

    /// out[j] (f32) = dot(hidden [width], embed[labels[j]]) in f32.
    pub fn labelLogits(k: *const Kernels, s: cuda.Stream, hidden: u64, embed: u64, labels: u64, out: u64, count: usize, width: usize) !void {
        if (count == 0) return;
        var a: cuda.Args = .{};
        for ([_]u64{ hidden, embed, labels, out }) |v| a.add(v);
        a.add(int(width));
        try go(k.label_logits_fn, s, count, 1, threads, &a);
    }

    /// out (f32) = logsumexp of logits f32 [count].
    pub fn logsumexp(k: *const Kernels, s: cuda.Stream, logits: u64, out: u64, count: usize) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(out);
        a.add(int(count));
        try go(k.logsumexp_fn, s, 1, 1, lse_threads, &a);
    }
};

fn int(x: usize) i32 {
    return @intCast(x);
}

fn blocks(n: usize, by: usize) usize {
    return (n + by - 1) / by;
}

fn go(f: cuda.Function, s: cuda.Stream, x: usize, y: usize, block: u32, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(x), .y = @intCast(y), .z = 1 }, .block = .{ .x = block } }, s, a);
}
