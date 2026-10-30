//! nemo_audio.cu's launches: q/k LayerNorm with partial RoPE, attention over a window's keys, the Sortformer glue.
const std = @import("std");
const cuda = @import("cuda");

const Args = cuda.Args;
const Stream = cuda.Stream;

/// The widest head (nemo_audio.cu's MAXD) and the attention block's threads.
pub const max_dim = 128;
const attention_threads = 128;
/// Keys one attention block holds scores for: a window plus the speaker cache and FIFO, with room.
pub const max_keys = 2048;

pub const Kernels = struct {
    mod: cuda.Module,
    qk_rope: cuda.Function,
    attention: cuda.Function,
    im2col: cuda.Function,
    relu: cuda.Function,
    zero_rows: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.nemo_audio);
        errdefer mod.unload();
        const k: Kernels = .{
            .mod = mod,
            .qk_rope = try mod.function("tf_na_qk_rope"),
            .attention = try mod.function("tf_na_attention"),
            .im2col = try mod.function("tf_na_im2col3"),
            .relu = try mod.function("tf_na_relu"),
            .zero_rows = try mod.function("tf_na_zero_rows"),
        };
        try k.attention.allowDynamicShared(max_keys * 4);
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
    }

    /// qkv [rows, 3, heads, dim] in place: q and k LayerNormed when `norm`, then RoPE on the first `rot` channels.
    pub fn qkRope(k: *const Kernels, s: Stream, qkv: u64, norm: ?[4]u64, cs: u64, sn: u64, rows: usize, heads: usize, dim: usize, rot: usize, eps: f32) !void {
        if (dim > max_dim or rot > dim or rot % 2 != 0) return error.UnsupportedShape;
        if (rows == 0) return;
        var a: Args = .{};
        a.add(qkv);
        for (norm orelse .{ 0, 0, 0, 0 }) |p| a.add(p);
        a.add(cs);
        a.add(sn);
        for ([_]usize{ heads, dim, rot }) |n| a.add(int(n));
        a.add(eps);
        try cuda.launch.launch(k.qk_rope, .{ .grid = .{ .x = @intCast(rows * heads * 2) }, .block = .{ .x = 32 } }, s, &a);
    }

    /// out [rows, heads, dim] = softmax(q k^T scale) v over keys [0, len) of qkv [rows, 3, heads, dim].
    pub fn attend(k: *const Kernels, s: Stream, qkv: u64, out: u64, rows: usize, heads: usize, dim: usize, len: usize, scale: f32) !void {
        if (dim > max_dim or len > rows or rows > max_keys) return error.UnsupportedShape;
        if (rows == 0) return;
        var a: Args = .{};
        a.add(qkv);
        a.add(out);
        for ([_]usize{ heads, dim, len }) |n| a.add(int(n));
        a.add(scale);
        try cuda.launch.launch(k.attention, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(heads) }, .block = .{ .x = attention_threads }, .shared = @intCast(rows * 4) }, s, &a);
    }

    /// y [rows, 3 c] from x [rows, c]: the three taps of a kernel-3, pad-1 conv, rows from `len` on read as zero.
    pub fn im2col3(k: *const Kernels, s: Stream, x: u64, y: u64, rows: usize, c: usize, len: usize) !void {
        var a: Args = .{};
        a.add(x);
        a.add(y);
        for ([_]usize{ rows, c, len }) |n| a.add(int(n));
        try cuda.launch.launch(k.im2col, .{ .grid = .{ .x = @intCast(blocks(rows * 3 * c)) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn reluInPlace(k: *const Kernels, s: Stream, x: u64, count: usize) !void {
        var a: Args = .{};
        a.add(x);
        a.add(@as(i64, @intCast(count)));
        try cuda.launch.launch(k.relu, .{ .grid = .{ .x = @intCast(blocks(count)) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// Rows [len, rows) of x [rows, c] set to zero.
    pub fn zeroRows(k: *const Kernels, s: Stream, x: u64, rows: usize, c: usize, len: usize) !void {
        if (len >= rows) return;
        var a: Args = .{};
        a.add(x);
        for ([_]usize{ rows, c, len }) |n| a.add(int(n));
        try cuda.launch.launch(k.zero_rows, .{ .grid = .{ .x = @intCast(blocks((rows - len) * c)) }, .block = .{ .x = 256 } }, s, &a);
    }
};

fn blocks(n: usize) usize {
    return @max(1, (n + 255) / 256);
}

fn int(x: usize) i32 {
    return @intCast(x);
}
