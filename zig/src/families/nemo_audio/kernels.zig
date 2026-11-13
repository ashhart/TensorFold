//! nemo_audio.cu's launches: mel features, their norm, q/k norm and rotary, head-64 attention, speaker head, fusion.
const std = @import("std");
const cuda = @import("cuda");

/// The most rows one head-64 attention call takes (the kernel's score row in shared memory).
pub const attention_rows = 1024;

fn int(x: usize) c_int {
    return @intCast(x);
}

fn ceil(n: usize, d: usize) u32 {
    return @intCast((n + d - 1) / d);
}

pub const Kernels = struct {
    mod: cuda.Module,
    mel_fn: cuda.Function,
    norm_fn: cuda.Function,
    qk_fn: cuda.Function,
    attn_fn: cuda.Function,
    taps_fn: cuda.Function,
    relu_fn: cuda.Function,
    preds_fn: cuda.Function,
    fuse_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.nemo_audio);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .mel_fn = try mod.function("tf_nemo_mel"),
            .norm_fn = try mod.function("tf_nemo_norm"),
            .qk_fn = try mod.function("tf_nemo_qk"),
            .attn_fn = try mod.function("tf_nemo_attn64"),
            .taps_fn = try mod.function("tf_nemo_taps"),
            .relu_fn = try mod.function("tf_nemo_relu"),
            .preds_fn = try mod.function("tf_nemo_preds"),
            .fuse_fn = try mod.function("tf_nemo_fuse"),
        };
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
    }

    /// Frames [first, first + frames) of out bf16 [frame][128] from x (f32 samples [lo, ...) of n); zeros from `valid`.
    pub fn mel(k: *const Kernels, s: cuda.Stream, x: u64, lo: usize, n: usize, window: u64, fb: u64, cs: u64, sn: u64, out: u64, first: usize, frames: usize, valid: usize, preemph: f32, guard: f32) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@as(i64, @intCast(lo)));
        a.add(@as(i64, @intCast(n)));
        for ([_]u64{ window, fb, cs, sn, out }) |p| a.add(p);
        a.add(int(first));
        a.add(int(valid));
        a.add(preemph);
        a.add(guard);
        try go(k.mel_fn, s, .{ .x = @intCast(frames) }, &a);
    }

    /// y bf16 [frames][128]: each feature of x normalised over its first `valid` frames, zeros after them.
    pub fn norm(k: *const Kernels, s: cuda.Stream, x: u64, y: u64, frames: usize, valid: usize, eps: f32) !void {
        if (valid < 2) return error.AudioTooShort;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(y);
        a.add(int(frames));
        a.add(int(valid));
        a.add(eps);
        try go(k.norm_fn, s, .{ .x = 128 / 32 }, &a);
    }

    /// qkv [rows][3 heads dim] in place: q and k heads LayerNormed (with `qk_norm`), then turned.
    pub fn qk(k: *const Kernels, s: cuda.Stream, qkv: u64, qk_norm: ?[4]u64, cs: u64, sn: u64, rows: usize, heads: usize, dim: usize, eps: f32) !void {
        if (dim != 64 and dim != 128) return error.UnsupportedShape;
        var a: cuda.Args = .{};
        a.add(qkv);
        for (qk_norm orelse [4]u64{ 0, 0, 0, 0 }) |p| a.add(p);
        a.add(cs);
        a.add(sn);
        for ([_]usize{ rows, heads, dim }) |v| a.add(int(v));
        a.add(eps);
        try go(k.qk_fn, s, .{ .x = @intCast(rows), .y = ceil(2 * heads, 8) }, &a);
    }

    /// out [rows][heads 64] = softmax(q k^T scale) v of qkv [rows][3 heads 64].
    pub fn attention(k: *const Kernels, s: cuda.Stream, qkv: u64, out: u64, rows: usize, heads: usize, scale: f32) !void {
        if (rows == 0) return;
        if (rows > attention_rows) return error.AudioRows;
        var a: cuda.Args = .{};
        a.add(qkv);
        a.add(out);
        a.add(int(rows));
        a.add(int(heads));
        a.add(scale);
        try go(k.attn_fn, s, .{ .x = ceil(rows, 8), .y = @intCast(heads) }, &a);
    }

    /// y [rows][3 width]: each row's three taps (the row before, itself, the row after) channel-major.
    pub fn taps(k: *const Kernels, s: cuda.Stream, x: u64, y: u64, rows: usize, width: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(y);
        a.add(int(rows));
        a.add(int(width));
        try go(k.taps_fn, s, .{ .x = @intCast(rows) }, &a);
    }

    pub fn relu(k: *const Kernels, s: cuda.Stream, x: u64, count: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@as(i64, @intCast(count)));
        try go(k.relu_fn, s, .{ .x = ceil(count, 256) }, &a);
    }

    /// preds f32 [rows][spk]: the mean of each row's `sub` sigmoids of logits bf16 [rows sub][spk].
    pub fn preds(k: *const Kernels, s: cuda.Stream, logits: u64, out: u64, rows: usize, spk: usize, sub: usize) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(out);
        for ([_]usize{ rows, spk, sub }) |v| a.add(int(v));
        try go(k.preds_fn, s, .{ .x = ceil(rows * spk, 256) }, &a);
    }

    /// out [rows][width] = an + LN(preds > thr) . kernel; sn (optional) [rows][8] the normed activity.
    pub fn fuse(k: *const Kernels, s: cuda.Stream, an: u64, preds_: u64, nw: u64, nb: u64, kernel: u64, out: u64, sn: ?u64, rows: usize, width: usize, thr: f32, eps: f32) !void {
        if (rows == 0) return;
        var a: cuda.Args = .{};
        for ([_]u64{ an, preds_, nw, nb, kernel, out, sn orelse 0 }) |p| a.add(p);
        a.add(int(width));
        a.add(thr);
        a.add(eps);
        try go(k.fuse_fn, s, .{ .x = @intCast(rows) }, &a);
    }
};

fn go(f: cuda.Function, s: cuda.Stream, grid: cuda.Dim3, a: *cuda.Args) !void {
    if (grid.x == 0) return;
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = 256 } }, s, a);
}

test "nemo audio launch geometry" {
    try std.testing.expectEqual(@as(u32, 104), ceil(828, 8));
    try std.testing.expectEqual(@as(u32, 2), ceil(2 * 8, 8));
    try std.testing.expectEqual(@as(u32, 3), ceil(2 * 10, 8));
}
