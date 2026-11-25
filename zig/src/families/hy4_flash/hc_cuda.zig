//! Hy4-flash hyper-connection CUDA launchers (`hy4_hc.cu`): stream mix, branch merge, and head collapse.
const std = @import("std");
const cuda = @import("cuda");

const launch = cuda.launch;

pub const streams = 4;
const threads = 256;

/// FP32 mix weights and scalars for a hyper-connection pre or head launch.
pub const Mix = struct {
    fn_: u64,
    base: u64,
    scale: u64,
    eps: f32,
    hc_eps: f32,
    magnitude: f32 = 0,
};

pub const Hc = struct {
    mod: cuda.Module,
    pre_fn: cuda.Function,
    head_fn: cuda.Function,
    merge_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Hc {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.hy4_hc);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .pre_fn = try mod.function("hy4_hc_pre"),
            .head_fn = try mod.function("hy4_hc_head"),
            .merge_fn = try mod.function("hy4_hc_merge"),
        };
    }

    pub fn unload(h: *Hc) void {
        h.mod.unload();
        h.* = undefined;
    }

    /// Collapse `x` bf16 [rows, 4, hidden] into `out` bf16 [rows, hidden] and `post` f32 [rows, 4].
    pub fn pre(h: *const Hc, s: cuda.Stream, m: Mix, x: u64, out: u64, post: u64, rows: usize, hidden: usize) !void {
        return launchMix(h.pre_fn, s, m, x, out, post, rows, hidden, 0);
    }

    /// Collapse final streams into `out` bf16 [rows, hidden] before final norm.
    pub fn head(h: *const Hc, s: cuda.Stream, m: Mix, x: u64, out: u64, rows: usize, hidden: usize) !void {
        return launchMix(h.head_fn, s, m, x, out, 0, rows, hidden, 0);
    }

    /// Merge branch `f` bf16 [rows, hidden] back into the four streams in `out`.
    pub fn merge(h: *const Hc, s: cuda.Stream, residual: u64, f: u64, post: u64, out: u64, rows: usize, hidden: usize) !void {
        return launchMerge(h.merge_fn, s, residual, f, post, out, rows, hidden, 0);
    }
};

/// Launch any `hy4_hc_pre` or `hy4_hc_head` entry point (production or contract control).
pub fn launchMix(f: cuda.Function, s: cuda.Stream, m: Mix, x: u64, out: u64, post: u64, rows: usize, hidden: usize, repeat: i32) !void {
    if (rows == 0) return;
    var a: launch.Args = .{};
    a.add(x);
    a.add(m.fn_);
    a.add(m.base);
    a.add(m.scale);
    a.add(m.eps);
    a.add(m.hc_eps);
    a.add(m.magnitude);
    a.add(out);
    a.add(post);
    a.add(@as(i32, @intCast(hidden)));
    a.add(repeat);
    return launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = threads } }, s, &a);
}

/// Launch any `hy4_hc_merge` entry point (production or contract control).
pub fn launchMerge(f: cuda.Function, s: cuda.Stream, residual: u64, branch: u64, post: u64, out: u64, rows: usize, hidden: usize, repeat: i32) !void {
    const count = rows * streams * hidden;
    if (count == 0) return;
    var a: launch.Args = .{};
    a.add(residual);
    a.add(branch);
    a.add(post);
    a.add(out);
    a.add(@as(i32, @intCast(rows)));
    a.add(@as(i32, @intCast(hidden)));
    a.add(repeat);
    return launch.launch(f, .{ .grid = .{ .x = @intCast((count + threads - 1) / threads), .y = 1, .z = 1 }, .block = .{ .x = threads } }, s, &a);
}

test "hc_cuda: stream and thread constants match kernel launch geometry" {
    try std.testing.expectEqual(@as(usize, 4), streams);
    try std.testing.expectEqual(@as(usize, 256), threads);
}
