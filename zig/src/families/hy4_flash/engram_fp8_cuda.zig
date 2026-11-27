//! CUDA launchers for Hy4-flash FP8 Engram: 100-byte row decode + per-head RMSNorm, rank-0 bias, and 4-stream gate.
const std = @import("std");
const cuda = @import("cuda");

const launch = cuda.launch;
const Stream = cuda.Stream;

pub const streams = 4;
pub const projections = 1 + streams;
pub const threads = 256;
pub const warps = threads / 32;
pub const norm_eps: f32 = std.math.floatEps(f32);

/// Device weight pointers for the post-allreduce 4-stream Engram gate.
pub const GateWeights = struct {
    norm_keys: u64,
    norm_queries: u64,
    pre_gamma: u64,
    stream_scales: u64,
    eps: f32 = norm_eps,
};

/// Launches `f` for 100-byte FP8 Engram row decode and per-head RMSNorm into BF16 `[rows, heads, dim]`.
pub fn launchEmbed(f: cuda.Function, s: Stream, raw: u64, head_norms: u64, eps: f32, out: u64, rows: usize, heads: usize, dim: usize, repeat: i32) !void {
    const pairs = rows * heads;
    if (pairs == 0 or dim == 0) return;
    var a: launch.Args = .{};
    a.add(raw);
    a.add(head_norms);
    a.add(eps);
    a.add(out);
    a.add(@as(i32, @intCast(pairs)));
    a.add(@as(i32, @intCast(heads)));
    a.add(@as(i32, @intCast(dim)));
    a.add(repeat);
    return launch.launch(f, .{ .grid = .{ .x = @intCast((pairs + warps - 1) / warps), .y = 1, .z = 1 }, .block = .{ .x = threads } }, s, &a);
}

/// Launches `f` to add rank-0 BF16 projection bias into FP32 partial sums `[rows, cols]`.
pub fn launchBias(f: cuda.Function, s: Stream, x: u64, b: u64, out: u64, rows: usize, cols: usize, repeat: i32) !void {
    const count = rows * cols;
    if (count == 0) return;
    var a: launch.Args = .{};
    a.add(x);
    a.add(b);
    a.add(out);
    a.add(@as(i32, @intCast(rows)));
    a.add(@as(i32, @intCast(cols)));
    a.add(repeat);
    return launch.launch(f, .{ .grid = .{ .x = @intCast((count + threads - 1) / threads), .y = 1, .z = 1 }, .block = .{ .x = threads } }, s, &a);
}

/// Launches `f` for unified RMSNorm, 4-stream signed-sqrt sigmoid gate, stream scales, mask, and residual add.
pub fn launchGate(f: cuda.Function, s: Stream, gw: GateWeights, streams_in: u64, proj: u64, mask: u64, engram_out: u64, streams_out: u64, rows: usize, hidden: usize, repeat: i32) !void {
    if (rows == 0 or hidden == 0) return;
    var a: launch.Args = .{};
    a.add(streams_in);
    a.add(proj);
    a.add(gw.norm_keys);
    a.add(gw.norm_queries);
    a.add(gw.pre_gamma);
    a.add(gw.stream_scales);
    a.add(mask);
    a.add(gw.eps);
    a.add(engram_out);
    a.add(streams_out);
    a.add(@as(i32, @intCast(hidden)));
    a.add(repeat);
    return launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = threads } }, s, &a);
}

/// Production Hy4-flash FP8 Engram CUDA module and kernel handles.
pub const Ops = struct {
    mod: cuda.Module,
    embed_fn: cuda.Function,
    bias_fn: cuda.Function,
    gate_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Ops {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.hy4_engram_fp8);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .embed_fn = try mod.function("hy4_engram_fp8_embed"),
            .bias_fn = try mod.function("hy4_engram_fp8_bias"),
            .gate_fn = try mod.function("hy4_engram_fp8_gate"),
        };
    }

    pub fn unload(o: *Ops) void {
        o.mod.unload();
        o.* = undefined;
    }

    pub fn embed(o: *const Ops, s: Stream, raw: u64, head_norms: u64, eps: f32, out: u64, rows: usize, heads: usize, dim: usize) !void {
        return launchEmbed(o.embed_fn, s, raw, head_norms, eps, out, rows, heads, dim, 0);
    }

    pub fn bias(o: *const Ops, s: Stream, x: u64, b: u64, out: u64, rows: usize, cols: usize) !void {
        return launchBias(o.bias_fn, s, x, b, out, rows, cols, 0);
    }

    pub fn gate(o: *const Ops, s: Stream, gw: GateWeights, streams_in: u64, proj: u64, mask: u64, engram_out: u64, streams_out: u64, rows: usize, hidden: usize) !void {
        return launchGate(o.gate_fn, s, gw, streams_in, proj, mask, engram_out, streams_out, rows, hidden, 0);
    }
};

test "engram_fp8_cuda launchers compile on host without embedded fatbins" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Ops);
    std.testing.refAllDecls(GateWeights);
}
