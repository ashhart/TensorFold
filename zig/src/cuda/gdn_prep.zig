//! gdn_prep.cu's Gated DeltaNet inputs for every CUDA family: each row's conv window, SiLU, q/k norms, g and beta.
const std = @import("std");
const kernels = @import("kernels.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");

const Driver = @import("driver.zig").Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

/// The kernel's buffers (addresses; strides in elements), as gdn_prep.cu's Io.
pub const Io = extern struct {
    qkv: u64, // [rows, qkv_ld] bf16: q (kh dk), k (kh dk), v (vh dk) channels
    qkv_ld: i64,
    state: u64, // [streams * keep, c] bf16: each stream's kept conv rows
    conv: u64, // [c, keep + 1] bf16 taps a channel
    win: u64, // [rows, keep + 1] i32: < keep a kept row of the row's stream, else qkv row (win - keep)
    sid: u64 = 0, // [rows] i32 stream (0: every row stream 0)
    a: u64, // [rows, a_ld] bf16
    a_ld: i64,
    b: u64, // [rows, b_ld] bf16
    b_ld: i64,
    a_log: u64, // [vh] fp32
    dt_bias: u64, // [vh] fp32
    q: u64, // [rows, kh, dk] bf16
    k: u64,
    v: u64, // [rows, vh, dk] bf16
    g: u64, // [rows, vh] fp32
    beta: u64,
};

/// Channels a row (c = 2 kh dk + vh dk), heads and kept conv rows (taps - 1).
pub const Shape = struct { kh: usize, vh: usize, dk: usize, keep: usize, eps: f32 = 1e-6 };

pub const GdnPrep = struct {
    mod: Module,
    fns: [3]Function, // dk 64, 128, 256

    pub fn load(d: *const Driver) !GdnPrep {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.gdn_prep);
        errdefer mod.unload();
        return .{ .mod = mod, .fns = .{ try mod.function("tf_gdn_prep_64"), try mod.function("tf_gdn_prep_128"), try mod.function("tf_gdn_prep_256") } };
    }

    pub fn unload(g: *GdnPrep) void {
        g.mod.unload();
    }

    /// Per row's conv window c = silu(conv): q = l2norm(c) / sqrt(dk), k = l2norm(c), v = c, g and beta.
    pub fn run(g: *const GdnPrep, s: Stream, io: Io, rows: usize, sh: Shape) !void {
        const which: usize = switch (sh.dk) {
            64 => 0,
            128 => 1,
            256 => 2,
            else => return error.GdnShape,
        };
        if (sh.keep + 1 > 8 or sh.keep == 0) return error.GdnShape;
        if (rows == 0) return;
        const heads = 2 * sh.kh + sh.vh;
        var a: launch_.Args = .{};
        a.add(io);
        a.add(@as(i32, @intCast(heads * sh.dk))); // c: a kept row's channels
        a.add(@as(i32, @intCast(sh.kh)));
        a.add(@as(i32, @intCast(sh.vh)));
        a.add(@as(i32, @intCast(sh.keep)));
        a.add(sh.eps);
        const t = launch_.rowThreads(rows, 128, 64); // a warp a head either way
        try launch_.launch(g.fns[which], .{ .grid = .{ .x = @intCast(rows), .y = @intCast((heads + t / 32 - 1) / (t / 32)), .z = 1 }, .block = .{ .x = t } }, s, &a);
    }
};
