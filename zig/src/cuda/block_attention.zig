//! block_attention.cu for every CUDA family: a DFlash-style drafter's block rows over their context and the block.
const std = @import("std");
const kernels = @import("kernels.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");

const Driver = @import("driver.zig").Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

/// A 32-key K and V tile (head dim 128), shared by the block's warps.
const smem: u32 = 2 * 32 * 16 * 16;

/// q [h, rows, 128], kb and vb [hk, rows, 128] (head-major); stream j's block rows j L .. j L + L - 1.
pub const Shape = struct { h: usize, hk: usize, rows: usize, block: usize, window: usize, scale: f32, causal: bool = false };

pub const BlockAttention = struct {
    mod: Module,
    fn_: Function,

    pub fn load(d: *const Driver) !BlockAttention {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.block_attention);
        errdefer mod.unload();
        const f = try mod.function("tf_block_attention");
        try f.allowDynamicShared(smem);
        return .{ .mod = mod, .fn_ = f };
    }

    pub fn unload(g: *BlockAttention) void {
        g.mod.unload();
    }

    /// out [rows, h, 128]; table i64 [streams, 2]: stream j's context K and V ([hk, lens[j], 128] bf16); lens i32.
    pub fn run(g: *const BlockAttention, s: Stream, q: u64, kb: u64, vb: u64, table: u64, lens: u64, out: u64, sh: Shape) !void {
        if (sh.hk == 0 or sh.h % sh.hk != 0 or sh.h / sh.hk > 8 or sh.block == 0 or sh.block > 16 or sh.rows % sh.block != 0) return error.BlockShape;
        if (sh.rows == 0) return;
        var a: launch_.Args = .{};
        for ([_]u64{ q, kb, vb, table, lens, out }) |v| a.add(v);
        a.add(sh.scale);
        for ([_]usize{ sh.window, sh.rows, sh.h, sh.hk, sh.block }) |v| a.add(@as(i32, @intCast(v)));
        a.add(@as(i32, @intFromBool(sh.causal)));
        const grid: launch_.Dim3 = .{ .x = @intCast(sh.rows / sh.block), .y = @intCast(sh.hk), .z = 1 };
        try launch_.launch(g.fn_, .{ .grid = grid, .block = .{ .x = @intCast(32 * (sh.h / sh.hk)) }, .shared = smem }, s, &a);
    }
};
