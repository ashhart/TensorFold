//! Softmax top-k routing with a sigmoid-gated shared expert (moe.cu), by the rules of tensorfold/cuda/moe.py.

const std = @import("std");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

/// Routed experts a row may pick from (a lane holds eight), and the picks a row may hold (a lane a slot).
pub const max_experts = 256;
pub const max_top_k = 31;

pub const Fns = struct {
    router_fn: Function,
    router_expert: ?Function = null,
    topk_fn: Function,
    combine_fn: Function,
    router_tiles_fn: Function,

    /// From the loaded moe module (kernels.moe).
    pub fn resolve(mod: Module) !Fns {
        const value = std.c.getenv("TF_CUDA_MOE_EXPERT_FIRST");
        return resolveMode(mod, expertFirst(if (value) |v| std.mem.span(v) else null));
    }

    pub fn expertFirst(value: ?[]const u8) bool {
        return value == null or !std.mem.eql(u8, value.?, "0");
    }

    pub fn resolveMode(mod: Module, on: bool) !Fns {
        return .{ .router_fn = try mod.function("tf_moe_router"), .router_expert = if (on) try mod.function("tf_moe_router_expert_first") else null, .topk_fn = try mod.function("tf_moe_topk"), .combine_fn = try mod.function("tf_moe_combine"), .router_tiles_fn = try mod.function("tf_moe_router_tiles") };
    }

    /// logits[r][e] = fp32 x[r] . w[e], the gate last; `params` word 2 caps rows; wide calls on the same-bits tiles.
    pub fn route(f: Fns, s: Stream, x: u64, x_stride: usize, rows: usize, w: u64, logits: u64, dims: usize, count: usize, params: u64) !void {
        if (f.router_expert == null and tiled(x, x_stride, w, rows, dims)) return f.routeTiles(s, x, rows, w, logits, dims, count, params);
        var a: launch_.Args = .{};
        a.add(x);
        a.add(int(x_stride));
        a.add(w);
        a.add(logits);
        for ([_]usize{ rows, dims, count }) |v| a.add(int(v));
        a.add(params);
        const rt: u32 = @intCast((rows + 15) / 16);
        const et: u32 = @intCast((count + 15) / 16);
        const cfg: launch_.Config = .{ .grid = if (f.router_expert != null) .{ .x = et, .y = rt } else .{ .x = rt, .y = et }, .block = .{ .x = 256 } };
        try launch_.launch(f.router_expert orelse f.router_fn, cfg, s, &a);
    }

    /// Each row's top-k picks and bf16-rounded weights ([rows][top_k + 1]); slot top_k is the shared expert.
    pub fn pick(f: Fns, s: Stream, logits: u64, picks: u64, wts: u64, rows: usize, experts: usize, top_k: usize, params: u64) !void {
        if (experts > max_experts or top_k > max_top_k) return error.RoutingTooWide;
        var a: launch_.Args = .{};
        a.add(logits);
        a.add(picks);
        a.add(wts);
        a.add(int(experts));
        a.add(int(top_k));
        a.add(params);
        try launch_.launch(f.topk_fn, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = 32 } }, s, &a);
    }

    /// out[r] = bf16(sum of w[r][s] * y[r][s]) over the slots in order; y fp32 (decode rows) or bf16 (prompt rows).
    pub fn combine(f: Fns, s: Stream, y: u64, y_f32: bool, wts: u64, out: u64, rows: usize, slots: usize, dims: usize, params: u64) !void {
        var a: launch_.Args = .{};
        a.add(y);
        a.add(@as(c_int, @intFromBool(y_f32)));
        a.add(wts);
        a.add(out);
        a.add(int(slots));
        a.add(int(dims));
        a.add(params);
        const blocks = @min((dims + 255) / 256, 16);
        try launch_.launch(f.combine_fn, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(blocks), .z = 1 }, .block = .{ .x = 256 } }, s, &a);
    }

    /// tf_moe_router_tiles over `rows` (tiled() holds: rows `dims` apart): the same logits, 128 rows by 128 a block.
    fn routeTiles(f: Fns, s: Stream, x: u64, rows: usize, w: u64, logits: u64, dims: usize, count: usize, params: u64) !void {
        var a: launch_.Args = .{};
        a.add(x);
        a.add(int(dims));
        a.add(w);
        a.add(logits);
        for ([_]usize{ rows, dims, count }) |v| a.add(int(v));
        a.add(params);
        const cfg: launch_.Config = .{ .grid = .{ .x = @intCast((rows + 127) / 128), .y = @intCast((count + 127) / 128), .z = 1 }, .block = .{ .x = 256 }, .shared = router_tiles_shared };
        try launch_.launch(f.router_tiles_fn, cfg, s, &a);
    }
};

fn int(x: usize) c_int {
    return @intCast(x);
}

/// Shared bytes tf_moe_router_tiles stages (common/router_tiles.cuh): two 32-input steps of 128 rows and 128 experts.
pub const router_tiles_shared: u32 = 2 * 2 * 128 * 20 * 4;

/// Rows from which the tiles beat tf_moe_router (tf-cuda-test moe-router-bench sets it).
pub const tiles_from = 257;

/// Whether a route takes the tiles: from tiles_from rows, a row of dims apart, 16-byte aligned, whole 32-input steps.
pub fn tiled(x: u64, x_stride: usize, w: u64, rows: usize, dims: usize) bool {
    return rows >= tiles_from and x_stride == dims and dims % 32 == 0 and (x | w) % 16 == 0;
}

test "router tiles take aligned calls from tiles_from rows only" {
    try std.testing.expect(tiled(0x1000, 2048, 0x2000, 257, 2048));
    try std.testing.expect(!tiled(0x1000, 2048, 0x2000, 256, 2048)); // decode, verify windows and short chunks keep it
    try std.testing.expect(!tiled(0x1008, 2048, 0x2000, 300, 2048));
    try std.testing.expect(!tiled(0x1000, 2064, 0x2000, 300, 2048)); // a padded stride keeps tf_moe_router
    try std.testing.expect(!tiled(0x1000, 2048, 0x2000, 300, 2050));
}
