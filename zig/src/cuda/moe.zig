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
    topk_fn: Function,
    combine_fn: Function,

    /// From the loaded moe module (kernels.moe).
    pub fn resolve(mod: Module) !Fns {
        return .{ .router_fn = try mod.function("tf_moe_router"), .topk_fn = try mod.function("tf_moe_topk"), .combine_fn = try mod.function("tf_moe_combine") };
    }

    /// logits[r][e] = fp32 x[r] . w[e], `count` rows of `dims` with the gate last; `params` word 2 (if set) caps rows.
    pub fn route(f: Fns, s: Stream, x: u64, x_stride: usize, rows: usize, w: u64, logits: u64, dims: usize, count: usize, params: u64) !void {
        var a: launch_.Args = .{};
        a.add(x);
        a.add(int(x_stride));
        a.add(w);
        a.add(logits);
        for ([_]usize{ rows, dims, count }) |v| a.add(int(v));
        a.add(params);
        const cfg: launch_.Config = .{ .grid = .{ .x = @intCast((rows + 15) / 16), .y = @intCast((count + 15) / 16), .z = 1 }, .block = .{ .x = 256 } };
        try launch_.launch(f.router_fn, cfg, s, &a);
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
};

fn int(x: usize) c_int {
    return @intCast(x);
}
