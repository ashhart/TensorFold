//! moe_glue.cu's routing and combine for every CUDA family: sigmoid top-k with bias, the weighted sum of slots.
const std = @import("std");
const kernels = @import("kernels.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const norms = @import("norms.zig");

const Driver = @import("driver.zig").Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;
const Kind = norms.Kind;

const threads = 256;
/// The most experts a row ranks and the most it keeps (moe_glue.cu MOST, the picked array).
pub const most_experts = 4096;
pub const most_topk = 64;
/// Experts a lane ranks in the warp form (moe_glue.cu PER): up to 512 experts take it.
const per_lane = 16;

/// Sigmoid routing: top `topk` by logit + bias, weights sigmoid(logit) [normalized] * scale; `shared` adds slot topk.
pub const Route = struct { experts: usize, topk: usize, shared: bool, normalize: bool = false, scale: f32 = 1 };

pub const MoeGlue = struct {
    mod: Module,
    topk_fn: Function,
    topk_warp_fn: Function,
    combine_fn: Function,

    pub fn load(d: *const Driver) !MoeGlue {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.moe_glue);
        errdefer mod.unload();
        return .{ .mod = mod, .topk_fn = try mod.function("tf_moe_topk_sigmoid"), .topk_warp_fn = try mod.function("tf_moe_topk_sigmoid_warp"), .combine_fn = try mod.function("tf_moe_combine") };
    }

    pub fn unload(g: *MoeGlue) void {
        g.mod.unload();
    }

    /// fp32 logits [rows, experts] (+ fp32 bias) into int32 picks and fp32 weights [rows, topk (+1 with shared)].
    pub fn topk(g: *const MoeGlue, s: Stream, logits: u64, bias: ?u64, r: Route, pick: u64, wts: u64, rows: usize) !void {
        if (r.experts > most_experts or r.topk == 0 or r.topk > most_topk or r.topk > r.experts) return error.RouteOutOfRange;
        if (rows == 0) return;
        const warp = r.experts <= 32 * per_lane and r.topk <= 32;
        var a: launch_.Args = .{};
        a.add(logits);
        a.add(bias orelse 0);
        a.add(pick);
        a.add(wts);
        if (warp) a.add(@as(i32, @intCast(rows)));
        a.add(@as(i32, @intCast(r.experts)));
        a.add(@as(i32, @intCast(r.topk)));
        a.add(@as(i32, @intFromBool(r.shared)));
        a.add(@as(i32, @intFromBool(r.normalize)));
        a.add(r.scale);
        if (warp) return launch_.launch(g.topk_warp_fn, .{ .grid = .{ .x = @intCast((rows + 7) / 8), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &a);
        try launch_.launch(g.topk_fn, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = threads } }, s, &a);
    }

    /// out [rows, dims] = the sum, in slot order, of w [rows, slots] (fp32) times y [rows, slots, dims].
    pub fn combine(g: *const MoeGlue, s: Stream, y: u64, y_kind: Kind, w: u64, out: u64, out_kind: Kind, rows: usize, slots: usize, dims: usize) !void {
        if (rows == 0) return;
        var a: launch_.Args = .{};
        a.add(y);
        a.add(@as(i32, @backingInt(y_kind)));
        a.add(w);
        a.add(out);
        a.add(@as(i32, @backingInt(out_kind)));
        a.add(@as(i32, @intCast(slots)));
        a.add(@as(i32, @intCast(dims)));
        const cols: u32 = @intCast((dims + threads - 1) / threads);
        try launch_.launch(g.combine_fn, .{ .grid = .{ .x = @intCast(rows), .y = cols, .z = 1 }, .block = .{ .x = threads } }, s, &a);
    }
};
