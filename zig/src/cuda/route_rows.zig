//! route_rows.cu for every CUDA family: router logits on mma tiles and softmax top-k rows, a row's bits its own.
const std = @import("std");
const kernels = @import("kernels.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");

const Driver = @import("driver.zig").Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

/// Logits a top-k row reads at most (the kernel's NT * MOST) and picks it keeps (its scratch).
pub const most_experts = 1024;
pub const most_picks = 128;

/// tf_route_logits' staged rows (16 inputs and 32 experts, padded by 8): two stages of 128, or three of 256.
const many_shared: u32 = 2 * 48 * (128 + 8) * 2;
const few_shared: u32 = 3 * 48 * (256 + 8) * 2;
/// tf_route_logits_wide's: 64 inputs and 32 experts, two stages of 128.
const wide_shared: u32 = 2 * 96 * (128 + 8) * 2;

pub const RouteRows = struct {
    mod: Module,
    logits_fn: Function,
    few_fn: Function,
    wide_fn: Function,
    topk_fn: Function,

    pub fn load(d: *const Driver) !RouteRows {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.route_rows);
        errdefer mod.unload();
        const r: RouteRows = .{ .mod = mod, .logits_fn = try mod.function("tf_route_logits"), .few_fn = try mod.function("tf_route_logits_few"), .wide_fn = try mod.function("tf_route_logits_wide"), .topk_fn = try mod.function("tf_route_topk") };
        try r.few_fn.allowDynamicShared(few_shared);
        try r.wide_fn.allowDynamicShared(wide_shared);
        return r;
    }

    pub fn unload(g: *RouteRows) void {
        g.mod.unload();
    }

    /// out [rows, e] fp32 = x [rows, d] (bf16, x_ld apart) times w [e, d] bf16 transposed; d % 16 == 0, 16-byte rows.
    pub fn logits(g: *const RouteRows, s: Stream, x: u64, x_ld: usize, w: u64, out: u64, rows: usize, d: usize, e: usize) !void {
        if (d % 16 != 0 or x_ld % 8 != 0 or x % 16 != 0 or w % 16 != 0) return error.RowShape;
        if (rows == 0 or e == 0) return;
        var a: launch_.Args = .{};
        a.add(x);
        a.add(@as(i32, @intCast(x_ld)));
        a.add(w);
        a.add(out);
        for ([_]usize{ rows, d, e }) |v| a.add(@as(i32, @intCast(v)));
        // up to 16 rows, deeper staging (the same sums: the router rows' reads set the time)
        const few = rows <= 16;
        if (rows >= 64) {
            // 64 rows a block: each router row read once per 64 input rows (the same sums)
            try launch_.launch(g.wide_fn, .{ .grid = .{ .x = @intCast((rows + 63) / 64), .y = @intCast((e + 31) / 32), .z = 1 }, .block = .{ .x = 128 }, .shared = wide_shared }, s, &a);
            return;
        }
        try launch_.launch(if (few) g.few_fn else g.logits_fn, .{ .grid = .{ .x = @intCast((rows + 15) / 16), .y = @intCast((e + 31) / 32), .z = 1 }, .block = .{ .x = 128 }, .shared = if (few) few_shared else many_shared }, s, &a);
    }

    /// Each row's top_k of its first `experts` logits into picks and weights [rows, slots]; `shared`: slot top_k too.
    pub fn topk(g: *const RouteRows, s: Stream, l: u64, ld: usize, experts: usize, top_k: usize, picks: u64, wts: u64, slots: usize, rows: usize, shared: bool) !void {
        const need = experts + @intFromBool(shared);
        if (experts > most_experts or top_k == 0 or top_k > most_picks or top_k > experts or slots < top_k + @intFromBool(shared) or ld < need) return error.RowShape;
        if (rows == 0) return;
        var a: launch_.Args = .{};
        a.add(l);
        for ([_]usize{ ld, experts, top_k }) |v| a.add(@as(i32, @intCast(v)));
        a.add(picks);
        a.add(wts);
        a.add(@as(i32, @intCast(slots)));
        a.add(@as(i32, @intFromBool(shared)));
        try launch_.launch(g.topk_fn, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = 128 } }, s, &a);
    }
};

test "the top-k scratch holds a 512-expert router's ten picks" {
    try std.testing.expect(most_experts >= 513 and most_picks >= 11);
}
