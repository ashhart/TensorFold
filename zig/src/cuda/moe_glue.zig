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
/// The score form's slices a row at most (moe_glue.cu MAXSK).
pub const most_slices = 8;

/// Sigmoid routing: top `topk` by logit (score: sigmoid of `slices` partials summed) + bias, weights scaled.
pub const Route = struct { experts: usize, topk: usize, shared: bool, normalize: bool = false, scale: f32 = 1, score: bool = false, shared_slots: usize = 1, slices: usize = 1, groups: usize = 1, keep: usize = 1 }; // groups > 1: the score form ranks only the `keep` groups of best top-2 sums

/// Groups a group-limited route holds at most (a warp's lanes).
pub const most_groups = 32;

pub const MoeGlue = struct {
    mod: Module,
    topk_fn: Function,
    topk_warp_fn: Function,
    topk_score_fn: Function,
    topk_score128_fn: Function, // the score form up to 128 experts
    topk_groups_fn: Function, // the score form, group-limited
    combine_fn: Function,

    pub fn load(d: *const Driver) !MoeGlue {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.moe_glue);
        errdefer mod.unload();
        return .{ .mod = mod, .topk_fn = try mod.function("tf_moe_topk_sigmoid"), .topk_warp_fn = try mod.function("tf_moe_topk_sigmoid_warp"), .topk_score_fn = try mod.function("tf_moe_topk_sigmoid_score_warp"), .topk_score128_fn = try mod.function("tf_moe_topk_sigmoid_score_warp_128"), .topk_groups_fn = try mod.function("tf_moe_topk_sigmoid_score_groups"), .combine_fn = try mod.function("tf_moe_combine") };
    }

    pub fn unload(g: *MoeGlue) void {
        g.mod.unload();
    }

    /// fp32 logits [rows, experts] (+ fp32 bias) into int32 picks and fp32 weights [rows, topk (+1 with shared)].
    pub fn topk(g: *const MoeGlue, s: Stream, logits: u64, bias: ?u64, r: Route, pick: u64, wts: u64, rows: usize) !void {
        if (r.experts > most_experts or r.topk == 0 or r.topk > most_topk or r.topk > r.experts) return error.RouteOutOfRange;
        try checkGroups(r);
        if (rows == 0) return;
        const warp = r.experts <= 32 * per_lane and r.topk <= 32;
        if (r.score and (!warp or r.shared_slots > 32)) return error.RouteOutOfRange;
        if (!r.score and (r.shared_slots != 1 or r.slices != 1)) return error.RouteOutOfRange;
        if (r.slices == 0 or r.slices > most_slices) return error.RouteOutOfRange;
        var a: launch_.Args = .{};
        a.add(logits);
        a.add(bias orelse 0);
        a.add(pick);
        a.add(wts);
        if (warp) a.add(@as(i32, @intCast(rows)));
        a.add(@as(i32, @intCast(r.experts)));
        a.add(@as(i32, @intCast(r.topk)));
        const shared: i32 = if (!r.shared) 0 else if (r.score) @intCast(r.shared_slots) else 1;
        a.add(shared);
        a.add(@as(i32, @intFromBool(r.normalize)));
        a.add(r.scale);
        if (r.score) a.add(@as(i32, @intCast(r.slices)));
        if (r.groups > 1) {
            a.add(@as(i32, @intCast(r.groups)));
            a.add(@as(i32, @intCast(r.keep)));
            return launch_.launch(g.topk_groups_fn, .{ .grid = .{ .x = @intCast((rows + 7) / 8), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &a);
        }
        if (warp) return launch_.launch(if (!r.score) g.topk_warp_fn else if (r.experts <= 128) g.topk_score128_fn else g.topk_score_fn, .{ .grid = .{ .x = @intCast((rows + 7) / 8), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &a);
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

/// A group-limited route: the score form's warp, equal groups of at least two, and room in the kept groups for top-k.
fn checkGroups(r: Route) !void {
    if (r.groups == 1 and r.keep == 1) return;
    if (!r.score or r.groups < 2 or r.groups > most_groups or r.experts > 32 * per_lane or r.experts % r.groups != 0) return error.RouteOutOfRange;
    if (r.experts / r.groups < 2 or r.keep == 0 or r.keep > r.groups or r.topk > r.keep * (r.experts / r.groups)) return error.RouteOutOfRange;
}

test "moe glue: group-limited routes outside the grouped kernel's form are refused" {
    const ling: Route = .{ .experts = 256, .topk = 8, .shared = true, .normalize = true, .scale = 2.5, .score = true, .groups = 8, .keep = 4 };
    try checkGroups(ling);
    try checkGroups(.{ .experts = 256, .topk = 8, .shared = false });
    var r = ling;
    r.score = false;
    try std.testing.expectError(error.RouteOutOfRange, checkGroups(r));
    for ([_][2]usize{ .{ 3, 2 }, .{ 64, 1 }, .{ 256, 2 }, .{ 8, 0 }, .{ 8, 9 }, .{ 128, 1 } }) |gk| {
        r = ling;
        r.groups = gk[0];
        r.keep = gk[1];
        try std.testing.expectError(error.RouteOutOfRange, checkGroups(r));
    }
    r = ling;
    r.experts = 1024;
    try std.testing.expectError(error.RouteOutOfRange, checkGroups(r));
    r = ling;
    r.keep = 1;
    r.topk = 33;
    try std.testing.expectError(error.RouteOutOfRange, checkGroups(r));
}

test "moe glue: more than 32 groups is refused on its own (the kept-group mask is 32 bits)" {
    var r: Route = .{ .experts = 512, .topk = 8, .shared = true, .score = true, .groups = 32, .keep = 4 };
    try checkGroups(r); // 32 groups of 16: the most the kernel takes
    r.groups = 64; // groups of 8, 4 kept: room for top 8, so only the group count refuses it
    try std.testing.expectError(error.RouteOutOfRange, checkGroups(r));
}

test "moe glue: groups of one expert are refused on their own (a group scores its two best)" {
    var r: Route = .{ .experts = 32, .topk = 8, .shared = false, .score = true, .groups = 16, .keep = 8 };
    try checkGroups(r); // groups of two
    r.groups = 32; // groups of one, 8 kept: room for top 8, so only the group size refuses it
    try std.testing.expectError(error.RouteOutOfRange, checkGroups(r));
}

test "moe glue: topk refuses a group-limited route before any launch, and admits Ling's" {
    const mg: MoeGlue = undefined; // rows 0: an admitted route returns before touching the module or the stream
    const ling: Route = .{ .experts = 512, .topk = 8, .shared = true, .normalize = true, .scale = 2.5, .score = true, .groups = 8, .keep = 4 };
    try mg.topk(undefined, 0, null, ling, 0, 0, 0);
    var r = ling;
    r.groups = 64;
    try std.testing.expectError(error.RouteOutOfRange, mg.topk(undefined, 0, null, r, 0, 0, 0));
    r = ling;
    r.score = false;
    try std.testing.expectError(error.RouteOutOfRange, mg.topk(undefined, 0, null, r, 0, 0, 0));
}
