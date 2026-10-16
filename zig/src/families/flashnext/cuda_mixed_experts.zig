//! Routed NVFP4 intermediate partitions and a separately owned MXFP8 shared expert.
const std = @import("std");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const routed = @import("cuda_modelopt_experts.zig");
const projections = @import("cuda_projection_load.zig");

pub const Owned = struct {
    experts: routed.Owned,
    shared: [3]wt.Q4,
    rank: usize,
    world: usize,

    pub fn free(self: *Owned) void {
        self.experts.free();
        for (&self.shared) |*p| p.free();
    }
};

pub fn load(a: std.mem.Allocator, d: *cuda.Driver, store: *wt.Store, prefix: []const u8, world: usize, rank: usize) !Owned {
    var e = try routed.load(a, d, store, prefix, world, rank);
    errdefer e.free();
    var shared: [3]wt.Q4 = @splat(wt.Q4.empty(d));
    errdefer for (&shared) |*p| p.free();
    // The shared branch participates exactly once in the sum across ranks.
    if (rank == 0) {
        var buf: [256]u8 = undefined;
        for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |name, i| {
            const n = if (i == 2) wt.dims else wt.moe_w;
            const k = if (i == 2) wt.moe_w else wt.dims;
            shared[i] = try projections.load(a, d, store, try std.fmt.bufPrint(&buf, "{s}.shared_expert.{s}", .{ prefix, name }), n, k);
            if (shared[i].native == null) return error.UnsupportedQuantization;
        }
    }
    return .{ .experts = e, .shared = shared, .rank = rank, .world = world };
}
