//! Whole routed experts per rank and a shared expert with optional exact output-column partitioning.
const std = @import("std");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const routed = @import("cuda_modelopt_experts.zig");
const projections = @import("cuda_projection_load.zig");

pub const Owned = struct {
    experts: routed.Owned,
    tfq3: ?cuda.tfq3.OwnedMoeLayer = null,
    tfq3_tables: ?cuda.DeviceBuffer = null,
    tfq4_tables: ?cuda.tfq4.UploadedTables = null,
    shared: [3]wt.Q4,
    rank: usize,
    world: usize,
    split_shared: bool = false,
    /// The shared expert's intermediate width: its gate_proj's rows (cuda_widths.zig's supported shared widths).
    shared_width: usize = wt.moe_w,

    pub fn free(self: *Owned) void {
        if (self.tfq3) |*t| t.deinit();
        if (self.tfq4_tables) |*b| b.free();
        if (self.tfq3_tables) |*b| b.free();
        self.experts.free();
        for (&self.shared) |*p| p.free();
    }
};

pub fn load(a: std.mem.Allocator, d: *cuda.Driver, store: *wt.Store, prefix: []const u8, world: usize, rank: usize, split: bool) !Owned {
    var e = try routed.load(a, d, store, prefix, world, rank);
    errdefer e.free();
    var shared: [3]wt.Q4 = @splat(wt.Q4.empty(d));
    errdefer for (&shared) |*p| p.free();
    var buf: [256]u8 = undefined;
    const gate = try std.fmt.bufPrint(&buf, "{s}.shared_expert.gate_proj.weight", .{prefix});
    const width = (try store.tensor(gate)).dim(0);
    if (!@import("cuda_widths.zig").sharedSupported(width)) {
        std.log.err("flashnext: {s} has {d} rows; the supported shared expert widths are 640 and 1280", .{ gate, width });
        return error.UnexpectedTensor;
    }
    // The shared branch participates exactly once in the sum across ranks.
    if (rank == 0 or split) {
        for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |name, i| {
            const n = if (i == 2) wt.dims else width;
            const k = if (i == 2) width else wt.dims;
            const key = try std.fmt.bufPrint(&buf, "{s}.shared_expert.{s}", .{ prefix, name });
            shared[i] = if (split) try projections.loadCut(a, d, store, key, n, k, try routed.Cut.init(n, world, rank), .rows) else try projections.load(a, d, store, key, n, k);
            if (shared[i].native == null) return error.UnsupportedQuantization;
        }
    }
    return .{ .experts = e, .shared = shared, .rank = rank, .world = world, .split_shared = split, .shared_width = width };
}
