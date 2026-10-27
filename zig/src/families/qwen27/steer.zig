//! A control vector's directions on the device: the layer each one steers and the strength it takes.
const std = @import("std");
const mtl = @import("metal");

/// How a direction is applied: the mode word the kernel reads.
pub const Mode = enum(u32) { project = 0, add = 1 };

/// One layer's direction: its layer id and the unit vector.
pub const Layer = struct { id: u32, dir: []const f32 };

/// What the plan is built from: the vector's directions and the model they steer.
pub const Source = struct {
    mode: Mode,
    alpha: f32,
    hidden: u32,
    layers_total: u32,
    layers: []const Layer,
};

/// The vector in force: one direction a device buffer, and the lookup that finds a layer's.
pub const Plan = struct {
    allocator: std.mem.Allocator,
    mode: Mode,
    alpha: f32,
    hidden: u32,
    ids: []u32,
    dirs: []mtl.Buffer,
    cover: []?usize,

    /// The direction that steers `layer`; null when the vector does not cover it.
    pub fn dir(p: *const Plan, layer: u32) ?mtl.Buffer {
        if (layer >= p.cover.len) return null;
        const index = p.cover[layer] orelse return null;
        return p.dirs[index];
    }

    /// The layers the vector covers, ascending.
    pub fn covered(p: *const Plan) []const u32 {
        return p.ids;
    }

    pub fn deinit(p: *Plan) void {
        for (p.dirs) |d| d.deinit();
        p.allocator.free(p.dirs);
        p.allocator.free(p.ids);
        p.allocator.free(p.cover);
        p.allocator.destroy(p);
    }
};

/// The plan for a vector: every direction on the device and the lookup that maps a layer to it.
pub fn init(allocator: std.mem.Allocator, device: mtl.Device, src: Source) !*Plan {
    if (src.hidden == 0 or src.layers_total == 0 or src.layers.len == 0) return error.BadSteerShape;
    for (src.layers) |l| {
        // Layer 0 is the embedding, which no block writes; the blocks are one to `layers_total`.
        if (l.id == 0 or l.id > src.layers_total) return error.BadSteerLayer;
        if (l.dir.len != src.hidden) return error.BadSteerWidth;
    }
    const plan = try allocator.create(Plan);
    errdefer allocator.destroy(plan);
    const ids = try allocator.alloc(u32, src.layers.len);
    errdefer allocator.free(ids);
    const dirs = try allocator.alloc(mtl.Buffer, src.layers.len);
    errdefer allocator.free(dirs);
    var live: usize = 0;
    errdefer for (dirs[0..live]) |d| d.deinit();
    const cover = try allocator.alloc(?usize, src.layers_total + 1);
    errdefer allocator.free(cover);
    @memset(cover, null);
    for (src.layers, 0..) |l, i| {
        dirs[i] = try device.buffer(@as(usize, src.hidden) * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        live += 1;
        @memcpy(dirs[i].slice(f32, src.hidden), l.dir);
        ids[i] = l.id;
        cover[l.id] = i;
    }
    plan.* = .{ .allocator = allocator, .mode = src.mode, .alpha = src.alpha, .hidden = src.hidden, .ids = ids, .dirs = dirs, .cover = cover };
    return plan;
}
