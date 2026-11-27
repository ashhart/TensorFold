//! Test-only TP=2 GLM-5.3-Flash NVFP4 prompt shapes; production dispatch stays unchanged.
const std = @import("std");

pub const Kind = enum { dense_up, dense_down, routed_up, routed_down };
pub const routed_experts = 288;
pub const plan_experts = 289;
pub const routed_slots = 8;
pub const slots = 9;
pub const Launch = struct { grid_x: u32, grid_y: u32 = 1, grid_z: u32 = 1, threads: u32, shared: u32 };
pub const Shape = struct {
    name: []const u8,
    kind: Kind,
    rows: usize,
    n: usize,
    k: usize,
    item_bound: usize = 0,
    planned_items: usize = 0,
    layers: usize,
    products: usize,
    pairs_per_row: usize,
    launches_8k: usize = 4,
    resident: bool = false,

    pub fn geometry(s: Shape, sms: usize) !Launch {
        if (sms == 0) return error.NoMultiprocessors;
        return switch (s.kind) {
            .dense_up, .dense_down => .{
                .grid_x = @intCast(@min(sms, ((s.rows + 127) / 128) * ((s.n + 255) / 256))),
                .threads = 256,
                .shared = 76848,
            },
            .routed_up => .{
                .grid_x = @intCast(s.item_bound * ((s.n / 32 + 3) / 4)),
                .threads = 128,
                .shared = if (s.resident) 17408 else 34816,
            },
            .routed_down => .{
                .grid_x = @intCast(s.item_bound * ((s.n / 32 + 7) / 8)),
                .threads = 256,
                .shared = if (s.resident) 17408 else 34816,
            },
        };
    }
};

pub const shapes = [_]Shape{
    .{ .name = "routed-up", .kind = .routed_up, .rows = 2048, .n = 1024, .k = 4096, .item_bound = 577, .planned_items = 320, .layers = 42, .products = 2, .pairs_per_row = 8 },
    .{ .name = "routed-down", .kind = .routed_down, .rows = 2048, .n = 4096, .k = 1024, .item_bound = 577, .planned_items = 320, .layers = 42, .products = 1, .pairs_per_row = 8 },
    .{ .name = "dense-up", .kind = .dense_up, .rows = 2048, .n = 6144, .k = 4096, .layers = 3, .products = 2, .pairs_per_row = 1 },
    .{ .name = "dense-down", .kind = .dense_down, .rows = 768, .n = 4096, .k = 6144, .layers = 3, .products = 1, .pairs_per_row = 1, .launches_8k = 8 },
    .{ .name = "dense-down-tail", .kind = .dense_down, .rows = 512, .n = 4096, .k = 6144, .layers = 3, .products = 1, .pairs_per_row = 1 },
};

pub fn find(name: []const u8) ?Shape {
    for (shapes) |s| if (std.mem.eql(u8, name, s.name)) return s;
    if (std.mem.endsWith(u8, name, "-r3")) {
        var s = find(name[0 .. name.len - 3]) orelse return null;
        if (s.kind != .routed_up and s.kind != .routed_down) return null;
        s.name = name;
        s.resident = true;
        return s;
    }
    return null;
}

pub fn byName(name: []const u8) Shape {
    return find(name) orelse unreachable;
}

/// Logical MACs on one TP=2 rank for an 8K prompt; the routed shared slot is excluded.
pub fn promptMacs8K(s: *const Shape) u128 {
    return @as(u128, s.rows) * s.launches_8k * s.pairs_per_row * s.products * s.n * s.k * s.layers;
}
