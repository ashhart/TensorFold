//! Family registrations use the loader's actual layout factory; shared registrations remain untouched.
const std = @import("std");
const cfg = @import("config.zig");
const weights = @import("weights.zig");
pub const fused = @import("fused_lane.zig");
pub const Variant = struct {
    name: []const u8,
    geo: cfg.Geometry,
    kind: u8,
    n: usize,
    k: usize,
    tap: bool = false,
    pub fn layout(v: Variant) fused.Layout {
        return weights.fusedLayout(v.geo, v.n, v.k, v.kind);
    }
    pub fn threadgroupBytes(v: Variant) usize {
        if (v.kind == 0) return 64 * 72 * 2;
        return std.mem.alignForward(usize, v.layout().threadgroupBytes() + 2, 256);
    }
};
fn variants(comptime geo: cfg.Geometry) [6]Variant {
    const sliding = geo.query_heads * cfg.sliding_dim + 2 * geo.sliding_kv_heads * cfg.sliding_dim;
    const global = geo.query_heads * cfg.global_dim + geo.global_kv_heads * cfg.global_dim;
    return .{
        .{ .name = "gemma4 " ++ geo.name ++ " rounded dense", .geo = geo, .kind = 0, .n = geo.hidden, .k = geo.hidden },
        .{ .name = "gemma4 " ++ geo.name ++ " fused1 sliding", .geo = geo, .kind = 1, .n = sliding, .k = geo.hidden },
        .{ .name = "gemma4 " ++ geo.name ++ " fused1 global", .geo = geo, .kind = 1, .n = global, .k = geo.hidden },
        .{ .name = "gemma4 " ++ geo.name ++ " fused1 tap", .geo = geo, .kind = 1, .n = sliding, .k = geo.hidden, .tap = true },
        .{ .name = "gemma4 " ++ geo.name ++ " fused2 geglu", .geo = geo, .kind = 2, .n = 2 * geo.intermediate, .k = geo.hidden },
        .{ .name = "gemma4 " ++ geo.name ++ " fused3 down", .geo = geo, .kind = 3, .n = geo.hidden, .k = geo.intermediate },
    };
}
pub const entries: []const Variant = &(variants(cfg.dense31b) ++ variants(cfg.unified12b));
pub fn requireComplete(list: []const Variant) !void {
    if (list.len != 12) return error.IncompleteGemmaContract;
    var seen: [2][6]bool = @splat(@splat(false));
    for (list) |v| {
        const gi: usize = if (v.geo.hidden == cfg.dense31b.hidden) 0 else if (v.geo.hidden == cfg.unified12b.hidden) 1 else return error.UnknownGeometry;
        const ki: usize = if (v.kind == 0) 0 else if (v.kind == 1) (if (v.tap) @as(usize, 3) else if (v.n == v.geo.query_heads * cfg.sliding_dim + 2 * v.geo.sliding_kv_heads * cfg.sliding_dim) @as(usize, 1) else 2) else if (v.kind == 2) 4 else if (v.kind == 3) 5 else return error.UnknownKind;
        if (seen[gi][ki]) return error.DuplicateGemmaContract;
        seen[gi][ki] = true;
        if (v.kind != 0) try v.layout().validate();
    }
    for (seen) |kinds| for (kinds) |has| if (!has) return error.IncompleteGemmaContract;
}

pub const adapt = @import("../../core/frags.zig").source;
