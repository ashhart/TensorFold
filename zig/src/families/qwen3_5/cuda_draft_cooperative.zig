//! Several mask-span views of one true prefix share a draft pass, then merge into one verified proposal tree.
const std = @import("std");
const frame = @import("cuda_draft_frame.zig");
const draft = @import("cuda_draft.zig");
const dr = @import("lanes").drafter;
const most = @import("cuda_engine.zig").most_window;
const max_nodes = 5 * most;

pub fn configure(d: anytype) !void {
    d.cooperative_lanes = try setting(try @import("sys").environment.given("TF_DRAFT_LANES"));
    d.refine = try @import("cuda_draft_refine.zig").enabled(try @import("sys").environment.given("TF_DRAFT_REFINE"));
    try @import("cuda_draft_feedback.zig").configure(d);
    if (d.cooperative_lanes > 1 and (d.c.block != 8 or d.tree_nodes == 0 or
        !@import("cuda_geometry.zig").dense27(d.e.g) or d.e.head != .a4)) return error.CooperativeNeedsBlock8Tree;
    if (d.refine and d.cooperative_lanes == 1) return error.RefinementNeedsLanes;
    if (try @import("sys").environment.given("TF_DRAFT_LANE_BASE_WEIGHT")) |raw| {
        d.cooperative_base_weight = try std.fmt.parseFloat(f64, raw);
        if (!std.math.isFinite(d.cooperative_base_weight) or d.cooperative_base_weight < 0 or d.cooperative_base_weight > 1)
            return error.BadCooperativeLanes;
    }
    d.frame_streams = d.cooperative_lanes;
    if (d.refine) std.log.info("DFlash2 seeded refinement: {d} lanes", .{d.cooperative_lanes});
}

pub fn setting(value: ?[]const u8) !u8 {
    const raw = value orelse return 1;
    const n = try std.fmt.parseInt(u8, raw, 10);
    if (n != 1 and n != 2 and n != 4) return error.BadCooperativeLanes;
    return n;
}

pub fn frames(streams: usize, lanes: u8) !usize {
    if (lanes != 1 and lanes != 2 and lanes != 4) return error.BadCooperativeLanes;
    if (streams == 0 or streams > frame.max_streams / lanes) return error.BadCooperativeLanes;
    return streams * lanes;
}

pub const Lane = struct { tokens: []const u32, parents: []const i32, own: []const f64, weight: ?f64 = null };
pub const Merged = struct {
    tokens: [most]u32 = undefined,
    parents: [most]i32 = undefined,
    own: [most]f64 = undefined,
    count: usize = 0,
};
const Node = struct { token: u32, parent: i32, mass: f64 = 0, depth: u32 };

/// Weighted prefix mass keeps parents ahead of children, including when several views propose the same prefix.
pub fn merge(lanes: []const Lane, budget: usize, base_weight: f64) !Merged {
    if (lanes.len < 2 or lanes.len > 5 or budget >= most) return error.BadCooperativeTree;
    if (!std.math.isFinite(base_weight) or base_weight < 0 or base_weight > 1) return error.BadCooperativeTree;
    var total_weight: f64 = 0;
    for (lanes) |lane| {
        if ((lane.weight != null) != (lanes[0].weight != null)) return error.BadCooperativeTree;
        if (lane.weight) |weight| {
            if (!std.math.isFinite(weight) or weight < 0 or weight > 1) return error.BadCooperativeTree;
            total_weight += weight;
        }
    }
    if (lanes[0].weight != null and @abs(total_weight - 1) > 1e-9) return error.BadCooperativeTree;
    var nodes: [max_nodes]Node = undefined;
    var count: usize = 0;
    for (lanes, 0..) |lane, lane_index| {
        if (lane.tokens.len > most or lane.tokens.len != lane.parents.len or lane.tokens.len != lane.own.len)
            return error.BadCooperativeTree;
        var mapping: [most]usize = undefined;
        var mass: [most]f64 = undefined;
        const weight = lane.weight orelse if (lane_index + 1 == lanes.len) base_weight else (1 - base_weight) / @as(f64, @floatFromInt(lanes.len - 1));
        for (lane.tokens, lane.parents, lane.own, 0..) |token, parent, own, i| {
            if (parent < -1 or parent >= @as(i32, @intCast(i)) or !std.math.isFinite(own)) return error.BadCooperativeTree;
            const up: i32 = if (parent < 0) -1 else @intCast(mapping[@intCast(parent)]);
            mass[i] = (if (parent < 0) @as(f64, 1) else mass[@intCast(parent)]) * @exp(@min(0, own));
            var found: ?usize = null;
            for (nodes[0..count], 0..) |n, j| if (n.parent == up and n.token == token) {
                found = j;
                break;
            };
            const at = found orelse blk: {
                if (count == nodes.len) return error.BadCooperativeTree;
                nodes[count] = .{ .token = token, .parent = up, .depth = if (up < 0) 0 else nodes[@intCast(up)].depth + 1 };
                count += 1;
                break :blk count - 1;
            };
            mapping[i] = at;
            nodes[at].mass += weight * mass[i];
        }
    }
    var order: [max_nodes]usize = undefined;
    for (order[0..count], 0..) |*v, i| v.* = i;
    std.mem.sort(usize, order[0..count], nodes[0..count], struct {
        fn less(ns: []const Node, a: usize, b: usize) bool {
            if (ns[a].mass != ns[b].mass) return ns[a].mass > ns[b].mass;
            if (ns[a].depth != ns[b].depth) return ns[a].depth < ns[b].depth;
            return a < b;
        }
    }.less);
    var out: Merged = .{ .count = @min(budget, count) };
    var selected: [max_nodes]i32 = @splat(-1);
    for (order[0..out.count], 0..) |at, i| {
        const n = nodes[at];
        if (n.parent >= 0 and selected[@intCast(n.parent)] < 0) return error.BadCooperativeTree;
        selected[at] = @intCast(i);
        out.tokens[i] = n.token;
        out.parents[i] = if (n.parent < 0) -1 else selected[@intCast(n.parent)];
        const parent_mass: f64 = if (n.parent < 0) 1 else nodes[@intCast(n.parent)].mass;
        out.own[i] = @log(@max(1e-100, @min(1, if (parent_mass > 0) n.mass / parent_mass else 0)));
    }
    return out;
}

pub fn forward(d: anytype, items: []const dr.Hold, contexts: []const *draft.Ctx) !void {
    if (d.cooperative_lanes == 1) return frame.forward(d, items, contexts);
    errdefer for (contexts) |ctx| {
        ctx.held_n = 0;
        ctx.grown = 0;
    };
    if (d.refine) return @import("cuda_draft_refine.zig").forward(d, items, contexts);
    const n = try frames(items.len, d.cooperative_lanes);
    if (n > d.frame_streams or contexts.len != items.len) return error.BadCooperativeLanes;
    var copies: [frame.max_streams]draft.Ctx = undefined;
    var pointers: [frame.max_streams]*draft.Ctx = undefined;
    var holds: [frame.max_streams]dr.Hold = undefined;
    for (items, contexts, 0..) |it, ctx, i| {
        if (it.lanes != null) return error.CooperativeNeedsGrownTree;
        for (0..d.cooperative_lanes) |lane| {
            const at = i * d.cooperative_lanes + lane;
            copies[at] = .{
                .ring = ctx.ring,
                .begin = ctx.begin,
                .end = ctx.end,
                .scanned = ctx.scanned,
                .wide = ctx.wide,
                .cooperative_owner = ctx,
                .attention_rows = if (d.cooperative_lanes == 2) (if (lane == 0) 4 else 8) else @intCast(2 * (lane + 1)),
            };
            pointers[at] = &copies[at];
            holds[at] = it;
        }
    }
    try frame.forward(d, holds[0..n], pointers[0..n]);
    for (contexts, 0..) |ctx, i| {
        var views: [4]Lane = undefined;
        for (views[0..d.cooperative_lanes], 0..) |*view, lane| {
            const c = &copies[i * d.cooperative_lanes + lane];
            view.* = .{ .tokens = c.held[0..c.grown], .parents = c.parents[0..c.grown], .own = c.own[0..c.grown] };
        }
        const merged = try merge(views[0..d.cooperative_lanes], d.tree_nodes, d.cooperative_base_weight);
        @memcpy(ctx.held[0..merged.count], merged.tokens[0..merged.count]);
        @memcpy(ctx.parents[0..merged.count], merged.parents[0..merged.count]);
        @memcpy(ctx.own[0..merged.count], merged.own[0..merged.count]);
        ctx.held_n = merged.count;
        ctx.grown = merged.count;
    }
}

test "cooperative trees deduplicate shared prefixes and price alternatives without losing their parents" {
    const a = Lane{ .tokens = &.{ 1, 2 }, .parents = &.{ -1, 0 }, .own = &.{ @log(@as(f64, 0.8)), @log(@as(f64, 0.5)) } };
    const b = Lane{ .tokens = &.{ 1, 3 }, .parents = &.{ -1, 0 }, .own = &.{ @log(@as(f64, 0.6)), @log(@as(f64, 0.9)) } };
    const out = try merge(&.{ a, b }, 3, 0.5);
    try std.testing.expectEqual(@as(usize, 3), out.count);
    try std.testing.expectEqualSlices(u32, &.{ 1, 3, 2 }, out.tokens[0..out.count]);
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 0 }, out.parents[0..out.count]);
    try std.testing.expectApproxEqAbs(@as(f64, 0.7), @exp(out.own[0]), 1e-12);
    const short = try merge(&.{ a, b }, 2, 0.5);
    try std.testing.expectEqualSlices(i32, &.{ -1, 0 }, short.parents[0..short.count]);
}

test "cooperative trees reject invalid parent references" {
    const bad = Lane{ .tokens = &.{1}, .parents = &.{0}, .own = &.{0} };
    try std.testing.expectError(error.BadCooperativeTree, merge(&.{ bad, bad }, 3, 0.5));
    try std.testing.expectError(error.BadCooperativeLanes, setting("3"));
}

test "cooperative trees accept normalized explicit lane weights and reject mixed or incomplete mixtures" {
    const a = Lane{ .tokens = &.{ 1, 2 }, .parents = &.{ -1, 0 }, .own = &.{ 0, @log(@as(f64, 0.5)) }, .weight = 0.25 };
    const b = Lane{ .tokens = &.{ 1, 3 }, .parents = &.{ -1, 0 }, .own = &.{ 0, @log(@as(f64, 0.5)) }, .weight = 0.25 };
    var c = Lane{ .tokens = &.{ 4, 5 }, .parents = &.{ -1, 0 }, .own = &.{ 0, @log(@as(f64, 0.8)) }, .weight = 0.5 };
    const merged = try merge(&.{ a, b, c }, 3, 0.5);
    try std.testing.expectEqualSlices(u32, &.{ 1, 4, 5 }, merged.tokens[0..merged.count]);
    try std.testing.expectEqualSlices(i32, &.{ -1, -1, 1 }, merged.parents[0..merged.count]);
    try std.testing.expectApproxEqAbs(@as(f64, 0.8), @exp(merged.own[2]), 1e-12);
    c.weight = 0.25;
    try std.testing.expectError(error.BadCooperativeTree, merge(&.{ a, b, c }, 3, 0.5));
    c.weight = null;
    try std.testing.expectError(error.BadCooperativeTree, merge(&.{ a, b, c }, 3, 0.5));
}
