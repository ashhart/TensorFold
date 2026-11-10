//! Draft once, condition several views on likely tentative prefixes, then verify one merged proposal tree.
const std = @import("std");
const frame = @import("cuda_draft_frame.zig");
const draft = @import("cuda_draft.zig");
const tree = @import("cuda_draft_tree.zig");
const coop = @import("cuda_draft_cooperative.zig");
const dr = @import("lanes").drafter;
const most = @import("cuda_engine.zig").most_window;
const lp_words = 43;

pub fn enabled(raw: ?[]const u8) !bool {
    const value = raw orelse return false;
    if (std.mem.eql(u8, value, "0")) return false;
    if (std.mem.eql(u8, value, "1")) return true;
    return error.BadRefinement;
}

pub const Seed = struct { tokens: [2]u32 = .{ 0, 0 }, mass: f64 = 0 };
pub const Seeds = struct { items: [4]Seed = @splat(.{}), count: usize = 0 };

/// Select distinct two-token prefixes using only the first pass's proposal probabilities.
pub fn choose(base: *const coop.Merged, lanes: usize) !Seeds {
    if ((lanes != 2 and lanes != 4) or base.count > most) return error.BadRefinement;
    var candidates: [most]Seed = undefined;
    var count: usize = 0;
    for (base.tokens[0..base.count], base.parents[0..base.count], base.own[0..base.count], 0..) |token, parent, own, i| {
        if (parent < -1 or parent >= @as(i32, @intCast(i)) or !std.math.isFinite(own)) return error.BadRefinement;
        if (parent < 0 or base.parents[@intCast(parent)] != -1) continue;
        const up: usize = @intCast(parent);
        const seed: Seed = .{ .tokens = .{ base.tokens[up], token }, .mass = @exp(@min(0, base.own[up]) + @min(0, own)) };
        var duplicate = false;
        for (candidates[0..count]) |prior| if (std.mem.eql(u32, &prior.tokens, &seed.tokens)) {
            duplicate = true;
            break;
        };
        if (!duplicate) {
            candidates[count] = seed;
            count += 1;
        }
    }
    std.mem.sort(Seed, candidates[0..count], {}, struct {
        fn less(_: void, a: Seed, b: Seed) bool {
            if (a.mass != b.mass) return a.mass > b.mass;
            if (a.tokens[0] != b.tokens[0]) return a.tokens[0] < b.tokens[0];
            return a.tokens[1] < b.tokens[1];
        }
    }.less);
    var selected: Seeds = .{ .count = @min(lanes, count) };
    @memcpy(selected.items[0..selected.count], candidates[0..selected.count]);
    return selected;
}

/// A seeded view holds its fixed prefix, then scores only later positions with their original sampling positions.
pub fn select(c: tree.Candidates, ctx: *draft.Ctx, budget: usize) !void {
    ctx.held_n = 0;
    ctx.grown = 0;
    if (budget == 0) return;
    const n: usize = ctx.seed_n;
    const rows = c.words.len / (lp_words * 4);
    if (n == 0 or n > ctx.seed_tokens.len or n >= rows or budget < n or budget > most or
        c.words.len % (lp_words * 4) != 0 or c.projected.len != rows * c.rank * 2) return error.BadRefinement;
    @memcpy(ctx.held[0..n], ctx.seed_tokens[0..n]);
    for (ctx.parents[0..n], ctx.own[0..n], 0..) |*parent, *own, i| {
        parent.* = @as(i32, @intCast(i)) - 1;
        own.* = 0;
    }
    var tail = c;
    tail.words = c.words[n * lp_words * 4 ..];
    tail.projected = c.projected[n * c.rank * 2 ..];
    tail.position = try std.math.add(u64, c.position, n);
    const count = tree.grow(tail, ctx.seed_tokens[n - 1], rows - n, budget - n, ctx.held[n..], ctx.parents[n..], ctx.own[n..]);
    for (ctx.parents[n..][0..count]) |*parent| {
        parent.* = if (parent.* < 0) @intCast(n - 1) else parent.* + @as(i32, @intCast(n));
    }
    ctx.held_n = n + count;
    ctx.grown = ctx.held_n;
}

pub fn forward(d: anytype, items: []const dr.Hold, contexts: []const *draft.Ctx) !void {
    const n = try coop.frames(items.len, d.cooperative_lanes);
    if (n > d.frame_streams or contexts.len != items.len) return error.BadRefinement;
    for (items, contexts) |it, ctx| if (it.lanes != null or ctx.seed_n != 0) return error.BadRefinement;
    errdefer for (contexts) |ctx| {
        ctx.held_n = 0;
        ctx.grown = 0;
    };
    try frame.forward(d, items, contexts);
    var bases: [frame.max_streams]coop.Merged = undefined;
    var baseline_weights: [frame.max_streams]f64 = undefined;
    var copies: [frame.max_streams]draft.Ctx = undefined;
    var pointers: [frame.max_streams]*draft.Ctx = undefined;
    var holds: [frame.max_streams]dr.Hold = undefined;
    var view_weights: [frame.max_streams]f64 = @splat(0);
    for (items, contexts, 0..) |it, ctx, i| {
        const base = &bases[i];
        base.count = ctx.grown;
        @memcpy(base.tokens[0..base.count], ctx.held[0..base.count]);
        @memcpy(base.parents[0..base.count], ctx.parents[0..base.count]);
        @memcpy(base.own[0..base.count], ctx.own[0..base.count]);
        const selected = try choose(base, d.cooperative_lanes);
        var mass: f64 = 0;
        for (selected.items[0..selected.count]) |seed| mass += seed.mass;
        const usable = mass > 0;
        baseline_weights[i] = if (usable) d.cooperative_base_weight else 1;
        for (0..d.cooperative_lanes) |lane| {
            const at = i * d.cooperative_lanes + lane;
            const active = usable and lane < selected.count;
            const seed = if (usable) selected.items[@min(lane, selected.count - 1)] else Seed{};
            copies[at] = .{
                .ring = ctx.ring,
                .begin = ctx.begin,
                .end = ctx.end,
                .scanned = ctx.scanned,
                .wide = ctx.wide,
                .cooperative_owner = ctx,
                .attention_rows = @intCast(d.c.block),
                .seed_n = if (usable) 2 else 0,
                .seed_tokens = seed.tokens,
            };
            pointers[at] = &copies[at];
            holds[at] = it;
            if (!active) holds[at].depth = 0;
            view_weights[at] = if (active) (1 - d.cooperative_base_weight) * seed.mass / mass else 0;
        }
    }
    try frame.forward(d, holds[0..n], pointers[0..n]);
    for (contexts, 0..) |ctx, i| {
        var views: [5]coop.Lane = undefined;
        for (views[0..d.cooperative_lanes], 0..) |*view, lane| {
            const at = i * d.cooperative_lanes + lane;
            const copy = &copies[at];
            view.* = .{ .tokens = copy.held[0..copy.grown], .parents = copy.parents[0..copy.grown], .own = copy.own[0..copy.grown], .weight = view_weights[at] };
        }
        const base = &bases[i];
        views[d.cooperative_lanes] = .{ .tokens = base.tokens[0..base.count], .parents = base.parents[0..base.count], .own = base.own[0..base.count], .weight = baseline_weights[i] };
        const merged = try coop.merge(views[0 .. d.cooperative_lanes + 1], d.tree_nodes, d.cooperative_base_weight);
        @memcpy(ctx.held[0..merged.count], merged.tokens[0..merged.count]);
        @memcpy(ctx.parents[0..merged.count], merged.parents[0..merged.count]);
        @memcpy(ctx.own[0..merged.count], merged.own[0..merged.count]);
        ctx.held_n = merged.count;
        ctx.grown = merged.count;
    }
}

test "DFlash refinement chooses distinct prefixes by original joint mass and refuses invalid parents" {
    var base: coop.Merged = .{ .count = 5 };
    @memcpy(base.tokens[0..5], &[_]u32{ 1, 4, 2, 5, 3 });
    @memcpy(base.parents[0..5], &[_]i32{ -1, -1, 0, 1, 0 });
    for (base.own[0..5], [_]f64{ 0.7, 0.3, 0.8, 0.6, 0.2 }) |*own, p| own.* = @log(p);
    const chosen = try choose(&base, 2);
    try std.testing.expectEqual(@as(usize, 2), chosen.count);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, &chosen.items[0].tokens);
    try std.testing.expectEqualSlices(u32, &.{ 4, 5 }, &chosen.items[1].tokens);
    try std.testing.expectApproxEqAbs(@as(f64, 0.56), chosen.items[0].mass, 1e-12);
    base.parents[1] = 1;
    try std.testing.expectError(error.BadRefinement, choose(&base, 4));
    try std.testing.expect(!try enabled(null));
    try std.testing.expect(try enabled("1"));
    try std.testing.expectError(error.BadRefinement, enabled("true"));
}

test "DFlash refinement skips seeded candidate rows and grafts later nodes under the fixed prefix" {
    var words: [7 * lp_words * 4]u8 = @splat(0);
    const w = std.mem.bytesAsSlice(u32, &words);
    for (0..7) |row| for (0..2) |j| {
        w[row * lp_words + 2 + j] = @intCast(if (row < 2) 40 + row * 2 + j else 10 + row * 2 + j);
        w[row * lp_words + 22 + j] = @bitCast(@as(f32, if (j == 0) 3 else 1));
    };
    const zeros: [64]u16 = @splat(0);
    const c: tree.Candidates = .{ .words = &words, .projected = std.mem.sliceAsBytes(zeros[0..7]), .topk = 2, .rank = 1, .predecessor = &zeros, .successor = &zeros };
    var ctx: draft.Ctx = .{ .ring = .{ .d = undefined, .ptr = 0, .len = 0 }, .seed_n = 2, .seed_tokens = .{ 1, 2 } };
    try select(c, &ctx, 9);
    try std.testing.expectEqual(@as(usize, 9), ctx.grown);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, ctx.held[0..2]);
    try std.testing.expectEqualSlices(i32, &.{ -1, 0 }, ctx.parents[0..2]);
    for (ctx.held[2..ctx.grown], ctx.parents[2..ctx.grown], 2..) |token, parent, i| {
        try std.testing.expect(token >= 14 and token < 40);
        try std.testing.expect(parent >= 1 and parent < @as(i32, @intCast(i)));
    }
    try select(c, &ctx, 0);
    try std.testing.expectEqual(@as(usize, 0), ctx.grown);
}
