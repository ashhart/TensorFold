//! Qwen27 grammar admission, host sampling and proposal pruning share the lane constraint.
const std = @import("std");
pub const supported = true;

pub fn draw(drawFn: anytype, a: std.mem.Allocator, logits: []const u16, sampling: anytype, position: u64, mask: []const u32) !u32 {
    if (mask.len != 0 and mask.len != (logits.len + 31) / 32) return error.InvalidQwenMasks;
    return drawFn(a, logits, sampling, position, mask);
}

pub fn prune(a: std.mem.Allocator, constraint: anytype, opening: ?u32, tree: anytype) !void {
    const c = constraint orelse return;
    const n = tree.tokens.len;
    if (tree.parents.len != n or tree.scores.len != n) return error.BadDraftParents;
    const map = try a.alloc(i32, n);
    defer a.free(map);
    @memset(map, -1);
    const path = try a.alloc(usize, n);
    defer a.free(path);
    var count: usize = 0;
    for (tree.parents, 0..) |parent, node| {
        if (parent < -1 or parent >= node) return error.BadDraftParents;
        if (parent >= 0 and map[@intCast(parent)] < 0) continue;
        var size: usize = 0;
        var at: i32 = @intCast(node);
        while (at >= 0) {
            path[size] = @intCast(at);
            size += 1;
            at = tree.parents[@intCast(at)];
        }
        if (try accepted(a, c, opening, tree.tokens, path[0..size])) {
            map[node] = @intCast(count);
            count += 1;
        }
    }
    if (count == n) return;
    const tokens = try a.alloc(u32, count);
    errdefer a.free(tokens);
    const parents = try a.alloc(i32, count);
    errdefer a.free(parents);
    const scores = try a.alloc(f64, count);
    errdefer a.free(scores);
    for (map, 0..) |mapped, node| if (mapped >= 0) {
        const i: usize = @intCast(mapped);
        tokens[i] = tree.tokens[node];
        parents[i] = if (tree.parents[node] < 0) -1 else map[@intCast(tree.parents[node])];
        scores[i] = tree.scores[node];
    };
    tree.deinit();
    tree.* = .{ .gpa = a, .tokens = tokens, .parents = parents, .scores = scores };
}

fn accepted(a: std.mem.Allocator, c: anytype, opening: ?u32, tokens: []const u32, path: []const usize) !bool {
    const active = c.active;
    defer c.active = active;
    var taken: usize = 0;
    const valid = walk(a, c, opening, tokens, path, &taken) catch |err| {
        try c.rules.vtable.rollback(c.rules.ptr, taken);
        return err;
    };
    try c.rules.vtable.rollback(c.rules.ptr, taken);
    return valid;
}

fn walk(a: std.mem.Allocator, c: anytype, opening: ?u32, tokens: []const u32, path: []const usize, taken: *usize) !bool {
    if (opening) |token| if (!try follow(a, c, token, taken)) return false;
    var i = path.len;
    while (i > 0) {
        i -= 1;
        if (!try follow(a, c, tokens[path[i]], taken)) return false;
    }
    return true;
}

fn follow(a: std.mem.Allocator, c: anytype, token: u32, taken: *usize) !bool {
    if (c.finished()) return false;
    if (try c.next(a)) |mask| {
        defer a.free(mask);
        if (token / 32 >= mask.len or mask[token / 32] & (@as(u32, 1) << @intCast(token % 32)) == 0) return false;
        if (!try c.follow(token)) return false;
        taken.* += 1;
    } else _ = try c.follow(token);
    return true;
}
