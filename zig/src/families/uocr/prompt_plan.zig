//! Grouped prompt descriptors, tested without Metal or a GPU.
const std = @import("std");
const sys = @import("sys");

pub fn enabled() !bool {
    return sys.environment.read(bool, "TF_UOCR_GROUPED_PROMPT", false);
}

pub const experts = 64;
pub const hidden = 1280;
pub const expert_ffn = 896;

/// The original GEMM's parameters, plus its expert and its place in the concatenated grid.
pub const Group = extern struct {
    p: [16]i32,
    group_y: u32,
    row0: u32,
    expert: u32,
    projection: u32,
};

pub const Plan = struct {
    groups: usize,
    threadgroups_y: usize,
};

/// Three projection-major descriptor sets; zero-row experts have no descriptor or launch padding.
pub fn make(start: []const usize, out: []Group) Plan {
    std.debug.assert(start.len == experts + 1 and out.len == 3 * experts);
    var result: Plan = .{ .groups = 0, .threadgroups_y = 0 };
    for (0..3) |projection| {
        var used: usize = 0;
        var group_y: usize = 0;
        for (0..experts) |e| {
            const m = start[e + 1] - start[e];
            if (m == 0) continue;
            const n: usize = if (projection == 2) hidden else expert_ffn;
            const k: usize = if (projection == 2) expert_ffn else hidden;
            const tm = (m + 63) / 64;
            var p: [16]i32 = @splat(0);
            const vals = [_]usize{ m, n, k, k, k, n, (n + 127) / 128, tm, 2, k / 256 };
            for (vals, 0..) |v, i| p[i] = @intCast(v);
            out[projection * experts + used] = .{
                .p = p,
                .group_y = @intCast(group_y),
                .row0 = @intCast(start[e]),
                .expert = @intCast(e),
                .projection = @intCast(projection),
            };
            used += 1;
            group_y += (tm + 3) / 4 * 4;
        }
        if (projection == 0) {
            result = .{ .groups = used, .threadgroups_y = group_y };
        } else {
            std.debug.assert(result.groups == used and result.threadgroups_y == group_y);
        }
    }
    return result;
}

test "uocr: grouped prompt plans preserve each expert's tiles and projection geometry" {
    var start: [65]usize = @splat(0);
    var groups: [3 * 64]Group = undefined;

    var plan = make(&start, &groups);
    try std.testing.expectEqual(0, plan.groups);
    try std.testing.expectEqual(0, plan.threadgroups_y);

    for (start[1..]) |*v| v.* = 1;
    plan = make(&start, &groups);
    try std.testing.expectEqual(1, plan.groups);
    try std.testing.expectEqual(4, plan.threadgroups_y);
    try std.testing.expectEqual(1, groups[0].p[0]);
    try std.testing.expectEqual(896, groups[0].p[1]);
    try std.testing.expectEqual(1280, groups[0].p[2]);
    try std.testing.expectEqual(1, groups[0].p[7]);
    try std.testing.expectEqual(0, groups[0].group_y);
    try std.testing.expectEqual(1, groups[64].projection);
    try std.testing.expectEqual(896, groups[128].p[2]);
    try std.testing.expectEqual(1280, groups[128].p[1]);

    for (start[1..]) |*v| v.* = 8;
    start[1] = 3;
    plan = make(&start, &groups);
    try std.testing.expectEqual(2, plan.groups);
    try std.testing.expectEqual(8, plan.threadgroups_y);
    try std.testing.expectEqual(3, groups[1].row0);
    try std.testing.expectEqual(4, groups[1].group_y);
    try std.testing.expectEqual(5, groups[1].p[0]);

    for (start[1..], 0..) |*v, i| v.* = @min(i + 1, 6);
    plan = make(&start, &groups);
    try std.testing.expectEqual(6, plan.groups);
    try std.testing.expectEqual(24, plan.threadgroups_y);
    try std.testing.expectEqual(5, groups[5].row0);
    try std.testing.expectEqual(5, groups[5].expert);
    try std.testing.expectEqual(20, groups[5].group_y);

    for (&start, 0..) |*v, i| v.* = i * 65;
    plan = make(&start, &groups);
    try std.testing.expectEqual(64, plan.groups);
    try std.testing.expectEqual(256, plan.threadgroups_y);
    try std.testing.expectEqual(2, groups[0].p[7]);
    try std.testing.expectEqual(252, groups[63].group_y);

    for (&start, 0..) |*v, i| v.* = i * 64;
    plan = make(&start, &groups);
    try std.testing.expectEqual(64, plan.groups);
    try std.testing.expectEqual(0, @mod(groups[0].p[0], 64));

    for (&start, 0..) |*v, i| v.* = i;
    plan = make(&start, &groups);
    try std.testing.expectEqual(64, plan.groups);
    try std.testing.expectEqual(256, plan.threadgroups_y);
    try std.testing.expectEqual(63, groups[63].expert);
}

test "uocr: grouped prompt switch defaults off and refuses malformed values" {
    const name = "TF_UOCR_GROUPED_PROMPT";
    const before: ?[:0]u8 = if (std.c.getenv(name)) |v| try std.testing.allocator.dupeSentinel(u8, std.mem.span(v), 0) else null;
    defer {
        if (before) |v| {
            _ = sys.env.setenv(name, v.ptr, 1);
            std.testing.allocator.free(v);
        } else _ = sys.env.unsetenv(name);
    }
    _ = sys.env.unsetenv(name);
    try std.testing.expect(!try enabled());
    _ = sys.env.setenv(name, "", 1);
    try std.testing.expect(!try enabled());
    _ = sys.env.setenv(name, "0", 1);
    try std.testing.expect(!try enabled());
    _ = sys.env.setenv(name, "1", 1);
    try std.testing.expect(try enabled());
    _ = sys.env.setenv(name, "2", 1);
    try std.testing.expectError(error.InvalidEnvironment, enabled());
}
