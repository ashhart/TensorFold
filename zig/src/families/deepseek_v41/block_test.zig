const std = @import("std");
const t = std.testing;
const block = @import("block.zig");
const calls = @import("block_calls.zig");
const listing = @import("widths_listing.zig");
const Config = @import("config.zig").Config;
fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try listing.parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try listing.parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    return w;
}
test "decode emitter rejects zero world before divisions" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const c: Config = .{};
    try t.expectError(error.InvalidBlock, block.emit(a, &c, &w, .{ .world = 0 }, &.{0}, 1, 0, false));
}
test "decode emitter rejects empty row windows" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const c: Config = .{};
    try t.expectError(error.InvalidBlock, block.emit(a, &c, &w, .{}, &.{0}, 0, 0, false));
}
test "whole backbone and DSpark emit named calls and bounded reusable buffer plans" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const c: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*layer, i| layer.* = @intCast(i);
    var plan: @import("buffers.zig").Plan = .{ .a = a };
    for ([_]i64{ 1, 2, 8, 16, 17, 32, 64 }) |n| {
        const cs = try block.emit(a, &c, &w, .{ .limit = 8192, .rope_rows = 10240 }, &backbone, n, 4080, true);
        try t.expect(cs.len > 0);
        try plan.add(cs);
        var extensions: usize = 0;
        var exchanges: usize = 0;
        for (cs) |call| {
            if (!call.triton and !call.glue) extensions += 1;
            if (calls.isExchange(call)) exchanges += 1;
        }
        try t.expect(extensions > 0 and exchanges > 0);
    }
    for ([_]u32{ 40, 41, 42 }) |layer| try plan.add(try block.emit(a, &c, &w, .{}, &.{layer}, 8, 4000, false));
    const totals = plan.totals();
    try t.expect(totals.roles[0] > 0 and totals.roles[1] > 0 and totals.roles[2] > 0);
}

test "expert scratch admission covers backbone and DSpark windows" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const c: Config = .{};
    try t.expectError(error.InvalidBlock, block.emit(a, &c, &w, .{ .expert_rows = 1, .expert_topp = null }, &.{0}, 8, 0, false));
    try t.expectError(error.InvalidBlock, block.emit(a, &c, &w, .{ .dspark_rows = 7 }, &.{40}, 8, 0, false));
    for ([_]block.Options{ .{ .expert_rows = -1 }, .{ .dspark_rows = -1 }, .{ .branch_rows = -1 }, .{ .expert_rows = std.math.maxInt(i64) }, .{ .dspark_rows = std.math.maxInt(i64) } }) |o| {
        try t.expectError(error.InvalidBlock, block.emit(a, &c, &w, o, &.{0}, 8, 0, false));
    }
    for ([_]u32{ 0, 40 }) |layer| {
        const cs = try block.emit(a, &c, &w, .{ .expert_rows = 8, .dspark_rows = 8, .expert_topp = null }, &.{layer}, 8, 0, false);
        var plan: @import("buffers.zig").Plan = .{ .a = a };
        try plan.add(cs);
        var seen: usize = 0;
        for (cs) |call| if (std.mem.eql(u8, call.name, "tensorfold_exl3_experts_v1.rot_in") or std.mem.eql(u8, call.name, "tensorfold_exl3_experts_v1.group_rot_in")) {
            const rows = call.args[7].arg.i;
            const hidden = call.args[8].arg.i;
            const slots = call.args[9].arg.i;
            for (call.args[5..7]) |arg| {
                const dst = arg.arg.t;
                const need: u64 = @intCast(rows * hidden * slots * 2);
                try t.expect(plan.sizes.get(dst.role.buf).? >= @as(u64, @intCast(dst.offset)) + need);
                seen += 1;
            }
        };
        try t.expect(seen >= 2);
    }
}
test "default expert routing is exact and pruning is explicit" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const c: Config = .{};
    const exact = try block.emit(a, &c, &w, .{}, &.{0}, 8, 0, false);
    for (exact) |call| try t.expect(!std.mem.eql(u8, call.name, "_prune"));
    const approx = try block.emit(a, &c, &w, .{ .expert_topp = 0.85 }, &.{0}, 8, 0, false);
    var prune = false;
    for (approx) |call| if (std.mem.eql(u8, call.name, "_prune")) {
        prune = true;
    };
    try t.expect(prune);
}

test "branch row threshold falls back to the main stream above its limit" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: @import("config.zig").Config = .{};
    const program = try block.emit(a, &cfg, &w, .{ .branches = true, .branch_rows = 1 }, &.{0}, 8, 0, false);
    for (program) |call| try std.testing.expect(!call.side and !call.fork and !call.join);
}
