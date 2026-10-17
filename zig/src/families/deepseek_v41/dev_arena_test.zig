//! TF_DSV41_ARENA changes allocations only: every emitted program and the buffer plan match the knob-off build.

const std = @import("std");
const testing = std.testing;
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const check = @import("calls_check.zig");
const rowmode = @import("rowmode.zig");
const dspark_emit = @import("dspark_emit.zig");
const dev_arena = @import("dev_arena.zig");
const Config = @import("config.zig").Config;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

test {
    _ = dev_arena;
}

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    return w;
}

fn options() block.Options {
    const limit: i64 = 1 << 20;
    var o: block.Options = .{ .limit = limit, .r1 = true, .mhc_defer = true, .expert_topp = 0.85, .index_budget = 64 << 20, .rows = true, .rope_rows = limit + 2048, .taps = true };
    const pages = @divExact(limit, block.Pool.page);
    o.pool = .{ .comp_pages = pages + 1, .ik_pages = pages + 1, .pts = pages };
    return o;
}

/// Every program kind, emitted with the environment as it is.
fn programs(a: std.mem.Allocator, cfg: *const Config, w: *const block.Widths) ![]const []const calls.Call {
    var out: std.ArrayList([]const calls.Call) = .empty;
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    const o = options();
    for ([_]i64{ 1, 5, 16, 17, 24, 64 }) |n| {
        const one = try block.emit(a, cfg, w, o, &backbone, n, 300_000, true);
        try out.append(a, one);
        if (n > 1) try out.append(a, try rowmode.transform(a, one, @intCast(n), .{ .slots = 4, .rmax = 64, .pts = o.pool.?.pts }));
    }
    try out.append(a, try dspark_emit.emitPass(a, cfg, w, o, cfg.dspark_block));
    try out.append(a, try dspark_emit.emitPassSlots(a, cfg, w, o, cfg.dspark_block, 4, 4));
    return out.items;
}

test "TF_DSV41_ARENA: every program's calls are the same with the knob set (only addresses move at run time)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    _ = unsetenv("TF_DSV41_ARENA");
    const off = try programs(a, &cfg, &w);
    for ([_][:0]const u8{ "1", "plain" }) |v| {
        _ = setenv("TF_DSV41_ARENA", v, 1);
        defer _ = unsetenv("TF_DSV41_ARENA");
        const on = try programs(a, &cfg, &w);
        try testing.expectEqual(off.len, on.len);
        for (off, on) |x, y| try testing.expectEqualDeep(x, y);
    }
}
