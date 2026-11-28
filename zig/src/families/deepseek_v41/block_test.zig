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

/// Every `_plain` launch of a float projection (`<prefix>.plain`), by its weight's name.
fn plainLaunches(a: std.mem.Allocator, cs: []const calls.Call) !std.StringArrayHashMapUnmanaged(calls.Call) {
    var out: std.StringArrayHashMapUnmanaged(calls.Call) = .empty;
    for (cs) |call| {
        if (!std.mem.eql(u8, call.name, "_plain")) continue;
        const name = call.args[2].arg.t.role.weight;
        if (!std.mem.endsWith(u8, name, ".plain")) continue;
        try out.put(a, name, call);
        // N is the row stride `_plain` stores with, and the grid's second dimension gives the outputs
        try t.expectEqual(call.args[4].arg.t.stride[0], call.args[6].arg.i);
        try t.expectEqual(@as(i64, 256), call.args[7].arg.i);
        try t.expect(!call.args[8].arg.b and call.args[9].arg.b);
        try t.expectEqual(call.args[4].arg.t.shape[1], call.grid[1]);
    }
    return out;
}

/// Fails when an argument reads an EXL3 part of an L2 float projection: its trellis, scales or a repacked form.
fn expectNoExl3(arg: calls.Arg) anyerror!void {
    const tensor = switch (arg) {
        .t, .opaque_table => |x| x,
        .list => |items| {
            for (items) |item| try expectNoExl3(item);
            return;
        },
        else => return,
    };
    const name = switch (tensor.role) {
        .weight => |n| n,
        .buf => |n| if (std.mem.startsWith(u8, n, "s.")) n[2..] else n,
        .empty => return,
    };
    var nb: [64]u8 = undefined;
    for (@import("float_linear.zig").projections) |proj|
        if (std.mem.startsWith(u8, name, try std.fmt.bufPrint(&nb, "L2.attn.{s}.", .{proj})))
            try t.expect(std.mem.endsWith(u8, name, ".plain"));
}

test "a float projection runs on `_plain` into its own columns, in a decode window and in a prompt segment" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var w = try widths(a);
    const c: Config = .{};
    const hd: i64 = c.head_dim;
    // layer 2 is a ratio-2 KV and index source, so its block holds all five projections names.plainOf covers
    for (@import("float_linear.zig").projections) |proj| try w.dense.put(a, try std.fmt.allocPrint(a, "L2.attn.{s}", .{proj}), 0);
    const window_calls = try block.emit(a, &c, &w, .{}, &.{2}, 8, 0, false);
    const window = try plainLaunches(a, window_calls);
    try t.expectEqual(@as(usize, 5), window.count());
    const gate = window.get("L2.attn.comp_wgate.plain").?;
    // the compressor's two projections share one row: the gate's half starts at its own column, with the row's stride
    try t.expectEqual(2 * hd, gate.args[6].arg.i);
    try t.expectEqual(2 * hd, gate.args[4].arg.t.offset);
    try t.expectEqual(hd, gate.grid[1]);
    try t.expectEqual(@as(i64, 0), window.get("L2.attn.comp_wkv.plain").?.args[4].arg.t.offset);
    try t.expectEqual(@as(i64, c.hidden), window.get("L2.attn.wq_a.plain").?.args[5].arg.i);
    try w.gm.put(a, 2, .{ (1 << 4) | (1 << 6) | (1 << 8), (1 << 4) | (1 << 10) });
    const segment_calls = try @import("block_prefill.zig").emitPrefill(a, &c, &w, .{}, &.{2}, 64, 0, false);
    const segment = try plainLaunches(a, segment_calls);
    for ([_][]const u8{ "L2.attn.wq_a.plain", "L2.attn.comp_wkv.plain", "L2.attn.comp_wgate.plain", "L2.attn.ix_wq_b.plain" }) |name|
        try t.expect(segment.get(name) != null);
    for ([_][]const calls.Call{ window_calls, segment_calls }) |program| for (program) |call| for (call.args) |named| try expectNoExl3(named.arg);
    // the widths the fixture holds: every projection keeps its EXL3 group and no float launch is emitted
    const kept = try plainLaunches(a, try block.emit(a, &c, &try widths(a), .{}, &.{2}, 8, 0, false));
    try t.expectEqual(@as(usize, 0), kept.count());
}

test "a float projection's L2 prefetch source is its fp32 matrix whole, never its EXL3 parts" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var w = try widths(a);
    const c: Config = .{};
    for (@import("float_linear.zig").projections) |proj| try w.dense.put(a, try std.fmt.allocPrint(a, "L2.attn.{s}", .{proj}), 0);
    var e: block.Emitter = .{ .a = a, .cfg = &c, .w = &w, .o = .{}, .n = 1, .start = 0 };
    const srcs = try e.l2pfSources(2, .x);
    // the x and q groups' float members; ix_wk has no prefetch site
    const want = [_]struct { []const u8, u64 }{
        .{ "L2.attn.wq_a", @as(u64, 4) * c.hidden * c.q_lora },
        .{ "L2.attn.comp_wkv", @as(u64, 4) * c.hidden * c.head_dim },
        .{ "L2.attn.comp_wgate", @as(u64, 4) * c.hidden * c.head_dim },
        .{ "L2.attn.ix_wq_b", @as(u64, 4) * c.q_lora * c.index_heads * c.index_dim },
    };
    for (want) |x| {
        var found = false;
        for (srcs) |s| if (std.mem.startsWith(u8, s.name, x[0])) {
            try t.expect(s.kind == .weight and std.mem.endsWith(u8, s.name, ".plain") and !found);
            try t.expectEqual(x[1], s.bytes);
            found = true;
        };
        try t.expect(found);
    }
}

test "with `r1` a float ix_wq_b reads the q rows its own `_rms` launch normed" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var w = try widths(a);
    const c: Config = .{};
    for (@import("float_linear.zig").projections) |proj| try w.dense.put(a, try std.fmt.allocPrint(a, "L2.attn.{s}", .{proj}), 0);
    // RMS_FOLD hands the indexer unnormed rows, and no rot_in is in a float projection's path to norm them
    var normed: ?[]const u8 = null;
    var plain_x: ?[]const u8 = null;
    for (try block.emit(a, &c, &w, .{ .r1 = true }, &.{2}, 8, 0, false)) |call| {
        if (std.mem.eql(u8, call.name, "_rms") and std.mem.eql(u8, call.args[2].arg.t.role.weight, "L2.attn.q_norm")) {
            try t.expectEqualStrings("L.qa", call.args[0].arg.t.role.buf);
            normed = call.args[3].arg.t.role.buf;
        }
        if (std.mem.eql(u8, call.name, "_plain") and std.mem.eql(u8, call.args[2].arg.t.role.weight, "L2.attn.ix_wq_b.plain")) {
            try t.expect(normed != null);
            plain_x = call.args[0].arg.t.role.buf;
        }
    }
    try t.expectEqualStrings(normed orelse return error.TestUnexpectedResult, plain_x orelse return error.TestUnexpectedResult);
}
