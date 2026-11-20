//! Executable IR contract for the additive paired site: direction rows, TP writer, lagged fold, tap and final.
const std = @import("std");
const t = std.testing;
const block = @import("block.zig");
const prefill = @import("block_prefill.zig");
const calls = @import("block_calls.zig");
const Config = @import("config.zig").Config;
const plan = @import("glp_plan.zig");
const ns = "tf_dsv41_glp_add_v1.";
const ids = [_]u32{ 1, 13, 36, 39 };

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try @import("widths_listing.zig").parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try @import("widths_listing.zig").parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    for (0..43) |layer| try w.gm.put(a, @intCast(layer), .{ (1 << 4) | (1 << 6) | (1 << 8), (1 << 4) | (1 << 10) });
    for (0..43) |layer| for (4..8) |group| {
        const first = try std.fmt.allocPrint(a, "L{d}.attn.wo_a.0", .{layer});
        try w.dense.put(a, try std.fmt.allocPrint(a, "L{d}.attn.wo_a.{d}", .{ layer, group }), try w.k2(first));
    };
    return w;
}

fn steering(a: std.mem.Allocator) !plan.Plan {
    const Layer = struct { id: u32, dir: []const f32 };
    const layers = try a.alloc(Layer, ids.len);
    const dir = try a.alloc(f32, 5120);
    @memset(dir, 0.25);
    for (layers, ids) |*layer, id| layer.* = .{ .id = id, .dir = dir };
    return (try plan.Plan.init(a, .{ .alpha = @as(f32, 0.5), .vector = .{
        .spec_version = @as(u32, 3),
        .hook_point = "hc_fold_pre_engram",
        .width = @as(usize, 5120),
        .layers = layers,
        .file_sha256 = @as([32]u8, @splat(3)),
    } })).?;
}

fn nth(cs: []const calls.Call, name: []const u8, wanted: usize) !usize {
    var seen: usize = 0;
    for (cs, 0..) |c, at| if (std.mem.eql(u8, c.name, name)) {
        if (seen == wanted) return at;
        seen += 1;
    };
    return error.MissingPairedEdit;
}
fn role(arg: calls.Arg) []const u8 {
    return switch (arg.t.role) {
        .buf => |b| b,
        .weight => |w| w,
        .empty => "",
    };
}
fn check(cs: []const calls.Call, rows: i64) !void {
    var folds: usize = 0;
    var writers: usize = 0;
    for (cs) |c| {
        folds += @intFromBool(std.mem.eql(u8, c.name, ns ++ "stream"));
        writers += @intFromBool(std.mem.eql(u8, c.name, ns ++ "writer"));
    }
    try t.expectEqual(ids.len, folds);
    try t.expectEqual(ids.len, writers);
    for (ids, 0..) |id, slot| {
        const fold = try nth(cs, ns ++ "stream", slot);
        const writer = try nth(cs, ns ++ "writer", slot);
        try t.expect(writer < fold);
        const stream = cs[fold];
        try t.expectEqualStrings("_site", cs[fold - 1].name);
        try t.expect(std.mem.startsWith(u8, role(stream.args[0].arg), "w.x"));
        try t.expectEqualStrings(plan.dirs_role, role(stream.args[2].arg));
        try t.expectEqual(@as(i64, @intCast(slot)) * 5120 * 4, stream.args[2].arg.t.offset);
        try t.expectEqual(@as(f64, 0.5), stream.args[3].arg.f);
        try t.expectEqual(rows, stream.args[4].arg.i);
        try t.expectEqual(@as(i64, 5120), stream.args[5].arg.i);
        if (id == 36) try t.expectEqualStrings("w.taps", role(stream.args[1].arg));
        const partials = cs[writer].args[0].arg.t;
        try t.expectEqualSlices(i64, &.{ 2, rows, 5120 }, partials.shape);
        try t.expectEqualStrings(plan.dirs_role, role(cs[writer].args[1].arg));
        try t.expectEqual(stream.args[2].arg.t.offset, cs[writer].args[1].arg.t.offset);
    }
    const last_fold = try nth(cs, ns ++ "stream", ids.len - 1);
    const final = try nth(cs, ns ++ "final", 0);
    try t.expectEqual(last_fold + 1, final);
    try t.expectEqualStrings("w.out", role(cs[final].args[3].arg));
    try t.expectEqualStrings("glue.kit_logits", cs[cs.len - 1].name);
}

test "decode programs pair the same direction before Engram, in the full TP writer, in DSpark taps and before head" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var w = try widths(a);
    const cfg: Config = .{};
    const p = try steering(a);
    var layers: [40]u32 = undefined;
    for (&layers, 0..) |*layer, i| layer.* = @intCast(i);
    for ([_]i64{ 1, 16, 17, 64 }) |rows|
        try check(try block.emit(a, &cfg, &w, .{ .glp = &p, .taps = true }, &layers, rows, 4032, true), rows);
}

test "whole prefill carries paired edits and the edited final across short and long segments" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var w = try widths(a);
    const cfg: Config = .{};
    const p = try steering(a);
    var layers: [40]u32 = undefined;
    for (&layers, 0..) |*layer, i| layer.* = @intCast(i);
    for ([_]i64{ 16, 300, 2048 }) |rows|
        try check(try prefill.emitPrefill(a, &cfg, &w, .{ .glp = &p, .taps = true }, &layers, rows, 0, true), rows);
}

test "a partial backbone, replay or trace cannot silently skip a paired edit" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var w = try widths(a);
    const cfg: Config = .{};
    const p = try steering(a);
    var layers: [40]u32 = undefined;
    for (&layers, 0..) |*layer, i| layer.* = @intCast(i);
    try t.expectError(error.InvalidBlock, block.emit(a, &cfg, &w, .{ .glp = &p }, layers[0..10], 16, 0, false));
    try t.expectError(error.InvalidBlock, block.emit(a, &cfg, &w, .{ .glp = &p, .trace = true }, &layers, 16, 0, true));
    try t.expectError(error.InvalidBlock, prefill.emitReplay(a, &cfg, &w, .{ .glp = &p }, layers[0..21], 300, 0, .encoder));
}
