//! A bank saved and reloaded is the bank that was taught: the same values, the same routes, the same composed bytes.
const std = @import("std");
const safety = @import("safety_test.zig");
const Bank = @import("bank.zig").Bank;
const whitener = @import("whitener.zig");
const t = std.testing;

fn sameWhitener(a: whitener.LayerWhitener, b: whitener.LayerWhitener) !void {
    try t.expectEqualSlices(u8, std.mem.sliceAsBytes(a.mean), std.mem.sliceAsBytes(b.mean));
    try t.expectEqualSlices(u8, std.mem.sliceAsBytes(a.basis), std.mem.sliceAsBytes(b.basis));
    try t.expectEqualSlices(u8, std.mem.sliceAsBytes(a.cal_tokens), std.mem.sliceAsBytes(b.cal_tokens));
    try t.expectEqual(@as(u32, @bitCast(a.norm_floor)), @as(u32, @bitCast(b.norm_floor)));
}

/// Every query's composed state from `b`, in order, with a miss as no bytes.
fn composeAll(b: *Bank, mock: *safety.Mock, queries: []const []const u32) !std.ArrayList(u8) {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(t.allocator);
    mock.composed = &out;
    defer mock.composed = null;
    for (queries) |q| {
        const snap = (try b.routeAndCompose(q)) orelse {
            try out.appendSlice(t.allocator, "miss");
            continue;
        };
        try out.appendSlice(t.allocator, std.mem.asBytes(&snap.at));
        mock.backend().dropSnap(mock, snap.saved);
    }
    return out;
}

test "bank exact: a saved and reloaded bank holds bit for bit the values, routes and composed states of the taught one" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(dir);
    var mock: safety.Mock = .{ .gpa = t.allocator, .arch = safety.Mock.spec };
    const fresh = try safety.taught(t.allocator, dir, &mock, false);
    defer fresh.deinit();
    const more = [_]u32{ 40, 41, 42, 43 };
    _ = try fresh.teachChunk(.{ .parent_id = 8, .body_ids = &more, .gate_ids = &more, .is_small_fact = true });
    const rule = [_]u32{ 50, 51, 52 };
    _ = try fresh.teachChunk(.{ .parent_id = 9, .body_ids = &rule, .gate_ids = &.{}, .is_standing = true });
    _ = try fresh.saveFull();
    const back = try Bank.init(t.allocator, t.io, dir, mock.backend());
    defer back.deinit();
    try t.expect(try back.loadIfPresent());

    try sameWhitener(fresh.w_lex.?, back.w_lex.?);
    try sameWhitener(fresh.w_sem.?, back.w_sem.?);
    try t.expectEqual(fresh.units.items.len, back.units.items.len);
    for (fresh.units.items, back.units.items) |a, b| {
        try t.expectEqual(a.unit_id, b.unit_id);
        try t.expectEqualSlices(u8, std.mem.sliceAsBytes(a.gate.lex_protos), std.mem.sliceAsBytes(b.gate.lex_protos));
        try t.expectEqualSlices(u8, std.mem.sliceAsBytes(a.gate.sem_protos), std.mem.sliceAsBytes(b.gate.sem_protos));
        try t.expectEqualSlices(u8, std.mem.sliceAsBytes(a.gate.lex_weights), std.mem.sliceAsBytes(b.gate.lex_weights));
        try t.expectEqualSlices(u8, std.mem.sliceAsBytes(a.gate.sem_weights), std.mem.sliceAsBytes(b.gate.sem_weights));
        try t.expectEqual(@as(u32, @bitCast(a.gate.neg_sem_max)), @as(u32, @bitCast(b.gate.neg_sem_max)));
    }
    const queries = [_][]const u32{ &.{ 30, 31, 32 }, &.{ 40, 41 }, &.{ 30, 41, 33 }, &.{ 45, 46, 47 }, &safety.fact_ids };
    var a = try composeAll(fresh, &mock, &queries);
    defer a.deinit(t.allocator);
    var b = try composeAll(back, &mock, &queries);
    defer b.deinit(t.allocator);
    try t.expect(a.items.len > 0);
    try t.expectEqualSlices(u8, a.items, b.items);
}

test "bank exact: a fact's recurrent change and conv state are stored whole, so composing it gives exactly the body's" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(dir);
    var mock: safety.Mock = .{ .gpa = t.allocator, .arch = safety.Mock.spec };
    const b = try safety.taught(t.allocator, dir, &mock, true);
    defer b.deinit();
    const back = try Bank.init(t.allocator, t.io, dir, mock.backend());
    defer back.deinit();
    try t.expect(try back.loadIfPresent());
    const RawState = @import("bank.zig").RawState;
    // the fact's state as the model computes it on the shared prefix
    var body = try RawState.init(t.allocator, mock.arch, safety.fact_ids.len);
    defer body.deinit(t.allocator);
    try mock.backend().compileState(&mock, &b.prefix_state.?, &safety.fact_ids, &body);
    const pref = &b.prefix_state.?;
    for ([_]*Bank{ b, back }) |bank| {
        var out = try bank.composeRawState(&.{0}); // the fact alone: one parent, weight 1
        defer out.deinit(t.allocator);
        for (out.rec_state, pref.rec_state, body.rec_state) |got, p, s| for (got, p, s) |g, pv, sv| {
            try t.expectEqual(@as(u32, @bitCast(@mulAdd(f32, 1.0, sv - pv, pv))), @as(u32, @bitCast(g)));
        };
        for (out.rec_conv, body.rec_conv) |got, want| try t.expectEqualSlices(u16, want, got);
    }
}
