//! Ordinary opens must have their own unscaled settings after a long or failed-long load.
const std = @import("std");
const w = @import("window_config.zig");
const t = std.testing;
fn failedLoad() !void {
    _ = try w.plan(600000, w.trained, 32, false);
    return error.LoadFailed;
}
test "flashnext window: long then ordinary open restores the trained unscaled setting" {
    const long = try w.plan(600000, w.trained, 32, false);
    try t.expect(long.settings.capacity > w.trained);
    try t.expectEqual(@as(f64, 4), long.settings.factor);
    const ordinary = try w.plan(4096, w.trained, 32, false);
    try t.expectEqual(w.trained, ordinary.settings.capacity);
    try t.expectEqual(@as(f64, 0), ordinary.settings.factor);
    try t.expectEqual(@as(f64, 4), long.settings.factor);
}
test "flashnext window: a failed long load cannot contaminate an ordinary open" {
    try t.expectError(error.LoadFailed, failedLoad());
    const ordinary = try w.plan(w.trained, w.trained, 32, false);
    try t.expectEqual(w.trained, ordinary.settings.capacity);
    try t.expectEqual(@as(f64, 0), ordinary.settings.factor);
}

test "flashnext window: immutable settings reject invalid capacities and long pairs" {
    const paired = try w.plan(600000, w.trained, 32, true);
    try t.expect(paired.refusal != null);
    try t.expectError(error.ContextTooLong, w.plan(std.math.maxInt(i64), w.trained, 32, false));
    try t.expectError(error.ContextTooLong, w.plan(-1, w.trained, 32, false));
    try t.expectError(error.ContextTooLong, (w.Settings{ .capacity = w.trained + 1 }).validate(false));
    try t.expectError(error.BadRotaryPolicy, (w.Settings{ .capacity = 600032, .factor = 0 }).validate(false));
    try t.expectError(error.LongWindowPairUnsupported, (w.Settings{ .capacity = 600032, .factor = 4 }).validate(true));
}

test "flashnext window: live runs keep distinct capacities and actual rotary buffers across another open" {
    const fz = @import("replay.zig");
    const metal = @import("metal");
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    metal.allocator = arena.allocator();
    const requested = try w.plan(600000, w.trained, 32, false);
    var long: fz.Run = .{ .arena = arena.allocator(), .device = .{}, .queue = .{}, .window = requested.settings };
    const before = try fz.ropeBuf(&long, 23.253496170043945, long.window.factor);
    const plain = try w.plan(4096, w.trained, 32, false);
    var ordinary: fz.Run = .{ .arena = arena.allocator(), .device = .{}, .queue = .{}, .window = plain.settings };
    const after = try fz.ropeBuf(&ordinary, 23.253496170043945, ordinary.window.factor);
    try t.expectEqual(@as(usize, 600032), long.window.capacity);
    try t.expectEqual(w.trained, ordinary.window.capacity);
    try t.expectEqual(@as(f32, 1), before.b.slice(f32, 35)[1]);
    try t.expectEqual(@as(f32, 0), after.b.slice(f32, 35)[1]);
    try t.expect(before.b.slice(f32, 35)[2] > 1);
    try t.expectEqual(@as(f32, 1), after.b.slice(f32, 35)[2]);
    try t.expectEqual(@as(usize, 35 * 4), before.b.length());
}

test "flashnext window: shared metadata uses each live engine's capacity after an ordinary open" {
    const fz = @import("replay.zig");
    const E = @import("engine.zig").Engine;
    const B = @import("backend.zig").Backend;
    const meta = @import("batch_meta.zig");
    const long = try w.plan(600000, w.trained, 32, false);
    const ordinary = try w.plan(4096, w.trained, 32, false);
    var lr: fz.Run = .{ .arena = undefined, .device = .{}, .queue = .{}, .window = long.settings };
    var sr: fz.Run = .{ .arena = undefined, .device = .{}, .queue = .{}, .window = ordinary.settings };
    var le: E = undefined;
    le.r = &lr;
    var se: E = undefined;
    se.r = &sr;
    var lb: B = .{ .gpa = t.allocator, .e = &le, .primary = undefined, .sessions = &.{}, .hidden = undefined, .grammar = undefined };
    var sb = lb;
    sb.e = &se;
    const rows = [_]meta.Window{.{ .slot = 0, .pos = w.trained + 2, .rows = 2 }};
    try t.expectEqual(@as(usize, 2), (try meta.forBackend(&lb, &rows)).total);
    try t.expectError(error.ContextFull, meta.forBackend(&sb, &rows));
    try t.expectEqual(@as(usize, 2), (try meta.forBackend(&lb, &rows)).total);
}

test "flashnext window: every changed rotary path has a pending resource record" {
    const registry = @import("contract").registry;
    for ([_][]const u8{ "flashnext vision fn_attn_prep_mrope", "flashnext rotary target ", "flashnext rotary mtp ", "flashnext rotary pooled ", "flashnext rotary absolute " }) |prefix| {
        var found = false;
        for (registry.all_metal) |r| if (std.mem.startsWith(u8, r.name, prefix)) {
            found = true;
            try t.expectEqual(@as(u32, 0), r.m1);
            try t.expectEqual(@as(u32, 0), r.m2);
        };
        try t.expect(found);
    }
}
