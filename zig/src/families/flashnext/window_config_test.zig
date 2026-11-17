//! Ordinary opens must have their own unscaled settings after a long or failed-long load.
const std = @import("std");
const w = @import("window_config.zig");
const t = std.testing;
fn failedLoad() !void {
    _ = try w.plan(600000, w.trained, 32, false);
    return error.LoadFailed;
}
test "Flash Next context: long then ordinary open restores the trained unscaled setting" {
    const long = try w.plan(600000, w.trained, 32, false);
    try t.expect(long.settings.capacity > w.trained);
    try t.expectEqual(@as(f64, 4), long.settings.factor);
    const ordinary = try w.plan(4096, w.trained, 32, false);
    try t.expectEqual(w.trained, ordinary.settings.capacity);
    try t.expectEqual(@as(f64, 0), ordinary.settings.factor);
    try t.expectEqual(@as(f64, 4), long.settings.factor);
}
test "Flash Next context: a failed long load cannot contaminate an ordinary open" {
    try t.expectError(error.LoadFailed, failedLoad());
    const ordinary = try w.plan(w.trained, w.trained, 32, false);
    try t.expectEqual(w.trained, ordinary.settings.capacity);
    try t.expectEqual(@as(f64, 0), ordinary.settings.factor);
}

test "Flash Next context: immutable settings reject invalid capacities and long pairs" {
    const paired = try w.plan(600000, w.trained, 32, true);
    try t.expect(paired.refusal != null);
    try t.expectError(error.ContextTooLong, w.plan(std.math.maxInt(i64), w.trained, 32, false));
    try t.expectError(error.ContextTooLong, w.plan(-1, w.trained, 32, false));
    try t.expectError(error.ContextTooLong, (w.Settings{ .capacity = w.trained + 1 }).validate(false));
    try t.expectError(error.BadRotaryPolicy, (w.Settings{ .capacity = 600032, .factor = 0 }).validate(false));
    try t.expectError(error.LongWindowPairUnsupported, (w.Settings{ .capacity = 600032, .factor = 4 }).validate(true));
}

test "Flash Next context: live runs keep distinct capacities and actual rotary buffers across another open" {
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

test "Flash Next context: shared metadata uses each live engine's capacity after an ordinary open" {
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

test "Flash Next context: every changed rotary path has a measured resource record" {
    const registry = @import("contract").registry;
    for ([_][]const u8{ "flashnext vision fn_attn_prep_mrope", "flashnext rotary target ", "flashnext rotary mtp ", "flashnext rotary pooled ", "flashnext rotary absolute " }) |prefix| {
        var found = false;
        for (registry.all_metal) |r| if (std.mem.startsWith(u8, r.name, prefix)) {
            found = true;
            try t.expect(r.m1 > 0 and r.m2 > 0); // recorded in e08019c9a
        };
        try t.expect(found);
    }
}

test "Flash Next context: one-plane aliases stay valid and two-part calls retain independent workspaces" {
    const E = @import("engine.zig").Engine;
    const P = @import("replay.zig").Prompt;
    const M = @import("replay.zig").Model;
    var first: P = undefined;
    var second: P = undefined;
    first.step = 8192;
    var model: M = undefined;
    model.pos = 0;
    var e: E = undefined;
    e.m = &model;
    e.pr = &first;
    e.pr2 = e.pr;
    e.segments = false;
    const one = e.chunkCall(16000);
    try t.expectEqual(@as(usize, 1), one.parts);
    const views = [2]*P{ e.pr, e.pr2 };
    try t.expectEqual(&first, views[0]);
    try t.expectEqual(&first, views[1]);
    e.pr2 = &second;
    e.segments = true;
    try t.expectEqual(@as(usize, 2), e.chunkCall(16000).parts);
    try t.expect(e.pr != e.pr2);
    const s: w.Settings = .{ .ssd_experts = 8 << 30 };
    try t.expect(!s.staggered() and !s.sibling(false));
    try t.expect((w.Settings{}).sibling(false));
    try t.expect(s.sibling(true));
    try t.expectError(error.FsrNeedsSingleMetalDevice, s.validate(true));
    try t.expect(!(w.Settings{ .ple_on_ssd = true }).sibling(false));
    try t.expect(!(w.Settings{ .fsr_trace = "trace.jsonl" }).sibling(false));
    try t.expect((w.Settings{}).sibling(true));
}
