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
    const ordinary = try w.plan(0, w.trained, 32, false);
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
    const plain = try w.plan(0, w.trained, 32, false);
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
    const ordinary = try w.plan(0, w.trained, 32, false);
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

test "Flash Next context: explicit smaller capacity keeps unscaled rotary and refuses a request past it" {
    const choice = try w.plan(65536, w.trained, 32, false);
    try t.expectEqual(@as(usize, 65568), choice.settings.capacity);
    try t.expectEqual(@as(f64, 0), choice.settings.factor);
    try choice.settings.validate(false);
    try t.expectEqual(@as(usize, 8192), (try w.plan(1, w.trained, 32, false)).settings.capacity);
    const edge = try w.plan(w.trained - 1, w.trained, 32, false);
    try t.expectEqual(@as(f64, 0), edge.settings.factor);
    const meta = @import("batch_meta.zig");
    try t.expectError(error.ContextFull, meta.Plan.init(&.{.{ .slot = 0, .pos = @intCast(choice.settings.capacity), .rows = 1 }}, @intCast(choice.settings.capacity)));
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

test "Flash Next context: edge windows preserve logical requests, physical padding and trained rotary" {
    for ([_]i64{ 1, 8159, 8160, 8161, 65535, 65536, w.trained - 1 }) |logical| {
        const choice = try w.plan(logical, w.trained, 32, false);
        try t.expect(choice.settings.capacity >= logical + 32);
        try t.expect(choice.settings.capacity >= w.minimum);
        try t.expectEqual(@as(usize, 0), choice.settings.capacity % 4);
        try t.expectEqual(@as(f64, 0), choice.settings.factor);
        try (try w.plan(logical, w.trained, 32, true)).settings.validate(true);
    }
    try t.expectEqual(w.trained, (try w.plan(0, w.trained, 32, false)).settings.capacity);
    try t.expectEqual(w.trained, (try w.plan(w.trained, w.trained, 32, false)).settings.capacity);
    try t.expectEqual(@as(f64, 4), (try w.plan(w.trained + 1, w.trained, 32, false)).settings.factor);
    try t.expectError(error.ContextTooLong, (w.Settings{ .capacity = w.minimum - 4 }).validate(false));
}

test "Flash Next context: a snapshot outside the smaller cache is refused before touching device state" {
    const fz = @import("replay.zig");
    const E = @import("engine.zig").Engine;
    const snapshots = @import("snapshot.zig");
    var model: fz.Model = undefined;
    model.cap = 8192;
    var e: E = undefined;
    e.m = &model;
    var state: snapshots.State = undefined;
    state.at = 8193;
    state.bytes = snapshots.bytes(state.at);
    try t.expectError(error.SnapshotSize, snapshots.restore(&e, &state));
    state.at = 4096;
    state.bytes = 1;
    try t.expectError(error.SnapshotSize, snapshots.restore(&e, &state));
}

test "Flash Next context: growth refuses insufficient destination headroom before submitting work" {
    const Session = @import("session.zig").Session;
    var src: Session = .{ .m = undefined };
    var dst: Session = .{ .m = undefined };
    src.m.state = 0;
    src.m.state_row = 0;
    src.m.cap = 8192;
    src.m.pos = 8180;
    dst.m.cap = 8191;
    var e: @import("engine.zig").Engine = undefined;
    try t.expectError(error.ContextFull, src.moveTo(&dst, &e));
}

test "Flash Next context: different physical caps cannot reuse disk identities with identical rotary" {
    const identity = @import("../../native/variant_identity.zig");
    var full = std.hash.Wyhash.init(17);
    var small = std.hash.Wyhash.init(17);
    identity.policy(&full, w.trained, 0, false);
    identity.policy(&small, 65568, 0, false);
    try t.expect(full.final() != small.final());
}

test "Flash Next context: growth records both KV head strides and preserves MTP and recurrence metadata" {
    const metal = @import("metal");
    const fz = @import("replay.zig");
    const Session = @import("session.zig").Session;
    const E = @import("engine.zig").Engine;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    metal.allocator = arena.allocator();
    var r: fz.Run = .{ .arena = arena.allocator(), .device = .{}, .queue = .{} };
    const Fixture = struct {
        fn buffer(rr: *fz.Run, n: usize, value: u8) !fz.Buf {
            const b = try rr.buffer(n);
            @memset(b.slice(u8, n), value);
            return .{ .b = b };
        }
        fn attention(rr: *fz.Run, layer: anytype, cap: usize, value: u8) !void {
            layer.keys = try buffer(rr, cap * 1024, value);
            layer.vals = try buffer(rr, cap * 1024, value);
            layer.raw = try buffer(rr, cap * 256, value);
            layer.pooled = try buffer(rr, cap / 4 * 256, value);
            layer.pooled_n = 8;
            if (value != 0) @memset(layer.keys.b.slice(u8, cap * 1024)[cap * 512 ..], 91);
        }
        fn session(rr: *fz.Run, cap: usize, value: u8) !Session {
            var s: Session = .{ .m = undefined };
            s.m.cap = cap;
            s.m.pos = 32;
            s.m.state = 0;
            s.m.state_row = 0;
            s.m.delta = 7;
            s.m.bpos = null;
            s.m.ple.hist = .{ 7, 11 };
            s.m.ple.cin = try buffer(rr, (fz.PLE_TAIL + fz.MAXR) * fz.WIDE * 2, value);
            s.last = try buffer(rr, fz.WIDE * 2, value);
            for (&s.m.layers, 0..) |*layer, i| {
                layer.linear = i % 4 != 3;
                if (layer.linear) {
                    layer.cs[0] = try buffer(rr, fz.CS_ROW, value);
                    layer.so[0] = try buffer(rr, fz.SO_ROW, value);
                } else try attention(rr, layer, cap, value);
            }
            try attention(rr, &s.m.mtp, cap, value);
            s.m.mtp.pos = 35;
            s.m.mtp.drafted = 2;
            s.held_n = 2;
            s.held[0] = 123;
            return s;
        }
    };
    var src = try Fixture.session(&r, 128, 37);
    var dst = try Fixture.session(&r, 256, 0);
    var e: E = undefined;
    e.r = &r;
    metal.source_count = 0;
    metal.first_blit = null;
    try src.moveTo(&dst, &e);
    try t.expectEqual(src.m.pos, dst.m.pos);
    try t.expectEqual(src.m.mtp.pos, dst.m.mtp.pos);
    try t.expectEqual(src.m.mtp.drafted, dst.m.mtp.drafted);
    try t.expectEqual(src.m.ple.hist, dst.m.ple.hist);
    try t.expectEqual(src.held_n, dst.held_n);
    try t.expectEqual(src.held[0], dst.held[0]);
    inline for (.{ &src.m.layers[3], &src.m.mtp }, .{ &dst.m.layers[3], &dst.m.mtp }) |from, to| {
        var heads: usize = 0;
        for (metal.sources[0..metal.source_count], 0..) |copy, i| if (copy.id == from.keys.b.id) {
            try t.expectEqual(heads * 128 * 512, copy.off);
            try t.expectEqual(to.keys.b.id, metal.destinations[i].id);
            try t.expectEqual(heads * 256 * 512, metal.destinations[i].off);
            try t.expectEqual(@as(usize, 66 * 512), metal.copy_bytes[i]);
            heads += 1;
        };
        try t.expectEqual(@as(usize, 2), heads);
        try t.expectEqual(@as(usize, 8), to.pooled_n);
    }
    var recurrent: usize = 0;
    for (metal.sources[0..metal.source_count], 0..) |copy, i| if (copy.id == src.m.layers[0].cs[0].b.id) {
        try t.expectEqual(dst.m.layers[0].cs[0].b.id, metal.destinations[i].id);
        try t.expectEqual(@as(usize, fz.CS_ROW), metal.copy_bytes[i]);
        recurrent += 1;
    };
    try t.expectEqual(@as(usize, 1), recurrent);
}
