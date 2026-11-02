//! A synthetic live pocket: an own-question neuron and a deliberately leaky neuron, read without teaching.
const std = @import("std");
const living = @import("learner.zig");
const fam = @import("family.zig");
const d = 64;

pub const Device = struct {
    up: [64 * d]f32 = @splat(0),
    down: [64 * d]f32 = @splat(0),
    pub fn init() Device {
        var m: Device = .{};
        m.up[0] = 1;
        m.up[d + 1] = 2;
        m.down[0] = 0.5;
        m.down[d + 1] = 0.25;
        return m;
    }
};

const T = struct {
    count: @import("pocket_clock.zig").Count = .{},
    pack: usize = 64,
    ids: []const u32 = &.{},
    scratch: [d]u16 = @splat(0),
    pub fn normed(t: *T, _: usize, row: usize) []const u16 {
        @memset(&t.scratch, 0);
        t.scratch[0] = if (t.ids[row] == 1) 0x3f80 else 0xbf80;
        t.scratch[1] = 0x3f80;
        return &t.scratch;
    }
    pub fn step(t: *T, ids: []const u32, start: usize, mode: @import("train_mode.zig").Mode, turn: @import("lesson_route.zig").Turn) !void {
        try std.testing.expectEqual(@import("train_mode.zig").Mode.keys, mode);
        try std.testing.expectEqual(ids.len - 1, start);
        try std.testing.expect(turn.whole);
        t.count.passes += 1;
        t.count.rows += ids.len - 1;
        t.ids = ids[0 .. ids.len - 1];
    }
};

const Host = struct {
    pub const Ctx = *Device;
    pub const Trainer = T;
    pub const family = @import("families.zig").nemotron_h;
    pub const max_rows = 64;
    pub fn seconds(_: std.Io) f64 {
        return 0;
    }
    pub fn trainer(a: std.mem.Allocator, _: std.Io, _: Ctx) !*T {
        const t = try a.create(T);
        t.* = .{};
        return t;
    }
    pub fn free(t: *T, a: std.mem.Allocator) void {
        a.destroy(t);
    }
    pub fn layout(_: Ctx) fam.Layout {
        var ls: fam.Layers = .empty;
        ls.set(1);
        return fam.Layout.fromExperts(d, 64, 3, ls);
    }
    pub fn pocketRead(m: Ctx, _: usize, up: []f32, down: []f32) usize {
        @memcpy(up, &m.up);
        @memcpy(down, &m.down);
        return 2;
    }
    pub fn pocketSave(_: Ctx) void {
        @panic("an audit cannot save");
    }
};

pub const Learner = living.Learner(Host);
pub const Neuron = struct { layer: u16 = 1, neuron: u16, key: enum { reply } = .reply };
pub const Peak = struct { activation: f32 = 0, row: u32 = 0 };

test "pocket audit: clean and leaky live neurons, final prompt row, exact weight and lesson immutability" {
    var m = Device.init();
    const before = m;
    var l = Learner.init(std.testing.allocator, std.testing.io, &m);
    defer l.deinit();
    l.lessons = 17;
    l.kept_rounds = 3;
    l.rounds = 9;
    l.at = 11;
    l.loss = 0.25;
    l.lesson = .{ .train = &.{.{ .ids = &.{ 8, 9 }, .start = 1, .decide = 0 }}, .steps = 37 };
    var audit = try Learner.Audit.init(&l);
    defer audit.deinit();
    const ns = [_]Neuron{ .{ .neuron = 0 }, .{ .neuron = 1 } };
    var out: [2]Peak = undefined;
    try audit.read(&l, &.{ 2, 1 }, &ns, &out);
    try std.testing.expectEqual(@as(f32, 1), out[0].activation);
    try std.testing.expectEqual(@as(u32, 1), out[0].row);
    try std.testing.expectEqual(@as(f32, 4), out[1].activation);
    try audit.read(&l, &.{ 2, 3, 4 }, &ns, &out);
    try std.testing.expectEqual(@as(f32, 0), out[0].activation);
    try std.testing.expectEqual(@as(f32, 4), out[1].activation);
    try std.testing.expectEqualSlices(f32, &before.up, &m.up);
    try std.testing.expectEqualSlices(f32, &before.down, &m.down);
    try std.testing.expectEqual(@as(u32, 17), l.lessons);
    try std.testing.expectEqual(@as(u32, 3), l.kept_rounds);
    try std.testing.expect(l.phase == .idle and !l.can_undo and l.lesson.steps == 37);
    try std.testing.expectEqualSlices(u32, &.{ 8, 9 }, l.lesson.train[0].ids);
    try std.testing.expect(l.rounds == 9 and l.at == 11 and l.loss == 0.25);
    try std.testing.expectEqual(@as(u64, 0), l.trainer.?.count.passes);
    try std.testing.expectEqual(@as(usize, 0), l.pockets.moe.items.len);
    try std.testing.expectEqual(@as(usize, 0), l.replay.items.len);
}

test "pocket audit: bad and missing rows, nonfinite activation, and unresolved pocket fail closed" {
    var m = Device.init();
    var l = Learner.init(std.testing.allocator, std.testing.io, &m);
    defer l.deinit();
    var audit = try Learner.Audit.init(&l);
    defer audit.deinit();
    var out: [1]Peak = undefined;
    const ns = [_]Neuron{.{ .neuron = 0 }};
    try std.testing.expectError(error.AuditPromptTooLong, audit.read(&l, &.{}, &ns, &out));
    try std.testing.expectError(error.AuditMissingNeuron, audit.read(&l, &.{2}, &[_]Neuron{.{ .neuron = 2 }}, &out));
    audit.pocket.up.items[0] = std.math.inf(f32);
    try std.testing.expectError(error.AuditNonfinite, audit.read(&l, &.{1}, &ns, &out));
    l.pockets.restore_pending = true;
    try std.testing.expectError(error.AuditUnresolved, Learner.Audit.init(&l));
    l.pockets.restore_pending = false;
}
