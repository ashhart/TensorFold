//! A delete on a pocket of three facts: their neurons zeroed and rounded, freed at the top, put back bit for bit.
const std = @import("std");
const pocket = @import("pocket.zig");
const forgetting = @import("pocket_forget.zig");
const whole = @import("whole.zig");
const examples = @import("lesson_examples.zig");
const family = @import("family.zig");
const shards = @import("shards.zig");
const F = @import("families.zig").nemotron_h;

const d = 64;
const layers = 4; // every MoE layer but the last holds a pocket: slots 0, 1 and 2

/// The trainer as the pocket reads it: nothing (the shape is the host's layout).
const Trainer = struct {};

const P = pocket.Pocket(Trainer, F);
const width = P.neurons * d;

/// A layer rounded as a save writes it: Nemotron's 4-bit groups of 64 (shards.zig).
fn round(up: []f32, down: []f32) !void {
    return shards.round(F, std.testing.allocator, d, &.{up}, down);
}

/// The backend: the pocket as the shards loaded it, and each layer's live copy as the forward reads it.
const Device = struct {
    loaded_up: [layers][width]f32 = @splat(@splat(0)),
    loaded_down: [layers][width]f32 = @splat(@splat(0)),
    loaded_used: [layers]usize = @splat(0),
    up: [layers][width]f32 = @splat(@splat(0)),
    down: [layers][width]f32 = @splat(@splat(0)),
    used: [layers]usize = @splat(0),
    uploads: usize = 0,
    saves: usize = 0,
    fail_at: usize = 0, // the upload, counted from one, that fails once (0: none)
    fail_from: usize = 0, // the upload, counted from one, from which every upload fails (0: none)
    failing: bool = false, // every upload fails

    pub fn pocketRead(h: *Device, l: usize, up: []f32, down: []f32) usize {
        @memcpy(up, &h.loaded_up[l]);
        @memcpy(down, &h.loaded_down[l]);
        h.up[l] = h.loaded_up[l];
        h.down[l] = h.loaded_down[l];
        h.used[l] = h.loaded_used[l];
        return h.loaded_used[l];
    }

    pub fn pocket(h: *Device, l: usize, up: []const f32, down: []const f32, used: usize) !void {
        if (h.failing or h.uploads + 1 == h.fail_at or (h.fail_from > 0 and h.uploads + 1 >= h.fail_from)) {
            h.uploads += 1;
            return error.UploadFailed;
        }
        @memcpy(&h.up[l], up);
        @memcpy(&h.down[l], down);
        h.used[l] = used;
        h.uploads += 1;
    }

    /// Every layer's shared expert carries a pocket; all but the last learn.
    pub fn layout(_: *Device) family.Layout {
        var has: family.Layers = .empty;
        for (0..layers) |l| has.set(l);
        return .fromExperts(d, 64, layers, has);
    }
};

/// The learner's host as a save reaches it: `Device` above, and a write into the shards that counts itself.
const SaveHost = struct {
    pub const Ctx = *Device;

    pub fn seconds(_: std.Io) f64 {
        return 0;
    }

    pub fn pocket(h: Ctx, l: usize, up: []const f32, down: []const f32, used: usize) !void {
        return h.pocket(l, up, down, used);
    }

    pub fn pocketRead(h: Ctx, l: usize, up: []f32, down: []f32) usize {
        return h.pocketRead(l, up, down);
    }

    pub fn layout(h: Ctx) family.Layout {
        return h.layout();
    }

    pub fn pocketSave(h: Ctx, _: std.mem.Allocator, _: std.Io, _: []const usize, _: usize, _: []f32, _: []f32) !usize {
        h.saves += 1;
        return 0;
    }
};

/// The learner as `whole.save` reads it: a pocket lesson kept, and its pocket.
const Learner = struct {
    pub const Host = SaveHost;
    method: whole.Method = .pocket,
    pocketed: bool = false,
    kept_rounds: u32 = 1,
    pockets: P = .{ .gpa = std.testing.allocator, .mode = .auto },
    trainer: ?*Trainer,
    b: *Device,
    gpa: std.mem.Allocator = std.testing.allocator,
    io: std.Io = std.testing.io,
    owned: examples.Lesson = .{},

    pub fn recalls(_: *Learner, _: []const examples.Example, _: bool) !bool {
        return true;
    }
};

/// Three facts as saved: A at (0, 0) and (1, 0); B at (0, 1) and (2, 0); C at (0, 2) and (1, 1), lessons 0, 1, 2.
fn three(h: *Device, p: *P, t: *Trainer) !void {
    var prng = std.Random.DefaultPrng.init(11);
    const r = prng.random();
    const held = [_][3]usize{ .{ 0, 0, 0 }, .{ 1, 0, 0 }, .{ 0, 1, 1 }, .{ 2, 0, 1 }, .{ 0, 2, 2 }, .{ 1, 1, 2 } };
    for (held) |x| {
        for (h.loaded_up[x[0]][x[1] * d ..][0..d], h.loaded_down[x[0]][x[1] * d ..][0..d]) |*u, *v| {
            u.* = r.floatNorm(f32) * 0.05;
            v.* = r.floatNorm(f32) * 0.2;
        }
        h.loaded_used[x[0]] = @max(h.loaded_used[x[0]], x[1] + 1);
    }
    // the shards hold 4-bit values: the pocket as a save left it
    for (0..layers) |l| try round(&h.loaded_up[l], &h.loaded_down[l]);
    try p.load(t, h);
    for (held, 0..) |x, i| try p.learned.append(p.gpa, .{ .slot = x[0], .neuron = x[1], .key = .reply, .start = i, .count = 1, .subject = 0, .lesson = @intCast(x[2]) });
    p.learned_before = 4; // C, the last lesson, found A's and B's
}

fn trainer() Trainer {
    return .{};
}

const b_neurons = [_][2]u16{ .{ 0, 1 }, .{ 2, 0 } };

test "a deleted fact's input weights go to zero, every output bit stays, its layers rounded as a save writes them" {
    var t = trainer();
    var h: Device = .{};
    var p: P = .{ .gpa = std.testing.allocator, .mode = .auto };
    defer p.deinit();
    try three(&h, &p, &t);
    // a lesson since the last save moved C's outputs in layer 0 off the 4-bit grid
    for (p.neuron(p.down.items, 0, 2, d)) |*v| v.* *= 1.001;
    const before_up = try std.testing.allocator.dupe(f32, p.up.items);
    defer std.testing.allocator.free(before_up);
    // what layer 0 is to hold: B's input weights zeroed, then the layer rounded on its own
    var want_up: [width]f32 = p.up.items[0..width].*;
    var want_down: [width]f32 = p.down.items[0..width].*;
    @memset(want_up[1 * d ..][0..d], 0);
    try round(&want_up, &want_down);
    try std.testing.expectEqual(@as(usize, 2), try forgetting.forget(&p, &t, &h, &b_neurons));
    try std.testing.expect(forgetting.pending(&p));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&want_up), std.mem.sliceAsBytes(p.up.items[0..width]));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&want_down), std.mem.sliceAsBytes(p.down.items[0..width]));
    try std.testing.expect(std.mem.allEqual(f32, p.neuron(p.up.items, 2, 0, d), 0));
    // layer 2 as saved: B's input weights zero, every output bit as it was, B's own too, so no 4-bit group moves
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&h.loaded_down[2]), std.mem.sliceAsBytes(p.down.items[2 * width ..][0..width]));
    // A's and C's input weights keep every bit; layer 1 was not touched at all
    for ([_][2]usize{ .{ 0, 0 }, .{ 0, 2 }, .{ 1, 0 }, .{ 1, 1 } }) |x| try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(before_up[(x[0] * P.neurons + x[1]) * d ..][0..d]), std.mem.sliceAsBytes(p.neuron(p.up.items, x[0], x[1], d)));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&h.loaded_down[1]), std.mem.sliceAsBytes(p.down.items[width..][0..width]));
    // layer 0's neuron 1 sits below C's, so it stays taken; layer 2 is empty again
    try std.testing.expectEqualSlices(usize, &.{ 3, 2, 0 }, p.used.items);
    // the forward reads exactly the pocket, and only the two layers changed were sent
    try std.testing.expectEqual(@as(usize, 2), h.uploads);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(p.down.items[0..width]), std.mem.sliceAsBytes(&h.down[0]));
    try std.testing.expectEqual(@as(usize, 0), h.used[2]);
    // B's records are gone, so no later fact on its freed neuron is refitted as B; its lesson's answers retire
    try std.testing.expectEqual(@as(usize, 4), p.learned.items.len);
    for (p.learned.items) |e| try std.testing.expect(e.lesson != 1);
    try std.testing.expectEqual(@as(usize, 2), p.learned_before);
    try std.testing.expectEqualSlices(u32, &.{1}, p.retired_lessons.items);
    forgetting.settle(&p);
    try std.testing.expect(!forgetting.pending(&p));
}

test "a lesson begun before a delete, then taken out, never brings back what the delete cut" {
    var t = trainer();
    var h: Device = .{};
    var p: P = .{ .gpa = std.testing.allocator, .mode = .auto };
    defer p.deinit();
    try three(&h, &p, &t);
    try @import("pocket_mark.zig").mark(&p); // a lesson began; a refused save left its mark in place
    try std.testing.expectEqual(@as(usize, 2), try forgetting.forget(&p, &t, &h, &b_neurons));
    try p.undo(&t, &h);
    for (b_neurons) |g| try std.testing.expect(std.mem.allEqual(f32, p.neuron(p.up.items, g[0], g[1], d), 0));
}

test "a delete put back restores every bit it changed, in the pocket and in the forward" {
    var t = trainer();
    var h: Device = .{};
    var p: P = .{ .gpa = std.testing.allocator, .mode = .auto };
    defer p.deinit();
    try three(&h, &p, &t);
    // C's outputs off the grid, as a lesson not yet saved leaves them: the delete's rounding moves them
    for (p.neuron(p.down.items, 0, 2, d)) |*v| v.* *= 1.001;
    try p.upload(&t, &h, 0);
    const gpa = std.testing.allocator;
    const up = try gpa.dupe(f32, p.up.items);
    defer gpa.free(up);
    const down = try gpa.dupe(f32, p.down.items);
    defer gpa.free(down);
    const used = try gpa.dupe(usize, p.used.items);
    defer gpa.free(used);
    const learned = try gpa.dupe(pocket.Learned, p.learned.items);
    defer gpa.free(learned);
    const live = h;
    try std.testing.expectEqual(@as(usize, 2), try forgetting.forget(&p, &t, &h, &b_neurons));
    try std.testing.expectEqual(@as(usize, 2), try forgetting.unforget(&p, &t, &h));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(up), std.mem.sliceAsBytes(p.up.items));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(down), std.mem.sliceAsBytes(p.down.items));
    try std.testing.expectEqualSlices(usize, used, p.used.items);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(learned), std.mem.sliceAsBytes(p.learned.items));
    try std.testing.expectEqual(@as(usize, 4), p.learned_before);
    try std.testing.expectEqual(@as(usize, 0), p.retired_lessons.items.len);
    for (0..layers - 1) |l| {
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&live.up[l]), std.mem.sliceAsBytes(&h.up[l]));
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&live.down[l]), std.mem.sliceAsBytes(&h.down[l]));
        try std.testing.expectEqual(live.used[l], h.used[l]);
    }
    try std.testing.expect(!forgetting.pending(&p));
    // with nothing to put back, a second put-back changes nothing
    const sent = h.uploads;
    try std.testing.expectEqual(@as(usize, 0), try forgetting.unforget(&p, &t, &h));
    try std.testing.expectEqual(sent, h.uploads);
    try std.testing.expectEqual(@as(usize, 6), p.learned.items.len);
}

test "a replaced fact's neurons are zero already: nothing to delete, nothing changed or kept" {
    var t = trainer();
    var h: Device = .{};
    var p: P = .{ .gpa = std.testing.allocator, .mode = .auto };
    defer p.deinit();
    try three(&h, &p, &t);
    // a newer fact replaced A: its neurons zeroed where they sit, their places taken
    try p.retire(&t, &h, &.{ .{ 0, 0 }, .{ 1, 0 } });
    const gpa = std.testing.allocator;
    const up = try gpa.dupe(f32, p.up.items);
    defer gpa.free(up);
    const down = try gpa.dupe(f32, p.down.items);
    defer gpa.free(down);
    const sent = h.uploads;
    try std.testing.expectEqual(@as(usize, 0), try forgetting.forget(&p, &t, &h, &.{ .{ 0, 0 }, .{ 1, 0 } }));
    try std.testing.expect(!forgetting.pending(&p));
    try std.testing.expectEqual(sent, h.uploads);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(up), std.mem.sliceAsBytes(p.up.items));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(down), std.mem.sliceAsBytes(p.down.items));
    try std.testing.expectEqual(@as(usize, 6), p.learned.items.len);
}

test "deleting every fact empties the pocket, and the delete still waits for a save to write it" {
    var t = trainer();
    var h: Device = .{};
    var p: P = .{ .gpa = std.testing.allocator, .mode = .auto };
    defer p.deinit();
    try three(&h, &p, &t);
    try std.testing.expectEqual(@as(usize, 6), try forgetting.forget(&p, &t, &h, &.{ .{ 0, 0 }, .{ 1, 0 }, .{ 0, 1 }, .{ 2, 0 }, .{ 0, 2 }, .{ 1, 1 } }));
    try std.testing.expect(p.empty());
    try std.testing.expect(forgetting.pending(&p));
    for (0..layers - 1) |l| try std.testing.expect(std.mem.allEqual(f32, &h.up[l], 0));
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, p.retired_lessons.items);
    try std.testing.expectEqual(@as(usize, 0), p.learned_before);
}

test "a save after the last fact is deleted writes the emptied pocket; an untaught one stays unwritten" {
    var t = trainer();
    var h: Device = .{};
    // a model never taught: nothing in the pocket, nothing written
    var untaught: Learner = .{ .trainer = &t, .b = &h };
    defer untaught.pockets.deinit();
    try untaught.pockets.load(&t, &h);
    try std.testing.expect(try whole.save(&untaught));
    try std.testing.expectEqual(@as(usize, 0), h.saves);
    var l: Learner = .{ .trainer = &t, .b = &h };
    defer l.pockets.deinit();
    try three(&h, &l.pockets, &t);
    _ = try forgetting.forget(&l.pockets, &t, &h, &.{ .{ 0, 0 }, .{ 1, 0 }, .{ 0, 1 }, .{ 2, 0 }, .{ 0, 2 }, .{ 1, 1 } });
    try std.testing.expect(try whole.save(&l));
    try std.testing.expectEqual(@as(usize, 1), h.saves);
    try std.testing.expect(!forgetting.pending(&l.pockets));
}

/// The pocket's state a delete may change, copied: its weights, counts, records and retired lessons.
const Copy = struct {
    up: []f32,
    down: []f32,
    used: []usize,
    learned: []pocket.Learned,
    learned_before: usize,
    retired: []u32,

    fn of(p: *const P) !Copy {
        const a = std.testing.allocator;
        return .{ .up = try a.dupe(f32, p.up.items), .down = try a.dupe(f32, p.down.items), .used = try a.dupe(usize, p.used.items), .learned = try a.dupe(pocket.Learned, p.learned.items), .learned_before = p.learned_before, .retired = try a.dupe(u32, p.retired_lessons.items) };
    }

    fn expectSame(c: Copy, p: *const P) !void {
        try c.expectFields(p);
        try std.testing.expect(!forgetting.pending(p));
    }

    /// The pocket in memory as copied, its snapshot kept or not.
    fn expectFields(c: Copy, p: *const P) !void {
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(c.up), std.mem.sliceAsBytes(p.up.items));
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(c.down), std.mem.sliceAsBytes(p.down.items));
        try std.testing.expectEqualSlices(usize, c.used, p.used.items);
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(c.learned), std.mem.sliceAsBytes(p.learned.items));
        try std.testing.expectEqual(c.learned_before, p.learned_before);
        try std.testing.expectEqualSlices(u32, c.retired, p.retired_lessons.items);
    }

    fn free(c: Copy) void {
        const a = std.testing.allocator;
        a.free(c.up);
        a.free(c.down);
        a.free(c.used);
        a.free(c.learned);
        a.free(c.retired);
    }
};

/// The forward's layers as the device holds them, for the pocket's slots.
fn expectLive(before: *const Device, now: *const Device) !void {
    for (0..layers - 1) |l| {
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&before.up[l]), std.mem.sliceAsBytes(&now.up[l]));
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&before.down[l]), std.mem.sliceAsBytes(&now.down[l]));
        try std.testing.expectEqual(before.used[l], now.used[l]);
    }
}

test "an allocation or an upload that fails anywhere in a delete leaves the pocket and the forward as they were" {
    var t = trainer();
    // each allocation of the delete in turn fails (the pocket's, the rounding's among them), until it goes through
    var failures: usize = 0;
    while (true) : (failures += 1) {
        var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{});
        const h = try std.testing.allocator.create(Device);
        defer std.testing.allocator.destroy(h);
        h.* = .{};
        var p: P = .{ .gpa = failing.allocator(), .mode = .auto };
        defer p.deinit();
        try three(h, &p, &t);
        const before = try Copy.of(&p);
        defer before.free();
        const live = try std.testing.allocator.create(Device);
        defer std.testing.allocator.destroy(live);
        live.* = h.*;
        failing.fail_index = failing.alloc_index + failures;
        if (forgetting.forget(&p, &t, h, &b_neurons)) |n| {
            try std.testing.expectEqual(@as(usize, 2), n);
            break;
        } else |e| try std.testing.expectEqual(error.OutOfMemory, e);
        try before.expectSame(&p);
        try expectLive(live, h);
    }
    try std.testing.expect(failures >= 5);
    // an upload that fails once, at either changed layer: the pocket and the forward put back
    for ([_]usize{ 1, 2 }) |at| {
        const h = try std.testing.allocator.create(Device);
        defer std.testing.allocator.destroy(h);
        h.* = .{};
        var p: P = .{ .gpa = std.testing.allocator, .mode = .auto };
        defer p.deinit();
        try three(h, &p, &t);
        const before = try Copy.of(&p);
        defer before.free();
        const live = try std.testing.allocator.create(Device);
        defer std.testing.allocator.destroy(live);
        live.* = h.*;
        h.fail_at = h.uploads + at;
        try std.testing.expectError(error.UploadFailed, forgetting.forget(&p, &t, h, &b_neurons));
        try before.expectSame(&p);
        try expectLive(live, h);
        // and a retry goes through
        try std.testing.expectEqual(@as(usize, 2), try forgetting.forget(&p, &t, h, &b_neurons));
    }
    // a device that takes no upload: put back in memory, the snapshot kept and the pocket stale until one lands
    const h = try std.testing.allocator.create(Device);
    defer std.testing.allocator.destroy(h);
    h.* = .{};
    var p: P = .{ .gpa = std.testing.allocator, .mode = .auto };
    defer p.deinit();
    try three(h, &p, &t);
    const before = try Copy.of(&p);
    defer before.free();
    const live = try std.testing.allocator.create(Device);
    defer std.testing.allocator.destroy(live);
    live.* = h.*;
    h.failing = true;
    try std.testing.expectError(error.PutBackNotOnDevice, forgetting.forget(&p, &t, h, &b_neurons));
    try before.expectFields(&p);
    try std.testing.expect(forgetting.stale(&p) and forgetting.pending(&p));
    h.failing = false;
    try std.testing.expectEqual(@as(usize, 2), try forgetting.unforget(&p, &t, h));
    try before.expectSame(&p);
    try expectLive(live, h);
}

test "a put-back the device stops taking after one layer is cut keeps its snapshot, refuses, and lands once it can" {
    var t = trainer();
    const h = try std.testing.allocator.create(Device);
    defer std.testing.allocator.destroy(h);
    h.* = .{};
    var l: Learner = .{ .trainer = &t, .b = h };
    defer l.pockets.deinit();
    try three(h, &l.pockets, &t);
    const before = try Copy.of(&l.pockets);
    defer before.free();
    const live = try std.testing.allocator.create(Device);
    defer std.testing.allocator.destroy(live);
    live.* = h.*;
    // the first changed layer (0) takes its cut; from the second upload on, the device takes nothing, put-back too
    h.fail_from = h.uploads + 2;
    try std.testing.expectError(error.PutBackNotOnDevice, forgetting.forget(&l.pockets, &t, h, &b_neurons));
    // memory is as it was and the snapshot is kept, while the device still holds layer 0 cut and layer 2 as it was
    try before.expectFields(&l.pockets);
    try std.testing.expect(whole.stale(&l) and forgetting.pending(&l.pockets));
    try std.testing.expect(std.mem.allEqual(f32, h.up[0][1 * d ..][0..d], 0));
    try std.testing.expect(!std.mem.allEqual(f32, live.up[0][1 * d ..][0..d], 0));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&live.up[2]), std.mem.sliceAsBytes(&h.up[2]));
    // until it lands, another delete is refused and a retry fails the same way, the snapshot kept
    try std.testing.expectError(error.PutBackNotOnDevice, forgetting.forget(&l.pockets, &t, h, &b_neurons));
    try std.testing.expectError(error.PutBackNotOnDevice, whole.heal(&l));
    try std.testing.expect(whole.stale(&l));
    try before.expectFields(&l.pockets);
    // the device takes uploads again: the retry lands every layer, and nothing is left to put back
    h.fail_from = 0;
    try whole.heal(&l);
    try std.testing.expect(!whole.stale(&l));
    try before.expectSame(&l.pockets);
    try expectLive(live, h);
}

test "a delete whose cut layer would round to other values when saved is put back whole, nothing changed" {
    var t = trainer();
    var h: Device = .{};
    var p: P = .{ .gpa = std.testing.allocator, .mode = .auto };
    defer p.deinit();
    try three(&h, &p, &t);
    var prng = std.Random.DefaultPrng.init(3);
    const r = prng.random();
    // slot 0's neurons far from zero next to their spread, until its cut layer would move when a save rounds it again
    var cut_up: [width]f32 = undefined;
    var cut_down: [width]f32 = undefined;
    var moves = false;
    for (0..20) |_| {
        const spread = std.math.pow(f32, 10, -4 + 6 * r.float(f32));
        const offset = spread * 256 * r.float(f32) * @as(f32, if (r.boolean()) 1 else -1);
        for (p.up.items[0..width], p.down.items[0..width]) |*u, *v| {
            u.* = offset + spread * r.floatNorm(f32);
            v.* = -offset + spread * r.floatNorm(f32);
        }
        @memcpy(&cut_up, p.up.items[0..width]);
        @memcpy(&cut_down, p.down.items[0..width]);
        @memset(cut_up[1 * d ..][0..d], 0); // B's neuron at slot 0
        try round(&cut_up, &cut_down);
        const once_up = cut_up;
        const once_down = cut_down;
        try round(&cut_up, &cut_down);
        moves = !std.mem.eql(u8, std.mem.sliceAsBytes(&once_up), std.mem.sliceAsBytes(&cut_up)) or !std.mem.eql(u8, std.mem.sliceAsBytes(&once_down), std.mem.sliceAsBytes(&cut_down));
        if (moves) break;
    }
    try std.testing.expect(moves);
    try p.upload(&t, &h, 0);
    const gpa = std.testing.allocator;
    const up = try gpa.dupe(f32, p.up.items);
    defer gpa.free(up);
    const down = try gpa.dupe(f32, p.down.items);
    defer gpa.free(down);
    const live = h;
    try std.testing.expectError(error.LayerRoundsAgain, forgetting.forget(&p, &t, &h, &b_neurons));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(up), std.mem.sliceAsBytes(p.up.items));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(down), std.mem.sliceAsBytes(p.down.items));
    for (0..layers - 1) |l| try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&live.up[l]), std.mem.sliceAsBytes(&h.up[l]));
    try std.testing.expect(!forgetting.pending(&p));
    try std.testing.expectEqual(@as(usize, 6), p.learned.items.len);
}
