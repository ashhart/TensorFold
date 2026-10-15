//! A delete on a pocket of three facts: their neurons zeroed and rounded, freed at the top, put back bit for bit.
const std = @import("std");
const cfg = @import("config.zig");
const pocket = @import("pocket.zig");
const forgetting = @import("pocket_forget.zig");
const whole = @import("whole.zig");
const examples = @import("lesson_examples.zig");
const Shards = @import("pocket_shards.zig").Io(struct {
    pub const shard_edit = @import("../../core/shard_edit.zig");
    pub const safetensors = @import("../../core/safetensors.zig");
    pub const affine4_host = @import("../../core/affine4_host.zig");
});

const d = 64;
const layers = 4; // every MoE layer but the last holds a pocket: slots 0, 1 and 2
const width = pocket.neurons * d;

/// The trainer as the pocket reads it: only the model's shape.
const Trainer = struct {
    c: cfg.Config,

    pub fn config(t: *const Trainer) cfg.Config {
        return t.c;
    }
};

const P = pocket.Pocket(Trainer);

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
    gpa: std.mem.Allocator = std.testing.allocator, // what rounding allocates from
    fail_at: usize = 0, // the upload, counted from one, that fails once (0: none)
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
        if (h.failing or h.uploads + 1 == h.fail_at) {
            h.uploads += 1;
            return error.UploadFailed;
        }
        @memcpy(&h.up[l], up);
        @memcpy(&h.down[l], down);
        h.used[l] = used;
        h.uploads += 1;
    }

    pub fn pocketRound(h: *Device, up: []f32, down: []f32) !void {
        return Shards.round(h.gpa, d, up, down);
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
    for (0..layers) |l| try Shards.round(std.testing.allocator, d, &h.loaded_up[l], &h.loaded_down[l]);
    try p.load(t, h);
    for (held, 0..) |x, i| try p.learned.append(p.gpa, .{ .slot = x[0], .neuron = x[1], .key = .reply, .start = i, .count = 1, .subject = 0, .lesson = @intCast(x[2]) });
    p.learned_before = 4; // C, the last lesson, found A's and B's
}

fn trainer() Trainer {
    var c: cfg.Config = .{ .hidden = d, .vocab = 16, .layers = layers, .mamba_heads = 1, .mamba_head_dim = 1, .groups = 1, .state = 1, .conv_kernel = 4, .heads = 1, .kv_heads = 1, .head_dim = 1, .experts = 1, .top_k = 1, .expert_width = 64, .shared_width = 64, .routed_scaling = 1, .eps = 1e-5 };
    for (0..layers) |l| c.kinds[l] = .moe;
    return .{ .c = c };
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
    try Shards.round(std.testing.allocator, d, &want_up, &want_down);
    try std.testing.expectEqual(@as(usize, 2), try forgetting.forget(&p, &t, &h, &b_neurons));
    try std.testing.expect(forgetting.pending(&p));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&want_up), std.mem.sliceAsBytes(p.up.items[0..width]));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&want_down), std.mem.sliceAsBytes(p.down.items[0..width]));
    try std.testing.expect(std.mem.allEqual(f32, p.neuron(p.up.items, 2, 0, d), 0));
    // layer 2 as saved: B's input weights zero, every output bit as it was, B's own too, so no 4-bit group moves
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&h.loaded_down[2]), std.mem.sliceAsBytes(p.down.items[2 * width ..][0..width]));
    // A's and C's input weights keep every bit; layer 1 was not touched at all
    for ([_][2]usize{ .{ 0, 0 }, .{ 0, 2 }, .{ 1, 0 }, .{ 1, 1 } }) |x| try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(before_up[(x[0] * pocket.neurons + x[1]) * d ..][0..d]), std.mem.sliceAsBytes(p.neuron(p.up.items, x[0], x[1], d)));
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
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(c.up), std.mem.sliceAsBytes(p.up.items));
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(c.down), std.mem.sliceAsBytes(p.down.items));
        try std.testing.expectEqualSlices(usize, c.used, p.used.items);
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(c.learned), std.mem.sliceAsBytes(p.learned.items));
        try std.testing.expectEqual(c.learned_before, p.learned_before);
        try std.testing.expectEqualSlices(u32, c.retired, p.retired_lessons.items);
        try std.testing.expect(!forgetting.pending(p));
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
    // each allocation of the delete in turn fails (the pocket's and the rounding's), until it goes through
    var failures: usize = 0;
    while (true) : (failures += 1) {
        var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{});
        const h = try std.testing.allocator.create(Device);
        defer std.testing.allocator.destroy(h);
        h.* = .{ .gpa = failing.allocator() };
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
    // a device that takes no upload: the pocket is put back, and the error is the first one
    const h = try std.testing.allocator.create(Device);
    defer std.testing.allocator.destroy(h);
    h.* = .{};
    var p: P = .{ .gpa = std.testing.allocator, .mode = .auto };
    defer p.deinit();
    try three(h, &p, &t);
    const before = try Copy.of(&p);
    defer before.free();
    h.failing = true;
    try std.testing.expectError(error.UploadFailed, forgetting.forget(&p, &t, h, &b_neurons));
    try before.expectSame(&p);
}
