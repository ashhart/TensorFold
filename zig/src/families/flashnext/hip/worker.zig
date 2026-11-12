//! A tensor-parallel rank above 0: runs the forwards rank 0 sends in its order, so the collectives pair up.
const std = @import("std");
const hip = @import("hip");
const Engine = @import("engine.zig").Engine;
const cache = @import("cache.zig");
const forward = @import("forward.zig");
const snapshot = @import("snapshot.zig");

/// A message's first word; rank 0 decides everything, a rank only applies it.
pub const Op = enum(u32) { stop, begin, fill, verify, keep, release, snap, restore, drop };

/// A message's step once it holds every fixed word that step reads: no word from the link is an index or tag before.
pub fn step(w: []const u32) error{ EmptyMessage, ShortMessage, UnknownStep }!Op {
    if (w.len == 0) return error.EmptyMessage;
    const op = std.enums.fromInt(Op, w[0]) orelse return error.UnknownStep;
    const need: usize = switch (op) {
        .stop => 1,
        .release, .drop => 2,
        .begin, .verify, .restore => 3,
        .fill, .keep, .snap => 4,
    };
    if (w.len < need) return error.ShortMessage;
    return op;
}

const Lane = struct {
    caches: cache.Caches,
    pending: [cache.max_window]u32 = undefined,
};

/// Rank 0's messages until it says stop: a stream begun, a prompt chunk, a window, a keep, a release.
pub fn follow(gpa: std.mem.Allocator, e: *Engine, link: hip.collective.Link) !void {
    var lanes: std.AutoHashMapUnmanaged(u32, *Lane) = .empty;
    defer {
        var it = lanes.valueIterator();
        while (it.next()) |l| {
            e.unlease(l.*.caches);
            gpa.destroy(l.*);
        }
        lanes.deinit(gpa);
    }
    var msg: std.ArrayList(u32) = .empty;
    defer msg.deinit(gpa);
    // rank 0's kept prompt states: this rank's own copy of each, under the slot rank 0 names
    var saved: std.AutoHashMapUnmanaged(u32, *snapshot.Saved) = .empty;
    defer {
        var it = saved.valueIterator();
        while (it.next()) |x| snapshot.drop(gpa, x.*);
        saved.deinit(gpa);
    }
    const ratio = e.cfg.index_ratio;
    const d = e.weights.model.dims;
    while (true) {
        msg.clearRetainingCapacity();
        try link.recv(gpa, &msg);
        const w = msg.items;
        switch (try step(w)) {
            .stop => return,
            .begin => {
                const lane = try gpa.create(Lane);
                errdefer gpa.destroy(lane);
                if (lanes.fetchRemove(w[1])) |kv| {
                    e.unlease(kv.value.caches);
                    gpa.destroy(kv.value);
                }
                lane.* = .{ .caches = try e.lease(w[2]) };
                errdefer e.unlease(lane.caches);
                try lanes.put(gpa, w[1], lane);
            },
            .fill => {
                const lane = lanes.get(w[1]) orelse return error.UnknownStream;
                const n = w[3];
                if (n == 0 or n > w.len - 4) return error.ShortMessage;
                const out = try e.prompt(&lane.caches, w[4..][0..n]);
                if (w[2] != 0) _ = try e.logits(e.ops(&e.arena), forward.at(out.mixed, (n - 1) * d.hidden), 1);
                try e.stream.synchronize();
            },
            .verify => {
                var segs: [forward.max_segs]forward.Seg = undefined;
                const count = w[1];
                if (count > segs.len) return error.TooManyWindows;
                const pick = std.enums.fromInt(@import("engine.zig").Pick, w[2]) orelse return error.UnknownStep;
                var at: usize = 3;
                for (segs[0..count]) |*g| {
                    if (w.len - at < 2) return error.ShortMessage;
                    const lane = lanes.get(w[at]) orelse return error.UnknownStream;
                    const n = w[at + 1];
                    if (n > lane.pending.len or n > w.len - at - 2) return error.ShortMessage;
                    @memcpy(lane.pending[0..n], w[at + 2 ..][0..n]);
                    g.* = .{ .c = &lane.caches, .tokens = lane.pending[0..n], .pos = lane.caches.len };
                    at += 2 + n;
                }
                // every rank meets the same shapes in the same order: its own choice is rank 0's
                _ = try e.choose(segs[0..count]);
                _ = try e.round(segs[0..count], pick);
                try e.stream.synchronize();
            },
            .keep => {
                const lane = lanes.get(w[1]) orelse return error.UnknownStream;
                if (w[3] > lane.pending.len) return error.ShortMessage;
                try e.keepWindow(&lane.caches, lane.pending[0..w[3]], w[2]);
            },
            .snap => {
                const lane = lanes.get(w[1]) orelse return error.UnknownStream;
                // a copy this rank cannot keep is a restore it refuses later, which every rank then hears
                const x = snapshot.save(gpa, &e.driver, e.stream, &lane.caches, null, w[3], ratio) catch continue;
                saved.put(gpa, w[2], x) catch snapshot.drop(gpa, x);
            },
            .restore => {
                const lane = lanes.get(w[1]) orelse return error.UnknownStream;
                // every rank resumes from the state, or every rank starts the prompt over
                snapshot.restoreAgreed(
                    gpa,
                    &e.driver,
                    e.stream,
                    saved.get(w[2]),
                    &lane.caches,
                    null,
                    ratio,
                    e.eos,
                    link,
                ) catch |err| if (err != error.RestoreRefused) return err;
            },
            .drop => if (saved.fetchRemove(w[1])) |kv| snapshot.drop(gpa, kv.value),
            .release => if (lanes.fetchRemove(w[1])) |kv| {
                try e.stream.synchronize();
                e.unlease(kv.value.caches);
                gpa.destroy(kv.value);
            },
        }
    }
}

test "security: a step from the link is refused unless its message holds the words it reads" {
    const t = std.testing;
    try t.expectError(error.EmptyMessage, step(&.{}));
    try t.expectError(error.UnknownStep, step(&.{9}));
    try t.expectError(error.UnknownStep, step(&.{0xffff_ffff}));
    try t.expectEqual(Op.stop, try step(&.{0}));
    const needs = [_]struct { Op, usize }{
        .{ .release, 2 }, .{ .drop, 2 }, .{ .begin, 3 }, .{ .verify, 3 }, .{ .restore, 3 }, .{ .fill, 4 }, .{ .keep, 4 }, .{ .snap, 4 },
    };
    var words = [_]u32{ 0, 1, 2, 3 };
    for (needs) |need| {
        words[0] = @backingInt(need[0]);
        try t.expectError(error.ShortMessage, step(words[0 .. need[1] - 1]));
        try t.expectEqual(need[0], try step(words[0..need[1]]));
    }
}
