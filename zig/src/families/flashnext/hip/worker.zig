//! A tensor-parallel rank above 0: runs the forwards rank 0 sends in its order, so the collectives pair up.
const std = @import("std");
const hip = @import("hip");
const Engine = @import("engine.zig").Engine;
const cache = @import("cache.zig");
const forward = @import("forward.zig");
const snapshot = @import("snapshot.zig");

/// A message's first word; rank 0 decides everything, a rank only applies it.
pub const Op = enum(u32) { stop, begin, fill, verify, keep, release, snap, restore, drop };

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
        if (w.len == 0) return error.EmptyMessage;
        switch (@as(Op, @fromBackingInt(@intCast(w[0])))) {
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
                const out = try e.prompt(&lane.caches, w[4..][0..n]);
                if (w[2] != 0) _ = try e.logits(e.ops(&e.arena), forward.at(out.mixed, (n - 1) * d.hidden), 1);
                try e.stream.synchronize();
            },
            .verify => {
                var segs: [forward.max_segs]forward.Seg = undefined;
                const count = w[1];
                const pick: @import("engine.zig").Pick = @fromBackingInt(w[2]);
                var at: usize = 3;
                for (segs[0..count]) |*g| {
                    const lane = lanes.get(w[at]) orelse return error.UnknownStream;
                    const n = w[at + 1];
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
                try e.keepWindow(&lane.caches, lane.pending[0..w[3]], w[2]);
            },
            .snap => {
                const lane = lanes.get(w[1]) orelse return error.UnknownStream;
                const x = try snapshot.save(gpa, &e.driver, e.stream, &lane.caches, null, w[3], ratio);
                try saved.put(gpa, w[2], x);
            },
            .restore => {
                const lane = lanes.get(w[1]) orelse return error.UnknownStream;
                try snapshot.restore(gpa, &e.driver, e.stream, saved.get(w[2]) orelse return error.UnknownSnapshot, &lane.caches, null, ratio);
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
