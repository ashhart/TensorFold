//! A family's passes as CUDA graphs keyed on their host values: the same key replays, new values update in place.
const std = @import("std");
const graph = @import("graph.zig");
const Stream = @import("stream.zig").Stream;

/// A key's two hashes: `shape` decides which kernels launch (the graph's slot), `moving` holds every other host value.
pub const Hash = struct {
    shape: std.hash.Wyhash = .init(0),
    moving: std.hash.Wyhash = .init(0),

    pub fn fixed(h: *Hash, v: u64) void {
        h.shape.update(std.mem.asBytes(&v));
    }

    /// Values a held graph takes in place (buffers, lengths); whether each is 0 stays a shape.
    pub fn values(h: *Hash, vs: []const u64) void {
        for (vs) |v| {
            h.moving.update(std.mem.asBytes(&v));
            h.fixed(@intFromBool(v != 0));
        }
    }
};

/// A key without what moves between runs of one held graph (`moving`): the graph's slot.
pub fn slotOf(key: anytype) @TypeOf(key) {
    var slot = key;
    slot.moving = 0;
    return slot;
}

/// Held graphs, each under its slot; `K` has a `moving: u64` field and compares with std.meta.eql.
pub fn Cache(comptime K: type, comptime max_graphs: usize) type {
    return struct {
        const Self = @This();
        /// A held graph and the whole key it was last recorded for.
        const Held = struct { key: K, exec: graph.Exec };
        map: std.AutoHashMapUnmanaged(K, Held) = .empty,
        /// Slots run once eagerly and are captured the second time (a one-off pass would never pay its capture back).
        seen: std.AutoHashMapUnmanaged(K, void) = .empty,
        captures: u64 = 0,
        replays: u64 = 0,
        /// Held graphs recorded again with new values and updated in place.
        updates: u64 = 0,
        /// A capture failed: every later pass runs eagerly (logged once).
        off: bool = false,

        pub fn clear(c: *Self, gpa: std.mem.Allocator) void {
            var it = c.map.valueIterator();
            while (it.next()) |h| h.exec.deinit();
            c.map.clearAndFree(gpa);
            c.seen.clearAndFree(gpa);
        }

        /// `pass.run()` (launches only: its uploads already queued) as `key`'s graph, captured first when new.
        pub fn run(c: *Self, gpa: std.mem.Allocator, stream: Stream, key: K, pass: anytype) !void {
            if (c.off) return pass.run();
            const slot = slotOf(key);
            if (c.map.getPtr(slot)) |h| {
                if (std.meta.eql(h.key, key)) {
                    c.replays += 1;
                    return h.exec.launchOn(stream);
                }
                refresh(stream, pass, &h.exec) catch |err| return c.fail(err, pass);
                h.key = key;
                c.updates += 1;
                return h.exec.launchOn(stream);
            }
            if (c.map.count() >= max_graphs or c.seen.count() >= 4 * max_graphs) c.clear(gpa);
            if (!c.seen.contains(slot)) {
                try c.seen.put(gpa, slot, {});
                return pass.run();
            }
            var exec = capture(stream, pass) catch |err| return c.fail(err, pass);
            errdefer exec.deinit();
            try c.map.put(gpa, slot, .{ .key = key, .exec = exec });
            c.captures += 1;
            if (c.captures % 64 == 1) std.log.info("{s} graphs: {d} captured, {d} updated, {d} replayed, {d} held", .{ @typeName(K), c.captures, c.updates, c.replays, c.map.count() });
            try exec.launchOn(stream);
        }

        /// The stream is out of capture again: this pass eagerly, and the ones after it.
        fn fail(c: *Self, err: anyerror, pass: anytype) !void {
            std.log.warn("{s} graphs off: capture failed ({s}), eager from here", .{ @typeName(K), @errorName(err) });
            c.off = true;
            return pass.run();
        }
    };
}

/// The pass's launches recorded into a graph, nothing launched; the stream left uncaptured.
fn record(stream: Stream, pass: anytype) !graph.Graph {
    try graph.beginCapture(stream, .thread_local);
    pass.run() catch |err| {
        if (graph.endCapture(stream)) |g| {
            var discarded = g;
            discarded.deinit();
        } else |_| {}
        return err;
    };
    return graph.endCapture(stream);
}

/// The pass's launches captured and instantiated (uploaded to the device), nothing launched.
fn capture(stream: Stream, pass: anytype) !graph.Exec {
    var g = try record(stream, pass);
    defer g.deinit();
    var exec = try g.instantiate();
    errdefer exec.deinit();
    try exec.upload(stream);
    return exec;
}

/// `exec` with the pass's values recorded now; a pass of other launches (another topology) gets a new executable.
fn refresh(stream: Stream, pass: anytype, exec: *graph.Exec) !void {
    var g = try record(stream, pass);
    defer g.deinit();
    if (try exec.update(g) == .success) return;
    var fresh = try g.instantiate();
    errdefer fresh.deinit();
    try fresh.upload(stream);
    exec.deinit();
    exec.* = fresh;
}
