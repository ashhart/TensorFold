//! Captured graphs by shape: a launch sequence recorded the second time its shape is met, replayed on every later one.

const std = @import("std");
const hip_graph = @import("graph.zig");
const Stream = @import("stream.zig").Stream;

/// Shapes kept at once; the least recently used one is dropped for a new one.
const max_entries = 64;

pub const State = enum { seen, ready, failed };

/// A cache of graphs keyed by `Key` (plain integers and booleans), each with the `Out` its capture produced.
pub fn Cache(comptime Key: type, comptime Out: type) type {
    return CacheOf(Key, Out, hip_graph.Exec);
}

/// The same over any replayable `Exec` (a graph, or a step's graphs between collectives): it has `deinit`.
pub fn CacheOf(comptime Key: type, comptime Out: type, comptime Exec: type) type {
    return struct {
        const Self = @This();

        pub const Entry = struct {
            key: Key,
            state: State = .seen,
            exec: ?Exec = null,
            out: Out = undefined,
            tick: u64 = 0,
        };

        gpa: std.mem.Allocator,
        entries: std.ArrayList(Entry) = .empty,
        clock: u64 = 0,
        /// Launches run, replayed and captured; the host's nanoseconds from a launch's start to its work queued.
        rounds: u64 = 0,
        submit_ns: u64 = 0,
        /// The host's nanoseconds recording and instantiating graphs.
        capture_ns: u64 = 0,
        replayed: u64 = 0,
        captured: u64 = 0,

        pub fn init(gpa: std.mem.Allocator) Self {
            return .{ .gpa = gpa };
        }

        pub fn deinit(g: *Self) void {
            for (g.entries.items) |*e| if (e.exec) |*x| x.deinit();
            g.entries.deinit(g.gpa);
        }

        /// The entry for `key`, made (state `seen`) if new.
        pub fn find(g: *Self, key: Key) !*Entry {
            g.clock += 1;
            for (g.entries.items) |*e| if (std.meta.eql(e.key, key)) {
                e.tick = g.clock;
                return e;
            };
            if (g.entries.items.len >= max_entries) {
                var oldest: usize = 0;
                for (g.entries.items, 0..) |e, i| if (e.tick < g.entries.items[oldest].tick) {
                    oldest = i;
                };
                if (g.entries.items[oldest].exec) |*x| x.deinit();
                _ = g.entries.swapRemove(oldest);
            }
            try g.entries.append(g.gpa, .{ .key = key, .tick = g.clock });
            return &g.entries.items[g.entries.items.len - 1];
        }

        /// Drops a shape's graph (a rank could not capture it): the shape runs eagerly from now on.
        pub fn revoke(g: *Self, e: *Entry) void {
            if (e.exec) |*x| x.deinit();
            e.exec = null;
            e.state = .failed;
            if (g.captured > 0) g.captured -= 1;
        }

        /// `body(ctx)` recorded on `stream`: its outputs and the graph (nothing ran), or null when the capture failed.
        pub fn record(stream: Stream, ctx: anytype, comptime body: fn (@TypeOf(ctx)) anyerror!Out) !?struct { out: Out, graph: hip_graph.Graph } {
            hip_graph.beginCapture(stream, .thread_local) catch return null;
            const recorded = body(ctx);
            const graph = hip_graph.endCapture(stream) catch return null;
            const out = recorded catch |err| {
                var g = graph;
                g.deinit();
                return if (err == error.OutOfDeviceMemory) err else null;
            };
            return .{ .out = out, .graph = graph };
        }

        /// Keeps `exec`, made by the caller, as the shape's replay with its outputs.
        pub fn hold(g: *Self, e: *Entry, exec: Exec, out: Out) void {
            e.exec = exec;
            e.out = out;
            e.state = .ready;
            g.captured += 1;
        }

        /// Instantiates a capture that just ended, keeps it with its outputs, and uploads it.
        pub fn keep(g: *Self, e: *Entry, graph: hip_graph.Graph, stream: Stream, out: Out) !void {
            var tmp = graph;
            defer tmp.deinit();
            errdefer e.state = .failed;
            var exec = try tmp.instantiate();
            errdefer exec.deinit();
            try exec.upload(stream);
            e.exec = exec;
            e.out = out;
            e.state = .ready;
            g.captured += 1;
        }
    };
}

test "a held replay is the shape's until the cache drops it" {
    const Fake = struct {
        drops: *usize,
        fn deinit(f: *@This()) void {
            f.drops.* += 1;
        }
    };
    var drops: usize = 0;
    var c = CacheOf(u32, u8, Fake).init(std.testing.allocator);
    const e = try c.find(7);
    c.hold(e, .{ .drops = &drops }, 3);
    const again = try c.find(7);
    try std.testing.expectEqual(State.ready, again.state);
    try std.testing.expectEqual(@as(u8, 3), again.out);
    c.revoke(again);
    try std.testing.expectEqual(@as(usize, 1), drops);
    try std.testing.expectEqual(State.failed, (try c.find(7)).state);
    c.deinit();
}
