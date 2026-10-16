//! Captured lane rounds: recorded the second time their key comes, as graphs cut at the collectives, then replayed.
const std = @import("std");
const hip = @import("hip");
const round = @import("round.zig");

/// Keys met once and not captured that are remembered: a key is captured when it comes back while remembered.
const remembered = 256;

/// What a lone stream's round takes from the host beside the plan: the span, its rows, Mamba states and snapshots.
pub const Key = struct { span: u32, width: u32, states: u64, snaps: u64 };

/// What a replay hands back: the round's final-normed rows and where the arena stands after the forward.
pub const Out = struct { hidden: hip.ops.Tensor, used: usize };

pub const Cache = hip.graph_cache.CacheOf(Key, Out, hip.graph_split.Split);

/// A lone stream's round key; shared rounds run eagerly (their keys seldom return: a capture would cost more).
pub fn keyOf(rows: []const round.Rows, span: usize) ?Key {
    if (rows.len != 1) return null;
    const r = rows[0];
    return .{
        .span = @intCast(span),
        .width = @intCast(r.tokens.len),
        .states = r.s.mem.base(),
        .snaps = if (r.snaps) |sn| sn.buf.base() else 0,
    };
}

pub const Graphs = struct {
    cache: Cache,
    met: [remembered]Key = undefined,
    met_n: usize = 0,
    met_at: usize = 0,

    pub fn init(gpa: std.mem.Allocator) Graphs {
        return .{ .cache = .init(gpa) };
    }

    pub fn deinit(g: *Graphs) void {
        const c = &g.cache;
        if (c.captured > 0) std.log.info(
            "round graphs: {d} captured ({d:.1} ms each), {d} of {d} rounds replayed",
            .{
                c.captured,
                @as(f64, @floatFromInt(c.capture_ns)) / @as(f64, @floatFromInt(c.captured)) / 1e6,
                c.replayed,
                c.rounds,
            },
        );
        c.deinit();
    }

    /// The key's captured (or failed) entry, if it has one.
    pub fn lookup(g: *Graphs, key: Key) ?*Cache.Entry {
        for (g.cache.entries.items) |*e| if (std.meta.eql(e.key, key)) {
            g.cache.clock += 1;
            e.tick = g.cache.clock;
            return e;
        };
        return null;
    }

    /// Whether `key` came before (and is forgotten now, its capture due); else it is remembered.
    pub fn again(g: *Graphs, key: Key) bool {
        for (g.met[0..g.met_n], 0..) |k, i| if (std.meta.eql(k, key)) {
            g.met_n -= 1;
            g.met[i] = g.met[g.met_n];
            if (g.met_at > g.met_n) g.met_at = g.met_n;
            return true;
        };
        if (g.met_n < remembered) {
            g.met[g.met_n] = key;
            g.met_n += 1;
        } else {
            g.met[g.met_at] = key;
            g.met_at = (g.met_at + 1) % remembered;
        }
        return false;
    }
};

test "a key is captured the second time it comes" {
    var g: Graphs = .init(std.testing.allocator);
    defer g.deinit();
    const a: Key = .{ .span = 128, .width = 1, .states = 4096, .snaps = 0 };
    var b = a;
    b.width = 2;
    try std.testing.expect(!g.again(a));
    try std.testing.expect(!g.again(b));
    try std.testing.expect(g.again(a));
    try std.testing.expect(!g.again(a));
    try std.testing.expect(g.lookup(a) == null);
}
