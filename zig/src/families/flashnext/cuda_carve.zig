//! TF_CUDA_CARVEOUT=1 (cuda/carveout.zig): a contiguous cache's first attention planes in the carveout.
const std = @import("std");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const prompt = @import("cuda_prompt.zig");
const Heads = @import("cuda_heads.zig").Heads;
const Part = @import("cuda_pages.zig").Part;

const attn_n = wt.attn_n;
const parts = [_]Part{ .key, .value, .index, .pooled };

comptime {
    std.debug.assert(parts.len * attn_n <= cuda.carveout.Spans.max); // every plane a span of its own
}

/// Planes of each part a cache keeps in the carveout, by `Part`: those of its first n attention layers.
pub const Counts = [parts.len]usize;

/// Bytes one attention layer's `part` takes in a contiguous cache of `cap` tokens (Engine.newSeqOn's sizes).
pub fn planeBytes(part: Part, cap: usize, h: Heads) usize {
    return switch (part) {
        .key, .value => cap * h.kvRow(),
        .index => cap * wt.index_dim * 2,
        .pooled => prompt.pooledBytes(cap),
    };
}

/// The planes of a `window`-token cache `room` bytes hold whole: key/value pairs a layer at a time, then indexer.
pub fn fit(room: usize, window: usize, h: Heads) Counts {
    var n: Counts = .{ 0, 0, 0, 0 };
    if (window == 0) return n;
    var left = room;
    const pair = al(planeBytes(.key, window, h)) + al(planeBytes(.value, window, h));
    const pairs = @min(attn_n, left / pair);
    n[@backingInt(Part.key)] = pairs;
    n[@backingInt(Part.value)] = pairs;
    left -= pairs * pair;
    for ([_]Part{ .index, .pooled }) |part| {
        const each = al(planeBytes(part, window, h));
        n[@backingInt(part)] = @min(attn_n, left / each);
        left -= n[@backingInt(part)] * each;
    }
    return n;
}

/// Bytes the planes `n` take at `cap` tokens: what a cache of that size leaves out of device memory.
pub fn kept(n: Counts, cap: usize, h: Heads) usize {
    var total: usize = 0;
    for (parts) |part| total += n[@backingInt(part)] * planeBytes(part, cap, h);
    return total;
}

fn al(n: usize) usize {
    return std.mem.alignForward(usize, n, cuda.carveout.alignment);
}

/// The carveout and the planes a contiguous cache keeps there, fitted to the engine's own window at load.
pub const Plan = struct {
    carve: *cuda.Carveout,
    n: Counts,
    heads: Heads, // the plane sizes follow the engine's heads

    pub fn of(carve: *cuda.Carveout, window: usize, h: Heads) Plan {
        return .{ .carve = carve, .n = fit(carve.bytes(), window, h), .heads = h };
    }

    /// The plan's planes for a `cap`-token cache into `into`, zeroed; another cache holding the carveout: none.
    pub fn take(p: Plan, into: *Carved, cap: usize) !void {
        if (p.carve.freeBytes() != p.carve.bytes()) return;
        into.carve = p.carve;
        const k = @backingInt(Part.key);
        const v = @backingInt(Part.value);
        for (0..p.n[k]) |layer| {
            into.planes[k][layer] = (try p.carve.take(planeBytes(.key, cap, p.heads))) orelse break;
            into.n[k] += 1; // counted before its value, so a failed take gives it back with the rest
            into.planes[v][layer] = (try p.carve.take(planeBytes(.value, cap, p.heads))) orelse {
                into.n[k] -= 1;
                p.carve.give(into.planes[k][layer]);
                break;
            };
            into.n[v] += 1;
        }
        for ([_]Part{ .index, .pooled }) |part| {
            const i = @backingInt(part);
            for (0..p.n[i]) |layer| {
                into.planes[i][layer] = (try p.carve.take(planeBytes(part, cap, p.heads))) orelse break;
                into.n[i] += 1;
            }
        }
        // Carveout.take zeroes on the legacy stream, which the engine's non-blocking stream never waits on
        try (cuda.Stream{ .d = p.carve.d, .handle = null }).synchronize();
    }

    /// The startup log's line: the planes, and how much of the engine's own window the memory budget counts there.
    pub fn log(p: Plan, window: usize, counted: usize) void {
        std.log.info("flashnext: the carveout holds a cache's keys and values of {d} attention layers, indexer keys of {d} and pooled keys of {d} ({d:.1} of its {d:.1} MiB at a {d}-token window); the memory budget counts {d:.1} MiB of the first stream's window there", .{ p.n[@backingInt(Part.key)], p.n[@backingInt(Part.index)], p.n[@backingInt(Part.pooled)], mib(kept(p.n, window, p.heads)), mib(p.carve.bytes()), window, mib(counted) });
    }
};

fn mib(bytes: usize) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 20);
}

/// A cache's planes in the carveout, given back when the cache goes.
pub const Carved = struct {
    carve: ?*cuda.Carveout = null,
    planes: [parts.len][attn_n]cuda.DeviceBuffer = undefined,
    n: Counts = .{ 0, 0, 0, 0 },

    /// Its planes of `part`, the first layers'.
    pub fn of(c: *const Carved, part: Part) []const cuda.DeviceBuffer {
        return c.planes[@backingInt(part)][0..c.n[@backingInt(part)]];
    }

    /// Every plane back to the carveout; the cache's work must have drained.
    pub fn give(c: *Carved) void {
        const carve = c.carve orelse return;
        for (parts) |part| for (c.of(part)) |b| carve.give(b);
        c.* = .{};
    }
};

/// The carveout: TF_CUDA_CARVEOUT=1, the card or a lending daemon, through carveout.zig's two calls; null when off.
pub fn open(gpa: std.mem.Allocator, ctx: *const cuda.Context, tp: usize) !?*cuda.Carveout {
    const want = (try cuda.carveout.requested(cuda.carveout.Settings.read(false, @import("sys").environment.Process{}))) orelse return null;
    if (tp != 1) return error.FlashNextCarveoutUnsupported; // one rank: the planes are this engine's own
    const c = try gpa.create(cuda.Carveout);
    errdefer gpa.destroy(c);
    c.* = try cuda.Carveout.from(ctx.d, want);
    return c;
}

pub fn close(gpa: std.mem.Allocator, c: *cuda.Carveout) void {
    c.close();
    gpa.destroy(c);
}

test "the planes are newSeq's: the attention part of seqBytes" {
    const h: Heads = .full;
    for ([_]usize{ 1, 5, 4096, wt.window }) |cap| {
        var each: usize = 0;
        var one: usize = 0;
        for (parts) |part| {
            each += planeBytes(part, cap, h);
            one += planeBytes(part, 1, h);
        }
        try std.testing.expectEqual(prompt.seqBytes(cap, h) - prompt.seqBytes(1, h), attn_n * (each - one));
    }
}

test "a full window's plan: whole planes, keys and values first" {
    // a daemon's 2,044 MiB: three K/V pairs (1,536 MiB), seven indexer keys, three pooled; 12 MiB holds no plane
    const h: Heads = .full;
    const lent = fit(2044 << 20, wt.window, h);
    try std.testing.expectEqual(Counts{ 3, 3, 7, 3 }, lent);
    try std.testing.expectEqual(@as(usize, 2032 << 20), kept(lent, wt.window, h));
    // the card's default 1,792 MiB: three pairs and four indexer keys, nothing left for a pooled plane
    try std.testing.expectEqual(Counts{ 3, 3, 4, 0 }, fit(1792 << 20, wt.window, h));
    // a short window keeps every plane there; no room, none
    try std.testing.expectEqual(Counts{ 12, 12, 12, 12 }, fit(2044 << 20, 8192, h));
    try std.testing.expectEqual(Counts{ 0, 0, 0, 0 }, fit(0, wt.window, h));
    try std.testing.expectEqual(Counts{ 0, 0, 7, 3 }, fit(511 << 20, wt.window, h));
}

test "every cache up to the window takes the window's planes from an empty carveout, in order" {
    const room: usize = 2044 << 20;
    const n = fit(room, wt.window, .full);
    for ([_]usize{ 1, 3, 4, 5, 2048, 8191, 131072, wt.window - 1, wt.window }) |cap| {
        var s: cuda.carveout.Spans = .{ .size = room };
        for (0..n[@backingInt(Part.key)]) |_| {
            try std.testing.expect(s.take(planeBytes(.key, cap, .full)) != null);
            try std.testing.expect(s.take(planeBytes(.value, cap, .full)) != null);
        }
        for ([_]Part{ .index, .pooled }) |part| for (0..n[@backingInt(part)]) |_| try std.testing.expect(s.take(planeBytes(part, cap, .full)) != null);
        try std.testing.expectEqual(kept(n, cap, .full), room - s.free());
        try std.testing.expect(kept(n, cap, .full) <= kept(n, wt.window, .full));
    }
}

test "a cache's carved planes, by part" {
    var c: Carved = .{};
    try std.testing.expectEqual(@as(usize, 0), c.of(.key).len);
    c.give(); // nothing taken: nothing to give
    c.n = .{ 2, 2, 1, 0 };
    try std.testing.expectEqual(@as(usize, 2), c.of(.value).len);
    try std.testing.expectEqual(@as(usize, 1), c.of(.index).len);
    try std.testing.expectEqual(@as(usize, 0), c.of(.pooled).len);
}
