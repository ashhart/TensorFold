//! Per-stream committed recurrence and KV state in host memory, with per-row window snapshots for partial acceptance.
const std = @import("std");
const c = @import("config.zig");
const mixers = @import("cpu_mixers.zig");

pub const delta_floats = c.linear_heads * c.linear_dim * c.linear_dim;
pub const delta_bytes = delta_floats * 4;
pub const conv_elems = (c.conv_taps - 1) * c.conv_dim;
pub const conv_bytes = conv_elems * 2;
pub const window_rows = 16;
pub const batch_rows = 32;
pub const attention_layers = c.layers / 4;

pub const Delta = struct { recurrence: []f32, conv: []u16 };
/// Keys and values of an attention layer: [kv head][capacity][head_dim] bf16 each.
pub const Attention = struct { keys: []u16, values: []u16 };
pub const Mixer = union(enum) { delta: Delta, attention: Attention };

pub const Cache = struct {
    gpa: std.mem.Allocator,
    capacity: usize,
    len: usize = 0,
    last: ?struct { start: usize, base: usize, rows: usize } = null,
    blocks: [c.layers]Mixer,
    logits: []u16,

    pub fn init(gpa: std.mem.Allocator, capacity: usize) !Cache {
        var out: Cache = .{ .gpa = gpa, .capacity = capacity, .blocks = undefined, .logits = &.{} };
        var made: usize = 0;
        errdefer out.free(made);
        out.logits = try gpa.alloc(u16, c.vocab);
        @memset(out.logits, 0);
        for (&out.blocks, 0..) |*b, i| {
            if (c.linear(i)) {
                const r = try gpa.alloc(f32, delta_floats);
                errdefer gpa.free(r);
                const conv = try gpa.alloc(u16, conv_elems);
                @memset(r, 0);
                @memset(conv, 0);
                b.* = .{ .delta = .{ .recurrence = r, .conv = conv } };
            } else {
                const n = capacity * c.kv_heads * c.head_dim;
                const keys = try gpa.alloc(u16, n);
                errdefer gpa.free(keys);
                const values = try gpa.alloc(u16, n);
                b.* = .{ .attention = .{ .keys = keys, .values = values } };
            }
            made += 1;
        }
        return out;
    }

    fn free(self: *Cache, blocks: usize) void {
        for (self.blocks[0..blocks]) |b| switch (b) {
            .delta => |d| {
                self.gpa.free(d.recurrence);
                self.gpa.free(d.conv);
            },
            .attention => |kv| {
                self.gpa.free(kv.keys);
                self.gpa.free(kv.values);
            },
        };
        if (self.logits.len > 0) self.gpa.free(self.logits);
    }

    pub fn deinit(self: *Cache) void {
        self.free(c.layers);
        self.* = undefined;
    }

    /// The pass's rows [base, base + rows) of the scratch committed: the last row's states, the logits set by the pass.
    pub fn commit(self: *Cache, scratch: *const Scratch, base: usize, rows: usize, record: bool) void {
        for (&self.blocks, scratch.snapshots) |*b, snapshot| {
            if (b.* == .delta) {
                const saved = if (record) base + rows - 1 else 0;
                @memcpy(b.delta.recurrence, snapshot.?.recurrence[saved * delta_floats ..][0..delta_floats]);
                @memcpy(b.delta.conv, snapshot.?.conv[(base + rows - 1) * conv_elems ..][0..conv_elems]);
            }
        }
        self.last = if (record) .{ .start = self.len, .base = base, .rows = rows } else null;
        self.len += rows;
    }

    /// Roll the last recorded window back to its first `path.len` rows (a chain's prefix: row i is row i).
    pub fn keep(self: *Cache, scratch: *const Scratch, path: []const u32) !void {
        const last = self.last orelse return error.NothingToKeep;
        if (path.len == 0 or path.len > last.rows) return error.NothingToKeep;
        for (path, 0..) |row, i| if (row != i) return error.UnsupportedQwenTree;
        for (&self.blocks, scratch.snapshots) |*b, snapshot| {
            if (b.* == .delta) {
                const src = last.base + path.len - 1;
                @memcpy(b.delta.recurrence, snapshot.?.recurrence[src * delta_floats ..][0..delta_floats]);
                @memcpy(b.delta.conv, snapshot.?.conv[src * conv_elems ..][0..conv_elems]);
            }
        }
        self.len = last.start + path.len;
        const row = last.base + path.len - 1;
        @memcpy(self.logits, scratch.logits[row * c.vocab ..][0..c.vocab]);
        self.last = null;
    }
};

/// Every (row, head) attention task's chunk partials, one slice a worker.
pub const Partials = []mixers.Partial;

pub const Scratch = struct {
    /// Every buffer below lives here; set last, once they are all made.
    arena: std.heap.ArenaAllocator,
    rows: usize,
    ids: []u32,
    h: []u16,
    x: []u16,
    r: []u16,
    qkv: []u16,
    z: []u16,
    a: []u16,
    b: []u16,
    q: []u16,
    k: []u16,
    v: []u16,
    g: []f32,
    beta: []u16,
    y: []u16,
    mix: []u16,
    qraw: []u16,
    knorm: []u16,
    queries: []u16,
    gate: []u16,
    up: []u16,
    act: []u16,
    /// The projection inputs prepared, rows × intermediate at most, and each group's sum.
    xp: []f32,
    xs: []f32,
    logits: []u16,
    /// Every (row, head) attention task's chunk partials, one slice a worker.
    partials: []mixers.Partial,
    snapshots: [c.layers]?Delta = @splat(null),

    pub fn init(gpa: std.mem.Allocator, rows: usize, capacity: usize, workers: usize) !Scratch {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        var out: Scratch = undefined;
        out.rows = rows;
        out.snapshots = @splat(null);
        out.ids = try a.alloc(u32, rows);
        inline for (.{ "h", "x", "r", "z", "q", "k", "v", "y", "mix", "queries" }) |name| @field(out, name) = try a.alloc(u16, rows * c.hidden);
        inline for (.{ "qkv", "gate", "up", "act" }) |name| @field(out, name) = try a.alloc(u16, rows * c.intermediate);
        out.qraw = try a.alloc(u16, rows * 2 * c.hidden);
        out.knorm = try a.alloc(u16, rows * c.kv_heads * c.head_dim);
        inline for (.{ "a", "b", "beta" }) |name| @field(out, name) = try a.alloc(u16, rows * c.linear_heads);
        out.g = try a.alloc(f32, rows * c.linear_heads);
        out.xp = try a.alloc(f32, rows * c.intermediate);
        out.xs = try a.alloc(f32, rows * c.intermediate / c.group);
        out.logits = try a.alloc(u16, batch_rows * c.vocab);
        out.partials = try a.alloc(mixers.Partial, workers * mixers.chunksFor(capacity));
        for (&out.snapshots, 0..) |*s, i| if (c.linear(i)) {
            s.* = .{ .recurrence = try a.alloc(f32, batch_rows * delta_floats), .conv = try a.alloc(u16, rows * conv_elems) };
        };
        out.arena = arena;
        return out;
    }

    pub fn deinit(self: *Scratch) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

/// Bytes a stream's Cache takes at `capacity` rows: its logits, delta layers' states, attention layers' KV.
pub fn cacheBytes(capacity: usize) u64 {
    var bytes: u64 = c.vocab * 2;
    for (0..c.layers) |i| bytes += if (c.linear(i)) delta_bytes + conv_bytes else 2 * capacity * c.kv_heads * c.head_dim * 2;
    return bytes;
}

pub const Snap = struct { at: u32, bytes: []u8 };

pub fn snapBytes(at: u32) u64 {
    return cacheBytes(at);
}

pub fn save(gpa: std.mem.Allocator, cache: *const Cache, at: u32) !*Snap {
    if (cache.len != at or at > cache.capacity) return error.QwenSnapshotPosition;
    const snap = try gpa.create(Snap);
    errdefer gpa.destroy(snap);
    snap.* = .{ .at = at, .bytes = try gpa.alloc(u8, snapBytes(at)) };
    transfer(cache, snap, false);
    return snap;
}

pub fn restore(cache: *Cache, snap: *const Snap) !void {
    if (cache.len != 0 or snap.at > cache.capacity or snap.bytes.len != snapBytes(snap.at)) return error.QwenSnapshotPosition;
    transfer(cache, snap, true);
    cache.len = snap.at;
    cache.last = null;
}

pub fn drop(gpa: std.mem.Allocator, snap: *Snap) void {
    gpa.free(snap.bytes);
    gpa.destroy(snap);
}

fn copy(live: []u8, saved: []u8, comptime restoring: bool) void {
    if (restoring) @memcpy(live, saved) else @memcpy(saved, live);
}

fn transfer(cache: *const Cache, snap: *const Snap, comptime restoring: bool) void {
    var off: usize = c.vocab * 2;
    copy(std.mem.sliceAsBytes(cache.logits), snap.bytes[0..off], restoring);
    for (cache.blocks) |block| switch (block) {
        .delta => |d| {
            copy(std.mem.sliceAsBytes(d.recurrence), snap.bytes[off..][0..delta_bytes], restoring);
            off += delta_bytes;
            copy(std.mem.sliceAsBytes(d.conv), snap.bytes[off..][0..conv_bytes], restoring);
            off += conv_bytes;
        },
        .attention => |kv| for ([_][]u16{ kv.keys, kv.values }) |buf| for (0..c.kv_heads) |h| {
            const size = @as(usize, snap.at) * c.head_dim * 2;
            copy(std.mem.sliceAsBytes(buf[h * cache.capacity * c.head_dim ..])[0..size], snap.bytes[off..][0..size], restoring);
            off += size;
        },
    };
    std.debug.assert(off == snap.bytes.len);
}

test "a stream's cache bytes: about 19.5 MB of states and logits, and 12 KiB of keys and values a row" {
    try std.testing.expectEqual(@as(u64, 18_874_368 + 663_552 + 496_640), cacheBytes(0));
    try std.testing.expectEqual(@as(u64, 12_288), cacheBytes(1) - cacheBytes(0));
}

test "a snapshot round-trips a cache's committed state" {
    const gpa = std.testing.allocator;
    var cache = try Cache.init(gpa, 8);
    defer cache.deinit();
    cache.blocks[0].delta.recurrence[5] = 1.5;
    cache.blocks[3].attention.keys[cache.capacity * c.head_dim + 7] = 0x3f80;
    cache.logits[9] = 0x4000;
    cache.len = 3;
    const snap = try save(gpa, &cache, 3);
    defer drop(gpa, snap);
    var fresh = try Cache.init(gpa, 16);
    defer fresh.deinit();
    try restore(&fresh, snap);
    try std.testing.expectEqual(@as(usize, 3), fresh.len);
    try std.testing.expectEqual(@as(f32, 1.5), fresh.blocks[0].delta.recurrence[5]);
    try std.testing.expectEqual(@as(u16, 0x3f80), fresh.blocks[3].attention.keys[fresh.capacity * c.head_dim + 7]);
    try std.testing.expectEqual(@as(u16, 0x4000), fresh.logits[9]);
    try std.testing.expectError(error.QwenSnapshotPosition, restore(&fresh, snap));
}
