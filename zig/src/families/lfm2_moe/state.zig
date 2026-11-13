//! Per-stream conv state and KV, with each window row's conv state kept for partial acceptance.
const std = @import("std");
const mtl = @import("metal");
const c = @import("config.zig");
const tile = @import("../../core/tile_qmm.zig");
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
/// The two Bx rows before the next token (bf16 [2, hidden]).
pub const conv_bytes = (c.conv_taps - 1) * c.hidden * 2;
pub const window_rows = 16;
pub const batch_rows = 32;
/// Rows a prompt attention pass takes at most: its partials are sized for these, a longer chunk goes in slices.
pub const attn_slice = 2048;
/// Key chunks a row's attention splits into at most (the kernel's CHUNKS), and keys a chunk (its PER).
pub const chunks = 32;
pub const chunk_keys = 4096;
/// Positions a stream's attention holds: the partials and the merge have room for `chunks` chunks, no more.
pub const max_context = chunks * chunk_keys;
/// Sub-spans a chunk, one a threadgroup for rows of up to `split_rows` (the kernel's SUBS), folded by the merge.
pub const subs = 8;
pub const split_rows = 8;
/// Query rows a prompt attention tile (two 8-row blocks a simdgroup); rows of up to `split_rows` take 8-row tiles.
pub const attn_rows = 16;

pub const Storage = struct {
    residency: ?mtl.ResidencySet.Follow = null,
    gpa: std.mem.Allocator,
    device: mtl.Device,
    buffers: std.ArrayList(mtl.Buffer) = .empty,

    pub fn alloc(self: *Storage, bytes: usize) !mtl.Buffer {
        const b = try self.device.buffer(@max(bytes, 16), opts);
        errdefer b.deinit();
        try self.buffers.append(self.gpa, b);
        if (self.residency) |f| f.join(f.ctx, b);
        return b;
    }

    pub fn setResident(self: *Storage, hook: ?mtl.ResidencySet.Follow) void {
        for (self.buffers.items) |b| {
            if (self.residency) |f| f.leave(f.ctx, b);
            if (hook) |f| f.join(f.ctx, b);
        }
        self.residency = hook;
    }

    pub fn deinit(self: *Storage) void {
        self.setResident(null);
        for (self.buffers.items) |b| b.deinit();
        self.buffers.deinit(self.gpa);
    }
};

pub const Attention = struct { keys: mtl.Buffer, values: mtl.Buffer };
/// Layers a drafter reads at most (each layer's output, hidden wide).
pub const max_taps = 8;
/// The forward's row destination addresses, preceded by raw bf16 rows only for generic scratch capture.
pub const Taps = struct {
    layers: [max_taps]u32 = @splat(0),
    count: u32 = 0,
    buf: mtl.Buffer,
    gpu_base: u64, // `buf`'s base address, cached for this allocation (not `table_at`)
    table_at: usize,
};
/// Latest scratch rows from `row`, or owned stage bytes from `stage`; gen 0 keeps a prompt anchor across forwards.
pub const TapRows = struct { start: usize, row: usize, rows: usize, gen: u64, stage: ?usize = null };
pub const Mixer = union(enum) { conv: mtl.Buffer, attention: Attention };

pub const Cache = struct {
    memory: Storage,
    capacity: usize,
    len: usize = 0,
    /// Native all-row owner, compared only; backend caches are destroyed before scratch.
    last: ?struct { start: usize, base: usize, rows: usize, committed: ?*const Scratch = null } = null,
    blocks: [c.layers]Mixer,
    logits: mtl.Buffer,
    /// Latest captured rows; generic capture alone copies the prompt anchor into `last_tap`.
    tap: ?TapRows = null,
    last_tap: ?mtl.Buffer = null,

    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, attention: *const [c.layers]bool, capacity: usize) !Cache {
        var out: Cache = .{ .memory = .{ .gpa = gpa, .device = device }, .capacity = capacity, .blocks = undefined, .logits = undefined };
        errdefer out.memory.deinit();
        out.logits = try out.memory.alloc(c.vocab * 2);
        for (&out.blocks, attention) |*b, a| {
            if (a) {
                const bytes = capacity * c.kv_heads * c.head_dim * 2;
                b.* = .{ .attention = .{ .keys = try out.memory.alloc(bytes), .values = try out.memory.alloc(bytes) } };
            } else {
                const conv = try out.memory.alloc(conv_bytes);
                @memset(conv.contents()[0..conv_bytes], 0);
                b.* = .{ .conv = conv };
            }
        }
        return out;
    }

    pub fn deinit(self: *Cache) void {
        self.memory.deinit();
    }

    fn restore(self: *Cache, scratch: *const Scratch, row: usize) void {
        for (&self.blocks, scratch.snapshots) |*b, snapshot| if (b.* == .conv) {
            @memcpy(b.conv.contents()[0..conv_bytes], (snapshot.?.contents() + row * conv_bytes)[0..conv_bytes]);
        };
    }

    pub fn commit(self: *Cache, scratch: *const Scratch, base: usize, rows: usize, record: bool) void {
        self.restore(scratch, if (record) base + rows - 1 else 0); // a prompt call snapshots its last row alone
        if (record) self.last = .{ .start = self.len, .base = base, .rows = rows } else self.last = null;
        self.len += rows;
    }

    pub fn keep(self: *Cache, scratch: *const Scratch, path: []const u32) !void {
        const last = self.last orelse return error.NothingToKeep;
        if (path.len == 0 or path.len > last.rows) return error.NothingToKeep;
        for (path, 0..) |p, i| if (p != i) return error.UnsupportedLfmTree;
        const row = last.base + path.len - 1;
        // Native verify already owns this full frame's conv/logits, independently of later scratch use.
        const committed = last.committed == scratch and path.len == last.rows and self.len == last.start + last.rows;
        if (!committed) {
            self.restore(scratch, row);
            @memcpy(self.logits.contents()[0 .. c.vocab * 2], (scratch.logits.contents() + row * c.vocab * 2)[0 .. c.vocab * 2]);
        }
        self.len = last.start + path.len;
        self.last = null;
    }
};

pub const Scratch = struct {
    memory: Storage,
    rows: usize,
    ids: mtl.Buffer,
    bad: mtl.Buffer, // set by lfm2_draft_ids when a held pick is flagged nonfinite or past the vocabulary
    h: mtl.Buffer,
    x: mtl.Buffer,
    r: mtl.Buffer,
    bcx: mtl.Buffer,
    y: mtl.Buffer,
    k: mtl.Buffer,
    v: mtl.Buffer,
    queries: mtl.Buffer,
    prompt_pick: mtl.Buffer,
    pm: mtl.Buffer,
    pl: mtl.Buffer,
    po: mtl.Buffer,
    gate: mtl.Buffer,
    up: mtl.Buffer,
    act: mtl.Buffer,
    route: mtl.Buffer,
    picks: mtl.Buffer,
    weights: mtl.Buffer,
    experts: mtl.Buffer,
    /// Prompt rows' picks by expert: starts, slot -> sorted, sorted -> token row, and the identity map.
    starts: mtl.Buffer,
    pos: mtl.Buffer,
    tokens: mtl.Buffer,
    ident: mtl.Buffer,
    work: mtl.Buffer,
    /// Window rows' picks by expert for the twin (tile_qmm.encodeTwinGroup).
    groups: mtl.Buffer,
    /// A projection's input rows' group sums for the tiles and their twin (f32 [rows, k / 64]).
    sums: mtl.Buffer,
    logits: mtl.Buffer,
    /// Each head row's greedy pick and nonfinite flag (u32 pairs), from a forward whose head takes every row.
    best: mtl.Buffer,
    snapshots: [c.layers]?mtl.Buffer = @splat(null),
    /// A recorded call's streams at every layer, each row's stream and the attention's row tiles (segments.zig).
    segs: mtl.Buffer,
    rowseg: mtl.Buffer,
    tiles: mtl.Buffer,
    /// A drafter's taps (null: none kept).
    taps: ?Taps = null,

    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, attention: *const [c.layers]bool, rows: usize) !Scratch {
        var out: Scratch = undefined;
        out.memory = .{ .gpa = gpa, .device = device };
        errdefer out.memory.deinit();
        out.rows = rows;
        out.snapshots = @splat(null);
        out.taps = null;
        const mem = &out.memory;
        const slots = rows * c.topk;
        out.ids = try mem.alloc(rows * 4);
        out.bad = try mem.alloc(4);
        inline for (.{ "h", "r", "y" }) |name| @field(out, name) = try mem.alloc(rows * c.hidden * 2);
        out.x = try mem.alloc((rows + attn_rows) * c.hidden * 2); // normalized Q reuses x; prompt attention reads a padded tile
        out.queries = try mem.alloc(rows * c.hidden * 2);
        out.bcx = try mem.alloc(rows * 3 * c.hidden * 2);
        out.k = try mem.alloc(rows * c.kv_heads * c.head_dim * 2);
        out.v = try mem.alloc(rows * c.kv_heads * c.head_dim * 2);
        const partials = partialCount(rows);
        out.prompt_pick = try mem.alloc(64);
        @memset(out.prompt_pick.contents()[0..64], 0);
        out.pm = try mem.alloc(partials * 4);
        out.pl = try mem.alloc(partials * 4);
        out.po = try mem.alloc(partials * c.head_dim * 4);
        const wide = @max(rows * c.intermediate, slots * c.moe_intermediate);
        inline for (.{ "gate", "up", "act" }) |name| @field(out, name) = try mem.alloc(wide * 2);
        out.route = try mem.alloc(rows * c.experts * 4);
        out.picks = try mem.alloc(slots * 4);
        out.weights = try mem.alloc(slots * 2);
        out.experts = try mem.alloc(slots * c.hidden * 2);
        out.starts = try mem.alloc((c.experts + 1) * 4);
        out.pos = try mem.alloc(slots * 4);
        out.tokens = try mem.alloc(slots * 4);
        out.ident = try mem.alloc(slots * 4);
        for (out.ident.slice(u32, slots), 0..) |*v, i| v.* = @intCast(i);
        out.work = try mem.alloc((slots / 32 + c.experts + 2) * 4);
        out.groups = try mem.alloc(tile.groupWords(@min(slots, tile.window_slots)) * 4);
        out.sums = try mem.alloc(@max(rows * (c.intermediate / 64), slots * (c.hidden / 64)) * 4);
        const logit_rows: usize = @min(rows, batch_rows);
        out.logits = try mem.alloc(logit_rows * c.vocab * 2);
        out.best = try mem.alloc(logit_rows * 2 * 4);
        out.segs = try mem.alloc(c.layers * batch_rows * @sizeOf(@import("segments.zig").Seg));
        out.rowseg = try mem.alloc(batch_rows * 4);
        out.tiles = try mem.alloc(batch_rows * 4);
        for (&out.snapshots, attention) |*s, a| if (!a) { // a recorded window's rows, or a prompt call's last row
            s.* = try mem.alloc(logit_rows * conv_bytes);
        };
        return out;
    }

    pub fn deinit(self: *Scratch) void {
        self.memory.deinit();
    }
};

/// Attention partials a scratch of `rows` holds: a prompt slice's, and a recorded call's streams kept apart.
pub fn partialCount(rows: usize) usize {
    const slice_rows: usize = @min(rows, attn_slice);
    // usize: @min with a constant narrows its type, and `most * subs` would overflow it.
    const most: usize = @min(rows, batch_rows);
    return @max(slice_rows, split_rows * subs, most * subs) * c.query_heads * chunks;
}

/// The served window: --context, else the checkpoint's, never past `max_context`; null when --context is outside it.
pub fn window(checkpoint: usize, requested: ?i64) ?usize {
    const most = @min(checkpoint, max_context);
    const w = requested orelse return most;
    return if (w >= 1 and w <= most) @as(usize, @intCast(w)) else null;
}

test "a shared round's 32 rows keep their attention partials apart" {
    // 32 rows of 8 sub-spans each, every row's 32 heads over 32 chunks; a prompt chunk's rows hold more.
    try std.testing.expectEqual(@as(usize, 32 * subs * c.query_heads * chunks), partialCount(32));
    try std.testing.expectEqual(@as(usize, split_rows * subs * c.query_heads * chunks), partialCount(1));
    try std.testing.expectEqual(@as(usize, 2048 * c.query_heads * chunks), partialCount(2048));
}

test "the served window stays inside the attention's 131,072 positions" {
    try std.testing.expectEqual(@as(usize, 131072), max_context);
    try std.testing.expectEqual(@as(?usize, 131072), window(131072, null));
    try std.testing.expectEqual(@as(?usize, 131072), window(131074, null));
    try std.testing.expectEqual(@as(?usize, 70000), window(131074, 70000));
    try std.testing.expectEqual(@as(?usize, 131072), window(262144, 131072));
    try std.testing.expectEqual(@as(?usize, null), window(131074, 131073));
    try std.testing.expectEqual(@as(?usize, null), window(8192, 8193));
    try std.testing.expectEqual(@as(?usize, null), window(131072, 0));
}

/// Bytes a stream's Cache takes at `capacity` rows: its logits, conv layers' windows, attention layers' KV.
pub fn cacheBytes(attention: *const [c.layers]bool, capacity: usize) u64 {
    var bytes: u64 = c.vocab * 2;
    for (attention) |a| bytes += if (a) 2 * capacity * c.kv_heads * c.head_dim * 2 else conv_bytes;
    return bytes;
}

test "a stream's cache bytes: logits and conv windows at any length, keys and values a row on attention layers" {
    var attention: [c.layers]bool = @splat(false);
    try std.testing.expectEqual(@as(u64, c.vocab * 2 + c.layers * conv_bytes), cacheBytes(&attention, 4096));
    attention[1] = true;
    attention[3] = true;
    try std.testing.expectEqual(@as(u64, 2 * 2 * c.kv_heads * c.head_dim * 2), cacheBytes(&attention, 1) - cacheBytes(&attention, 0));
}

fn fillKeepRows(s: *Scratch) void {
    for (0..s.rows) |row| {
        @memset((s.logits.contents() + row * c.vocab * 2)[0 .. c.vocab * 2], @intCast(11 + row));
        for (s.snapshots, 0..) |snapshot, layer| if (snapshot) |b|
            @memset((b.contents() + row * conv_bytes)[0..conv_bytes], @intCast(31 + row + layer));
    }
}

fn expectKeepRow(cache: *const Cache, s: *const Scratch, conv_row: usize, logit_row: usize, len: usize) !void {
    try std.testing.expectEqual(len, cache.len);
    try std.testing.expectEqualSlices(u8, (s.logits.contents() + logit_row * c.vocab * 2)[0 .. c.vocab * 2], cache.logits.contents()[0 .. c.vocab * 2]);
    for (cache.blocks, s.snapshots) |b, snapshot| if (b == .conv)
        try std.testing.expectEqualSlices(u8, (snapshot.?.contents() + conv_row * conv_bytes)[0..conv_bytes], b.conv.contents()[0..conv_bytes]);
}

fn commitKeep(cache: *Cache, s: *const Scratch, base: usize, rows: usize, record: bool, logit_row: usize, native: bool) void {
    @memcpy(cache.logits.contents()[0 .. c.vocab * 2], (s.logits.contents() + logit_row * c.vocab * 2)[0 .. c.vocab * 2]);
    cache.commit(s, base, rows, record);
    if (native) cache.last.?.committed = s;
}

test "lfm2_moe: full keep validates before consuming, then partial and prompt commits replace the frame" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const attention: [c.layers]bool = @splat(false);
    var s = try Scratch.init(std.testing.allocator, device, &attention, 4);
    defer s.deinit();
    fillKeepRows(&s);
    var cache = try Cache.init(std.testing.allocator, device, &attention, 32);
    defer cache.deinit();
    cache.len = 5;
    commitKeep(&cache, &s, 1, 3, true, 3, true);
    for ([_][]const u32{ &.{}, &.{ 9, 8, 7, 6 } }) |path|
        try std.testing.expectError(error.NothingToKeep, cache.keep(&s, path));
    for ([_][]const u32{ &.{1}, &.{ 0, 2 }, &.{ 0, 1, 1 } }) |path|
        try std.testing.expectError(error.UnsupportedLfmTree, cache.keep(&s, path));
    try expectKeepRow(&cache, &s, 3, 3, 8);
    try cache.keep(&s, &.{ 0, 1, 2 });
    try expectKeepRow(&cache, &s, 3, 3, 8);
    try std.testing.expectError(error.NothingToKeep, cache.keep(&s, &.{ 0, 1, 2 }));
    commitKeep(&cache, &s, 0, 3, true, 2, true);
    try cache.keep(&s, &.{ 0, 1 });
    try expectKeepRow(&cache, &s, 1, 1, 10);
    try std.testing.expectError(error.NothingToKeep, cache.keep(&s, &.{0}));
    commitKeep(&cache, &s, 2, 1, true, 2, true);
    try cache.keep(&s, &.{0});
    try expectKeepRow(&cache, &s, 2, 2, 11);
    commitKeep(&cache, &s, 0, 2, true, 1, true);
    commitKeep(&cache, &s, 0, 3, false, 0, false);
    try expectKeepRow(&cache, &s, 0, 0, 16);
    try std.testing.expectError(error.NothingToKeep, cache.keep(&s, &.{0}));
}

test "lfm2_moe: last-only heads, replaced scratch and rewound lengths retain restoring keeps" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const attention: [c.layers]bool = @splat(false);
    var s = try Scratch.init(std.testing.allocator, device, &attention, 4);
    defer s.deinit();
    fillKeepRows(&s);
    var cache = try Cache.init(std.testing.allocator, device, &attention, 32);
    defer cache.deinit();
    // A recorded last-only head writes row zero, even for later segments.
    for ([_]usize{ 0, 2 }) |base| {
        commitKeep(&cache, &s, base, 2, true, 0, false);
        try expectKeepRow(&cache, &s, base + 1, 0, cache.len);
        try cache.keep(&s, &.{ 0, 1 });
        try expectKeepRow(&cache, &s, base + 1, base + 1, cache.len);
    }
    commitKeep(&cache, &s, 0, 2, true, 1, false);
    // An unmarked generic frame still reads the supplied scratch after it is reused.
    @memset(s.logits.contents()[0 .. 4 * c.vocab * 2], 81);
    for (s.snapshots) |snapshot| if (snapshot) |b| @memset(b.contents()[0 .. 4 * conv_bytes], 82);
    try cache.keep(&s, &.{ 0, 1 });
    try expectKeepRow(&cache, &s, 1, 1, 6);
    commitKeep(&cache, &s, 0, 2, true, 1, true);
    const other = s; // an explicitly supplied different scratch remains the restore source
    @memset(s.logits.contents()[0 .. 4 * c.vocab * 2], 91);
    for (s.snapshots) |snapshot| if (snapshot) |b| @memset(b.contents()[0 .. 4 * conv_bytes], 92);
    try cache.keep(&other, &.{ 0, 1 });
    try expectKeepRow(&cache, &other, 1, 1, 8);
    commitKeep(&cache, &s, 0, 2, true, 1, true);
    cache.len -= 1;
    @memset(cache.logits.contents()[0 .. c.vocab * 2], 0);
    for (cache.blocks) |b| @memset(b.conv.contents()[0..conv_bytes], 0);
    try cache.keep(&s, &.{ 0, 1 });
    try expectKeepRow(&cache, &s, 1, 1, 10);
}

test "lfm2_moe: native full keep owns its bytes across scratch reuse, generic keep still restores" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const attention: [c.layers]bool = @splat(false);
    var s = try Scratch.init(std.testing.allocator, device, &attention, 2);
    defer s.deinit();
    fillKeepRows(&s);
    var native = try Cache.init(std.testing.allocator, device, &attention, 8);
    defer native.deinit();
    var generic = try Cache.init(std.testing.allocator, device, &attention, 8);
    defer generic.deinit();
    commitKeep(&native, &s, 0, 2, true, 1, true);
    commitKeep(&generic, &s, 0, 2, true, 1, false);
    @memset(s.logits.contents()[0 .. 2 * c.vocab * 2], 91);
    for (s.snapshots) |snapshot| if (snapshot) |b| @memset(b.contents()[0 .. 2 * conv_bytes], 92);
    try native.keep(&s, &.{ 0, 1 });
    try std.testing.expectEqual(generic.len, native.len);
    try std.testing.expectEqualSlices(u8, generic.logits.contents()[0 .. c.vocab * 2], native.logits.contents()[0 .. c.vocab * 2]);
    for (generic.blocks, native.blocks) |a, b|
        try std.testing.expectEqualSlices(u8, a.conv.contents()[0..conv_bytes], b.conv.contents()[0..conv_bytes]);
    try std.testing.expectError(error.NothingToKeep, native.keep(&s, &.{ 0, 1 }));
    try generic.keep(&s, &.{ 0, 1 });
    try expectKeepRow(&generic, &s, 1, 1, 2);
    // A later generic commit must also clear the earlier native frame's ownership.
    commitKeep(&native, &s, 0, 2, true, 0, false);
    @memset(s.logits.contents()[0 .. 2 * c.vocab * 2], 101);
    try native.keep(&s, &.{ 0, 1 });
    try expectKeepRow(&native, &s, 1, 1, 4);
}
