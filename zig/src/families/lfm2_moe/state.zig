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
pub const Mixer = union(enum) { conv: mtl.Buffer, attention: Attention };

pub const Cache = struct {
    memory: Storage,
    capacity: usize,
    len: usize = 0,
    last: ?struct { start: usize, base: usize, rows: usize } = null,
    blocks: [c.layers]Mixer,
    logits: mtl.Buffer,

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
        self.restore(scratch, row);
        self.len = last.start + path.len;
        @memcpy(self.logits.contents()[0 .. c.vocab * 2], (scratch.logits.contents() + row * c.vocab * 2)[0 .. c.vocab * 2]);
        self.last = null;
    }
};

pub const Scratch = struct {
    memory: Storage,
    rows: usize,
    ids: mtl.Buffer,
    h: mtl.Buffer,
    x: mtl.Buffer,
    r: mtl.Buffer,
    bcx: mtl.Buffer,
    y: mtl.Buffer,
    q: mtl.Buffer,
    k: mtl.Buffer,
    v: mtl.Buffer,
    queries: mtl.Buffer,
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
    /// Prompt rows' picks by expert: starts, sorted -> slot, slot -> sorted, sorted -> token row, and the identity map.
    starts: mtl.Buffer,
    order: mtl.Buffer,
    pos: mtl.Buffer,
    tokens: mtl.Buffer,
    ident: mtl.Buffer,
    work: mtl.Buffer,
    /// Window rows' picks by expert for the twin (tile_qmm.encodeTwinGroup).
    groups: mtl.Buffer,
    /// A projection's input rows' group sums for the tiles and their twin (f32 [rows, k / 64]).
    sums: mtl.Buffer,
    logits: mtl.Buffer,
    snapshots: [c.layers]?mtl.Buffer = @splat(null),

    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, attention: *const [c.layers]bool, rows: usize) !Scratch {
        var out: Scratch = undefined;
        out.memory = .{ .gpa = gpa, .device = device };
        errdefer out.memory.deinit();
        out.rows = rows;
        out.snapshots = @splat(null);
        const mem = &out.memory;
        const slots = rows * c.topk;
        out.ids = try mem.alloc(rows * 4);
        inline for (.{ "h", "x", "r", "y", "q" }) |name| @field(out, name) = try mem.alloc(rows * c.hidden * 2);
        out.queries = try mem.alloc((rows + attn_rows) * c.hidden * 2); // a prompt attention tile reads attn_rows from its first
        out.bcx = try mem.alloc(rows * 3 * c.hidden * 2);
        out.k = try mem.alloc(rows * c.kv_heads * c.head_dim * 2);
        out.v = try mem.alloc(rows * c.kv_heads * c.head_dim * 2);
        const slice_rows: usize = @min(rows, attn_slice);
        const partials = @max(slice_rows, split_rows * subs) * c.query_heads * chunks;
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
        out.order = try mem.alloc(slots * 4);
        out.pos = try mem.alloc(slots * 4);
        out.tokens = try mem.alloc(slots * 4);
        out.ident = try mem.alloc(slots * 4);
        for (out.ident.slice(u32, slots), 0..) |*v, i| v.* = @intCast(i);
        out.work = try mem.alloc((slots / 32 + c.experts + 2) * 4);
        out.groups = try mem.alloc(tile.groupWords(@min(slots, tile.window_slots)) * 4);
        out.sums = try mem.alloc(@max(rows * (c.intermediate / 64), slots * (c.hidden / 64)) * 4);
        const logit_rows: usize = @min(rows, batch_rows);
        out.logits = try mem.alloc(logit_rows * c.vocab * 2);
        for (&out.snapshots, attention) |*s, a| if (!a) { // a recorded window's rows, or a prompt call's last row
            s.* = try mem.alloc(logit_rows * conv_bytes);
        };
        return out;
    }

    pub fn deinit(self: *Scratch) void {
        self.memory.deinit();
    }
};

/// The served window: --context, else the checkpoint's, never past `max_context`; null when --context is outside it.
pub fn window(checkpoint: usize, requested: ?i64) ?usize {
    const most = @min(checkpoint, max_context);
    const w = requested orelse return most;
    return if (w >= 1 and w <= most) @as(usize, @intCast(w)) else null;
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
