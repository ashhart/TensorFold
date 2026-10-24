//! Per-stream keys and values with the stream's M-RoPE layout, and the prompt and window scratch.
const std = @import("std");
const mtl = @import("metal");
const c = @import("config.zig");
const rope = @import("../../core/qwen_vl/rope.zig");
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

pub const window_rows = 16;
pub const batch_rows = 32;
/// Rows a prompt call takes at most (a longer prompt goes in chunks).
pub const prompt_rows = 2048;
/// Positions a stream holds at most.
pub const max_context = 131072;

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

/// Keys and values [kv head, capacity, head dim] bf16.
pub const Attention = struct { keys: mtl.Buffer, values: mtl.Buffer };

/// Bytes one row's keys and values take over every layer.
pub const row_bytes = c.layers * 2 * c.kv_heads * c.head_dim * 2;

pub const Cache = struct {
    memory: Storage,
    capacity: usize,
    len: usize = 0,
    /// Where each row sits in M-RoPE positions: the prompt's images, and the shift every later row carries.
    layout: rope.Layout = .{},
    last: ?struct { start: usize, base: usize, rows: usize } = null,
    blocks: [c.layers]Attention,
    logits: mtl.Buffer,

    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, capacity: usize) !Cache {
        var out: Cache = .{ .memory = .{ .gpa = gpa, .device = device }, .capacity = capacity, .blocks = undefined, .logits = undefined };
        errdefer out.memory.deinit();
        out.logits = try out.memory.alloc(c.vocab * 2);
        const bytes = capacity * c.kv_heads * c.head_dim * 2;
        for (&out.blocks) |*b| b.* = .{ .keys = try out.memory.alloc(bytes), .values = try out.memory.alloc(bytes) };
        return out;
    }

    pub fn deinit(self: *Cache) void {
        self.memory.deinit();
    }

    /// Row `p`'s (t, h, w).
    pub fn position(self: *const Cache, p: usize) [3]i32 {
        return self.layout.position(@intCast(p));
    }

    pub fn commit(self: *Cache, base: usize, rows: usize, record: bool) void {
        if (record) self.last = .{ .start = self.len, .base = base, .rows = rows } else self.last = null;
        self.len += rows;
    }

    /// A verified window's accepted rows (a chain from its first): the rest are dropped, the last row's logits kept.
    pub fn keep(self: *Cache, scratch: *const Scratch, path: []const u32) !void {
        const last = self.last orelse return error.NothingToKeep;
        if (path.len == 0 or path.len > last.rows) return error.NothingToKeep;
        for (path, 0..) |p, i| if (p != i) return error.UnsupportedCosmosTree;
        const row = last.base + path.len - 1;
        self.len = last.start + path.len;
        @memcpy(self.logits.contents()[0 .. c.vocab * 2], (scratch.logits.contents() + row * c.vocab * 2)[0 .. c.vocab * 2]);
        self.last = null;
    }
};

pub const Scratch = struct {
    memory: Storage,
    rows: usize,
    ids: mtl.Buffer,
    pos: mtl.Buffer, // i32 [rows, 3]
    slots: mtl.Buffer, // u32 [rows]: each row's cache slot
    map: mtl.Buffer, // i32 [rows]: each row's image row, or -1
    h: mtl.Buffer,
    x: mtl.Buffer,
    r: mtl.Buffer,
    y: mtl.Buffer,
    q: mtl.Buffer,
    k: mtl.Buffer,
    v: mtl.Buffer,
    queries: mtl.Buffer,
    act: mtl.Buffer,
    /// A projection's input rows' group sums for the tiles and their twin (f32 [rows, k / 64]).
    sums: mtl.Buffer,
    logits: mtl.Buffer,

    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, rows: usize) !Scratch {
        var out: Scratch = undefined;
        out.memory = .{ .gpa = gpa, .device = device };
        errdefer out.memory.deinit();
        out.rows = rows;
        const mem = &out.memory;
        out.ids = try mem.alloc(rows * 4);
        out.pos = try mem.alloc(rows * 3 * 4);
        out.slots = try mem.alloc(rows * 4);
        out.map = try mem.alloc(rows * 4);
        // queries has 32 rows of slack: the last flash tile reads whole 8-row query tiles past the call's rows
        inline for (.{ "h", "x", "r", "y", "q", "queries" }) |name| @field(out, name) = try mem.alloc((rows + 32) * c.hidden * 2);
        out.k = try mem.alloc(rows * c.kv_heads * c.head_dim * 2);
        out.v = try mem.alloc(rows * c.kv_heads * c.head_dim * 2);
        out.act = try mem.alloc(rows * c.intermediate * 2);
        out.sums = try mem.alloc(rows * (c.intermediate / 64) * 4);
        out.logits = try mem.alloc(@as(usize, @min(rows, batch_rows)) * c.vocab * 2);
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
