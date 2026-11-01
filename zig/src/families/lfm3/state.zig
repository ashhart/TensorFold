//! Per-stream conv state and KV, with each window row's conv state kept for partial acceptance.
const std = @import("std");
const mtl = @import("metal");
const c = @import("config.zig");
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
/// The six Bx rows before the next token (bf16 [6, hidden]).
pub const conv_bytes = (c.conv_taps - 1) * c.hidden * 2;
pub const window_rows = 16;
pub const batch_rows = 32;
/// Rows a prompt attention pass takes at most: its partials are sized for these, a longer chunk goes in slices.
pub const attn_slice = 64;
/// Key chunks a row's attention splits into at most (the kernel's CHUNKS), and keys a chunk (its PER).
pub const chunks = 64;
pub const chunk_keys = 4096;
/// Positions a stream's attention holds: the partials and the merge have room for `chunks` chunks, no more.
pub const max_context = chunks * chunk_keys;
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

    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, layer_types: *const [c.layers]c.LayerType, capacity: usize) !Cache {
        if (capacity == 0 or capacity > max_context) return error.InvalidLfmContext;
        var out: Cache = .{ .memory = .{ .gpa = gpa, .device = device }, .capacity = capacity, .blocks = undefined, .logits = undefined };
        errdefer out.memory.deinit();
        out.logits = try out.memory.alloc(c.vocab * 2);
        for (&out.blocks, layer_types) |*b, kind| {
            if (kind != .conv) {
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
    conv_segments: ?mtl.Buffer = null,
    conv_rows: ?mtl.Buffer = null,
    attention_segments: mtl.Buffer,
    attention_rows: mtl.Buffer,
    pm: mtl.Buffer,
    pl: mtl.Buffer,
    po: mtl.Buffer,
    attention_risk: mtl.Buffer,
    gate: mtl.Buffer,
    up: mtl.Buffer,
    act: mtl.Buffer,
    head_gate: mtl.Buffer,
    sums: ?mtl.Buffer = null,
    binned_sums: ?mtl.Buffer = null,
    binned_meta: ?mtl.Buffer = null,
    logits: mtl.Buffer,
    snapshots: [c.layers]?mtl.Buffer = @splat(null),

    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, layer_types: *const [c.layers]c.LayerType, rows: usize) !Scratch {
        if (rows == 0) return error.LfmRowsExceeded;
        var out: Scratch = undefined;
        out.memory = .{ .gpa = gpa, .device = device };
        errdefer out.memory.deinit();
        out.rows = rows;
        out.conv_segments = null;
        out.conv_rows = null;
        out.snapshots = @splat(null);
        const mem = &out.memory;
        out.ids = try mem.alloc(rows * 4);
        inline for (.{ "h", "x", "r", "y", "q" }) |name| @field(out, name) = try mem.alloc(rows * c.hidden * 2);
        out.queries = try mem.alloc(rows * c.hidden * 2);
        out.k = try mem.alloc(rows * c.kv_heads * c.head_dim * 2);
        out.v = try mem.alloc(rows * c.kv_heads * c.head_dim * 2);
        const slice_rows: usize = @min(rows, attn_slice);
        const partials = slice_rows * c.query_heads * chunks;
        out.attention_segments = try mem.alloc(c.layers * batch_rows * @sizeOf(@import("kernels.zig").AttentionSegment));
        out.attention_rows = try mem.alloc(batch_rows * 4);
        out.pm = try mem.alloc(partials * 4);
        out.pl = try mem.alloc(partials * 4);
        const logit_rows: usize = @min(rows, batch_rows);
        const logits_bytes = logit_rows * c.vocab * 2;
        out.po = try mem.alloc(@max(partials * c.head_dim * 4, logits_bytes));
        out.attention_risk = try mem.alloc(partials * @sizeOf(u32));
        inline for (.{ "gate", "up", "act" }) |name| @field(out, name) = try mem.alloc(rows * c.intermediate * 2);
        comptime std.debug.assert(c.intermediate >= 3 * c.hidden);
        out.bcx = out.act; // Mixer input expires before the FFN activation writes this storage.
        out.sums = null;
        out.binned_sums = null;
        out.binned_meta = null;
        out.head_gate = try mem.alloc(rows * c.query_heads * 2);
        out.logits = out.po; // Attention partials expire before the final head writes logits.
        for (&out.snapshots, layer_types) |*s, kind| if (kind == .conv) { // a recorded window's rows, or a prompt call's last row
            s.* = try mem.alloc(logit_rows * conv_bytes);
        };
        return out;
    }

    pub fn prepareSharedConv(self: *Scratch) !void {
        if (self.conv_segments == null)
            self.conv_segments = try self.memory.alloc(c.layers * batch_rows * @sizeOf(@import("kernels.zig").ConvSegment));
        if (self.conv_rows == null) self.conv_rows = try self.memory.alloc(batch_rows * 4);
    }

    pub fn prepareBinnedProjectionCandidate(self: *Scratch) !void {
        const bytes = self.rows * (c.intermediate / 32) * 8;
        if (self.binned_sums == null) self.binned_sums = try self.memory.alloc(bytes);
        if (self.binned_meta == null) self.binned_meta = try self.memory.alloc(bytes);
    }

    pub fn prepareProjectionCandidate(self: *Scratch) !void {
        if (self.sums == null) self.sums = try self.memory.alloc(self.rows * (c.intermediate / 32) * 4);
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

/// Bytes a stream's cache takes: fixed convolution history, full-addressed KV and the current logits.
pub fn cacheBytes(layer_types: *const [c.layers]c.LayerType, capacity: usize) u64 {
    var bytes: u64 = c.vocab * 2;
    for (layer_types) |kind| bytes += if (kind != .conv) 2 * capacity * c.kv_heads * c.head_dim * 2 else conv_bytes;
    return bytes;
}

test "LFM3 context and storage retain the full and sliding layers at the checkpoint window" {
    try std.testing.expectEqual(@as(usize, 262144), max_context);
    try std.testing.expectEqual(@as(?usize, 262144), window(262144, null));
    try std.testing.expectEqual(@as(?usize, 128000), window(128000, null));
    try std.testing.expectEqual(@as(?usize, null), window(262144, 262145));
    try std.testing.expectEqual(@as(?usize, null), window(128000, 0));
    try std.testing.expectEqual(@as(usize, 24576), conv_bytes);
    try std.testing.expectEqual(@as(u64, 11 * 2 * c.kv_heads * c.head_dim * 2), cacheBytes(&c.schedule, 1) - cacheBytes(&c.schedule, 0));
}
