//! Per-stream committed recurrence and KV state, with a shared pool of window snapshots for partial acceptance.
const std = @import("std");
const mtl = @import("metal");
const c = @import("config.zig");
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
pub const window_rows = 16;
pub const batch_rows = 32;

pub const Storage = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    buffers: std.ArrayList(mtl.Buffer) = .empty,

    pub fn alloc(self: *Storage, bytes: usize) !mtl.Buffer {
        const b = try self.device.buffer(@max(bytes, 16), opts);
        errdefer b.deinit();
        try self.buffers.append(self.gpa, b);
        return b;
    }

    pub fn deinit(self: *Storage) void {
        for (self.buffers.items) |b| b.deinit();
        self.buffers.deinit(self.gpa);
    }
};

pub const Delta = struct { recurrence: mtl.Buffer, conv: mtl.Buffer };
pub const Attention = struct { keys: mtl.Buffer, values: mtl.Buffer };
pub const Mixer = union(enum) { delta: Delta, attention: Attention };

pub const Cache = struct {
    memory: Storage,
    g: c.Geometry,
    capacity: usize,
    len: usize = 0,
    last: ?struct { start: usize, base: usize, rows: usize } = null,
    blocks: []Mixer,
    logits: mtl.Buffer,

    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, g: c.Geometry, capacity: usize) !Cache {
        var out: Cache = .{ .memory = .{ .gpa = gpa, .device = device }, .g = g, .capacity = capacity, .blocks = try gpa.alloc(Mixer, g.layers), .logits = undefined };
        errdefer gpa.free(out.blocks);
        errdefer out.memory.deinit();
        out.logits = try out.memory.alloc(c.vocab * 2);
        for (out.blocks, 0..) |*b, i| {
            if (c.linear(i)) {
                const r = try out.memory.alloc(g.deltaBytes());
                const conv = try out.memory.alloc(g.convBytes());
                @memset(r.contents()[0..g.deltaBytes()], 0);
                @memset(conv.contents()[0..g.convBytes()], 0);
                b.* = .{ .delta = .{ .recurrence = r, .conv = conv } };
            } else {
                const bytes = capacity * g.kvInner() * 2;
                b.* = .{ .attention = .{ .keys = try out.memory.alloc(bytes), .values = try out.memory.alloc(bytes) } };
            }
        }
        return out;
    }

    pub fn deinit(self: *Cache) void {
        self.memory.deinit();
        self.memory.gpa.free(self.blocks);
    }

    /// The forward wrote each recurrence's last-row state in place (gdn_chain state_final); conv windows copy here.
    pub fn commit(self: *Cache, scratch: *const Scratch, base: usize, rows: usize, record: bool) void {
        const cn = self.g.convBytes();
        for (self.blocks, scratch.snapshots) |*b, snapshot| {
            if (b.* == .delta) {
                const src = snapshot.?.conv.contents() + (base + rows - 1) * cn;
                @memcpy(b.delta.conv.contents()[0..cn], src[0..cn]);
            }
        }
        if (record) self.last = .{ .start = self.len, .base = base, .rows = rows } else self.last = null;
        self.len += rows;
    }

    /// Keeps a prefix on the host; returns the row whose recurrence forward.keep copies back, null if all kept.
    pub fn keep(self: *Cache, scratch: *const Scratch, path: []const u32) !?usize {
        const last = self.last orelse return error.NothingToKeep;
        if (path.len == 0 or path.len > last.rows) return error.NothingToKeep;
        for (path, 0..) |row, i| if (row != i) return error.UnsupportedQwenTree;
        self.len = last.start + path.len;
        self.last = null;
        if (path.len == last.rows) return null;
        const cn = self.g.convBytes();
        const row = last.base + path.len - 1;
        for (self.blocks, scratch.snapshots) |*b, snapshot| {
            if (b.* == .delta) @memcpy(b.delta.conv.contents()[0..cn], (snapshot.?.conv.contents() + row * cn)[0..cn]);
        }
        @memcpy(self.logits.contents()[0 .. c.vocab * 2], (scratch.logits.contents() + row * c.vocab * 2)[0 .. c.vocab * 2]);
        return row;
    }
};

pub const Scratch = struct {
    memory: Storage,
    g: c.Geometry,
    rows: usize,
    ids: mtl.Buffer,
    windows: mtl.Buffer,
    dims: mtl.Buffer,
    h: mtl.Buffer, // residual [rows, hidden]
    x: mtl.Buffer, // normed [rows, hidden]
    r: mtl.Buffer, // a mixer's or MLP's output [rows, hidden]
    qkv: mtl.Buffer, // [rows, convDim]
    z: mtl.Buffer, // [rows, vInner]
    a: mtl.Buffer,
    b: mtl.Buffer,
    q: mtl.Buffer, // DeltaNet queries, or normed attention queries [rows, max(kInner, qInner)]
    k: mtl.Buffer, // DeltaNet keys, or attention keys before their norm [rows, max(kInner, kvInner)]
    v: mtl.Buffer, // DeltaNet or attention values [rows, max(vInner, kvInner)]
    g_: mtl.Buffer, // decay gates f32 [rows, value heads]
    beta: mtl.Buffer,
    y: mtl.Buffer, // a mixer's heads [rows, inner]
    mix: mtl.Buffer, // gated heads, the output projection's input [rows, inner]
    qraw: mtl.Buffer, // attention queries and their gates [rows, 2 * qInner]
    knorm: mtl.Buffer, // [rows, kvInner]
    queries: mtl.Buffer, // rotated queries by head [query heads, rows, head_dim]
    gate: mtl.Buffer,
    up: mtl.Buffer,
    act: mtl.Buffer,
    pm: mtl.Buffer,
    pl: mtl.Buffer,
    po: mtl.Buffer,
    logits: mtl.Buffer,
    snapshots: []?Delta,

    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, g: c.Geometry, rows: usize, capacity: usize) !Scratch {
        var out: Scratch = undefined;
        out.memory = .{ .gpa = gpa, .device = device };
        errdefer out.memory.deinit();
        out.g = g;
        out.rows = rows;
        out.snapshots = try gpa.alloc(?Delta, g.layers);
        errdefer gpa.free(out.snapshots);
        @memset(out.snapshots, null);
        const mem = &out.memory;
        out.ids = try mem.alloc(rows * 4);
        out.windows = try mem.alloc(rows * c.conv_taps * 4);
        out.dims = try mem.alloc(8 * 4);
        inline for (.{ "h", "x", "r" }) |name| @field(out, name) = try mem.alloc(rows * g.hidden * 2);
        out.qkv = try mem.alloc(rows * g.convDim() * 2);
        out.z = try mem.alloc(rows * g.vInner() * 2);
        out.q = try mem.alloc(rows * @max(g.kInner(), g.qInner()) * 2);
        out.k = try mem.alloc(rows * @max(g.kInner(), g.kvInner()) * 2);
        out.v = try mem.alloc(rows * @max(g.vInner(), g.kvInner()) * 2);
        inline for (.{ "y", "mix" }) |name| @field(out, name) = try mem.alloc(rows * g.inner() * 2);
        out.qraw = try mem.alloc(rows * 2 * g.qInner() * 2);
        out.knorm = try mem.alloc(rows * g.kvInner() * 2);
        out.queries = try mem.alloc(rows * g.qInner() * 2);
        inline for (.{ "gate", "up", "act" }) |name| @field(out, name) = try mem.alloc(rows * g.intermediate * 2);
        inline for (.{ "a", "b", "beta" }) |name| @field(out, name) = try mem.alloc(rows * g.linear_v_heads * 2);
        out.g_ = try mem.alloc(rows * g.linear_v_heads * 4);
        const chunks = (capacity + 127) / 128;
        out.pm = try mem.alloc(rows * g.query_heads * chunks * 4);
        out.pl = try mem.alloc(rows * g.query_heads * chunks * 4);
        out.po = try mem.alloc(rows * g.query_heads * chunks * c.head_dim * 4);
        out.logits = try mem.alloc(batch_rows * c.vocab * 2);
        for (out.snapshots, 0..) |*s, i| if (c.linear(i)) {
            s.* = .{ .recurrence = try mem.alloc(batch_rows * g.deltaBytes()), .conv = try mem.alloc(rows * g.convBytes()) };
        };
        return out;
    }

    pub fn deinit(self: *Scratch) void {
        self.memory.deinit();
        self.memory.gpa.free(self.snapshots);
    }
};
