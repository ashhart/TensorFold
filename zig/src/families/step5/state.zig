//! A stream's keys and values (every layer, every position), and the scratch one window of up to 16 rows runs in.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");

/// Rows a forward call takes at most: the router's and the attention's launch limit.
pub const window_rows = 16;

pub const Cache = struct {
    keys: mtl.Buffer,
    values: mtl.Buffer,
    logits: mtl.Buffer,
    capacity: usize,
    len: usize = 0,

    pub fn init(device: mtl.Device, c: *const cfg.Config, capacity: usize) !Cache {
        const bytes = @as(usize, c.layers) * capacity * c.kvWidth() * 2;
        const keys = try device.buffer(bytes, mtl.ResourceOptions.private);
        errdefer keys.deinit();
        const values = try device.buffer(bytes, mtl.ResourceOptions.private);
        errdefer values.deinit();
        return .{ .keys = keys, .values = values, .logits = try device.buffer(@as(usize, c.vocab) * 2, mtl.ResourceOptions.shared), .capacity = capacity };
    }

    /// Byte offset of layer `l`'s position `p` in keys or values.
    pub fn at(self: *const Cache, c: *const cfg.Config, l: usize, p: usize) usize {
        return (l * self.capacity + p) * c.kvWidth() * 2;
    }

    pub fn deinit(self: *Cache) void {
        self.keys.deinit();
        self.values.deinit();
        self.logits.deinit();
    }
};

pub const Scratch = struct {
    ids: mtl.Buffer,
    resid: mtl.Buffer, // fp32 [rows, hidden]
    x: mtl.Buffer, // bf16 normed rows
    q: mtl.Buffer,
    qn: mtl.Buffer,
    k: mtl.Buffer,
    v: mtl.Buffer,
    g: mtl.Buffer,
    o: mtl.Buffer,
    y: mtl.Buffer, // a sublayer's bf16 output [rows, hidden]
    shared: mtl.Buffer,
    gate: mtl.Buffer,
    up: mtl.Buffer,
    act: mtl.Buffer,
    eg: mtl.Buffer, // [rows * topk, moe_intermediate] each
    eu: mtl.Buffer,
    ea: mtl.Buffer,
    ey: mtl.Buffer, // [rows * topk, hidden]
    logits32: mtl.Buffer,
    pick: mtl.Buffer,
    wts: mtl.Buffer,
    route: [7]mtl.Buffer, // the router's per-Mac groups, unused on one Mac
    mask: mtl.Buffer,
    head: mtl.Buffer,

    pub fn init(device: mtl.Device, c: *const cfg.Config, capacity: usize) !Scratch {
        const r: usize = window_rows;
        const h: usize = c.hidden;
        const slots = r * c.topk;
        const sizes = [_]usize{
            r * 4,                     r * h * 4,                 r * h * 2,          r * c.heads * c.head_dim * 2, r * c.heads * c.head_dim * 2,
            r * c.kvWidth() * 2,       r * c.kvWidth() * 2,       r * c.heads * 2,    r * c.heads * c.head_dim * 2, r * h * 2,
            r * h * 2,                 r * c.intermediate * 2,    r * c.intermediate * 2, r * c.intermediate * 2,  slots * c.moe_intermediate * 2,
            slots * c.moe_intermediate * 2, slots * c.moe_intermediate * 2, slots * h * 2, r * c.experts * 4,       slots * 4,
            slots * 4,
        };
        var s: Scratch = undefined;
        var made: usize = 0;
        const fields = comptime fieldNames();
        errdefer inline for (fields, 0..) |f, i| if (i < made) @field(s, f).deinit();
        inline for (fields, 0..) |f, i| {
            @field(s, f) = try device.buffer(sizes[i], mtl.ResourceOptions.shared);
            made += 1;
        }
        var routed: usize = 0;
        errdefer for (s.route[0..routed]) |b| b.deinit();
        for (&s.route) |*b| {
            b.* = try device.buffer(@as(usize, c.experts) * window_rows * 4 + 4096, mtl.ResourceOptions.shared);
            routed += 1;
        }
        s.mask = try device.buffer(2 * r * capacity, mtl.ResourceOptions.shared);
        errdefer s.mask.deinit();
        s.head = try device.buffer(@as(usize, c.vocab) * 2, mtl.ResourceOptions.shared);
        return s;
    }

    fn fieldNames() [21][]const u8 {
        return .{ "ids", "resid", "x", "q", "qn", "k", "v", "g", "o", "y", "shared", "gate", "up", "act", "eg", "eu", "ea", "ey", "logits32", "pick", "wts" };
    }

    pub fn deinit(s: *Scratch) void {
        inline for (comptime fieldNames()) |f| @field(s, f).deinit();
        for (s.route) |b| b.deinit();
        s.mask.deinit();
        s.head.deinit();
    }
};
