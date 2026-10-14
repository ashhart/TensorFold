//! Gemma 4 dense decode state: each stream's caches (a ring a sliding layer, a buffer a global one) and scratch.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");

const opts = mtl.ResourceOptions.private | mtl.ResourceOptions.untracked;
const shared = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// Window rows at most: the lane projection's widest, and one verify window.
pub const max_rows = 16;
/// A sliding layer's ring: the window plus rows a window may write past it (gilby's RingKVCache, WINDOW_ROWS).
pub const ring = cfg.window + 128;

/// One stream's keys and values, per layer [kv heads, slots, head dim] bf16, and how many positions it holds.
pub const Cache = struct {
    k: [cfg.layers]mtl.Buffer,
    v: [cfg.layers]mtl.Buffer,
    capacity: usize,
    len: usize = 0,

    pub fn init(device: mtl.Device, capacity: usize) !Cache {
        var c: Cache = .{ .k = undefined, .v = undefined, .capacity = capacity };
        var made: usize = 0;
        errdefer for (0..made) |i| {
            c.k[i].deinit();
            c.v[i].deinit();
        };
        for (0..cfg.layers) |i| {
            const slots = slotsOf(i, capacity);
            const bytes = cfg.kvHeads(i) * slots * cfg.headDim(i) * 2;
            c.k[i] = try device.buffer(bytes, opts);
            c.v[i] = device.buffer(bytes, opts) catch |e| {
                c.k[i].deinit();
                return e;
            };
            made += 1;
        }
        return c;
    }

    pub fn deinit(c: *Cache) void {
        for (0..cfg.layers) |i| {
            c.k[i].deinit();
            c.v[i].deinit();
        }
    }
};

/// Cache slots for layer `i`: the ring on a sliding layer, the whole capacity on a global one.
pub fn slotsOf(i: usize, capacity: usize) usize {
    return if (cfg.global(i)) capacity else ring;
}

/// Activations for one window of up to `max_rows` rows, shared by every layer.
pub const Scratch = struct {
    ids: mtl.Buffer,
    pos: mtl.Buffer,
    /// Per layer kind (0 sliding, 1 global): each row's first key, and the attention chunks' meta.
    lo: [2]mtl.Buffer,
    meta: [2]mtl.Buffer,
    h: [2]mtl.Buffer,
    normed: [2]mtl.Buffer,
    qkv: mtl.Buffer,
    q: mtl.Buffer,
    k: mtl.Buffer,
    v: mtl.Buffer,
    attn: mtl.Buffer,
    o: mtl.Buffer,
    hn: mtl.Buffer,
    n_mlp: mtl.Buffer,
    gate_up: mtl.Buffer,
    act: mtl.Buffer,
    y: mtl.Buffer,
    pm: mtl.Buffer,
    pl: mtl.Buffer,
    po: mtl.Buffer,
    logits: mtl.Buffer,
    /// Attention partial slots: query heads x rows x chunks, chunks of 64 keys over the capacity.
    slots: usize,

    pub fn init(device: mtl.Device, capacity: usize) !Scratch {
        const r = max_rows;
        const d = cfg.hidden;
        const qd = cfg.query_heads * cfg.global_dim;
        const slots = cfg.query_heads * r * (capacity / 64 + 2);
        var s: Scratch = undefined;
        s.slots = slots;
        inline for (.{ .{ "ids", r * 4, true }, .{ "pos", r * 4, true }, .{ "qkv", r * (qd + 2 * 16 * 256) * 2, false }, .{ "q", r * qd * 2, false }, .{ "k", r * 16 * 256 * 2, false }, .{ "v", r * 16 * 256 * 2, false }, .{ "attn", r * qd * 2, false }, .{ "o", r * d * 2, false }, .{ "hn", r * d * 2, false }, .{ "n_mlp", r * d * 2, false }, .{ "gate_up", r * 2 * cfg.intermediate * 2, false }, .{ "act", r * cfg.intermediate * 2, false }, .{ "y", r * d * 2, false }, .{ "pm", slots * 4, false }, .{ "pl", slots * 4, false }, .{ "po", slots * cfg.global_dim * 4, false }, .{ "logits", r * cfg.vocab * 2, true } }) |f|
            @field(s, f[0]) = try device.buffer(f[1], if (f[2]) shared else opts);
        for (&s.lo) |*b| b.* = try device.buffer(r * 4, shared);
        for (&s.meta) |*b| b.* = try device.buffer(16, shared);
        for (&s.h) |*b| b.* = try device.buffer(r * d * 2, opts);
        for (&s.normed) |*b| b.* = try device.buffer(r * d * 2, opts);
        return s;
    }

    pub fn deinit(s: *Scratch) void {
        inline for (.{ "ids", "pos", "qkv", "q", "k", "v", "attn", "o", "hn", "n_mlp", "gate_up", "act", "y", "pm", "pl", "po", "logits" }) |f| @field(s, f).deinit();
        for (s.lo ++ s.meta) |b| b.deinit();
        for (s.h) |b| b.deinit();
        for (s.normed) |b| b.deinit();
    }
};
