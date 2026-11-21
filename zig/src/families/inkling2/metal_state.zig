//! Inkling 2 on Metal: one stream's exact state (keys/values, short-convolution histories), laid out so a rollback is a length reset.
//!
//! Every per-token quantity of the model is either a KV row or one of four short-convolution inputs (key, value,
//! attention output, MLP output). Global layers keep a KV row per position; sliding layers a ring wide enough that
//! rows of a rejected window never overwrite a key a re-run still reads. Convolution histories are rings of the last
//! `hist_ring` raw inputs: a forward reads positions len-3..len-1 and stores its own rows afterwards, so positions below
//! the new length stay intact for `max_rollback` rows.
const std = @import("std");
const mtl = @import("metal");
const config = @import("config.zig");

/// The most rows a `keep` may drop (a verify window: the pending token and its drafts).
pub const max_rollback = 16;
/// Slots of every convolution history: the 3 rows a window reads plus `max_rollback` plus slack, a power of two.
pub const hist_ring = 32;
/// Global layers' slots come in whole blocks of this many positions (the attention kernels' key blocks).
pub const global_block = 256;

pub const Geometry = struct {
    /// Positions a stream holds (global layers' keys); a stream's context limit.
    capacity: u32,
    /// The most rows of one forward over every item (Options.rows); sliding rings are sized by it.
    rows: u32,
};

/// Slots of a sliding layer's ring: its window, one forward's rows and a window's rollback, in whole 64s.
pub fn ringSlots(c: *const config.Text, rows: u32) u32 {
    return std.mem.alignForward(u32, c.window + @max(rows, max_rollback) + max_rollback, 64);
}

/// Slots of layer `i`'s keys: a ring (sliding) or one per position (global).
pub fn slots(c: *const config.Text, i: u32, g: Geometry) u32 {
    return if (config.isGlobal(i)) std.mem.alignForward(u32, g.capacity, global_block) else ringSlots(c, g.rows);
}

/// 0 where slots are positions, else the ring's slot count.
pub fn ring(c: *const config.Text, i: u32, g: Geometry) u32 {
    return if (config.isGlobal(i)) 0 else ringSlots(c, g.rows);
}

/// Width of one KV row of layer `i` in elements.
pub fn kvWidth(c: *const config.Text, i: u32) usize {
    const h = c.heads(i);
    return @as(usize, h.kv) * h.dim;
}

/// Byte offsets of layer `i`'s four histories in its history buffer, and its size.
pub const Hist = struct {
    k: usize,
    v: usize,
    attn: usize,
    mlp: usize,
    bytes: usize,
};

pub fn hist(c: *const config.Text, i: u32) Hist {
    const kv = kvWidth(c, i) * 2 * hist_ring;
    const full = @as(usize, c.hidden) * 2 * hist_ring;
    return .{ .k = 0, .v = kv, .attn = 2 * kv, .mlp = 2 * kv + full, .bytes = 2 * kv + 2 * full };
}

/// Bytes of layer `i`'s keys, and of its values (the same).
pub fn kvBytes(c: *const config.Text, i: u32, g: Geometry) usize {
    return @as(usize, slots(c, i, g)) * kvWidth(c, i) * 2;
}

/// Bytes a stream holds: every layer's keys and values and its histories.
pub fn bytes(cfg: *const config.Config, g: Geometry) u64 {
    var total: u64 = 0;
    for (0..cfg.text.executionLayers()) |l| {
        const i = cfg.text.physicalLayer(@intCast(l));
        total += 2 * kvBytes(&cfg.text, i, g) + hist(&cfg.text, i).bytes;
    }
    return total;
}

pub const State = struct {
    gpa: std.mem.Allocator,
    geometry: Geometry,
    layers: u32,
    /// Positions held: the next forward's rows start here.
    len: u32 = 0,
    /// Lowest length a `keep` may return to (see Model.keep).
    floor: u32 = 0,
    /// A forward that touched this stream failed: its rows may be half written, so only `clear` or a restore reuse it.
    failed: bool = false,
    /// Per layer: keys then values (private), and the four histories (shared, zeroed at reset).
    kv: []mtl.Buffer,
    hist: []mtl.Buffer,
    follow: ?mtl.ResidencySet.Follow = null,

    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, cfg: *const config.Config, g: Geometry) !State {
        const n = cfg.text.executionLayers();
        if (g.capacity == 0 or g.rows == 0) return error.InvalidGeometry;
        const kv = try gpa.alloc(mtl.Buffer, n);
        errdefer gpa.free(kv);
        const h = try gpa.alloc(mtl.Buffer, n);
        errdefer gpa.free(h);
        var made: usize = 0;
        errdefer for (0..made) |l| {
            kv[l].deinit();
            h[l].deinit();
        };
        for (0..n) |l| {
            const i = cfg.text.physicalLayer(@intCast(l));
            kv[l] = try device.buffer(@max(2 * kvBytes(&cfg.text, i, g), 16), mtl.ResourceOptions.private);
            errdefer kv[l].deinit();
            h[l] = try device.buffer(hist(&cfg.text, i).bytes, mtl.ResourceOptions.shared);
            @memset(h[l].contents()[0..h[l].length()], 0);
            made += 1;
        }
        return .{ .gpa = gpa, .geometry = g, .layers = n, .kv = kv, .hist = h };
    }

    /// Every buffer enrolled with (or released from) a residency hook, the spark-style follow.
    pub fn setResident(self: *State, hook: ?mtl.ResidencySet.Follow) void {
        if (self.follow) |f| self.apply(f, false);
        self.follow = hook;
        if (hook) |f| self.apply(f, true);
    }

    fn apply(self: *State, f: mtl.ResidencySet.Follow, joining: bool) void {
        for (self.kv, self.hist) |k, h| {
            mtl.resident.fields.one(k, f, joining);
            mtl.resident.fields.one(h, f, joining);
        }
    }

    /// A fresh stream: no positions, zero histories (positions before 0 read as zeros).
    pub fn clear(self: *State) void {
        for (self.hist) |h| @memset(h.contents()[0..h.length()], 0);
        self.len = 0;
        self.floor = 0;
        self.failed = false;
    }

    pub fn deinit(self: *State) void {
        self.setResident(null);
        for (self.kv, self.hist) |k, h| {
            k.deinit();
            h.deinit();
        }
        self.gpa.free(self.kv);
        self.gpa.free(self.hist);
    }
};

test "inkling2 state: rings, histories and a stream's bytes follow the geometry" {
    var t: config.Text = std.mem.zeroes(config.Text);
    t.window = 1024;
    t.layers = 48;
    t.hidden = 4608;
    t.local = .{ .q = 48, .kv = 8, .dim = 128 };
    t.global = .{ .q = 48, .kv = 8, .dim = 128 };
    const g: Geometry = .{ .capacity = 5000, .rows = 128 };
    // window + one forward's rows + a window's rollback, in whole 64s
    try std.testing.expectEqual(@as(u32, 1216), ringSlots(&t, g.rows));
    try std.testing.expectEqual(@as(u32, 5120), slots(&t, 5, g));
    try std.testing.expectEqual(@as(u32, 0), ring(&t, 5, g));
    try std.testing.expectEqual(@as(u32, 1216), ring(&t, 0, g));
    // a decode-sized forward still keeps the rollback slack
    try std.testing.expectEqual(@as(u32, 1088), ringSlots(&t, 4));
    // the convolution histories: 2 KV inputs and 2 hidden inputs a layer, 32 slots each
    const h = hist(&t, 0);
    try std.testing.expectEqual(@as(usize, 2 * 1024 * 2 * 32 + 2 * 4608 * 2 * 32), h.bytes);
    try std.testing.expectEqual(h.bytes - 4608 * 2 * 32, h.mlp);
    var total: u64 = 0;
    for (0..48) |l| total += 2 * kvBytes(&t, @intCast(l), g) + hist(&t, @intCast(l)).bytes;
    const cfg: config.Config = .{ .text = t, .vision = undefined, .mtp = null, .eos = 0, .image_token = 0 };
    try std.testing.expectEqual(total, bytes(&cfg, g));
}
