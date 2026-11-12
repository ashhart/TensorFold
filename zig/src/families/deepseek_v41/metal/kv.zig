//! A slot's per-layer state on Metal: the window ring, and at KV sources the compressed pool, index keys and carry.
const std = @import("std");
const mtl = @import("metal");
const Config = @import("dsv41").config.Config;
const pipelines = @import("pipelines.zig");

/// Window slots a layer keeps: the attention window plus one launch's rows, so no row's keys are overwritten early.
pub fn windowSlots(c: *const Config) u32 {
    return c.window + pipelines.max_rows;
}

/// Bytes of one index key: 64 bytes of E2M1 codes, then four E8M0 scales.
pub const index_key_bytes = 68;

/// One layer's regions. The window is mirrored (slot s and s + slots hold one key) so any 128 positions are one run.
pub const Region = struct {
    window: mtl.Buffer,
    /// compressed latents [positions / ratio][head_dim] bf16, at KV sources only
    pool: ?mtl.Buffer = null,
    /// index keys [positions / ratio][68], at KV sources only
    keys: ?mtl.Buffer = null,
    /// the ratio-2 compressor's carry: kv [2][head_dim] then score [2][head_dim], fp32
    carry: ?mtl.Buffer = null,
    /// the ratio-2 compressor's inputs of the last launch: kv [rows][head_dim] then score [rows][head_dim], fp32
    inputs: ?mtl.Buffer = null,
};

/// The compressed rows a KV source holds for `limit` positions.
pub fn poolRows(c: *const Config, layer: u32, limit: u32) u32 {
    return limit / c.compressRatio(layer) + 1;
}

/// Bytes of a slot's DSpark taps: [max_rows][targets][hidden] bf16, the launch's rows (the drafter's main_hidden).
pub fn tapBytes(c: *const Config) usize {
    return @as(usize, pipelines.max_rows) * c.dspark_targets.len * c.hidden * 2;
}

pub const Slot = struct {
    regions: []Region,
    limit: u32,
    a: std.mem.Allocator,
    /// the last launch's DSpark taps (null without targets): row r, target t at (r * targets + t) * hidden
    taps: ?mtl.Buffer = null,

    /// Every backbone layer's regions for positions [0, limit) and the taps, zeroed; the carry's scores start at -inf.
    pub fn init(a: std.mem.Allocator, device: mtl.Device, c: *const Config, limit: u32) !Slot {
        const regions = try a.alloc(Region, c.layers);
        errdefer a.free(regions);
        var made: usize = 0;
        errdefer for (regions[0..made]) |*r| free(r);
        const opts = mtl.ResourceOptions.shared;
        for (regions, 0..) |*r, l| {
            const layer: u32 = @intCast(l);
            const wbytes = 2 * @as(usize, windowSlots(c)) * c.head_dim * 2;
            r.* = .{ .window = try device.buffer(wbytes, opts) };
            made = l + 1;
            @memset(r.window.contents()[0..wbytes], 0);
            if (!c.isKvSource(layer)) continue;
            const rows = poolRows(c, layer, limit);
            r.pool = try device.buffer(@as(usize, rows) * c.head_dim * 2, opts);
            @memset(r.pool.?.contents()[0 .. @as(usize, rows) * c.head_dim * 2], 0);
            r.keys = try device.buffer(@as(usize, rows) * index_key_bytes, opts);
            @memset(r.keys.?.contents()[0 .. @as(usize, rows) * index_key_bytes], 0);
            if (c.compressRatio(layer) > 1) {
                r.carry = try device.buffer(4 * @as(usize, c.head_dim) * 4, opts);
                resetCarry(r, c);
                r.inputs = try device.buffer(inputBytes(c), opts);
                @memset(r.inputs.?.contents()[0..inputBytes(c)], 0);
            }
        }
        var taps: ?mtl.Buffer = null;
        if (tapBytes(c) > 0) {
            taps = try device.buffer(tapBytes(c), opts);
            @memset(taps.?.contents()[0..tapBytes(c)], 0);
        }
        return .{ .regions = regions, .limit = limit, .a = a, .taps = taps };
    }

    fn free(r: *Region) void {
        if (r.inputs) |b| b.deinit();
        r.window.deinit();
        if (r.pool) |b| b.deinit();
        if (r.keys) |b| b.deinit();
        if (r.carry) |b| b.deinit();
    }

    pub fn deinit(s: *Slot) void {
        for (s.regions) |*r| free(r);
        s.a.free(s.regions);
        if (s.taps) |t| t.deinit();
    }

    /// The region a layer's compressed attention reads: its KV source's.
    pub fn source(s: *const Slot, c: *const Config, layer: u32) ?*const Region {
        const src = c.kvSource(layer) orelse return null;
        return &s.regions[src];
    }

    /// Every region of `from` copied over this slot's, in a blit on `cb` (snapshots, test resets).
    pub fn copyFrom(s: *const Slot, cb: mtl.CommandBuffer, from: *const Slot) void {
        const b = cb.blit();
        for (s.regions, from.regions) |to, fr| {
            b.copy(fr.window, 0, to.window, 0, to.window.length());
            inline for (.{ "pool", "keys", "carry" }) |f| {
                if (@field(to, f)) |t| b.copy(@field(fr, f).?, 0, t, 0, t.length());
            }
        }
        if (s.taps) |t| b.copy(from.taps.?, 0, t, 0, t.length());
        b.end();
    }
};

/// Bytes of a ratio-2 region's compressor inputs: kv then score rows of one launch, fp32.
pub fn inputBytes(c: *const Config) usize {
    return 2 * @as(usize, pipelines.max_rows) * c.head_dim * 4;
}

/// A carry as a new slot's: kv 0, scores -inf.
pub fn resetCarry(r: *const Region, c: *const Config) void {
    const v = (r.carry orelse return).slice(f32, 4 * c.head_dim);
    @memset(v[0 .. 2 * c.head_dim], 0);
    @memset(v[2 * c.head_dim ..], -std.math.inf(f32));
}

/// After keeping `kept` rows of a launch from `start`: the carry holds the last kept even row, as if none followed.
pub fn rewindCarry(r: *const Region, c: *const Config, start: u64, kept: u32) void {
    const inputs = r.inputs orelse return;
    if (kept == 0 or (start + kept) % 2 == 0) return; // an even next row overwrites both slots before pooling
    const D: usize = c.head_dim;
    const v = r.carry.?.slice(f32, 4 * D);
    const in = inputs.slice(f32, 2 * pipelines.max_rows * D);
    @memcpy(v[0..D], in[(kept - 1) * D ..][0..D]);
    @memcpy(v[2 * D ..][0..D], in[(pipelines.max_rows + kept - 1) * D ..][0..D]);
}

/// The two window slots of position `pos`.
pub fn windowRows(c: *const Config, pos: u32) [2]u32 {
    const n = windowSlots(c);
    return .{ pos % n, pos % n + n };
}
