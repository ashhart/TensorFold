//! Where an imported state lands: a descriptor per Metal engine layout (`release`, `packed`) over host buffer slices.
const std = @import("std");
const g = @import("geometry.zig");
const c = @import("codec.zig");
const fp4 = @import("fp4.zig");

pub const Error = error{ TargetTooSmall, LayoutMismatch, NoDrafterRings };

pub const Release = struct {
    /// window + pipelines.max_rows: positions p live in slot p % window_slots and p % window_slots + window_slots
    window_slots: u32 = 144,
    /// cfg.window: the drafter's ring row of position p is p % ds_window
    ds_window: u32 = 128,
    layers: []const Layer,
    /// each drafter stage's slot ring: at least ds_window rows of head_dim bf16
    drafter: []const []u8 = &.{},

    pub const Layer = struct {
        window: []u8,
        pool: []u8 = &.{},
        keys: []u8 = &.{},
        carry: []u8 = &.{},
    };
};

pub const Packed = struct {
    live: []u8,
    storage: []u8,
    /// 40 (backbone rings) or 43 (with the drafter's)
    layers: u32,
    /// rows of the ratio-1 owner (an owner of ratio r holds capacity / r)
    capacity: u32,

    pub const ring_stride = 528;
    pub const main_stride = 288;
    pub const index_stride = 68;
    pub const window = 128;
    pub const capture_rows = 8;
    pub const owners = [_]u32{ 2, 8, 14, 20 };
    pub const ratios = [_]u32{ 2, 2, 2, 1 };

    pub fn ringBytes(p: Packed) usize {
        return @as(usize, p.layers) * window * ring_stride;
    }
    pub const partial_bytes = 7 * 2 * 512 * 4;
    pub const capture_bytes = capture_rows * 2 * 512 * 4;
    pub fn liveBytes(p: Packed) usize {
        return p.ringBytes() + partial_bytes + owners.len * capture_bytes;
    }
    pub fn storageBytes(capacity: u32) usize {
        var n: usize = 0;
        for (ratios) |r| n += @as(usize, capacity / r) * (main_stride + index_stride);
        return n;
    }
    pub fn mainOff(p: Packed, id: usize) usize {
        var off: usize = 0;
        for (ratios[0..id]) |r| off += @as(usize, p.capacity / r) * (main_stride + index_stride);
        return off;
    }
    pub fn partialOff(p: Packed, id: usize) usize {
        var off = p.ringBytes();
        for (ratios[0..id]) |r| off += 2 * r * 512 * 4;
        return off;
    }
    pub fn captureOff(p: Packed, id: usize) usize {
        return p.ringBytes() + partial_bytes + id * capture_bytes;
    }
};

pub const Layout = union(enum) {
    release: Release,
    @"packed": Packed,

    /// Checks buffers against `geo` and `upto`, then cleans the slot's window, carry and capture; pool/key rows stay.
    pub fn begin(l: Layout, geo: g.Geometry, upto: u64) Error!void {
        switch (l) {
            .release => |r| {
                if (r.layers.len != geo.layers or r.window_slots == 0 or r.ds_window == 0) return error.LayoutMismatch;
                for (r.layers, 0..) |layer, i| {
                    if (layer.window.len < 2 * @as(usize, r.window_slots) * 1024) return error.TargetTooSmall;
                    @memset(layer.window, 0);
                    const src = geo.sourceIndex(@intCast(i)) orelse continue;
                    const rows: usize = @intCast(geo.rowsOf(src, upto));
                    if (layer.pool.len < rows * 1024 or layer.keys.len < rows * 68) return error.TargetTooSmall;
                    if (geo.sources[src].ratio == 2) {
                        if (layer.carry.len < 4 * 512 * 4) return error.TargetTooSmall;
                        @memset(layer.carry[0 .. 2 * 512 * 4], 0);
                        for (0..2 * 512) |k| std.mem.writeInt(u32, layer.carry[(2 * 512 + k) * 4 ..][0..4], @bitCast(-std.math.inf(f32)), .little);
                    }
                }
                for (r.drafter) |d| if (d.len < @as(usize, r.ds_window) * 1024) return error.TargetTooSmall;
            },
            .@"packed" => |p| {
                if (geo.window != Packed.window or geo.head_dim != 512 or geo.sources.len != Packed.owners.len) return error.LayoutMismatch;
                for (geo.sources, Packed.owners, Packed.ratios) |s, o, r| if (s.layer != o or s.ratio != r) return error.LayoutMismatch;
                if ((p.layers != 40 and p.layers != 43) or p.layers < geo.layers or p.capacity == 0) return error.LayoutMismatch;
                if (upto > p.capacity or p.live.len < p.liveBytes() or p.storage.len < Packed.storageBytes(p.capacity)) return error.TargetTooSmall;
                @memset(p.live[0..p.liveBytes()], 0);
            },
        }
    }

    pub fn hasDrafter(l: Layout, stages: u32) bool {
        return switch (l) {
            .release => |r| r.drafter.len >= stages,
            .@"packed" => |p| p.layers >= 40 + stages,
        };
    }

    /// Position `pos`'s window key of a backbone layer.
    pub fn setWindow(l: Layout, layer: u32, pos: u64, row: *const [512]u16) void {
        switch (l) {
            .release => |r| {
                const n: u64 = r.window_slots;
                const w = r.layers[layer].window;
                inline for (.{ pos % n, pos % n + n }) |slot| c.storeRow(512, row, w[@as(usize, @intCast(slot)) * 1024 ..]);
            },
            .@"packed" => |p| ring(p, layer, pos, row),
        }
    }

    /// Position `pos`'s key of DSpark stage `stage`.
    pub fn setDrafter(l: Layout, stage: u32, pos: u64, row: *const [512]u16, layers: u32) void {
        switch (l) {
            .release => |r| c.storeRow(512, row, r.drafter[stage][@as(usize, @intCast(pos % r.ds_window)) * 1024 ..]),
            .@"packed" => |p| ring(p, layers + stage, pos, row),
        }
    }

    fn ring(p: Packed, layer: u32, pos: u64, row: *const [512]u16) void {
        const at = (@as(usize, layer) * Packed.window + @as(usize, @intCast(pos % Packed.window))) * Packed.ring_stride;
        fp4.window(row, p.live[at..][0..Packed.ring_stride]);
    }

    /// Compressed latent row `r` of source `src`.
    pub fn setPool(l: Layout, geo: g.Geometry, src: u32, r: u64, row: *const [512]u16) void {
        switch (l) {
            .release => |rl| c.storeRow(512, row, rl.layers[geo.sources[src].layer].pool[@as(usize, @intCast(r)) * 1024 ..]),
            .@"packed" => |p| fp4.mainRow(row, p.storage[p.mainOff(src) + @as(usize, @intCast(r)) * Packed.main_stride ..][0..Packed.main_stride]),
        }
    }

    /// Index key row `r` of source `src`.
    pub fn setIndex(l: Layout, geo: g.Geometry, src: u32, r: u64, row: *const [128]u16) void {
        switch (l) {
            .release => |rl| fp4.releaseIndexKey(row, rl.layers[geo.sources[src].layer].keys[@as(usize, @intCast(r)) * 68 ..][0..68]),
            .@"packed" => |p| {
                const rows: usize = p.capacity / Packed.ratios[src];
                fp4.index(row, p.storage[p.mainOff(src) + rows * Packed.main_stride + @as(usize, @intCast(r)) * Packed.index_stride ..][0..Packed.index_stride]);
            },
        }
    }

    /// The ratio-2 compressor row of position upto-1 (f32 [512] kv and gate, little-endian) in slot (upto-1) % 2.
    pub fn setCarry(l: Layout, geo: g.Geometry, src: u32, upto: u64, kv: []const u8, gate: []const u8) void {
        const slot: usize = @intCast((upto - 1) % 2);
        switch (l) {
            .release => |rl| {
                const carry = rl.layers[geo.sources[src].layer].carry;
                @memcpy(carry[slot * 2048 ..][0..2048], kv);
                @memcpy(carry[(2 + slot) * 2048 ..][0..2048], gate);
            },
            .@"packed" => |p| {
                const at = p.partialOff(src);
                @memcpy(p.live[at + slot * 2048 ..][0..2048], kv);
                @memcpy(p.live[at + (2 + slot) * 2048 ..][0..2048], gate);
                // the raw-row capture a rollback re-cuts a straddling group from: one slot a position, mod 8
                const cap = p.captureOff(src) + @as(usize, @intCast((upto - 1) % Packed.capture_rows)) * 2 * 2048;
                @memcpy(p.live[cap..][0..2048], kv);
                @memcpy(p.live[cap + 2048 ..][0..2048], gate);
            },
        }
    }
};
