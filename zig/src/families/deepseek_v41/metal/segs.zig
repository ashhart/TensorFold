//! A launch over several slots in one dispatch a stage: each layer's slot table, which the slot kernels index by row.
const std = @import("std");
const mtl = @import("metal");
const ctx_ = @import("ctx.zig");
const scratch = @import("scratch.zig");
const pipelines = @import("pipelines.zig");

const Ctx = ctx_.Ctx;
const Ref = ctx_.Ref;
const R = pipelines.max_rows;

/// The kernels' TfDsv41Seg: a slot's buffers at one layer (GPU addresses, 0 for none), then its rows from `base`.
pub const Seg = extern struct {
    window: u64 = 0,
    pool: u64 = 0,
    keys: u64 = 0,
    carry: u64 = 0,
    inputs: u64 = 0,
    /// the rows the attention reads after the window: the KV source's compressed rows, else the window
    comp: u64 = 0,
    /// the index keys of the layer's KV source
    ikeys: u64 = 0,
    base: u32,
    rows: u32,
    /// keys the slot's last row reaches at the layer's compress ratio, and its first position plus one
    nkeys: u32 = 0,
    start: u32 = 0,
};

comptime {
    std.debug.assert(@sizeOf(Seg) == scratch.seg_bytes);
}

/// Whether a launch takes the slot kernels: rows of more than one slot.
pub fn many(x: *const Ctx) bool {
    return x.segs.len > 1;
}

fn addr(b: ?mtl.Buffer) u64 {
    return if (b) |x| x.gpuAddress() else 0;
}

/// Every buffer of the launch's slots declared on `e` in one call: run once an encoder before its first `table`.
pub fn declare(x: *const Ctx, e: mtl.ComputeEncoder) void {
    if (!many(x)) return;
    var ids: [512]mtl.objc.Id = undefined;
    var n: usize = 0;
    const use = mtl.ResourceUsage.read | mtl.ResourceUsage.write;
    for (x.segs) |sg| for (sg.slot.regions) |*region| {
        if (n + 5 > ids.len) {
            e.useResources(ids[0..n], use);
            n = 0;
        }
        ids[n] = region.window.id;
        n += 1;
        inline for (.{ "pool", "keys", "carry", "inputs" }) |f| if (@field(region, f)) |b| {
            ids[n] = b.id;
            n += 1;
        };
    };
    e.useResources(ids[0..n], use);
}

/// Layer `l`'s table of the launch's slots written (each slot once); `declare` made its buffers resident.
pub fn table(x: *const Ctx, l: u32) !Ref {
    const c = x.c;
    if (x.segs.len > R) return error.BadRows;
    const ratio = c.compressRatio(l);
    const at: Ref = ctx_.plus(x.s.segs, @as(usize, l) * R * scratch.seg_bytes);
    const out: []Seg = @alignCast(std.mem.bytesAsSlice(Seg, x.s.bytes(at, x.segs.len * scratch.seg_bytes)));
    for (x.segs, 0..) |sg, j| {
        for (x.segs[0..j]) |prev| if (prev.slot == sg.slot) return error.SlotTwice;
        const region = &sg.slot.regions[l];
        const src = sg.slot.source(c, l);
        out[j] = .{
            .window = region.window.gpuAddress(),
            .pool = addr(region.pool),
            .keys = addr(region.keys),
            .carry = addr(region.carry),
            .inputs = addr(region.inputs),
            .comp = if (src) |r| r.pool.?.gpuAddress() else region.window.gpuAddress(),
            .ikeys = if (src) |r| addr(r.keys) else 0,
            .base = sg.r0,
            .rows = sg.n,
            .nkeys = if (ratio > 0) (sg.p0 + sg.n) / ratio else 0,
            .start = sg.p0 + 1,
        };
    }
    return at;
}

/// The row -> slot map of the launch (Meta.rowseg, written with the row records).
pub fn rowSlots(x: *const Ctx) Ref {
    return x.s.metaAt(scratch.Meta.rowseg);
}
