//! Metal-side importer: checks each neutral part against the plan, lands it once; `complete` when all parts landed.
const std = @import("std");
const g = @import("geometry.zig");
const c = @import("codec.zig");
const plan = @import("plan.zig");
const metal = @import("metal.zig");

pub const Error = error{ BadPart, Duplicate, Incomplete, NonFinite, OutOfMemory } || metal.Error || error{ BadGeometry, BadShape };

pub const Options = struct {
    /// land the DSpark blocks' rings when the layout holds them (a wrong ring costs draft acceptance, never a token)
    drafter: bool = true,
};

pub const Result = struct {
    upto: u64,
    /// the drafter's valid key positions [ds_first, ds_end) (equal: none landed)
    ds_first: u64,
    ds_end: u64,
    /// the parts carried drafter rings the layout could not hold, or the caller declined
    drafter_dropped: bool,
};

const Key = struct { shard: u32, layer: u32, kind: plan.Kind, first_page: u32 };

pub const Import = struct {
    gpa: std.mem.Allocator,
    shape: plan.Shape,
    layout: metal.Layout,
    opts: Options,
    known: std.AutoHashMapUnmanaged(Key, bool) = .empty,
    /// parts not yet landed; shards' parts may land from several threads (`put` of different shards never overlap)
    remaining: std.atomic.Value(usize) = .init(0),
    ds_landed: bool = false,

    /// `shape.shard` is ignored (a decoder pulls every shard); `shape.shards` / `split` / `ds_rows` say what will come.
    pub fn init(gpa: std.mem.Allocator, shape: plan.Shape, layout: metal.Layout, opts: Options) Error!Import {
        var im: Import = .{ .gpa = gpa, .shape = shape, .layout = layout, .opts = opts };
        im.shape.shard = 0;
        try im.shape.check();
        errdefer im.known.deinit(gpa);
        try layout.begin(im.shape.geo, im.shape.upto);
        var s: u32 = 0;
        while (s < im.shape.shards) : (s += 1) {
            var sh = im.shape;
            sh.shard = s;
            const list = try plan.specs(gpa, sh);
            defer gpa.free(list);
            for (list) |spec| try im.known.put(gpa, .{ .shard = s, .layer = spec.layer, .kind = spec.kind, .first_page = spec.first_page }, false);
        }
        im.remaining.store(im.known.count(), .monotonic);
        return im;
    }

    pub fn deinit(im: *Import) void {
        im.known.deinit(im.gpa);
    }

    /// One part of shard `shard`: `rows` x `cols` of `dtype`, `bytes` long.
    pub fn put(im: *Import, shard: u32, layer: u32, name: []const u8, dtype: plan.Dtype, rows: u64, cols: u64, bytes: []const u8) Error!void {
        if (shard >= im.shape.shards) return error.BadPart;
        var sh = im.shape;
        sh.shard = shard;
        const spec = plan.find(sh, layer, name) orelse return error.BadPart;
        if (dtype != spec.dtype or rows != spec.rows or cols != spec.cols or bytes.len != spec.bytes()) return error.BadPart;
        const key: Key = .{ .shard = shard, .layer = layer, .kind = spec.kind, .first_page = spec.first_page };
        const slot = im.known.getPtr(key) orelse return error.BadPart;
        if (slot.*) return error.Duplicate;
        try im.apply(sh, spec, bytes);
        slot.* = true;
        _ = im.remaining.fetchSub(1, .acq_rel);
    }

    fn apply(im: *Import, sh: plan.Shape, spec: plan.Spec, bytes: []const u8) Error!void {
        const geo = sh.geo;
        switch (spec.kind) {
            .swa => for (0..spec.rows) |i| {
                const row = c.loadRow(512, bytes[i * 1024 ..]);
                try finite(&row);
                im.layout.setWindow(spec.layer, sh.upto - spec.rows + i, &row);
            },
            .ring => {
                if (!im.opts.drafter or !im.layout.hasDrafter(geo.mtp_layers)) return;
                im.ds_landed = true;
                try landRing(im.layout, geo, sh.upto, spec.layer, spec.rows, bytes);
            },
            .carry => im.layout.setCarry(geo, geo.sourceIndex(spec.layer).?, sh.upto, bytes[0..2048], bytes[2048..4096]),
            .comp, .ixk => {
                const src = geo.sourceIndex(spec.layer).?;
                var it = plan.partPages(sh, src, spec.kind, spec.first_page).?;
                var at: usize = 0;
                while (it.next()) |run| for (0..run.n) |k| {
                    if (spec.kind == .comp) {
                        const row = c.loadRow(512, bytes[at..]);
                        try finite(&row);
                        im.layout.setPool(geo, src, run.row + k, &row);
                        at += 1024;
                    } else {
                        const row = c.loadRow(128, bytes[at..]);
                        try finite(&row);
                        im.layout.setIndex(geo, src, run.row + k, &row);
                        at += 256;
                    }
                };
            },
        }
    }

    /// A drafter ring part: `rows` bf16 rows, the keys of positions [upto - rows, upto) of stage `layer - geo.layers`.
    pub fn landRing(layout: metal.Layout, geo: g.Geometry, upto: u64, layer: u32, rows: u64, bytes: []const u8) Error!void {
        for (0..rows) |i| {
            const row = c.loadRow(512, bytes[i * 1024 ..]);
            try finite(&row);
            layout.setDrafter(layer - geo.layers, upto - rows + i, &row, geo.layers);
        }
    }

    fn finite(row: []const u16) Error!void {
        for (row) |v| if (!c.finiteBf16(v)) return error.NonFinite;
    }

    pub fn complete(im: *const Import) Error!Result {
        if (im.remaining.load(.acquire) != 0) return error.Incomplete;
        const upto = im.shape.upto;
        const landed = im.ds_landed;
        return .{ .upto = upto, .ds_first = if (landed) upto - im.shape.ds_rows else upto, .ds_end = upto, .drafter_dropped = im.shape.ds_rows > 0 and !landed };
    }
};

/// Engram lookback ids after `upto` prompt tokens: the last `n` (fewer for a short prompt), as the CUDA engine's tail.
pub fn lookbackIds(tokens: []const u32, upto: u64, n: u32) []const u32 {
    const end: usize = @intCast(@min(upto, tokens.len));
    return tokens[end -| n..end];
}
