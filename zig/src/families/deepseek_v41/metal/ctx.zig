//! What a layer's encoders share in one launch: the config, kernels, buffers, the slot's state and the rows' geometry.
const std = @import("std");
const mtl = @import("metal");
const Config = @import("dsv41").config.Config;
const pipelines = @import("pipelines.zig");
const scratch = @import("scratch.zig");
const kv = @import("kv.zig");
const rope = @import("rope.zig");
const weights = @import("weights.zig");

pub const Ref = pipelines.glue.Ref;
const fp8 = pipelines.fp8;

/// One slot's rows of a launch: its state, first position and row count; rows follow in order from row `r0`.
pub const Seg = struct { slot: *const kv.Slot, p0: u32, r0: u32, n: u32 };

pub const Ctx = struct {
    c: *const Config,
    p: *const pipelines.Pipelines,
    s: *const scratch.Scratch,
    slot: *const kv.Slot,
    tables: *const rope.Tables,
    /// the launch's first position and its row count (consecutive positions of one stream)
    p0: u32,
    rows: u32,
    /// rows of several slots, one segment each in row order (empty: the launch is one stream's `p0`, `rows`)
    segs: []const Seg = &.{},
    /// first row of this context inside the launch's buffers (nonzero only in a segment's context of `parts`)
    r0: u32 = 0,

    /// Total rows of the launch.
    pub fn total(x: *const Ctx) u32 {
        return if (x.segs.len == 0) x.rows else x.segs[x.segs.len - 1].r0 + x.segs[x.segs.len - 1].n;
    }

    /// The contexts of the slot-bound stages: one a segment, each over its rows only (just this one without segments).
    pub fn parts(x: *const Ctx, out: *[pipelines.max_rows]Ctx) []const Ctx {
        if (x.segs.len == 0) {
            out[0] = x.*;
            return out[0..1];
        }
        for (x.segs, 0..) |sg, i| {
            out[i] = x.*;
            out[i].slot = sg.slot;
            out[i].p0 = sg.p0;
            out[i].rows = sg.n;
            out[i].r0 = sg.r0;
            out[i].segs = &.{};
        }
        return out[0..x.segs.len];
    }

    /// A row-indexed buffer at this context's first row, `bytes` a row.
    pub fn row(x: *const Ctx, r: Ref, bytes: usize) Ref {
        return plus(r, @as(usize, x.r0) * bytes);
    }

    /// The row records at Meta index `at`, `stride` entries a row, from this context's first row.
    pub fn metaRow(x: *const Ctx, at: usize, stride: usize) Ref {
        return x.s.metaAt(at + stride * x.r0);
    }

    /// A dense FP8 projection of the launch's rows: x [rows][k] bf16 to y [rows][n].
    pub fn proj(x: *const Ctx, e: mtl.ComputeEncoder, in: Ref, w: weights.Fp8, out: Ref, f32_out: bool) !void {
        try x.projBatch(e, in, w, out, f32_out, .{});
    }

    /// `proj` over a batch of same-shape launches in one dispatch (fp8.Batch strides from the first one).
    pub fn projBatch(x: *const Ctx, e: mtl.ComputeEncoder, in: Ref, w: weights.Fp8, out: Ref, f32_out: bool, b: fp8.Batch) !void {
        const call: fp8.Call = .{ .k = w.k, .n = w.n, .rows = x.rows, .split = fp8.splitOf(w.k), .out = if (f32_out) .f32 else .bf16 };
        try fp8.encodeWhole(&x.p.fp8, e, cv(fp8.Ref, in), cv(fp8.Ref, w.w), cv(fp8.Ref, w.s), cv(fp8.Ref, out), call, b);
    }

    /// The row records of a compress ratio (1 or 2) at Meta index `base`.
    pub fn byRatio(x: *const Ctx, base: usize, ratio: u32) Ref {
        return x.s.metaAt(base + (ratio - 1) * pipelines.max_rows + x.r0);
    }

    pub fn meta(x: *const Ctx, at: usize) Ref {
        return x.s.metaAt(at + x.r0);
    }

    pub fn table(x: *const Ctx, layer: u32) Ref {
        return .{ .buf = x.tables.of(if (x.c.compressRatio(layer) > 0) .compressed else .window) };
    }

    /// Keys the launch's last row reaches at a compress ratio.
    pub fn keys(x: *const Ctx, ratio: u32) u32 {
        return (x.p0 + x.rows) / ratio;
    }
};

/// A buffer place as another helper's Ref type.
pub fn cv(comptime T: type, r: Ref) T {
    return .{ .buf = r.buf, .off = r.off };
}

/// A buffer at an offset from a place.
pub fn plus(r: Ref, bytes: usize) Ref {
    return .{ .buf = r.buf, .off = r.off + bytes };
}

pub fn whole(b: mtl.Buffer) Ref {
    return .{ .buf = b };
}

test "a launch's keys follow its last row" {
    const c: Config = .{};
    var x: Ctx = undefined;
    x.c = &c;
    x.p0 = 9;
    x.rows = 4;
    try std.testing.expectEqual(@as(u32, 13), x.keys(1));
    try std.testing.expectEqual(@as(u32, 6), x.keys(2));
}
