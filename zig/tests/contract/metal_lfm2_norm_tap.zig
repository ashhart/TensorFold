//! Norm/tap capture to per-row destinations replaces norm plus copies, bit for bit.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;
const hidden = 2048;
const topk = 4;
const max_cols = 4;
/// A launch's raw-state destination. `mask` names the destination columns written, duplicates included.
const Layout = struct { dst_cols: usize, mask: u32 };
const layouts = [_]Layout{
    .{ .dst_cols = 1, .mask = 0b1 }, // one column
    .{ .dst_cols = 4, .mask = 0b1001 }, // first and last column of four
    .{ .dst_cols = 4, .mask = 0b0110 }, // one layer captured twice, guard columns either side
};

/// A row with no destination keeps its columns' prior values.
fn dark(row: usize) bool {
    return row % 5 == 3;
}
const shapes = [_]c.Shape{
    .{ .rows = 1, .n = hidden, .k = hidden },
    .{ .rows = 10, .n = hidden, .k = hidden },
    .{ .rows = 32, .n = hidden, .k = hidden },
    .{ .rows = 2048, .n = hidden, .k = hidden },
};
pub const Which = enum { norm, norm_first, combine };
fn depth(_: c.Shape) usize {
    return 14;
}
pub fn spec(w: Which) c.Spec {
    return .{
        .name = switch (w) {
            .norm => "metal lfm2 norm tap lfm2_norm_tap",
            .norm_first => "metal lfm2 norm tap first lfm2_norm_tap",
            .combine => "metal lfm2 combine norm tap lfm2_combine_norm_tap",
        },
        .accum = .f32,
        .depth = depth,
        .ops = 8,
        .mids = 1,
        .out = .bf16,
        .served = &shapes,
        .replaces = "LFM2 norm followed by tap copy",
        .nan = .{ .stated = "all output bytes equal the original norm and copy, including NaNs" },
        .sums = .{ .bf16 = 0, .why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps" },
    };
}
pub fn controlName(w: Which) []const u8 {
    return switch (w) {
        .norm => "control: metal lfm2 norm tap wrong column",
        .norm_first => "control: metal lfm2 norm tap first wrong column",
        .combine => "control: metal lfm2 combine norm tap wrong column",
    };
}
pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    which: Which,
    control: bool,
    fused: mtl.Pipeline,
    old: mtl.Pipeline,
    tap: mtl.Pipeline,
    r: ?rows.Rows = null,
    initial: []u16 = &.{},
    prior: []u16 = &.{},
    residual: rig_.Ref = undefined,
    weights: rig_.Ref = undefined,
    router: rig_.Ref = undefined,
    /// Per layout the normalized output, the updated state and the destination; `tab` holds each row's address.
    x: [layouts.len]rig_.Ref = undefined,
    hs: [layouts.len]rig_.Ref = undefined,
    dst: [layouts.len]rig_.Ref = undefined,
    tab: rig_.Ref = undefined,
    bx: rig_.Ref = undefined,
    bh: rig_.Ref = undefined,
    bdst: rig_.Ref = undefined,
    baseline: std.ArrayList(u8) = .empty,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, w: Which, control: bool) !Probe {
        const needle = "size_t(ctz(m)) * D";
        if (control and std.mem.indexOf(u8, sources.lfm2, needle) == null) return error.ControlEditMissing;
        const text = if (control) try std.mem.replaceOwned(u8, a, sources.lfm2, needle, "size_t(ctz(m) ^ uint(mask > 1)) * D") else sources.lfm2;
        defer if (control) a.free(text);
        const fused = try rig_.pipeline(device, text, if (w == .combine) "lfm2_combine_norm_tap" else "lfm2_norm_tap");
        errdefer fused.deinit();
        const old = try rig_.pipeline(device, sources.lfm2, if (w == .combine) "lfm2_combine_norm" else "lfm2_norm");
        errdefer old.deinit();
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .which = w, .control = control, .fused = fused, .old = old, .tap = try rig_.pipeline(device, sources.lfm2, "lfm2_tap") };
    }
    fn drop(p: *Probe) void {
        if (p.r) |r| r.deinit();
        p.r = null;
        p.a.free(p.initial);
        p.a.free(p.prior);
        p.initial = &.{};
        p.prior = &.{};
        p.rig.reset();
    }
    pub fn deinit(p: *Probe) void {
        p.drop();
        p.rig.deinit();
        p.baseline.deinit(p.a);
        p.fused.deinit();
        p.old.deinit();
        p.tap.deinit();
    }
    const vt = g.hooks(Probe);
    pub fn kernel(p: *Probe) g.Kernel {
        var s = spec(p.which);
        if (p.control) s.name = controlName(p.which);
        return .{ .ctx = p, .vt = &vt, .spec = s, .control = if (p.control) .shapes else null };
    }
    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.drop();
        if (shape.n != hidden or shape.k != hidden or shape.rows == 0) return false;
        const r = try rows.Rows.init(p.a, .rms, shape);
        p.r = r;
        r.fill(fill, seed);
        p.initial = try p.a.dupe(u16, r.x);
        const n = r.outputs();
        p.prior = try p.a.alloc(u16, n * max_cols);
        @memset(p.prior, 0x3f80);
        const res = try p.a.alloc(u16, n * if (p.which == .combine) @as(usize, topk) else 1);
        defer p.a.free(res);
        @memset(res, 0);
        const rw = try p.a.alloc(u16, shape.rows * topk);
        defer p.a.free(rw);
        @memset(rw, 0);
        // Normal cases exercise residual and expert combine; stress leaves h unchanged so Rows can simulate its sums.
        if (fill != .stress) {
            const expert = [_]f32{ 0.125, -0.0625, 0.03125, 0.015625 };
            const combined: f32 = 0.046875;
            for (0..n) |i| {
                if (p.which != .combine) {
                    res[i] = c.toBf16(0.125);
                } else for (expert, 0..) |v, k| {
                    res[(i / hidden * topk + k) * hidden + i % hidden] = c.toBf16(v);
                }
                r.x[i] = if (p.which == .norm_first) p.initial[i] else c.toBf16(@as(f32, @floatCast(c.fromBf16(p.initial[i]))) + if (p.which == .norm) @as(f32, 0.125) else combined);
            }
            for (0..shape.rows) |row| for (0..topk) |k| {
                rw[row * topk + k] = c.toBf16(if (k == 0) 0.5 else 1);
            };
        }
        if (fill == .nonfinite) {
            // Exercise non-finites in the other operands, not only the hidden row.
            r.w[1] = 0x7fc0;
            if (shape.rows > 4) {
                res[4 * hidden * if (p.which == .combine) @as(usize, topk) else 1] = 0x7f80;
                if (p.which == .combine) rw[4 * topk + 1] = 0xff80;
            }
        }
        p.residual = try p.rig.input(std.mem.sliceAsBytes(res));
        p.weights = try p.rig.input(std.mem.sliceAsBytes(r.w));
        p.router = try p.rig.input(std.mem.sliceAsBytes(rw));
        // x of the first layout must be first: gate 4 reads its normalized outputs for the sum test.
        for (layouts, 0..) |l, i| {
            p.x[i] = try p.rig.output(n * 2);
            p.hs[i] = try p.rig.output(n * 2);
            p.dst[i] = try p.rig.output(n * l.dst_cols * 2);
        }
        p.tab = try p.rig.scratch(shape.rows * 8);
        p.bx = try p.rig.scratch(n * 2);
        p.bh = try p.rig.scratch(n * 2);
        p.bdst = try p.rig.scratch(n * max_cols * 2);
        return true;
    }
    pub fn count(p: *Probe) usize {
        return p.r.?.outputs();
    }
    /// One layout through the fused kernel, or through the original norm and the copy kernel.
    fn launch(p: *Probe, li: usize, fused: bool) !void {
        const n = p.count();
        const l = layouts[li];
        const x = if (fused) p.x[li] else p.bx;
        const dst = if (fused) p.dst[li] else p.bdst;
        const h = if (fused) p.hs[li] else p.bh;
        @memcpy(p.rig.bytes(h, n * 2), std.mem.sliceAsBytes(p.initial));
        @memcpy(p.rig.bytes(dst, n * l.dst_cols * 2), std.mem.sliceAsBytes(p.prior[0 .. n * l.dst_cols]));
        const table = std.mem.bytesAsSlice(u64, p.rig.bytes(p.tab, p.r.?.shape.rows * 8));
        for (table, 0..) |*at, row| at.* = if (dark(row)) 0 else dst.buf.gpuAddress() + dst.off + row * l.dst_cols * hidden * 2;
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(if (fused) p.fused else p.old);
        if (p.which != .combine) {
            e.setBuffer(h.buf, h.off, 0);
            e.setBuffer(p.residual.buf, p.residual.off, 1);
            e.setBuffer(p.weights.buf, p.weights.off, 2);
            e.setValue(p.r.?.eps, 3);
            e.setValue(@as(u32, @intFromBool(p.which != .norm_first)), 4);
        } else {
            e.setBuffer(p.residual.buf, p.residual.off, 0);
            e.setBuffer(p.router.buf, p.router.off, 1);
            e.setBuffer(h.buf, h.off, 2);
            e.setBuffer(p.weights.buf, p.weights.off, 3);
            e.setValue(p.r.?.eps, 4);
        }
        e.setBuffer(x.buf, x.off, 5);
        if (fused) {
            e.useResource(dst.buf, mtl.ResourceUsage.read | mtl.ResourceUsage.write);
            e.setBuffer(p.tab.buf, p.tab.off, 6);
            e.setValue(l.mask, 7);
        }
        e.dispatchGroups(mtl.Size.of(p.r.?.shape.rows, 1, 1), mtl.Size.of(hidden / 4, 1, 1));
        if (!fused) for (0..l.dst_cols) |col| {
            if (l.mask >> @intCast(col) & 1 == 0) continue;
            e.setPipeline(p.tap);
            e.setBuffer(h.buf, h.off, 0);
            e.setBuffer(dst.buf, dst.off, 1);
            e.setValue([2]u32{ @intCast(l.dst_cols * hidden), @intCast(col * hidden) }, 2);
            e.dispatchThreads(mtl.Size.of(hidden / 8, p.r.?.shape.rows, 1), mtl.Size.of(256, 1, 1));
        };
        try p.rig.finish(pass);
    }
    pub fn run(p: *Probe) !void {
        for (0..layouts.len) |i| try p.launch(i, true);
    }
    /// The original norm, which updates its own copy of the state, then the copy kernel into each requested column.
    pub fn replaced(p: *Probe) ![]const u8 {
        p.baseline.clearRetainingCapacity();
        const n = p.count();
        for (layouts, 0..) |l, i| {
            // The tap copy reads the original norm's updated state: the buffer the old kernel wrote.
            try p.launch(i, false);
            const width = l.dst_cols * hidden;
            const dst = p.rig.bytes(p.bdst, n * l.dst_cols * 2);
            for (0..p.r.?.shape.rows) |row| if (dark(row)) @memcpy(dst[row * width * 2 ..][0 .. width * 2], std.mem.sliceAsBytes(p.prior[row * width ..][0..width]));
            try p.baseline.appendSlice(p.a, p.rig.bytes(p.bx, n * 2));
            try p.baseline.appendSlice(p.a, p.rig.bytes(p.bh, n * 2));
            try p.baseline.appendSlice(p.a, dst);
        }
        return p.baseline.items;
    }
    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        const actual = try p.a.dupe(u8, try p.rig.outputBytes());
        defer p.a.free(actual);
        if (!std.mem.eql(u8, actual, try p.replaced())) return "fused outputs differ from norm and tap";
        return null;
    }
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        try p.r.?.reference(want, mag);
    }
    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        try p.r.?.simulate(acc, order, out);
    }
};
