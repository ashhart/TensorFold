//! The mixed-row head norm: unchanged arithmetic, frozen-byte equality, FP64 bounds and guarded strided inputs.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const ct = @import("contract");
const rr = @import("rig.zig");
const c = ct.spec;
const g = ct.gates;
const q = tf.qwen27;
const D = 256;
fn depth(_: c.Shape) usize {
    return 14;
}
pub fn label(bad: bool) []const u8 {
    return if (bad) "control: q27 mixed head norm missing native rounding" else "metal q27 mixed head norm";
}
pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rr.Rig,
    bad: bool,
    pipe: mtl.Pipeline,
    old: mtl.Pipeline,
    r: ?ct.rows.Rows = null,
    heads: u32 = 0,
    rows: u32 = 0,
    x: rr.Ref = undefined,
    contiguous: rr.Ref = undefined,
    w: rr.Ref = undefined,
    out: rr.Ref = undefined,
    before: rr.Ref = undefined,
    const vt = g.hooks(Probe);
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, bad: bool) !Probe {
        const raw = q.runtime_sources.glue;
        const needle = "q27_round(value * inverse[0], a.activation)";
        if (std.mem.count(u8, raw, needle) != 1) return error.MixedControlEdit;
        const text = if (bad) try std.mem.replaceOwned(u8, a, raw, needle, "(value * inverse[0])") else try a.dupe(u8, raw);
        defer a.free(text);
        const combined = try std.mem.concat(a, u8, &.{ text, @embedFile("q27_mixed_reference.metal") });
        defer a.free(combined);
        const lib = try mtl.Library.fromSource(device, combined, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const pipe = try mtl.Pipeline.init(device, lib, "q27_head_norm", false);
        errdefer pipe.deinit();
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .bad = bad, .pipe = pipe, .old = try mtl.Pipeline.init(device, lib, "q27_head_norm_before_mixed", false) };
    }
    pub fn deinit(p: *Probe) void {
        if (p.r) |r| r.deinit();
        p.pipe.deinit();
        p.old.deinit();
        p.rig.deinit();
    }
    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .ctx = p, .vt = &vt, .control = if (p.bad) .shapes else null, .spec = .{
            .name = label(p.bad),
            .replaces = "ec344c2bab q27_head_norm on unstacked rows",
            .accum = .f32,
            .depth = depth,
            .ops = 8,
            .mids = 1,
            .out = .bf16,
            .served = &.{ .{ .rows = 16, .n = 4, .k = D }, .{ .rows = 1, .n = 24, .k = D }, .{ .rows = 16, .n = 2, .k = D }, .{ .rows = 16, .n = 12, .k = D }, .{ .rows = 128, .n = 16, .k = D }, .{ .rows = 2048, .n = 4, .k = D }, .{ .rows = 2048, .n = 24, .k = D }, .{ .rows = 1, .n = 32, .k = D } },
            .tiles = .{ .{ .size = 16 }, .{ .step = 2, .min = 2 }, .{ .size = D, .step = D, .min = D } },
        } };
    }
    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 2048 or shape.k != D or shape.n == 0 or shape.n > 32) return false;
        if (p.r) |r| r.deinit();
        p.r = null;
        p.rig.reset();
        p.rows = @intCast(shape.rows);
        p.heads = @intCast(shape.n);
        const r = try ct.rows.Rows.init(p.a, .rms, .{ .rows = shape.rows * shape.n, .n = D, .k = D });
        p.r = r;
        r.fill(fill, seed);
        if (fill == .stress) @import("q27_mixed_inputs.zig").stress(r.x, r.w);
        const mixed = try p.a.alloc(u16, r.x.len * 2);
        defer p.a.free(mixed);
        @memset(mixed, 0x7fc5);
        const is_q = shape.n >= 12;
        for (0..shape.rows * shape.n) |head| {
            const base = if (is_q) head * 2 * D else (head / shape.n) * 2 * shape.n * D + (head % shape.n) * D;
            @memcpy(mixed[base..][0..D], r.x[head * D ..][0..D]);
        }
        p.x = try p.rig.input(std.mem.sliceAsBytes(mixed));
        p.contiguous = try p.rig.input(std.mem.sliceAsBytes(r.x));
        p.w = try p.rig.input(std.mem.sliceAsBytes(r.w));
        p.before = try p.rig.scratch(r.x.len * 2);
        p.out = try p.rig.output(r.x.len * 2);
        const pass = p.rig.begin();
        pass.e.setPipeline(p.old);
        pass.e.setBuffer(p.contiguous.buf, p.contiguous.off, 0);
        pass.e.setBuffer(p.w.buf, p.w.off, 1);
        pass.e.setValue(q.glue.Head{ .width = D, .rows = p.rows * p.heads, .stride = D, .activation = .bf16, .gain = .bf16, .eps = r.eps }, 2);
        pass.e.setBuffer(p.before.buf, p.before.off, 3);
        pass.e.dispatchGroups(.of(p.rows * p.heads, 1, 1), .of(64, 1, 1));
        try p.rig.finish(pass);
        return true;
    }
    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        const is_q = p.heads >= 12;
        e.setPipeline(p.pipe);
        e.setBuffer(p.x.buf, p.x.off, 0);
        e.setBuffer(p.w.buf, p.w.off, 1);
        e.setValue(q.glue.Head{ .width = D, .rows = p.rows * p.heads, .stride = if (is_q) 2 * D else D, .heads = if (is_q) 1 else p.heads, .row_stride = if (is_q) 0 else 2 * p.heads * D, .activation = .bf16, .gain = .bf16, .eps = p.r.?.eps }, 2);
        e.setBuffer(p.out.buf, p.out.off, 3);
        e.dispatchGroups(.of(p.rows * p.heads, 1, 1), .of(64, 1, 1));
        try p.rig.finish(pass);
        if (!p.bad) {
            if (!std.mem.eql(u8, p.rig.bytes(p.out, p.r.?.x.len * 2), try p.replaced())) return error.MixedNormBytesDiffer;
            try p.independent();
        }
    }
    fn independent(p: *Probe) !void {
        const n = p.count();
        const out = p.rig.bytes(p.out, n * 2);
        const want = try p.a.alloc(f64, n);
        defer p.a.free(want);
        const mag = try p.a.alloc(f64, n);
        defer p.a.free(mag);
        try p.reference(want, mag);
        const spec = p.kernel().spec;
        const shape = c.Shape{ .rows = p.rows, .n = p.heads, .k = D };
        for (want, mag, 0..) |w, m, i| {
            const got = c.read(.bf16, out, i);
            if (c.classOf(got) != c.classOf(w)) return error.MixedNormReferenceClass;
            if (std.math.isFinite(w) and @abs(got - w) > c.bound(spec, shape, w, m)) {
                std.debug.print("MIXED_NORM_FP64 index={d} got={d} want={d} bound={d}\n", .{ i, got, w, c.bound(spec, shape, w, m) });
                return error.MixedNormReferenceBound;
            }
        }
    }
    pub fn replaced(p: *Probe) ![]const u8 {
        return p.rig.bytes(p.before, p.r.?.x.len * 2);
    }
    pub fn count(p: *Probe) usize {
        return p.r.?.outputs();
    }
    pub fn reference(p: *Probe, out: []f64, mag: []f64) !void {
        try p.r.?.reference(out, mag);
    }
    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        try p.r.?.simulate(acc, order, out);
    }
};
