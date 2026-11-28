//! Serial extended-range replay versus parallel repair, including forced repair and a planted part-order control.
const std = @import("std");
const mtl = @import("metal");
const ct = @import("contract");
const base = @import("metal_lfm3.zig");
const rig = @import("rig.zig");
const sources = @import("kernel_sources");
const c = ct.spec;
const g = ct.gates;
const Args = @import("tensorfold").lfm3.kernels.AttentionArgs;
const Segment = @import("tensorfold").lfm3.kernels.AttentionSegment;

pub fn label(shared: bool, forced: bool, bad: bool) []const u8 {
    if (bad) return "lfm3 repair-bad-order LFM3 range parallel control";
    if (shared) return if (forced) "lfm3 repair LFM3 range parallel shared forced" else "lfm3 repair LFM3 range parallel shared ordinary";
    return if (forced) "lfm3 repair LFM3 range parallel direct forced" else "lfm3 repair LFM3 range parallel direct ordinary";
}

pub const Probe = struct {
    inner: base.Probe,
    flagged: mtl.Pipeline,
    partial: mtl.Pipeline,
    merge: mtl.Pipeline,
    segment_partial: ?mtl.Pipeline = null,
    shared: bool,
    forced: bool,
    bad: bool,
    previous: rig.Ref = undefined,
    exponents: rig.Ref = undefined,
    flags: rig.Ref = undefined,
    table: rig.Ref = undefined,
    rows: rig.Ref = undefined,

    const vt = blk: {
        var hooks = g.hooks(base.Probe);
        hooks.prepare = prepare;
        hooks.run = run;
        hooks.replaced = replaced;
        break :blk hooks;
    };

    fn parent(ctx: *anyopaque) *Probe {
        const inner: *base.Probe = @ptrCast(@alignCast(ctx));
        return @fieldParentPtr("inner", inner);
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, shared: bool, forced: bool, bad: bool) !Probe {
        const serial = try std.mem.concat(a, u8, &.{ if (forced) "#define L3_RANGE_FORCE_REPAIR 1\n" else "", sources.lfm3, "\n", sources.lfm3_attention });
        defer a.free(serial);
        var inner = try base.Probe.initSource(a, device, queue, .attention_full, serial);
        errdefer inner.deinit();
        const candidate = try std.mem.concat(a, u8, &.{ if (bad) "#define L3_REPAIR_BAD_ORDER 1\n" else "", serial });
        defer a.free(candidate);
        const flagged = try rig.pipeline(device, candidate, if (shared) "lfm3_attn_segments_merge_flagged" else "lfm3_attn_merge_flagged");
        errdefer flagged.deinit();
        const partial = try rig.pipeline(device, candidate, if (shared) "lfm3_attn_segments_repair_partial" else "lfm3_attn_repair_partial");
        errdefer partial.deinit();
        const merge = try rig.pipeline(device, candidate, if (shared) "lfm3_attn_segments_repair_merge" else "lfm3_attn_repair_merge");
        errdefer merge.deinit();
        var out: Probe = .{ .inner = inner, .flagged = flagged, .partial = partial, .merge = merge, .shared = shared, .forced = forced, .bad = bad };
        if (shared) out.segment_partial = try rig.pipeline(device, serial, "lfm3_attn_segments");
        return out;
    }

    pub fn deinit(p: *Probe) void {
        p.inner.deinit();
        p.flagged.deinit();
        p.partial.deinit();
        p.merge.deinit();
        if (p.segment_partial) |pipeline| pipeline.deinit();
    }

    pub fn kernel(p: *Probe) g.Kernel {
        var out = p.inner.kernel();
        out.vt = &vt;
        out.spec.name = label(p.shared, p.forced, p.bad);
        out.spec.replaces = "serial LFM3 extended-range attention repair on identical inputs";
        out.spec.served = &.{ .{ .rows = 17, .n = 2048, .k = 2048 }, .{ .rows = 17, .n = 2048, .k = 8192 }, .{ .rows = 17, .n = 2048, .k = 32768 } };
        out.control = if (p.bad) .shapes else null;
        return out;
    }

    fn prepare(ctx: *anyopaque, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const p = parent(ctx);
        if (!try p.inner.prepare(shape, fill, seed)) return false;
        const b = &p.inner;
        const partials = shape.rows * 16 * 64;
        p.previous = try b.rig.scratch(shape.rows * 2048 * 2);
        p.exponents = try b.rig.scratch(partials * 128 * 4);
        p.flags = try b.rig.scratch(shape.rows * 16 * 4);
        if (!p.shared) return true;
        var table: [2]Segment = undefined;
        var ids: [17]u32 = undefined;
        const split = @max(@as(usize, 1), shape.rows / 2);
        for (&table, 0..) |*seg, i| {
            const start = if (i == 0) 0 else split;
            const history = shape.k + start;
            seg.* = .{ .keys = b.args[1].buf.gpuAddress() + b.args[1].off, .values = b.args[2].buf.gpuAddress() + b.args[2].off, .base = @intCast(start), .rows = @intCast(if (i == 0) split else shape.rows - split), .args = .{ .len = @intCast(history), .cap = @intCast(shape.k + shape.rows), .scale = 1.0 / @sqrt(@as(f32, 128)), .window = 0, .chunk_base = 0 } };
        }
        for (ids[0..shape.rows], 0..) |*id, i| id.* = if (i < split) 0 else 1;
        p.table = try b.rig.input(std.mem.asBytes(&table));
        p.rows = try b.rig.input(std.mem.sliceAsBytes(ids[0..shape.rows]));
        return true;
    }

    fn bind(e: mtl.ComputeEncoder, refs: []const rig.Ref) void {
        for (refs, 0..) |ref, i| e.setBuffer(ref.buf, ref.off, i);
    }

    fn args(p: *const Probe) Args {
        return .{ .len = @intCast(p.inner.data.history), .cap = @intCast(p.inner.data.history + p.inner.data.rows), .scale = 1.0 / @sqrt(@as(f32, 128)), .window = 0, .chunk_base = 0 };
    }

    fn full(p: *Probe, e: mtl.ComputeEncoder) void {
        const b = &p.inner;
        const a = args(p);
        const used = (b.data.history + b.data.rows + 4095) / 4096;
        e.setPipeline(b.pipeline);
        bind(e, &.{ b.args[0], b.args[1], b.args[2] });
        e.setBytes(std.mem.asBytes(&a), 3);
        for (b.extra, 4..) |ref, i| e.setBuffer(ref.buf, ref.off, i);
        e.setBuffer(b.risk.buf, b.risk.off, 7);
        e.dispatchGroups(.of(16, b.data.rows, used), .of(128, 1, 1));
        e.setPipeline(p.flagged);
        bind(e, &b.extra);
        e.setBytes(std.mem.asBytes(&a), 3);
        e.setBuffer(b.out.buf, b.out.off, 4);
        e.setBuffer(b.risk.buf, b.risk.off, 5);
        e.setBuffer(p.flags.buf, p.flags.off, 6);
        e.dispatchGroups(.of(16, b.data.rows, 1), .of(32, 1, 1));
        e.setPipeline(p.partial);
        bind(e, &.{ b.args[0], b.args[1], b.args[2] });
        e.setBytes(std.mem.asBytes(&a), 3);
        for (b.extra, 4..) |ref, i| e.setBuffer(ref.buf, ref.off, i);
        e.setBuffer(p.exponents.buf, p.exponents.off, 7);
        e.setBuffer(p.flags.buf, p.flags.off, 8);
        e.dispatchGroups(.of(16, b.data.rows, used), .of(128, 1, 1));
        e.setPipeline(p.merge);
        bind(e, &b.extra);
        e.setBuffer(p.exponents.buf, p.exponents.off, 3);
        e.setBytes(std.mem.asBytes(&a), 4);
        e.setBuffer(b.out.buf, b.out.off, 5);
        e.setBuffer(p.flags.buf, p.flags.off, 6);
        e.dispatchGroups(.of(16, b.data.rows, 1), .of(32, 1, 1));
    }

    fn segmented(p: *Probe, e: mtl.ComputeEncoder) void {
        const b = &p.inner;
        const used = (b.data.history + b.data.rows + 4095) / 4096;
        e.useResource(b.args[1].buf, mtl.ResourceUsage.read);
        e.useResource(b.args[2].buf, mtl.ResourceUsage.read);
        e.setPipeline(p.segment_partial.?);
        bind(e, &.{ b.args[0], p.table, p.rows, b.risk });
        for (b.extra, 4..) |ref, i| e.setBuffer(ref.buf, ref.off, i);
        e.dispatchGroups(.of(16, b.data.rows, used), .of(128, 1, 1));
        e.setPipeline(p.flagged);
        bind(e, &b.extra);
        e.setBuffer(p.table.buf, p.table.off, 3);
        e.setBuffer(b.out.buf, b.out.off, 4);
        e.setBuffer(p.rows.buf, p.rows.off, 5);
        e.setBuffer(b.risk.buf, b.risk.off, 6);
        e.setBuffer(p.flags.buf, p.flags.off, 7);
        e.dispatchGroups(.of(16, b.data.rows, 1), .of(32, 1, 1));
        e.setPipeline(p.partial);
        bind(e, &.{ b.args[0], p.table, p.rows });
        for (b.extra, 3..) |ref, i| e.setBuffer(ref.buf, ref.off, i);
        e.setBuffer(p.exponents.buf, p.exponents.off, 6);
        e.setBuffer(p.flags.buf, p.flags.off, 7);
        e.dispatchGroups(.of(16, b.data.rows, used), .of(128, 1, 1));
        e.setPipeline(p.merge);
        bind(e, &b.extra);
        e.setBuffer(p.exponents.buf, p.exponents.off, 3);
        e.setBuffer(p.table.buf, p.table.off, 4);
        e.setBuffer(b.out.buf, b.out.off, 5);
        e.setBuffer(p.rows.buf, p.rows.off, 6);
        e.setBuffer(p.flags.buf, p.flags.off, 7);
        e.dispatchGroups(.of(16, b.data.rows, 1), .of(32, 1, 1));
    }

    fn run(ctx: *anyopaque) !void {
        const p = parent(ctx);
        const pass = p.inner.rig.begin();
        if (p.shared) p.segmented(pass.e) else p.full(pass.e);
        try p.inner.rig.finish(pass);
    }

    fn replaced(ctx: *anyopaque) ![]const u8 {
        const p = parent(ctx);
        const saved = p.inner.out;
        p.inner.out = p.previous;
        defer p.inner.out = saved;
        try p.inner.run();
        return p.inner.rig.bytes(p.previous, p.inner.data.rows * 2048 * 2);
    }
};
