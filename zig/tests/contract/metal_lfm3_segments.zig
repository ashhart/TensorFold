//! Shared-stream attention reuses the serial arithmetic, with a byte replacement gate and a missing-tail control.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const ct = @import("contract");
const base_ = @import("metal_lfm3.zig");
const rig_ = @import("rig.zig");
const Segment = @import("tensorfold").lfm3.kernels.AttentionSegment;
const c = ct.spec;
const g = ct.gates;

pub fn name(sliding: bool, bad: bool) []const u8 {
    return if (bad) "control: LFM3 shared attention missing tail" else if (sliding) "metal LFM3 shared attention sliding" else "metal LFM3 shared attention full";
}

pub const Probe = struct {
    inner: base_.Probe,
    partial: mtl.Pipeline,
    merge: mtl.Pipeline,
    bad: bool,
    table: rig_.Ref = undefined,
    rows: rig_.Ref = undefined,
    previous: rig_.Ref = undefined,
    const vt = blk: {
        var hooks = g.hooks(base_.Probe);
        hooks.prepare = prepare;
        hooks.run = run;
        hooks.replaced = replaced;
        break :blk hooks;
    };
    fn parent(ctx: *anyopaque) *Probe {
        const inner: *base_.Probe = @ptrCast(@alignCast(ctx));
        return @fieldParentPtr("inner", inner);
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, sliding: bool, bad: bool) !Probe {
        var inner = try base_.Probe.init(a, device, queue, if (sliding) .attention_sliding else .attention_full);
        errdefer inner.deinit();
        const partial = try rig_.pipeline(device, sources.lfm3_attention, "lfm3_attn_segments");
        errdefer partial.deinit();
        const text = try std.mem.concat(a, u8, &.{ if (bad) "#define L3_SEGMENT_BAD_TAIL 1\n" else "", sources.lfm3_attention });
        defer a.free(text);
        return .{ .inner = inner, .partial = partial, .merge = try rig_.pipeline(device, text, "lfm3_attn_segments_merge"), .bad = bad };
    }
    pub fn deinit(p: *Probe) void {
        p.inner.deinit();
        p.partial.deinit();
        p.merge.deinit();
    }
    pub fn kernel(p: *Probe) g.Kernel {
        var out = p.inner.kernel();
        out.vt = &vt;
        out.spec.name = name(p.inner.kind == .attention_sliding, p.bad);
        out.spec.replaces = "serial LFM3 attention on identical row histories";
        out.control = if (p.bad) .shapes else null;
        return out;
    }
    fn prepare(ctx: *anyopaque, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const p = parent(ctx);
        if (!try p.inner.prepare(shape, fill, seed)) return false;
        const b = &p.inner;
        p.previous = try b.rig.scratch(shape.rows * 2048 * 2);
        var table: [2]Segment = undefined;
        var ids: [17]u32 = undefined;
        const split = @max(@as(usize, 1), shape.rows / 2);
        for (&table, 0..) |*seg, i| {
            const start = if (i == 0) 0 else split;
            const history = shape.k + start;
            seg.* = .{ .keys = b.args[1].buf.gpuAddress() + b.args[1].off, .values = b.args[2].buf.gpuAddress() + b.args[2].off, .base = @intCast(start), .rows = @intCast(if (i == 0) split else shape.rows - split), .args = .{ .len = @intCast(history), .cap = @intCast(shape.k + shape.rows), .scale = 1.0 / @sqrt(@as(f32, 128)), .window = if (b.kind == .attention_sliding) 256 else 0, .chunk_base = @intCast(if (b.kind == .attention_sliding) (history + 1 -| 256) / 4096 else 0) } };
        }
        for (ids[0..shape.rows], 0..) |*id, i| id.* = if (i < split) 0 else 1;
        p.table = try b.rig.input(std.mem.asBytes(&table));
        p.rows = try b.rig.input(std.mem.sliceAsBytes(ids[0..shape.rows]));
        return true;
    }
    fn run(ctx: *anyopaque) !void {
        const p = parent(ctx);
        const b = &p.inner;
        const pass = b.rig.begin();
        const e = pass.e;
        e.setPipeline(p.partial);
        e.useResource(b.args[1].buf, mtl.ResourceUsage.read);
        e.useResource(b.args[2].buf, mtl.ResourceUsage.read);
        e.setBuffer(b.args[0].buf, b.args[0].off, 0);
        e.setBuffer(p.table.buf, p.table.off, 1);
        e.setBuffer(p.rows.buf, p.rows.off, 2);
        e.setBuffer(b.risk.buf, b.risk.off, 3);
        for (b.extra, 4..) |ref, slot| e.setBuffer(ref.buf, ref.off, slot);
        e.dispatchGroups(.of(16, b.data.rows, (b.data.history + b.data.rows + 4095) / 4096), .of(128, 1, 1));
        e.setPipeline(p.merge);
        for (b.extra, 0..) |ref, slot| e.setBuffer(ref.buf, ref.off, slot);
        e.setBuffer(p.table.buf, p.table.off, 3);
        e.setBuffer(b.out.buf, b.out.off, 4);
        e.setBuffer(p.rows.buf, p.rows.off, 5);
        e.setBuffer(b.args[0].buf, b.args[0].off, 6);
        e.setBuffer(b.risk.buf, b.risk.off, 7);
        e.dispatchGroups(.of(16, b.data.rows, 1), .of(32, 1, 1));
        try b.rig.finish(pass);
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
