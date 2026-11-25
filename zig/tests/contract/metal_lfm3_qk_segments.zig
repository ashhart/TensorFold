//! Shared qk runs each row with its own stream's length and cache, bit-equal to lfm3_qk run stream by stream.
const std = @import("std");
const mtl = @import("metal");
const ct = @import("contract");
const base = @import("metal_lfm3.zig");
const rig = @import("rig.zig");
const Segment = @import("tensorfold").lfm3.kernels.AttentionSegment;
const c = ct.spec;
const g = ct.gates;

pub fn name(sliding: bool, bad: bool) []const u8 {
    return if (bad) "control: LFM3 shared qk wrong stream" else if (sliding) "metal LFM3 shared qk sliding" else "metal LFM3 shared qk full";
}

/// One key row of one head: 128 bf16 values.
const head_bytes = 128 * 2;

pub const Probe = struct {
    inner: base.Probe,
    pipeline: mtl.Pipeline,
    bad: bool,
    table: rig.Ref = undefined,
    rows: rig.Ref = undefined,
    previous: [3]rig.Ref = undefined,
    parts: [3]struct { start: usize, rows: usize, len: usize } = undefined,
    used: usize = 0,
    bytes: std.ArrayList(u8) = .empty,
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
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, sliding: bool, bad: bool) !Probe {
        var inner = try base.Probe.init(a, device, queue, if (sliding) .qk_sliding else .qk_full);
        errdefer inner.deinit();
        const source = try base.Probe.source(a);
        defer a.free(source);
        const text = try std.mem.concat(a, u8, &.{ if (bad) "#define L3_QK_BAD_STREAM 1\n" else "", source });
        defer a.free(text);
        const pipeline = try rig.pipeline(device, text, "lfm3_qk_segments");
        errdefer pipeline.deinit();
        if (pipeline.maxThreads() < 32 or rig.tgMemory(pipeline) != 0) return error.SharedQkLimits;
        return .{ .inner = inner, .pipeline = pipeline, .bad = bad };
    }
    pub fn deinit(p: *Probe) void {
        p.bytes.deinit(p.inner.a);
        p.inner.deinit();
        p.pipeline.deinit();
    }
    pub fn kernel(p: *Probe) g.Kernel {
        var out = p.inner.kernel();
        out.vt = &vt;
        out.spec.name = name(p.inner.kind == .qk_sliding, p.bad);
        out.spec.replaces = "serial lfm3_qk on each stream's rows, length and cache";
        out.control = if (p.bad) .shapes else null;
        return out;
    }
    // Up to 3 streams, each as long as half its first row; a cache starts (start - len) key rows in.
    fn prepare(ctx: *anyopaque, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const p = parent(ctx);
        if (!try p.inner.prepare(shape, fill, seed)) return false;
        const b = &p.inner;
        const a = b.arena.allocator();
        p.previous[0] = try b.rig.scratch(shape.rows * 2048 * 2);
        p.previous[1] = try b.rig.scratch(shape.rows * 512 * 2);
        p.previous[2] = try b.rig.scratch(shape.rows * 512 * 2);
        p.used = @min(3, shape.rows);
        const pos = try a.alloc(u32, shape.rows);
        var table: [3]Segment = undefined;
        var ids: [17]u32 = undefined;
        var start: usize = 0;
        for (0..p.used) |i| {
            const rows = if (i + 1 == p.used) shape.rows - start else @max(1, (shape.rows - start) / 2);
            const len = start / 2;
            p.parts[i] = .{ .start = start, .rows = rows, .len = len };
            const shift = (start - len) * head_bytes;
            table[i] = .{ .keys = b.extra[0].buf.gpuAddress() + b.extra[0].off + shift, .values = b.extra[1].buf.gpuAddress() + b.extra[1].off + shift, .base = @intCast(start), .rows = @intCast(rows), .args = .{ .len = @intCast(len), .cap = @intCast(shape.rows), .scale = 0, .window = 0 } };
            @memset(ids[start..][0..rows], @intCast(i));
            for (pos[start..][0..rows], len..) |*v, at| v.* = @intCast(at);
            start += rows;
        }
        b.data.pos = pos;
        p.table = try b.rig.input(std.mem.asBytes(&table)[0 .. p.used * @sizeOf(Segment)]);
        p.rows = try b.rig.input(std.mem.sliceAsBytes(ids[0..shape.rows]));
        return true;
    }
    fn args(p: *const Probe, len: usize) extern struct { len: u32, cap: u32, eps: f32, log2_theta: f32, rotary_pairs: u32 } {
        const full = p.inner.kind == .qk_full;
        return .{ .len = @intCast(len), .cap = @intCast(p.inner.data.rows), .eps = 1e-5, .log2_theta = std.math.log2(@as(f32, if (full) 1e6 else 1e4)), .rotary_pairs = if (full) 32 else 64 };
    }
    fn run(ctx: *anyopaque) !void {
        const p = parent(ctx);
        const b = &p.inner;
        const pass = b.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipeline);
        e.useResource(b.extra[0].buf, mtl.ResourceUsage.write);
        e.useResource(b.extra[1].buf, mtl.ResourceUsage.write);
        for ([_]rig.Ref{ b.args[0], b.args[1], b.args[2], b.args[3], b.args[3] }, 0..) |ref, i| e.setBuffer(ref.buf, ref.off, i);
        const qk = p.args(0);
        e.setBytes(std.mem.asBytes(&qk), 5);
        e.setBuffer(b.out.buf, b.out.off, 6);
        e.setBuffer(p.table.buf, p.table.off, 7);
        e.setBuffer(p.rows.buf, p.rows.off, 8);
        e.dispatchGroups(.of(20, b.data.rows, 1), .of(32, 1, 1));
        try b.rig.finish(pass);
    }
    /// lfm3_qk once a stream, into copies of the outputs, with the same lengths and caches.
    fn replaced(ctx: *anyopaque) ![]const u8 {
        const p = parent(ctx);
        const b = &p.inner;
        const pass = b.rig.begin();
        const e = pass.e;
        e.setPipeline(b.pipeline);
        for (p.parts[0..p.used]) |part| {
            e.setBuffer(b.args[0].buf, b.args[0].off + part.start * 2048 * 2, 0);
            e.setBuffer(b.args[1].buf, b.args[1].off + part.start * 512 * 2, 1);
            e.setBuffer(b.args[2].buf, b.args[2].off + part.start * 512 * 2, 2);
            e.setBuffer(b.args[3].buf, b.args[3].off, 3);
            e.setBuffer(b.args[3].buf, b.args[3].off, 4);
            const qk = p.args(part.len);
            e.setBytes(std.mem.asBytes(&qk), 5);
            e.setBuffer(p.previous[0].buf, p.previous[0].off + part.start * 2048 * 2, 6);
            const shift = (part.start - part.len) * head_bytes;
            e.setBuffer(p.previous[1].buf, p.previous[1].off + shift, 7);
            e.setBuffer(p.previous[2].buf, p.previous[2].off + shift, 8);
            e.dispatchGroups(.of(20, part.rows, 1), .of(32, 1, 1));
        }
        try b.rig.finish(pass);
        p.bytes.clearRetainingCapacity();
        try p.bytes.appendSlice(b.a, b.rig.bytes(p.previous[0], b.data.rows * 2048 * 2));
        try p.bytes.appendSlice(b.a, b.rig.bytes(p.previous[1], b.data.rows * 512 * 2));
        try p.bytes.appendSlice(b.a, b.rig.bytes(p.previous[2], b.data.rows * 512 * 2));
        return p.bytes.items;
    }
};
