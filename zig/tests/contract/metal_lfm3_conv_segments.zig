//! Shared convolution retains each cache's seven-tap equation, exact serial bytes and guarded snapshots.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const ct = @import("contract");
const base = @import("metal_lfm3.zig");
const math = @import("lfm3_math.zig");
const rig = @import("rig.zig");
const Segment = @import("tensorfold").lfm3.kernels.ConvSegment;
const c = ct.spec;
const g = ct.gates;

pub fn name(bad: bool) []const u8 {
    return if (bad) "control: LFM3 shared conv wrong state" else "metal LFM3 shared conv";
}

pub const Probe = struct {
    inner: base.Probe,
    pipeline: mtl.Pipeline,
    bad: bool,
    table: rig.Ref = undefined,
    rows: rig.Ref = undefined,
    previous: [2]rig.Ref = undefined,
    parts: [4]struct { data: math.Data, state: rig.Ref, start: usize } = undefined,
    used: usize = 0,
    bytes: std.ArrayList(u8) = .empty,
    const vt = blk: {
        var hooks = g.hooks(base.Probe);
        hooks.prepare = prepare;
        hooks.run = run;
        hooks.reference = reference;
        hooks.simulate = simulate;
        hooks.replaced = replaced;
        break :blk hooks;
    };
    fn parent(ctx: *anyopaque) *Probe {
        const inner: *base.Probe = @ptrCast(@alignCast(ctx));
        return @fieldParentPtr("inner", inner);
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, bad: bool) !Probe {
        var inner = try base.Probe.init(a, device, queue, .conv);
        errdefer inner.deinit();
        const text = try std.mem.concat(a, u8, &.{ if (bad) "#define L3_CONV_BAD_STATE 1\n" else "", sources.lfm3 });
        defer a.free(text);
        const pipeline = try rig.pipeline(device, text, "lfm3_conv_segments");
        errdefer pipeline.deinit();
        const threads = pipeline.maxThreads();
        const bytes = rig.tgMemory(pipeline);
        if (threads < 256 or bytes != 0) return error.SharedConvLimits;
        std.debug.print("SHARED_CONV_LIMITS bad={} launched=256 max_threads={d} static_tg_bytes={d}\n", .{ bad, threads, bytes });
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
        out.spec.name = name(p.bad);
        out.spec.replaces = "serial seven-tap convolution on each independent cache and snapshot span";
        out.control = if (p.bad) .shapes else null;
        return out;
    }
    fn prepare(ctx: *anyopaque, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const p = parent(ctx);
        if (!try p.inner.prepare(shape, fill, seed)) return false;
        const b = &p.inner;
        const a = b.arena.allocator();
        p.previous[0] = try b.rig.scratch(shape.rows * 2048 * 2);
        p.previous[1] = try b.rig.scratch(shape.rows * 6 * 2048 * 2);
        p.used = @min(4, shape.rows);
        var table: [4]Segment = undefined;
        var ids: [17]u32 = undefined;
        var start: usize = 0;
        for (0..p.used) |i| {
            const rows = if (i + 1 == p.used) shape.rows - start else @max(1, (shape.rows - start) / 2);
            const state = try a.alloc(u16, b.data.z.len);
            for (state, 0..) |*v, j| v.* = b.data.z[(j + i * 257) % state.len];
            const ref = try b.rig.input(std.mem.sliceAsBytes(state));
            var data = b.data;
            data.rows = rows;
            data.x = b.data.x[start * 6144 ..][0 .. rows * 6144];
            data.z = state;
            p.parts[i] = .{ .data = data, .state = ref, .start = start };
            table[i] = .{ .state = ref.buf.gpuAddress() + ref.off, .base = @intCast(start), .rows = @intCast(rows) };
            @memset(ids[start..][0..rows], @intCast(i));
            start += rows;
        }
        p.table = try b.rig.input(std.mem.asBytes(&table)[0 .. p.used * @sizeOf(Segment)]);
        p.rows = try b.rig.input(std.mem.sliceAsBytes(ids[0..shape.rows]));
        return true;
    }
    fn run(ctx: *anyopaque) !void {
        const p = parent(ctx);
        const b = &p.inner;
        const pass = b.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipeline);
        for (p.parts[0..p.used]) |part| e.useResource(part.state.buf, mtl.ResourceUsage.read);
        for ([_]rig.Ref{ b.args[0], p.table, b.args[1], b.out, b.extra[0], p.rows }, 0..) |ref, i|
            e.setBuffer(ref.buf, ref.off, i);
        e.dispatchThreads(.of(2048, b.data.rows, 1), .of(256, 1, 1));
        try b.rig.finish(pass);
    }
    fn evaluate(p: *Probe, acc: c.Accum, order: g.Order, want: []f64, mag: []f64) !void {
        const b = &p.inner;
        for (p.parts[0..p.used]) |part| {
            const n = part.data.rows * 2048;
            const w = try b.a.alloc(f64, n * 7);
            defer b.a.free(w);
            const m = try b.a.alloc(f64, n * 7);
            defer b.a.free(m);
            try math.evaluate(b.a, part.data, acc, order, w, m);
            @memcpy(want[part.start * 2048 ..][0..n], w[0..n]);
            @memcpy(mag[part.start * 2048 ..][0..n], m[0..n]);
            const offset = b.data.rows * 2048 + part.start * 6 * 2048;
            @memcpy(want[offset..][0 .. n * 6], w[n..]);
            @memcpy(mag[offset..][0 .. n * 6], m[n..]);
        }
    }
    fn reference(ctx: *anyopaque, want: []f64, mag: []f64) !void {
        return parent(ctx).evaluate(.none, .sequential, want, mag);
    }
    fn simulate(ctx: *anyopaque, acc: c.Accum, order: g.Order, want: []f64) !void {
        const p = parent(ctx);
        const mag = try p.inner.a.alloc(f64, want.len);
        defer p.inner.a.free(mag);
        try p.evaluate(acc, order, want, mag);
    }
    fn replaced(ctx: *anyopaque) ![]const u8 {
        const p = parent(ctx);
        const b = &p.inner;
        const pass = b.rig.begin();
        const e = pass.e;
        e.setPipeline(b.pipeline);
        for (p.parts[0..p.used]) |part| {
            e.setBuffer(b.args[0].buf, b.args[0].off + part.start * 6144 * 2, 0);
            e.setBuffer(part.state.buf, part.state.off, 1);
            e.setBuffer(b.args[1].buf, b.args[1].off, 2);
            e.setBuffer(p.previous[0].buf, p.previous[0].off + part.start * 2048 * 2, 3);
            e.setBuffer(p.previous[1].buf, p.previous[1].off + part.start * 6 * 2048 * 2, 4);
            e.setValue(@as(u32, 0), 5);
            e.dispatchThreads(.of(2048, part.data.rows, 1), .of(256, 1, 1));
        }
        try b.rig.finish(pass);
        p.bytes.clearRetainingCapacity();
        try p.bytes.appendSlice(b.a, b.rig.bytes(p.previous[0], b.data.rows * 2048 * 2));
        try p.bytes.appendSlice(b.a, b.rig.bytes(p.previous[1], b.data.rows * 6 * 2048 * 2));
        return p.bytes.items;
    }
};
