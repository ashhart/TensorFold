const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const ct = @import("contract");
const base = @import("metal_lfm3.zig");
const math = @import("lfm3_math.zig");
const rig = @import("rig.zig");
const c = ct.spec;
const g = ct.gates;
pub fn name(bad: bool) []const u8 {
    return if (bad) "control: LFM3 post norm removed rounding" else "metal LFM3 post norm exact fusion";
}
pub const Probe = struct {
    inner: base.Probe,
    fused: mtl.Pipeline,
    norm: mtl.Pipeline,
    bad: bool,
    residual: rig.Ref = undefined,
    previous: rig.Ref = undefined,
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
        return @fieldParentPtr("inner", @as(*base.Probe, @ptrCast(@alignCast(ctx))));
    }
    pub fn init(a: std.mem.Allocator, d: mtl.Device, q: mtl.Queue, bad: bool) !Probe {
        var inner = try base.Probe.init(a, d, q, .post_add);
        errdefer inner.deinit();
        const text = try std.mem.concat(a, u8, &.{ if (bad) "#define L3_POSTNORM_SKIP_ROUND\n" else "", sources.lfm3 });
        defer a.free(text);
        const fused = try rig.pipeline(d, text, "lfm3_post_norm");
        errdefer fused.deinit();
        return .{ .inner = inner, .fused = fused, .norm = try rig.pipeline(d, sources.lfm3, "lfm3_norm"), .bad = bad };
    }
    pub fn deinit(p: *Probe) void {
        p.inner.deinit();
        p.fused.deinit();
        p.norm.deinit();
    }
    pub fn kernel(p: *Probe) g.Kernel {
        var k = p.inner.kernel();
        k.vt = &vt;
        k.spec.name = name(p.bad);
        k.spec.replaces = "separate production post-add and next RMSNorm";
        k.spec.mids = 4;
        k.spec.sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "both original reduction trees retained; replacement requires identical bytes" };
        k.control = if (p.bad) .shapes else null;
        return k;
    }
    fn prepare(ctx: *anyopaque, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const p = parent(ctx);
        if (!try p.inner.prepare(shape, fill, seed)) return false;
        p.residual = try p.inner.rig.scratch(shape.rows * 4096);
        p.previous = try p.inner.rig.scratch(shape.rows * 4096);
        return true;
    }
    fn run(ctx: *anyopaque) !void {
        const p = parent(ctx);
        const b = &p.inner;
        @memcpy(b.rig.bytes(p.residual, b.data.rows * 4096), std.mem.sliceAsBytes(b.data.z));
        const pass = b.rig.begin();
        const e = pass.e;
        e.setPipeline(p.fused);
        e.setBuffer(b.args[0].buf, b.args[0].off, 0);
        e.setBuffer(b.args[3].buf, b.args[3].off, 1);
        e.setBuffer(p.residual.buf, p.residual.off, 2);
        e.setValue(@as(f32, 1e-5), 3);
        e.setBuffer(b.args[1].buf, b.args[1].off, 4);
        e.setBuffer(b.out.buf, b.out.off, 5);
        e.dispatchGroups(.of(b.data.rows, 1, 1), .of(256, 1, 1));
        try b.rig.finish(pass);
    }
    fn replaced(ctx: *anyopaque) ![]const u8 {
        const p = parent(ctx);
        const b = &p.inner;
        const saved = b.out;
        b.out = p.residual;
        defer b.out = saved;
        try b.run();
        const pass = b.rig.begin();
        const e = pass.e;
        e.setPipeline(p.norm);
        e.setBuffer(p.residual.buf, p.residual.off, 0);
        e.setBuffer(p.residual.buf, p.residual.off, 1);
        e.setBuffer(b.args[1].buf, b.args[1].off, 2);
        e.setValue(@as(f32, 1e-5), 3);
        e.setValue(@as(u32, 0), 4);
        e.setBuffer(p.previous.buf, p.previous.off, 5);
        e.dispatchGroups(.of(b.data.rows, 1, 1), .of(256, 1, 1));
        try b.rig.finish(pass);
        return b.rig.bytes(p.previous, b.data.rows * 4096);
    }
    fn evaluate(ctx: *anyopaque, acc: c.Accum, order: g.Order, want: []f64, mag: []f64) !void {
        const b = &parent(ctx).inner;
        try math.evaluate(b.a, b.data, acc, order, want, mag);
        for (0..b.data.rows) |r| {
            var terms: [2048]f64 = undefined;
            for (&terms, 0..) |*v, j| v.* = want[r * 2048 + j] * want[r * 2048 + j];
            var square: f64 = 0;
            if (acc == .none) {
                for (terms) |v| square += v;
            } else square = ct.rows.sumIn(acc, order, &terms);
            const inv = 1 / @sqrt(square / 2048 + 1e-5);
            for (0..2048) |j| {
                const at = r * 2048 + j;
                want[at] = math.bf(math.bf(want[at] * inv) * math.val(b.data.w[j]));
                mag[at] = @abs(want[at]);
            }
        }
    }
    fn reference(ctx: *anyopaque, want: []f64, mag: []f64) !void {
        return evaluate(ctx, .none, .sequential, want, mag);
    }
    fn simulate(ctx: *anyopaque, acc: c.Accum, order: g.Order, out: []f64) !void {
        const b = &parent(ctx).inner;
        const mag = try b.a.alloc(f64, out.len);
        defer b.a.free(mag);
        try evaluate(ctx, acc, order, out, mag);
    }
};
