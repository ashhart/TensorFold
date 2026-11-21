//! GLM's one-row router and indexer-gate logits (glm_route_one.metal) against float64 and tf_route_logits' bits.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const bits = @import("metal_glm_bits.zig");
const rig_ = @import("rig.zig");
const Logits = @import("metal_flint_route.zig").Logits;
const c = contract.spec;
const g = contract.gates;
const route = tf.moe_route;
const kernels = tf.glm.kernels;

/// Rows: one-row windows, each its own launch as moe.zig and forward.zig's mlaCache launch it; n: outputs; k: hidden.
fn servedOf(comptime s: route.Shape) [3]c.Shape {
    return .{ .{ .rows = 16, .n = s.experts, .k = s.hidden }, .{ .rows = 1, .n = s.experts, .k = s.hidden }, .{ .rows = 6, .n = s.experts, .k = s.hidden } };
}
const served = [2][3]c.Shape{ servedOf(kernels.route_shape), servedOf(kernels.igate_shape) };

/// The loader's router library at a shape: core/moe_route.metal's logits and select, glm_route_one.metal appended.
pub fn text(a: std.mem.Allocator, s: route.Shape) ![]u8 {
    const core = try route.source(a, s);
    defer a.free(core);
    return std.mem.concat(a, u8, &.{ core, sources.glm_route_one });
}

pub const RouteOne = struct {
    inner: Logits,
    rig: *rig_.Rig,
    one: mtl.Pipeline,
    igate: bool,
    control: bool,
    base: rig_.Ref = undefined,

    fn shape(t: *const RouteOne) route.Shape {
        return if (t.igate) kernels.igate_shape else kernels.route_shape;
    }
    pub fn name(igate: bool, control: bool) []const u8 {
        if (control) return if (igate) "control: metal glm launch route glm_route_logits_one indexer gate, one simdgroup's partial left out" else "control: metal glm launch route glm_route_logits_one router, one simdgroup's partial left out";
        return if (igate) "metal glm launch route glm_route_logits_one indexer gate 128 outputs, bf16 out" else "metal glm launch route glm_route_logits_one router 288 experts, fp32 out";
    }
    pub fn create(s: *bits.Suite, igate: bool, control: bool) !*RouteOne {
        const loader = try s.production();
        const own = try text(s.a, if (igate) kernels.igate_shape else kernels.route_shape);
        defer s.a.free(own);
        _ = try loader.require(own);
        const body = if (control) try bits.edited(s.a, own, "void glm_route_logits_one(", "for (int k = 1; k < 8; k++)", "for (int k = 2; k < 8; k++)") else try s.a.dupe(u8, own);
        defer s.a.free(body);
        const one = try bits.pipeline(s.device, body, "glm_route_logits_one");
        errdefer one.deinit();
        const old = try bits.pipeline(s.device, own, route.names[0]);
        errdefer old.deinit();
        const t = try s.a.create(RouteOne);
        const Router = @FieldType(Logits, "r");
        const r: Router = .{ .label = if (igate) "glm igate" else "glm", .shape = if (igate) kernels.igate_shape else kernels.route_shape };
        t.* = .{ .inner = .{ .a = s.a, .rig = .{ .a = s.a, .device = s.device, .queue = s.queue }, .pipe = old, .r = r, .served = served[@intFromBool(igate)] }, .rig = undefined, .one = one, .igate = igate, .control = control };
        t.rig = &t.inner.rig;
        return t;
    }
    pub fn deinit(t: *RouteOne) void {
        t.inner.deinit();
        t.one.deinit();
    }
    const vt = g.hooks(RouteOne);
    pub fn kernel(t: *RouteOne) g.Kernel {
        var sp = Logits.spec(name(t.igate, t.control), t.inner.r, &served[@intFromBool(t.igate)]);
        sp.tiles = .{ .{}, .{}, .{} };
        sp.replaces = "tf_route_logits at one row";
        return .{ .spec = sp, .ctx = t, .vt = &vt, .control = if (t.control) .shapes else null };
    }
    fn outBytes(t: *const RouteOne) usize {
        return if (t.igate) 2 else 4;
    }
    pub fn prepare(t: *RouteOne, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!try t.inner.prepare(s, fill, seed)) return false;
        t.base = try t.rig.scratch(s.rows * s.n * t.outBytes());
        return true;
    }
    /// Each row a one-row window: moe_route.logits at one row, on that row's input and output.
    fn launch(t: *RouteOne, pipe: mtl.Pipeline, out: rig_.Ref) !void {
        const sh = t.shape();
        const pass = t.rig.begin();
        for (0..t.inner.rows) |r| {
            const x: rig_.Ref = .{ .buf = t.inner.ins[0].buf, .off = t.inner.ins[0].off + r * sh.hidden * 2 };
            const o: rig_.Ref = .{ .buf = out.buf, .off = out.off + r * sh.experts * t.outBytes() };
            route.logits(pass.e, pipe, sh, x, t.inner.ins[1], o, 1);
        }
        try t.rig.finish(pass);
    }
    pub fn run(t: *RouteOne) !void {
        try t.launch(t.one, t.inner.out);
    }
    pub fn replaced(t: *RouteOne) ![]const u8 {
        const n = t.inner.count() * t.outBytes();
        @memset(t.rig.bytes(t.base, n), 0xff);
        try t.launch(t.inner.pipe, t.base);
        return t.rig.bytes(t.base, n);
    }
    pub fn count(t: *RouteOne) usize {
        return t.inner.count();
    }
    pub fn reference(t: *RouteOne, want: []f64, mag: []f64) !void {
        try t.inner.reference(want, mag);
    }
    pub fn intervals(t: *RouteOne, low: []f64, high: []f64) !void {
        try t.inner.intervals(low, high);
    }
    pub fn bounds(t: *RouteOne, want: []const f64, mag: []const f64, allow: []f64) !void {
        try t.inner.bounds(want, mag, allow);
    }
    pub fn simulate(t: *RouteOne, acc: c.Accum, order: g.Order, out: []f64) !void {
        try t.inner.simulate(acc, order, out);
    }
};

pub fn collect(s: *bits.Suite, only: ?[]const u8) !void {
    for ([_]bool{ false, true }) |igate| for ([_]bool{ false, true }) |control| {
        if (s.wanted(only, RouteOne.name(igate, control))) try s.own(RouteOne, try RouteOne.create(s, igate, control));
    };
}

/// Gate 5's registrations `glm launch route router|igate <function>`: the loader's router library at that shape.
pub const prefix = "glm launch route ";
pub fn limitsText(a: std.mem.Allocator, _: mtl.Device, label: []const u8) !bits.Texts {
    const t = try text(a, if (std.mem.indexOf(u8, label, " igate ") != null) kernels.igate_shape else kernels.route_shape);
    return .{ .here = t, .pre_m5 = t };
}
