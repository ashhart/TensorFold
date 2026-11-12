//! Bit-preserving launch contract: a window's experts grouped by expert against the recorded 4/32 expert kernels.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const Fixture = tf.flashnext_grouped_fixture.Fixture;
const fx = tf.flashnext_grouped_fixture;

pub const Probe = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    control: bool = false,
    indexed: bool = false,
    fixture: ?*Fixture = null,
    actual: []const u8 = &.{},
    baseline: []const u8 = &.{},
    inputs_: std.ArrayList(u8) = .empty,

    pub fn deinit(p: *Probe) void {
        if (p.fixture) |fixture| fixture.deinit(p.a);
        p.inputs_.deinit(p.a);
    }
    pub fn name(control: bool, indexed: bool) []const u8 {
        if (indexed) return if (control) "control: Flash Next grouped experts through the scale index, gate and up tables swapped" else "Flash Next q4 experts grouped by expert through the scale index";
        return if (control) "control: Flash Next grouped experts, gate and up scales swapped" else "Flash Next q4 experts grouped by expert";
    }
    fn depth(_: c.Shape) usize {
        return 0;
    }
    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .ctx = p, .control = if (p.control) .shapes else null, .spec = .{
            .name = name(p.control, p.indexed),
            .accum = .none,
            .depth = depth,
            .out = .raw,
            .replaces = "the recorded 4/32 expert gate/up and down kernels a row at a time, on the same weights, logits and rows",
            .nan = .{ .stated = "every output bit of the recorded kernels, non-finite scales included, is retained" },
            .served = &.{ .{ .rows = 5, .n = fx.inner, .k = fx.d }, .{ .rows = 8, .n = fx.inner, .k = fx.d }, .{ .rows = 13, .n = fx.inner, .k = fx.d }, .{ .rows = 16, .n = fx.inner, .k = fx.d } },
            .sums = .{ .why = "grouping changes which threadgroup runs a (row, expert) pair, not a sum; the recorded kernels' sums are not requalified here" },
        }, .vt = &vt };
    }
    fn of(ctx: *anyopaque) *Probe {
        return @ptrCast(@alignCast(ctx));
    }
    fn prepare(ctx: *anyopaque, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const p = of(ctx);
        if (shape.rows == 0 or shape.rows > 16 or shape.n != fx.inner or shape.k != fx.d) return false;
        if (p.fixture) |fixture| {
            fixture.deinit(p.a);
            p.fixture = null;
        }
        p.fixture = try Fixture.init(p.a, p.device, p.queue, shape.rows, seed, fill == .nonfinite);
        p.actual = &.{};
        p.baseline = &.{};
        return true;
    }
    fn run(ctx: *anyopaque) !void {
        const p = of(ctx);
        const fixture = p.fixture.?;
        fixture.wrong_scales = false;
        p.baseline = try fixture.run(false, false);
        fixture.wrong_scales = p.control;
        p.actual = try fixture.run(true, p.indexed);
    }
    fn output(ctx: *anyopaque) ![]const u8 {
        return of(ctx).actual;
    }
    fn inputs(ctx: *anyopaque) ![]const u8 {
        const p = of(ctx);
        p.inputs_.clearRetainingCapacity();
        try p.fixture.?.inputs(p.a, &p.inputs_);
        return p.inputs_.items;
    }
    fn intact(ctx: *anyopaque) !bool {
        return of(ctx).fixture.?.intact();
    }
    fn overlap(_: *anyopaque) bool {
        return false;
    }
    fn count(ctx: *anyopaque) usize {
        return of(ctx).actual.len;
    }
    fn reference(_: *anyopaque, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }
    fn check(ctx: *anyopaque, _: g.Gate) !?[]const u8 {
        const p = of(ctx);
        if (!std.mem.eql(u8, p.actual, p.baseline)) return "the grouped launches changed the recorded kernels' output bits";
        return null;
    }
    fn replaced(ctx: *anyopaque) ![]const u8 {
        return of(ctx).baseline;
    }
    const vt: g.VTable = .{ .prepare = prepare, .run = run, .output = output, .inputs = inputs, .intact = intact, .overlap = overlap, .count = count, .reference = reference, .check = check, .replaced = replaced };
};
