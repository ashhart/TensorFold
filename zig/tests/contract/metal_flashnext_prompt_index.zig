//! Bit-preserving read contract: the indexed 4/32 prompt gathers against the same gathers reading scales direct.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const fx = tf.flashnext_prompt_index_fixture;
const Fixture = fx.Fixture;

pub const Probe = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    control: bool = false,
    fixture: ?*Fixture = null,
    actual: []const u8 = &.{},
    baseline: []const u8 = &.{},
    inputs_: std.ArrayList(u8) = .empty,

    pub fn deinit(p: *Probe) void {
        if (p.fixture) |fixture| fixture.deinit(p.a);
        p.inputs_.deinit(p.a);
    }
    pub fn name(control: bool) []const u8 {
        return if (control) "control: Flash Next q4 prompt gather index, two table pairs swapped" else "Flash Next q4 prompt gathers through the scale index";
    }
    fn depth(_: c.Shape) usize {
        return 0;
    }
    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .ctx = p, .control = if (p.control) .shapes else null, .spec = .{
            .name = name(p.control),
            .accum = .none,
            .depth = depth,
            .out = .raw,
            .replaces = "the 4/32 prompt gathers (32, 64 and 128-row tiles, fused gate/up) reading each group's scale and bias direct, on the same stored bytes and rows",
            .nan = .{ .stated = "every output bit of the direct kernels, non-finite scales included, is retained" },
            .served = &.{ .{ .rows = 40, .n = fx.inner, .k = fx.d }, .{ .rows = 97, .n = fx.inner, .k = fx.d }, .{ .rows = 200, .n = fx.inner, .k = fx.d } },
            .sums = .{ .why = "the index changes where two operands are read, not a sum; the direct kernels' sums are not requalified here" },
        }, .vt = &vt };
    }
    fn of(ctx: *anyopaque) *Probe {
        return @ptrCast(@alignCast(ctx));
    }
    fn prepare(ctx: *anyopaque, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const p = of(ctx);
        if (shape.rows < fx.experts or shape.rows > 512 or shape.n != fx.inner or shape.k != fx.d) return false;
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
        fixture.wrong_entry = false;
        p.baseline = try fixture.run(false);
        fixture.wrong_entry = p.control;
        p.actual = try fixture.run(true);
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
        if (!std.mem.eql(u8, p.actual, p.baseline)) return "the indexed read changed the direct gathers' output bits";
        return null;
    }
    fn replaced(ctx: *anyopaque) ![]const u8 {
        return of(ctx).baseline;
    }
    const vt: g.VTable = .{ .prepare = prepare, .run = run, .output = output, .inputs = inputs, .intact = intact, .overlap = overlap, .count = count, .reference = reference, .check = check, .replaced = replaced };
};
