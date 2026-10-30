//! Bit-preserving address-view contract: the oracle executes the unchanged resident expert arithmetic.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
pub const Probe = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    bits: usize,
    control: bool = false,
    fixture: ?*tf.flashnext_fsr_fixture.Fixture = null,
    actual: []const u8 = &.{},
    baseline: []const u8 = &.{},
    inputs_: std.ArrayList(u8) = .empty,

    pub fn deinit(p: *Probe) void {
        if (p.fixture) |fixture| fixture.deinit(p.a);
        p.inputs_.deinit(p.a);
    }
    pub fn name(bits: usize, control: bool) []const u8 {
        return if (control)
            (if (bits == 4) "control: Flash Next FSR q4 wrong slot map" else "control: Flash Next FSR q6 wrong slot map")
        else
            (if (bits == 4) "Flash Next FSR q4 exact stored expert view" else "Flash Next FSR q6 exact stored expert view");
    }
    fn depth(_: c.Shape) usize {
        return 0;
    }
    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .ctx = p, .control = if (p.control) .shapes else null, .spec = .{
            .name = name(p.bits, p.control),
            .accum = .none,
            .depth = depth,
            .out = .raw,
            .replaces = "the resident expert bodies on the same stored bytes, routes and input rows",
            .nan = .{ .stated = "every resident output bit, including non-finite encodings, is retained" },
            .served = &.{ .{ .rows = 1, .n = 64, .k = 64 }, .{ .rows = 2, .n = 64, .k = 64 }, .{ .rows = 8, .n = 64, .k = 64 } },
            .sums = .{ .why = "the address substitution has no arithmetic; this contract does not requalify the unchanged resident sums" },
        }, .vt = &vt };
    }
    fn of(ctx: *anyopaque) *Probe {
        return @ptrCast(@alignCast(ctx));
    }
    fn prepare(ctx: *anyopaque, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const p = of(ctx);
        if (shape.rows == 0 or shape.rows > 8 or shape.n != 64 or shape.k != 64) return false;
        if (p.fixture) |fixture| {
            fixture.deinit(p.a);
            p.fixture = null;
        }
        p.fixture = try tf.flashnext_fsr_fixture.Fixture.init(p.a, p.device, p.queue, shape.rows, p.bits, seed, fill == .nonfinite);
        p.actual = &.{};
        p.baseline = &.{};
        return true;
    }
    fn run(ctx: *anyopaque) !void {
        const p = of(ctx);
        const fixture = p.fixture.?;
        fixture.wrong_map = false;
        p.baseline = try fixture.run(false);
        fixture.wrong_map = p.control;
        p.actual = try fixture.run(true);
    }
    fn output(ctx: *anyopaque) ![]const u8 {
        return of(ctx).actual;
    }
    fn inputs(ctx: *anyopaque) ![]const u8 {
        const p = of(ctx);
        const fixture = p.fixture.?;
        p.inputs_.clearRetainingCapacity();
        try p.inputs_.appendSlice(p.a, std.mem.sliceAsBytes(fixture.initial));
        for (fixture.full) |layer| for (layer) |ref| try p.inputs_.appendSlice(p.a, fixture.bytes(ref));
        for (fixture.router_weights) |ref| try p.inputs_.appendSlice(p.a, fixture.bytes(ref));
        try p.inputs_.appendSlice(p.a, fixture.bytes(fixture.row_count));
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
        if (!std.mem.eql(u8, p.actual, p.baseline)) return "the slotted view changed resident output bits";
        return null;
    }
    fn replaced(ctx: *anyopaque) ![]const u8 {
        return of(ctx).baseline;
    }
    const vt: g.VTable = .{ .prepare = prepare, .run = run, .output = output, .inputs = inputs, .intact = intact, .overlap = overlap, .count = count, .reference = reference, .check = check, .replaced = replaced };
};
