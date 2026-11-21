//! Post-block taps keep the exact BF16 residuals in row-major trained-layer order without changing the target.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const fl = @import("metal_flint.zig");
const c = contract.spec;
const g = contract.gates;
const base = "metal flint glue fl_tap";
const Tap = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    control: bool,
    label: [160]u8 = undefined,
    x: []f32 = &.{},
    shape: c.Shape = undefined,
    input: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, control: bool) !Tap {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .control = control, .pipe = try fl.gluePipe(a, device, "fl_tap", if (control) .{ .kernel = "fl_tap", .from = "x[i.y * a.x + i.x]", .to = "x[i.y * a.x + (i.x + 1) % a.x]" } else null) };
    }
    pub fn deinit(t: *Tap) void {
        t.a.free(t.x);
        t.rig.deinit();
        t.pipe.deinit();
    }
    const vt = g.hooks(Tap);
    pub fn kernel(t: *Tap) g.Kernel {
        return .{ .spec = .{ .name = fl.named(base, if (t.control) "next column" else null, &t.label), .out = .bf16, .accum = .none, .depth = fl.noSums, .tiles = .{ .{}, .{ .size = 256 }, .{} }, .served = &.{ .{ .rows = 16, .n = 2048, .k = 1 }, .{ .rows = 2048, .n = 2048, .k = 1 }, .{ .rows = 1, .n = 2048, .k = 1 }, .{ .rows = 32, .n = 2048, .k = 1 } } }, .ctx = t, .vt = &vt, .control = if (t.control) .shapes else null };
    }
    pub fn prepare(t: *Tap, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        t.a.free(t.x);
        t.x = &.{};
        t.rig.reset();
        t.shape = shape;
        t.x = try t.a.alloc(f32, shape.rows * shape.n * 5);
        var prng = std.Random.DefaultPrng.init(seed);
        for (t.x) |*x| x.* = (prng.random().float(f32) - 0.5) * 8;
        if (fill == .nonfinite) for (t.x[0..3], fl.planted) |*x, v| {
            x.* = v;
        };
        t.input = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.out = try t.rig.output(t.x.len * 2);
        return true;
    }
    pub fn run(t: *Tap) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        const n = t.shape.n;
        for (0..5) |tap| {
            p.e.setBuffer(t.input.buf, t.input.off + tap * t.shape.rows * n * 4, 0);
            p.e.setBuffer(t.out.buf, t.out.off, 1);
            p.e.setValue([4]u32{ @intCast(n), @intCast(t.shape.rows), @intCast(5 * n), @intCast(tap * n) }, 2);
            p.e.dispatchThreads(mtl.Size.of(n, t.shape.rows, 1), mtl.Size.of(256, 1, 1));
        }
        try t.rig.finish(p);
    }
    pub fn count(t: *Tap) usize {
        return t.x.len;
    }
    pub fn reference(t: *Tap, want: []f64, mag: []f64) !void {
        const n = t.shape.n;
        for (want, mag, 0..) |*w, *m, i| {
            const row = i / (5 * n);
            const tap = i / n % 5;
            w.* = fl.bf(t.x[(tap * t.shape.rows + row) * n + i % n]);
            m.* = 0;
        }
    }
};
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    taps: fl.List(Tap) = .{},
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        if (fl.wanted(only, base)) for ([_]bool{ false, true }) |control| {
            try s.taps.add(s.a, s.kernels, try Tap.init(s.a, s.device, s.queue, control));
        };
    }
    pub fn deinit(s: *Suite) void {
        s.taps.deinit(s.a);
    }
};
pub fn limitsText(_: std.mem.Allocator, _: mtl.Device, _: []const u8) !?fl.Texts {
    return null;
}
