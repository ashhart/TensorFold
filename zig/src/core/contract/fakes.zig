//! CPU kernels for the contract's own tests: correct row operations and one planted fault for each gate.
const std = @import("std");
const c = @import("contract.zig");
const g = @import("gates.zig");
const rows = @import("rows.zig");

/// What a CPU kernel gets wrong: nothing, max dropping NaN, order by run, a lost tail, fp16 sums, a write to its input.
pub const Fault = enum { none, fmax, unordered, tail, fp16, writes_input };

pub const Fake = struct {
    a: std.mem.Allocator,
    op: rows.Op,
    fault: Fault = .none,
    r: ?rows.Rows = null,
    out: []u8 = &.{},
    seen: []u8 = &.{},
    runs: usize = 0,

    pub fn deinit(f: *Fake) void {
        if (f.r) |r| r.deinit();
        f.a.free(f.out);
        f.a.free(f.seen);
        f.r = null;
        f.out = &.{};
        f.seen = &.{};
    }

    fn of(ctx: *anyopaque) *Fake {
        return @ptrCast(@alignCast(ctx));
    }

    fn prepare(ctx: *anyopaque, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const f = of(ctx);
        f.deinit();
        f.r = try rows.Rows.init(f.a, f.op, shape);
        f.r.?.fill(fill, seed);
        f.out = try f.a.alloc(u8, 4 * f.r.?.outputs());
        @memset(f.out, 0xff);
        f.seen = try f.a.alloc(u8, 2 * (f.r.?.x.len + f.r.?.w.len));
        return true;
    }

    fn run(ctx: *anyopaque) !void {
        const f = of(ctx);
        const r = f.r.?;
        const k = r.shape.k;
        f.runs += 1;
        for (0..r.shape.rows) |row| {
            const x = r.x[row * k ..][0..k];
            var y: f32 = 0;
            if (r.op == .max) {
                y = -std.math.inf(f32);
                for (x) |h| {
                    const v = c.fromBf16(h);
                    y = if (f.fault == .fmax) fmax(y, v) else if (std.math.isNan(v) or std.math.isNan(y)) std.math.nan(f32) else @max(y, v);
                }
            } else {
                var h16: f16 = 0;
                const end = if (f.fault == .tail and k % 32 != 0) k - 1 else k;
                for (0..end) |j| {
                    const i = if (f.fault == .unordered) (j + f.runs * 7) % end else j;
                    const t = c.fromBf16(x[i]) * (if (r.op == .dot) c.fromBf16(r.w[i]) else 1);
                    y += t;
                    h16 += @floatCast(t);
                }
                if (f.fault == .fp16) y = h16;
            }
            std.mem.writeInt(u32, f.out[4 * row ..][0..4], @bitCast(y), .little);
        }
        if (f.fault == .writes_input) r.x[0] = 0;
    }

    fn output(ctx: *anyopaque) ![]const u8 {
        return of(ctx).out;
    }

    fn inputs(ctx: *anyopaque) ![]const u8 {
        const f = of(ctx);
        const r = f.r.?;
        @memcpy(f.seen[0 .. 2 * r.x.len], std.mem.sliceAsBytes(r.x));
        @memcpy(f.seen[2 * r.x.len ..], std.mem.sliceAsBytes(r.w));
        return f.seen;
    }

    fn intact(_: *anyopaque) !bool {
        return true;
    }

    fn overlap(_: *anyopaque) bool {
        return false;
    }

    fn count(ctx: *anyopaque) usize {
        return of(ctx).r.?.outputs();
    }

    fn reference(ctx: *anyopaque, want: []f64, mag: []f64) !void {
        return of(ctx).r.?.reference(want, mag);
    }

    fn simulate(ctx: *anyopaque, acc: c.Accum, order: g.Order, out: []f64) !void {
        return of(ctx).r.?.simulate(acc, order, out);
    }

    pub const vt = g.VTable{ .prepare = prepare, .run = run, .output = output, .inputs = inputs, .intact = intact, .overlap = overlap, .count = count, .reference = reference, .simulate = simulate };

    pub fn kernel(f: *Fake, name: []const u8, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(name, f.op), .ctx = f, .vt = &vt, .control = control };
    }
};

/// IEEE maxNum, as C's fmax and Metal's max: a NaN operand loses to a number.
fn fmax(a: f32, b: f32) f32 {
    if (std.math.isNan(a)) return b;
    if (std.math.isNan(b)) return a;
    return @max(a, b);
}

fn chain(s: c.Shape) usize {
    return s.k;
}

fn spec(name: []const u8, op: rows.Op) c.Spec {
    return .{
        .name = name,
        .accum = if (op == .max) .none else .f32,
        .depth = chain,
        .out = .f32,
        .tiles = .{ .{ .size = 16 }, .{}, .{ .size = 32 } },
        .served = &.{ .{ .rows = 8, .n = 1, .k = 4096 }, .{ .rows = 3, .n = 1, .k = 640 } },
    };
}

test "correct CPU kernels pass gates 1-4 and every planted fault goes red at its gate" {
    const a = std.testing.allocator;
    var good = [_]Fake{ .{ .a = a, .op = .sum }, .{ .a = a, .op = .dot }, .{ .a = a, .op = .max } };
    var bad = [_]Fake{
        .{ .a = a, .op = .max, .fault = .fmax },
        .{ .a = a, .op = .sum, .fault = .unordered },
        .{ .a = a, .op = .sum, .fault = .writes_input },
        .{ .a = a, .op = .sum, .fault = .tail },
        .{ .a = a, .op = .dot, .fault = .fp16 },
    };
    defer for (&good) |*f| f.deinit();
    defer for (&bad) |*f| f.deinit();
    const want = [_]g.Gate{ .nonfinite, .repeat, .repeat, .shapes, .sums };
    var ks: [8]g.Kernel = undefined;
    for (&good, 0..) |*f, i| ks[i] = f.kernel(@tagName(f.op), null);
    for (&bad, want, 0..) |*f, gate, i| ks[3 + i] = f.kernel(@tagName(f.fault), gate);
    var t: g.Tally = .{};
    g.quiet = true;
    defer g.quiet = false;
    try g.runAll(a, &ks, 0x5eed, &t);
    try std.testing.expectEqual(@as(usize, 3), t.passed);
    try std.testing.expectEqual(@as(usize, 5), t.caught);
    try std.testing.expectEqual(@as(usize, 0), t.problems());
}

test "a tail fault passes at whole tiles: the edges are what catch it" {
    const a = std.testing.allocator;
    var f = Fake{ .a = a, .op = .sum, .fault = .tail };
    defer f.deinit();
    var k = f.kernel("tail at tiles only", null);
    k.spec.tiles = .{ .{}, .{}, .{} };
    k.spec.served = &.{.{ .rows = 4, .n = 1, .k = 4096 }};
    g.quiet = true;
    defer g.quiet = false;
    try std.testing.expectEqual(g.Verdict.pass, try g.shapes(a, k, 1));
}

test "a prepare that runs out of memory leaves nothing to free twice" {
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    var f = Fake{ .a = failing.allocator(), .op = .sum };
    defer f.deinit();
    const shape: c.Shape = .{ .rows = 4, .n = 1, .k = 64 };
    try std.testing.expect(try Fake.prepare(&f, shape, .random, 1));
    failing.fail_index = failing.alloc_index;
    try std.testing.expectError(error.OutOfMemory, Fake.prepare(&f, shape, .random, 2));
}
