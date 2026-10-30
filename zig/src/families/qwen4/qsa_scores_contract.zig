//! QSA scores have an independent f64 dot, IEEE max-number ReLU and exact-integer accumulator stress.
const std = @import("std");
const ct = @import("core").contract;
const c = ct.spec;
pub const Control = @import("qsa_control_source.zig").Control;
pub const Fixture = struct {
    rows: usize,
    blocks: usize,
    query: []u16,
    pooled: []u16,
    complete: []i32,
    pub fn make(a: std.mem.Allocator, rows: usize, blocks: usize, fill: ct.gates.Fill, seed: u64) !Fixture {
        const f: Fixture = .{ .rows = rows, .blocks = blocks, .query = try a.alloc(u16, rows * 4 * 128), .pooled = try a.alloc(u16, blocks * 128), .complete = try a.alloc(i32, rows) };
        @memset(f.complete, @intCast(blocks));
        var random = std.Random.DefaultPrng.init(seed);
        for (f.query, 0..) |*v, i| v.* = c.toBf16(if (fill == .nonfinite) 1 else if (fill == .stress) (if (i % 128 < 64) @as(f32, 127) else 126) else random.random().float(f32) * 2 - 1);
        for (f.pooled, 0..) |*v, i| v.* = c.toBf16(if (fill == .nonfinite) 1 else if (fill == .stress) (if (i % 128 < 64) @as(f32, 127) else -128) else random.random().float(f32) * 2 - 1);
        if (fill == .nonfinite) for ([_]u16{ 0x7fc0, 0x7f80, 0xff80 }, 0..) |v, i| {
            f.pooled[i * 128] = v;
        };
        return f;
    }
    pub fn reference(f: Fixture, want: []f64, mag: []f64, acc: c.Accum, order: ct.rows.Order) void {
        for (0..f.rows) |r| for (0..f.blocks) |b| {
            var heads: [4]f64 = undefined;
            var magnitude: f64 = 0;
            for (&heads, 0..) |*head, h| {
                var terms: [128]f64 = undefined;
                for (&terms, 0..) |*v, d| {
                    v.* = @as(f64, c.fromBf16(f.query[(r * 4 + h) * 128 + d])) * c.fromBf16(f.pooled[b * 128 + d]);
                    magnitude += @abs(v.*);
                }
                const dot = ct.rows.sumIn(acc, order, &terms);
                head.* = if (std.math.isNan(dot)) 0 else @max(dot, 0);
            }
            const at = r * f.blocks + b;
            want[at] = ct.rows.sumIn(acc, order, &heads) / @sqrt(@as(f64, 128));
            mag[at] = magnitude / @sqrt(@as(f64, 128));
        };
    }
};
pub fn Case(comptime contract: type, comptime B: type) type {
    const g = contract.gates;
    return struct {
        const Self = @This();
        a: std.mem.Allocator,
        rig: B.Rig,
        backend: B,
        control: Control = .production,
        label: [160]u8 = undefined,
        arena: ?std.heap.ArenaAllocator = null,
        fixture: Fixture = undefined,
        repeat: u32 = 0,
        pub fn deinit(s: *Self) void {
            s.rig.deinit();
            if (s.arena) |*a| a.deinit();
        }
        fn depth(_: c.Shape) usize {
            return 128 + 4;
        }
        pub fn kernel(s: *Self) g.Kernel {
            const name = std.fmt.bufPrint(&s.label, "{s} Qwen4 QSA scores {t}", .{ B.name, s.control }) catch unreachable;
            return .{ .ctx = s, .vt = &vt, .control = switch (s.control) {
                .production => null,
                .nan => .nonfinite,
                .repeat => .repeat,
                .tail => .shapes,
                .sums => .sums,
            }, .spec = .{
                .name = name,
                .accum = .f32,
                .depth = depth,
                .ops = 6,
                .out = .f32,
                .nan = .{ .stated = "IEEE max-number ReLU maps a NaN dot to zero and preserves positive infinity." },
                .tiles = .{ .{ .size = 8 }, .{ .size = 64, .min = 129 }, .{} },
                .served = B.served,
            } };
        }
        const vt = g.hooks(Self);
        pub fn prepare(s: *Self, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
            if (shape.rows == 0 or shape.rows > 128 or shape.n < 129 or shape.n > 4097 or shape.k != 128) return false;
            s.rig.reset();
            if (s.arena) |*a| a.deinit();
            s.arena = .init(s.a);
            s.repeat = 0;
            s.fixture = try Fixture.make(s.arena.?.allocator(), shape.rows, shape.n, fill, seed);
            try s.backend.bind(&s.rig, s.fixture, s.control);
            return true;
        }
        pub fn count(s: *Self) usize {
            return s.fixture.rows * s.fixture.blocks;
        }
        pub fn run(s: *Self) !void {
            try s.backend.launch(&s.rig, s.fixture, s.repeat);
            s.repeat += 1;
        }
        pub fn reference(s: *Self, want: []f64, mag: []f64) !void {
            s.fixture.reference(want, mag, .none, .sequential);
        }
        pub fn simulate(s: *Self, acc: c.Accum, order: contract.rows.Order, out: []f64) !void {
            const mag = try s.a.alloc(f64, out.len);
            defer s.a.free(mag);
            s.fixture.reference(out, mag, acc, order);
        }
        pub fn check(s: *Self, at: g.Gate) !?[]const u8 {
            if (at != .nonfinite and at != .shapes) return null;
            const want = try s.a.alloc(f64, s.count());
            defer s.a.free(want);
            const mag = try s.a.alloc(f64, want.len);
            defer s.a.free(mag);
            try s.reference(want, mag);
            const out = try s.rig.outputBytes();
            var positive: usize = 0;
            var finite: usize = 0;
            for (want, mag, 0..) |v, magnitude, i| {
                if (c.classOf(v) != c.classOf(c.read(.f32, out, i))) return "QSA score max-number nonfinite class differs";
                if (std.math.isInf(v) and v > 0) positive += 1;
                if (std.math.isFinite(v)) {
                    const actual = c.read(.f32, out, i);
                    const limit = if (at == .nonfinite) c.boundExact(s.kernel().spec, v) else c.bound(s.kernel().spec, .{ .rows = s.fixture.rows, .n = s.fixture.blocks, .k = 128 }, v, magnitude);
                    if (!(@abs(actual - v) <= limit)) return "QSA finite score leaves its independent f64 bound";
                    finite += 1;
                }
            }
            return if (at == .nonfinite and (positive == 0 or finite == 0)) "QSA score fixture must prove both positive infinity and finite scores" else null;
        }
    };
}
test "Qwen4 QSA scores: integer stress proves exact F32 sums and separates every reduced order" {
    const a = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const f = try Fixture.make(arena.allocator(), 1, 129, .stress, 0x367);
    var want: [129]f64 = undefined;
    var mag: [129]f64 = undefined;
    f.reference(&want, &mag, .none, .sequential);
    for ([_]ct.rows.Order{ .sequential, .pairwise, .strided }) |order| {
        var out: [129]f64 = undefined;
        f.reference(&out, &mag, .f32, order);
        try std.testing.expectEqualSlices(f64, &want, &out);
        for ([_]c.Accum{ .f16, .bf16 }) |acc| {
            f.reference(&out, &mag, acc, order);
            for (out, want) |got, v| try std.testing.expect(!std.math.isFinite(got) or @abs(got - v) > 1000 * c.unit(.f32) * @abs(v));
        }
    }
}
