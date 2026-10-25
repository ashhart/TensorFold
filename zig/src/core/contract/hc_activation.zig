//! HC division is BF16 before activation; injection rounds sigmoid once more before its exact factor of two.
const std = @import("std");
const h = @import("../hc_activation.zig");
const q = @import("../quant/quant.zig");
pub const Control = enum(u2) { production, nan, repeat, tail };
pub const Fixture = struct {
    p: h.Params,
    x: []u16,
    pub fn init(a: std.mem.Allocator, p: h.Params, fill: anytype, seed: u64) !Fixture {
        const x = try a.alloc(u16, (try h.bytes(p)) / 2);
        var rng = std.Random.DefaultPrng.init(seed);
        for (x) |*v| v.* = q.convert.bf16(@as(f32, @floatFromInt(rng.random().intRangeAtMost(i32, -128, 128))) / 8);
        if (fill == .nonfinite) for ([_]u16{ 0x7fc0, 0x7f80, 0xff80, 0x8000 }, 0..) |v, i| {
            if (i < x.len) x[i] = v;
        };
        x[x.len - 1] = 0x4100;
        return .{ .p = p, .x = x };
    }
    pub fn reference(f: Fixture, want: []f64, mag: []f64) void {
        for (f.x, want, mag) |bits, *w, *m| {
            const input: f64 = @as(f32, @bitCast(@as(u32, bits) << 16));
            const d: f64 = @as(f32, @bitCast(@as(u32, q.convert.bf16(@floatCast(input / @as(f64, @floatFromInt(f.p.streams))))) << 16));
            const e = @exp(-@abs(d));
            const sigmoid = (if (d < 0) e else @as(f64, 1)) / (1 + e);
            w.* = if (f.p.injection == 1) 2 * @as(f64, @as(f32, @bitCast(@as(u32, q.convert.bf16(@floatCast(sigmoid))) << 16))) else (if (d < 0) d * e else d) / (1 + e);
            m.* = @abs(w.*);
        }
    }
};
pub fn Case(comptime contract: type, comptime B: type) type {
    const c = contract.spec;
    const g = contract.gates;
    return struct {
        const Self = @This();
        a: std.mem.Allocator,
        rig: B.Rig,
        backend: B,
        streams: c_int,
        injection: bool,
        control: Control = .production,
        label: [128]u8 = undefined,
        arena: ?std.heap.ArenaAllocator = null,
        fixture: Fixture = undefined,
        repetition: c_int = 0,
        pub fn deinit(s: *Self) void {
            s.rig.deinit();
            if (s.arena) |*arena| arena.deinit();
        }
        fn depth(_: c.Shape) usize {
            return 0;
        }
        pub fn kernel(s: *Self) g.Kernel {
            const label = std.fmt.bufPrint(&s.label, "{s} HC activation streams{d} inject{} {t}", .{ B.name, s.streams, s.injection, s.control }) catch unreachable;
            return .{ .ctx = s, .vt = &vt, .control = switch (s.control) {
                .production => null,
                .nan => .nonfinite,
                .repeat => .repeat,
                .tail => .shapes,
            }, .spec = .{
                .name = label,
                .accum = .none,
                .out = .bf16,
                .depth = depth,
                .ops = 12,
                .mids = if (s.injection) 1 else 0,
                .tiles = .{ .{ .size = 8 }, .{ .size = 256 }, .{} },
                .served = if (s.injection) &.{ .{ .rows = 4, .n = 4, .k = 1 }, .{ .rows = 128, .n = 8, .k = 1 } } else &.{ .{ .rows = 4, .n = 257, .k = 1 }, .{ .rows = 8, .n = 320, .k = 1 }, .{ .rows = 128, .n = 640, .k = 1 } },
                .sums = .{ .why = "Scalar division and activation have no accumulation." },
            } };
        }
        const vt = g.hooks(Self);
        pub fn prepare(s: *Self, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
            if (shape.rows == 0 or shape.rows > 128 or shape.n == 0 or shape.n > 640 or shape.k != 1) return false;
            s.rig.reset();
            if (s.arena) |*arena| arena.deinit();
            s.arena = .init(s.a);
            s.repetition = 0;
            s.fixture = try Fixture.init(s.arena.?.allocator(), .{ .rows = @intCast(shape.rows), .cols = @intCast(shape.n), .streams = s.streams, .injection = @intFromBool(s.injection) }, fill, seed);
            try s.backend.bind(&s.rig, s.fixture);
            return true;
        }
        pub fn count(s: *Self) usize {
            return s.fixture.x.len;
        }
        pub fn run(s: *Self) !void {
            try s.backend.launch(&s.rig, s.fixture.p, s.control, s.repetition);
            s.repetition += 1;
        }
        pub fn reference(s: *Self, want: []f64, mag: []f64) !void {
            s.fixture.reference(want, mag);
        }
    };
}
