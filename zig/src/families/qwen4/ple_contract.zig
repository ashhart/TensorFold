//! PLE gates, convolution outputs and kept tails use one independent formula at the native launch boundary.
const std = @import("std");
pub const reference = @import("ple_reference.zig");
pub const Fixture = reference.Fixture;
pub const Part = reference.Part;
pub const Control = @import("ple_control_source.zig").Control;
pub fn Case(comptime ct: type, comptime B: type) type {
    return struct {
        const Self = @This();
        a: std.mem.Allocator,
        rig: B.Rig,
        backend: B,
        part: Part = .output,
        control: Control = .production,
        label: [160]u8 = undefined,
        arena: ?std.heap.ArenaAllocator = null,
        fixture: Fixture = undefined,
        repeat: u32 = 0,
        pub fn deinit(s: *Self) void {
            s.rig.deinit();
            if (s.arena) |*a| a.deinit();
            if (@hasDecl(B, "deinit")) s.backend.deinit();
        }
        fn depth(shape: ct.spec.Shape) usize {
            return shape.n;
        }
        pub fn kernel(s: *Self) ct.gates.Kernel {
            const label = std.fmt.bufPrint(&s.label, "{s} Qwen4 PLE {t} {t}", .{ B.name, s.part, s.control }) catch unreachable;
            return .{ .ctx = s, .vt = &vt, .control = switch (s.control) {
                .production => null,
                .nan => .nonfinite,
                .repeat => .repeat,
                .tail => .shapes,
                .sums => .sums,
            }, .spec = .{
                .name = label,
                .accum = .f32,
                .out = .bf16,
                .depth = depth,
                .tiles = .{ .{ .size = 16 }, .{}, .{} },
                .served = B.served,
                .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "BF16 nodes can hide reduced sums; explicit overflow controls exercise the native F32 reducers." },
            } };
        }
        const vt = ct.gates.hooks(Self);
        pub fn prepare(s: *Self, shape: ct.spec.Shape, fill: ct.gates.Fill, seed: u64) !bool {
            if (shape.rows == 0 or shape.rows > 128 or !((shape.n == 256 and shape.k == 2) or (shape.n == 5120 and shape.k == 4))) return false;
            s.rig.reset();
            if (s.arena) |*a| a.deinit();
            s.arena = .init(s.a);
            s.repeat = 0;
            s.fixture = try Fixture.make(s.arena.?.allocator(), .{ .rows = shape.rows, .hidden = shape.n, .streams = shape.k }, s.part, fill, seed);
            try s.backend.bind(&s.rig, s.fixture, s.control);
            return true;
        }
        pub fn count(s: *Self) usize {
            return s.fixture.want.len;
        }
        pub fn run(s: *Self) !void {
            try s.backend.launch(&s.rig, s.fixture, s.repeat);
            s.repeat += 1;
        }
        pub fn reference(s: *Self, want: []f64, mag: []f64) !void {
            @memcpy(want, s.fixture.want);
            for (want, mag) |v, *m| m.* = @abs(v);
        }
        pub fn bounds(s: *Self, _: []const f64, _: []const f64, out: []f64) !void {
            @memcpy(out, s.fixture.error_bound);
        }
        pub fn simulate(s: *Self, acc: ct.spec.Accum, order: ct.rows.Order, out: []f64) !void {
            var arena: std.heap.ArenaAllocator = .init(s.a);
            defer arena.deinit();
            const mem = arena.allocator();
            try s.fixture.compute(mem, acc, order, out, try mem.alloc(f64, out.len));
        }
    };
}

pub fn configuration(a: std.mem.Allocator, shape: reference.Shape) !@import("config.zig").Config {
    var problem: @import("config.zig").Problem = .{};
    var config = try @import("config.zig").parse(a, @embedFile("fixtures/tiny-dense.json"), &problem);
    config.hidden = shape.hidden;
    config.streams = shape.streams;
    config.ple_conv = shape.taps;
    config.ngram = shape.dilation;
    config.eps = shape.eps;
    return config;
}
