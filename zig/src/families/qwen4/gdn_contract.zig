//! Readout and every saved recurrent row reach their native launch with independent f64 and propagated bounds.
const std = @import("std");
const common = @import("gdn_reference.zig");
const gdn = @import("cuda_gdn.zig");
pub const Fixture = common.Fixture;
pub const Part = common.Part;
pub const Control = @import("gdn_control_source.zig").Control;
pub fn Case(comptime contract: type, comptime B: type) type {
    const c = contract.spec;
    const g = contract.gates;
    return struct {
        const Self = @This();
        a: std.mem.Allocator,
        rig: B.Rig,
        backend: B,
        gate: gdn.Gate = .silu,
        flags: u4 = 11,
        part: Part = .readout,
        control: Control = .production,
        label: [160]u8 = undefined,
        arena: ?std.heap.ArenaAllocator = null,
        fixture: Fixture = undefined,
        repetition: u32 = 0,
        pub fn deinit(s: *Self) void {
            s.rig.deinit();
            if (s.arena) |*arena| arena.deinit();
            if (@hasDecl(B, "deinit")) s.backend.deinit();
        }
        fn depth(shape: c.Shape) usize {
            return shape.rows * 128;
        }
        pub fn kernel(s: *Self) g.Kernel {
            const label = std.fmt.bufPrint(&s.label, "{s} Qwen4 GDN {t} weights{d} {t} {t}", .{ B.name, s.gate, s.flags, s.part, s.control }) catch unreachable;
            return .{ .ctx = s, .vt = &vt, .control = switch (s.control) {
                .production => null,
                .nan => .nonfinite,
                .repeat => .repeat,
                .tail => .shapes,
                .sums => .sums,
            }, .spec = .{
                .name = label,
                .accum = if (s.part == .conv) .none else .f32,
                .out = if (s.part == .state) .f32 else .bf16,
                .depth = depth,
                .tiles = .{ .{ .size = 16 }, .{}, .{} },
                .served = B.served,
                .sums = if (s.part == .conv) .{ .f16 = 0, .bf16 = 0, .why = "Raw convolution tails only copy BF16 input bits." } else .{ .bf16 = 0, .exact = false, .why = "The normalized BF16 readout can hide BF16 sums; F16 square-sum overflow is separated and tested." },
            } };
        }
        const vt = g.hooks(Self);
        pub fn prepare(s: *Self, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
            if (shape.rows == 0 or shape.rows > 16 or !((shape.n == 1 and shape.k == 2) or (shape.n == 16 and shape.k == 48))) return false;
            s.rig.reset();
            if (s.arena) |*arena| arena.deinit();
            s.arena = .init(s.a);
            s.repetition = 0;
            s.fixture = try Fixture.make(s.arena.?.allocator(), .{ .rows = shape.rows, .keys = shape.n, .values = shape.k, .eps = 1e-6, .gate = s.gate, .weight_flags = s.flags }, s.part, fill, seed);
            try s.backend.bind(&s.rig, s.fixture, s.control);
            return true;
        }
        pub fn count(s: *Self) usize {
            return s.fixture.want.len;
        }
        pub fn run(s: *Self) !void {
            try s.backend.launch(&s.rig, s.fixture, s.repetition);
            s.repetition += 1;
        }
        pub fn reference(s: *Self, want: []f64, mag: []f64) !void {
            @memcpy(want, s.fixture.want);
            for (want, mag) |v, *m| m.* = @abs(v);
        }
        pub fn bounds(s: *Self, _: []const f64, _: []const f64, allow: []f64) !void {
            @memcpy(allow, s.fixture.error_bound);
        }
        pub fn simulate(s: *Self, acc: c.Accum, order: contract.rows.Order, out: []f64) !void {
            var temporary: std.heap.ArenaAllocator = .init(s.a);
            defer temporary.deinit();
            const errors = try temporary.allocator().alloc(f64, out.len);
            try s.fixture.compute(temporary.allocator(), acc, order, out, errors);
        }
    };
}
