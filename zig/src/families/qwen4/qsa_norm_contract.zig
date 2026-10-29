//! QSA query, key, index-query and pooled-key Cases share independent BF16-node references.
const std = @import("std");
pub const reference = @import("qsa_norm_reference.zig");
pub const Fixture = reference.Fixture;
pub const Part = reference.Part;
pub const Control = @import("qsa_control_source.zig").Control;
pub fn Case(comptime ct: type, comptime B: type) type {
    return struct {
        const Self = @This();
        a: std.mem.Allocator,
        rig: B.Rig,
        backend: B,
        part: Part = .query,
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
        fn depth(_: ct.spec.Shape) usize {
            return 256;
        }
        pub fn kernel(s: *Self) ct.gates.Kernel {
            const name = std.fmt.bufPrint(&s.label, "{s} Qwen4 QSA {t} {t}", .{ B.name, s.part, s.control }) catch unreachable;
            return .{ .ctx = s, .vt = &vt, .control = switch (s.control) {
                .production => null,
                .nan => .nonfinite,
                .repeat => .repeat,
                .tail => .shapes,
                .sums => .sums,
            }, .spec = .{
                .name = name,
                .accum = .f32,
                .out = .bf16,
                .depth = depth,
                .tiles = .{ .{ .size = 16 }, .{}, .{} },
                .served = B.served,
                .sums = .{ .bf16 = 0, .exact = false, .why = "BF16 normalization can mask BF16 sums; the fixture separates F16 square-sum overflow." },
            } };
        }
        const vt = ct.gates.hooks(Self);
        pub fn prepare(s: *Self, shape: ct.spec.Shape, fill: ct.gates.Fill, seed: u64) !bool {
            if (shape.rows == 0 or shape.rows > 128 or shape.n != 24 or shape.k != 4) return false;
            s.rig.reset();
            if (s.arena) |*a| a.deinit();
            s.arena = .init(s.a);
            s.repeat = 0;
            s.fixture = try Fixture.make(s.arena.?.allocator(), s.part, shape.rows, fill, seed);
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
            @memcpy(out, s.fixture.errors);
        }
        pub fn simulate(s: *Self, acc: ct.spec.Accum, order: ct.rows.Order, out: []f64) !void {
            const errors = try s.a.alloc(f64, out.len);
            defer s.a.free(errors);
            try s.fixture.compute(acc, order, out, errors);
        }
    };
}
