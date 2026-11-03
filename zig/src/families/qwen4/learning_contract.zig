//! Native learner primitives use the shared five-gate harness, guarded backend buffers and expression-derived bounds.
const std = @import("std");
const core = @import("core");
const contract = core.contract;
const data = @import("learning_contract_data.zig");
pub fn Case(comptime B: type, comptime spec: type) type {
    return struct {
        const Self = @This();
        pub const Fixture = data.Fixture(spec);
        const vtable = contract.gates.hooks(Self);
        const zero_vtable = blk: {
            var v = vtable;
            v.check = zeroCheck;
            break :blk v;
        };
        fn zeroCheck(ctx: *anyopaque, _: contract.gates.Gate) anyerror!?[]const u8 {
            const c: *Self = @ptrCast(@alignCast(ctx));
            for (c.fixture.?.values, 8..) |values, i| for (values, 0..) |_, j| {
                if (@as(u64, @bitCast(c.fixture.?.output(i, j))) != 0) return "zero must overwrite every output with positive zero";
            };
            return null;
        }
        a: std.mem.Allocator,
        backend: B,
        rig: struct {
            base: B.Rig,
            parent: ?*Self = null,
            pub fn outputBytes(r: *@This()) ![]const u8 {
                return r.parent.?.output();
            }
            pub fn inputBytes(r: *@This()) ![]const u8 {
                return r.base.inputBytes();
            }
            pub fn intact(r: *@This()) !bool {
                return r.base.intact();
            }
            pub fn overlap(r: *@This()) bool {
                return r.base.overlap();
            }
        },
        op: spec.Op,
        control: core.living_rows.Control = .production,
        iteration: u32 = 0,
        sigmoid: bool = false,
        geglu: bool = false,
        gemma_attention: bool = false,
        fixture: ?Fixture = null,
        converted: std.ArrayList(f32) = .empty,
        label: []const u8,
        shapes: [4]contract.spec.Shape = .{ .{ .rows = 1, .n = 1, .k = 64 }, .{ .rows = 3, .n = 1, .k = 64 }, .{ .rows = 17, .n = 1, .k = 128 }, .{ .rows = 33, .n = 1, .k = 256 } },
        pub fn useGemmaAttention(c: *Self) void {
            c.gemma_attention = true;
            c.shapes = .{ .{ .rows = 1, .n = 32, .k = 256 }, .{ .rows = 3, .n = 16, .k = 256 }, .{ .rows = 17, .n = 32, .k = 512 }, .{ .rows = 33, .n = 16, .k = 512 } };
        }
        pub fn deinit(c: *Self) void {
            if (c.fixture) |*f| f.deinit();
            c.rig.base.deinit();
            c.converted.deinit(c.a);
        }
        fn depth(shape: contract.spec.Shape) usize {
            return 16 + shape.rows * shape.k;
        }
        pub fn kernel(c: *Self) contract.gates.Kernel {
            const constant_zero = if (spec == core.living_gemma) c.op == .zero else false;
            return .{ .control = switch (c.control) {
                .production => null,
                .nan => .nonfinite,
                .repeat => .repeat,
                .tail => .shapes,
                .sums => .sums,
            }, .spec = .{ .name = c.label, .accum = if (data.reduces(spec, c.op)) .f32 else .none, .depth = depth, .out = .f32, .nan = if (constant_zero) .{ .stated = "no inputs; every output is positive zero" } else .propagate, .served = &c.shapes, .tiles = .{ .{ .size = 16, .min = 1 }, .{}, .{ .size = 64, .step = 64, .min = 64 } } }, .ctx = c, .vt = if (constant_zero) &zero_vtable else &vtable };
        }
        pub fn prepare(c: *Self, shape: contract.spec.Shape, fill: contract.gates.Fill, seed: u64) !bool {
            c.rig.parent = c;
            c.iteration = 0;
            c.rig.base.reset();
            if (c.fixture) |*f| {
                f.deinit();
                c.fixture = null;
            }
            const gemma = spec == core.living_qsa and c.gemma_attention;
            if (shape.rows == 0 or shape.rows > 512 or shape.k < 64 or (shape.k > 256 and !(gemma and shape.k == 512)) or shape.k % 64 != 0) return false;
            if (gemma and ((shape.n != 16 and shape.n != 32) or (shape.k != 256 and shape.k != 512))) return false;
            var p = data.params(spec, c.op, shape.rows, shape.k);
            if (spec == core.living_gdn) {
                p.u[4] = 1;
                p.u[6] = 1;
                p.u[5] = @intFromBool(c.sigmoid);
            }
            if (spec == core.living_qsa) {
                p.u[5] = @intFromBool(B.probability_cut);
                if (shape.k == 256) {
                    p.u[1] = 24;
                    p.u[2] = 4;
                }
            }
            if (spec == core.living_rows and c.op == .linear) p.u[1] = @intCast(shape.k);
            if (spec == core.living_rows) {
                if (c.geglu) p.u[12] = 1;
            }
            if (spec == core.living_qsa) {
                if (c.gemma_attention) {
                    p.u[1] = @intCast(shape.n);
                    p.u[2] = @intCast(if (shape.k == 512) (if (shape.n == 16) @as(usize, 1) else 4) else (if (shape.n == 16) @as(usize, 8) else 16));
                    p.u[3] = @intCast(shape.k);
                    p.u[5] = if (B.probability_cut) 2 else 0;
                    p.u[6] = p.u[3] / 128;
                    p.f[0] = 1;
                }
            }
            c.fixture = try Fixture.init(c.a, c.op, p, seed, fill == .nonfinite);
            if (fill == .stress) c.fixture.?.stress();
            try c.fixture.?.reference();
            try c.backend.bind(&c.rig.base, &c.fixture.?);
            return true;
        }
        pub fn run(c: *Self) !void {
            c.iteration += 1;
            c.fixture.?.raw.p.u[13] = c.iteration;
            try c.backend.launch(&c.rig.base, &c.fixture.?);
        }
        pub fn simulate(c: *Self, acc: contract.spec.Accum, order: contract.gates.Order, out: []f64) !void {
            try c.fixture.?.simulate(acc, order, out);
        }
        pub fn count(c: *Self) usize {
            var n: usize = 0;
            for (c.fixture.?.values) |v| n += v.len;
            return n;
        }
        fn output(c: *Self) ![]const u8 {
            const f = &c.fixture.?;
            try c.converted.resize(c.a, c.count());
            var at: usize = 0;
            for (f.values, 8..) |values, i| for (0..values.len) |j| {
                c.converted.items[at] = @floatCast(f.output(i, j));
                at += 1;
            };
            return std.mem.sliceAsBytes(c.converted.items);
        }
        pub fn reference(c: *Self, want: []f64, mag: []f64) !void {
            var at: usize = 0;
            for (c.fixture.?.values) |values| for (values) |v| {
                want[at] = v.value;
                mag[at] = @abs(v.value);
                at += 1;
            };
        }
        pub fn bounds(c: *Self, _: []const f64, _: []const f64, allow: []f64) !void {
            var at: usize = 0;
            for (c.fixture.?.values) |values| for (values) |v| {
                if (std.math.isFinite(v.value) and !std.math.isFinite(v.err)) return error.UnboundedPrimitiveReference;
                allow[at] = v.err;
                at += 1;
            };
        }
    };
}
