//! The core route select (core/moe_route.metal's tf_route_select) at GLM's and Step-5's shapes against main's kernel.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const bits = @import("metal_glm_bits.zig");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const route = tf.moe_route;

/// Main's core/moe_route.metal: the select each row's simdgroup ran before the simd picks and prefix offsets.
pub const main_route = @embedFile("baselines/moe_route_main.metal.txt");
const M = 16; // the shapes' max_rows
const scale: f32 = 2.5; // config.zig's routed_scale (both families)

/// GLM's router (MLX's options; expert parallel splits the held experts) and Step-5's (precise math, every expert).
pub const Family = enum { glm, step5 };
fn shapeOf(f: Family) route.Shape {
    return if (f == .glm) tf.glm.kernels.route_shape else .{ .hidden = 4096, .experts = 352, .topk = 8 };
}
fn options(f: Family) mtl.CompileOptions {
    return if (f == .glm) mtl.CompileOptions.mlx() else .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise };
}

/// Rows: the window's rows; n: the experts; k: the held experts (0 all, 1 the lower half, 2 the upper: two Macs).
fn servedOf(comptime f: Family) [if (f == .glm) 8 else 4]c.Shape {
    const E = shapeOf(f).experts;
    return if (f == .glm) .{ .{ .rows = 16, .n = E, .k = 0 }, .{ .rows = 1, .n = E, .k = 0 }, .{ .rows = 3, .n = E, .k = 0 }, .{ .rows = 8, .n = E, .k = 0 }, .{ .rows = 1, .n = E, .k = 1 }, .{ .rows = 16, .n = E, .k = 1 }, .{ .rows = 2, .n = E, .k = 2 }, .{ .rows = 16, .n = E, .k = 2 } } else .{ .{ .rows = 16, .n = E, .k = 0 }, .{ .rows = 1, .n = E, .k = 0 }, .{ .rows = 5, .n = E, .k = 0 }, .{ .rows = 12, .n = E, .k = 0 } };
}
const served_glm = servedOf(.glm);
const served_step5 = servedOf(.step5);

/// The family's router library (core/moe_route.zig's defines) over `body`: today's core or main's frozen one.
pub fn text(a: std.mem.Allocator, f: Family, body: ?[]const u8) ![]u8 {
    const t = try route.source(a, shapeOf(f));
    if (body == null) return t;
    defer a.free(t);
    return std.mem.concat(a, u8, &.{ t[0 .. t.len - sources.core_moe_route.len], body.? });
}

pub const Case = struct {
    family: Family,
    control: bool,
    new: mtl.Pipeline,
    old: mtl.Pipeline,

    pub fn init(s: *bits.Suite, family: Family, control: bool) !Case {
        const own = try text(s.a, family, null);
        defer s.a.free(own);
        const body = if (control) try bits.edited(s.a, own, "void tf_route_select(", "offs[0][g] = b0 + held;", "offs[0][g] = b0;") else try s.a.dupe(u8, own);
        defer s.a.free(body);
        const frozen = try text(s.a, family, main_route);
        defer s.a.free(frozen);
        const new = try compile(s.device, body, family);
        errdefer new.deinit();
        return .{ .family = family, .control = control, .new = new, .old = try compile(s.device, frozen, family) };
    }
    fn compile(device: mtl.Device, t: []const u8, f: Family) !mtl.Pipeline {
        const lib = try mtl.Library.fromSource(device, t, options(f));
        defer lib.deinit();
        return mtl.Pipeline.init(device, lib, route.names[1], false);
    }
    pub fn deinit(x: *Case) void {
        x.new.deinit();
        x.old.deinit();
    }
    pub fn name(family: Family, control: bool) []const u8 {
        if (control) return if (family == .glm) "control: metal glm launch picks tf_route_select glm 288 experts, a group's held count dropping its last lane" else "control: metal glm launch picks tf_route_select step5 352 experts, a group's held count dropping its last lane";
        return if (family == .glm) "metal glm launch picks tf_route_select glm 288 experts top 8" else "metal glm launch picks tf_route_select step5 352 experts top 8";
    }
    pub fn spec(x: *const Case) c.Spec {
        return bits.spec(name(x.family, x.control), "main's tf_route_select", "main's picks (ties to the lower id, its NaN rule), weights, held groups and members, pick lists, counts and word", if (x.family == .glm) &served_glm else &served_step5, "the weights' fp32 sum and scale in main's order; the rest compares and counts");
    }
    /// The outputs: picks, weights, held ids, members, count, this Mac's and the peer's pick lists, counts, the word.
    const sizes = [_]usize{ M * 8 * 4, M * 8 * 4, M * 8 * 4, M * 8 * M * 4, 16, M * 8 * 4, M * 8 * 4, 16, 16 };
    pub fn layout(x: *const Case, s: c.Shape) ?bits.Layout {
        if (s.rows == 0 or s.rows > M or s.n != shapeOf(x.family).experts or s.k > 2) return null;
        var l: bits.Layout = .{};
        _ = l.add(.input, M * s.n * 4, .f32);
        _ = l.add(.input, s.n * 4, .f32);
        for (sizes, 0..) |n, i| _ = l.add(.output, n, if (i == 1) .f32 else .int);
        return l;
    }
    /// tf-glm-route-select's draws by row: random, levels (ties), edge values (NaN, -0, +-90, -inf bias), all equal.
    pub fn fill(_: *const Case, s: c.Shape, f: g.Fill, seed: u64, data: [][]u8) !void {
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        const lg = std.mem.bytesAsSlice(f32, data[0]);
        const bs = std.mem.bytesAsSlice(f32, data[1]);
        const d = if (f == .nonfinite) 2 else seed % 4;
        for (lg, 0..) |*v, i| v.* = switch ((d + i / s.n) % 4) {
            0 => (rnd.float(f32) - 0.5) * 8,
            1 => @round((rnd.float(f32) - 0.5) * 6) / 2,
            2 => if (i % 97 == 0) -0.0 else if (i % 89 == 0) 90 else if (i % 83 == 0) -90 else if (i % 79 == 0) 1e-30 else if (i % 211 == 5) std.math.nan(f32) else (rnd.float(f32) - 0.5) * 4,
            else => 0.25,
        };
        for (bs, 0..) |*v, i| v.* = switch (d) {
            0 => (rnd.float(f32) - 0.5) * 0.2,
            1, 3 => @as(f32, @floatFromInt(i % 4)) * 0.125,
            else => if (i == 7 or i == 200) -std.math.inf(f32) else if (i % 13 == 0) 0.0 else (rnd.float(f32) - 0.5) * 0.1,
        };
    }
    pub fn encode(x: *const Case, e: mtl.ComputeEncoder, r: []const rig_.Ref, s: c.Shape, new: bool) void {
        const E: u32 = @intCast(s.n);
        const own: [2]u32 = switch (s.k) {
            0 => .{ 0, E },
            1 => .{ 0, E / 2 },
            else => .{ E / 2, E },
        };
        route.select(e, if (new) x.new else x.old, r[0], r[1], scale, @intCast(s.rows), own, route.Outputs(rig_.Ref){ .pick = r[2], .wts = r[3], .ids = r[4], .members = r[5], .count = r[6], .mine = r[7], .theirs = r[8], .counts = r[9], .word = r[10] });
    }
};

pub fn collect(s: *bits.Suite, only: ?[]const u8) !void {
    if (s.wanted(only, Case.name(.glm, false))) {
        const loader = try s.production();
        const own = try text(s.a, .glm, null);
        defer s.a.free(own);
        var found = false;
        for (loader.texts.items) |t| found = found or std.mem.startsWith(u8, t, own);
        if (!found) return error.NotTheLoadersLibrary; // GLM's router library is this text with glm_route_one appended
    }
    for ([_]Family{ .glm, .step5 }) |family| for ([_]bool{ false, true }) |control| {
        if (s.wanted(only, Case.name(family, control))) try s.add(Case, try Case.init(s, family, control));
    };
}

/// Gate 5's registrations `glm launch picks glm|step5 <function>`: that family's router library.
pub const prefix = "glm launch picks ";
pub fn limitsText(a: std.mem.Allocator, _: mtl.Device, label: []const u8) !bits.Texts {
    const t = try text(a, if (std.mem.indexOf(u8, label, " step5 ") != null) .step5 else .glm, null);
    return .{ .here = t, .pre_m5 = t };
}
