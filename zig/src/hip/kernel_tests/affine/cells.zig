//! The prompt cells the tuner times: dense rows and served routed tokens.

const std = @import("std");
const tuning = @import("../../tuning/tuning.zig");
const moe = @import("../../moe.zig");
const cases = @import("cases.zig");
const affine = @import("../../launches/affine.zig");
const registry = @import("core").registry;

/// A product geometry on the prompt path: dense, or a served model's routed gate and up (or up) or down.
pub const Class = struct {
    name: []const u8,
    n: usize,
    k: usize,
    bits: u8,
    group: u16,
    routed: bool = false,
    down: bool = false,
    /// A routed class's plan: its routed experts, a token's slots, whose last is the shared expert if `shared`.
    experts: usize = 0,
    slots: usize = 0,
    shared: bool = false,
    /// The plan is sized by the tokens, not a power-of-two bucket.
    exact: bool = false,

    /// The case of `count` rows (dense) or tokens (routed) of this class.
    pub fn case(c: Class, count: usize) cases.Case {
        if (!c.routed) return .{ .name = c.name, .path = .prefill, .rows = count, .n = c.n, .k = c.k, .bits = c.bits, .group = c.group };
        return .{ .name = c.name, .path = .prefill, .rows = count, .n = c.n, .k = c.k, .bits = c.bits, .group = c.group, .experts = c.experts, .slots = c.slots, .shared = c.shared, .served = true, .exact = c.exact, .down = c.down };
    }
};

const qwen: Class = .{ .name = "", .n = 0, .k = 0, .bits = 4, .group = 64, .routed = true, .experts = 256, .slots = 9, .shared = true };
const nemotron: Class = .{ .name = "", .n = 0, .k = 0, .bits = 4, .group = 64, .routed = true, .experts = 128, .slots = 6 };
const flashnext: Class = .{ .name = "", .n = 0, .k = 0, .bits = 4, .group = 32, .routed = true, .experts = 512, .slots = 11, .shared = true, .exact = true };

fn with(c: Class, name: []const u8, n: usize, k: usize, down: bool) Class {
    var out = c;
    out.name = name;
    out.n = n;
    out.k = k;
    out.down = down;
    return out;
}

/// The served routed classes, Qwen3.6-35B-A3B's first; Nemotron's shared expert runs outside its plan.
pub const routed = [_]Class{
    with(qwen, "35b served gate_up", 1024, 2048, false),
    with(qwen, "35b served down", 2048, 512, true),
    with(nemotron, "nemotron served up", 1856, 2688, false),
    with(nemotron, "nemotron served down", 2688, 1856, true),
    with(flashnext, "flash next served gate_up", 1280, 2560, false),
    with(flashnext, "flash next served down", 2560, 640, true),
};

/// The rows and tokens the tuner starts from; it bisects only between them, so its cells stay at or below the top.
pub const coarse_rows = [_]usize{ 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256 };
pub const coarse_tokens = [_]usize{ 1, 2, 4, 8, 16, 32, 33, 63, 64, 65, 128, 256, 512, 1024, 2048 };

/// The most rows or tokens the tuner bisects between.
pub fn sampledMax(c: Class) usize {
    return if (c.routed) 2048 else 256;
}

/// The distinct dense prompt geometries of `cases.prefill` and `cases.short`, then the routed classes.
pub fn classes(buf: []Class) []Class {
    var len: usize = 0;
    inline for (.{ &cases.prefill, &cases.short }) |list| for (list) |c| {
        if (c.routed()) continue;
        const g: Class = .{ .name = c.name, .n = c.n, .k = c.k, .bits = c.bits, .group = c.group };
        for (buf[0..len]) |o| {
            if (o.n == g.n and o.k == g.k and o.bits == g.bits and o.group == g.group and !o.routed) break;
        } else {
            buf[len] = g;
            len += 1;
        }
    };
    for (routed) |r| {
        buf[len] = r;
        len += 1;
    }
    return buf[0..len];
}

pub const max_classes = cases.prefill.len + cases.short.len + routed.len;

/// The `cases.prefill` rows of a dense class past `sampledMax`, timed as they are.
pub fn longRows(c: Class, buf: []usize) []usize {
    var len: usize = 0;
    if (c.routed) return buf[0..0];
    for (cases.prefill) |p| {
        if (p.routed() or p.n != c.n or p.k != c.k or p.bits != c.bits or p.group != c.group or p.rows <= sampledMax(c)) continue;
        if (std.mem.indexOfScalar(usize, buf[0..len], p.rows) != null) continue;
        buf[len] = p.rows;
        len += 1;
    }
    return buf[0..len];
}

/// The item count serving passes the registry for `tokens` tokens of a routed class (`moe.run`, `expertParts`).
pub fn selectorCount(c: Class, tokens: usize) usize {
    if (tokens == 1) return c.slots;
    const size = if (c.exact) tokens else moe.bucket(tokens);
    return moe.capacity(size * c.slots, c.experts + @intFromBool(c.shared));
}

/// The cells on both sides of a bound `b` (rows b and b + 1, or tokens at or below and just above it), both sampled.
pub fn sides(c: Class, b: u32, out: *[2]usize) bool {
    if (b == 0) return false;
    if (!c.routed) {
        out.* = .{ b, b + 1 };
    } else {
        var below: usize = 0;
        var t: usize = 1;
        while (t <= sampledMax(c) and selectorCount(c, t) <= b) : (t += 1) below = t;
        if (below == 0 or t > sampledMax(c)) return false;
        out.* = .{ below, t };
    }
    return out[1] <= sampledMax(c);
}

/// The tuner's starting cells: the coarse counts, the long rows and both sides of each prompt bound of either table.
pub fn coarse(c: Class, list: *std.ArrayList(usize), gpa: std.mem.Allocator) !void {
    const base: []const usize = if (c.routed) &coarse_tokens else &coarse_rows;
    for (base) |v| try put(list, v, gpa);
    var buf: [cases.prefill.len]usize = undefined;
    for (longRows(c, &buf)) |v| try put(list, v, gpa);
    for ([_]*const tuning.Table{ &tuning.gfx1100, &tuning.gfx1201 }) |t| for (t.rows) |r| {
        if (!isPrompt(r.id, c.routed)) continue;
        for ([_]u32{ if (c.routed) r.items else r.rows, r.min_rows }) |b| {
            var two: [2]usize = undefined;
            if (sides(c, b, &two)) for (two) |v| try put(list, v, gpa);
        }
    };
    // a routed class: both sides of the whole pairs-per-item bounds and either table's, wherever they change answer
    if (c.routed) {
        var bs: [8 + 2 * tuning.gfx1201.rows.len]f32 = undefined;
        var n: usize = 0;
        for (1..9) |b| {
            bs[n] = @floatFromInt(b);
            n += 1;
        }
        for ([_]*const tuning.Table{ &tuning.gfx1100, &tuning.gfx1201 }) |tb| for (tb.rows) |r| if (r.pairs_per_item != 0) {
            bs[n] = r.pairs_per_item;
            n += 1;
        };
        for (bs[0..n]) |b| for (1..sampledMax(c)) |t| {
            if (admits(c, b, t) != admits(c, b, t + 1)) {
                try put(list, t, gpa);
                try put(list, t + 1, gpa);
            }
        };
    }
    std.mem.sort(usize, list.items, {}, std.sort.asc(usize));
}

/// Whether a pairs-per-item bound `b` takes `tokens` tokens of a routed class, as the registry compares.
fn admits(c: Class, b: f32, tokens: usize) bool {
    const pairs: f64 = @floatFromInt(tokens * c.slots);
    return pairs <= @as(f64, b) * @as(f64, @floatFromInt(selectorCount(c, tokens)));
}

/// The shape serving asks the registry about for `tokens` tokens of a routed class, from one of its timed shapes.
pub fn routedShape(c: Class, template: registry.Shape, tokens: usize) registry.Shape {
    var s = template;
    s.items = @intCast(selectorCount(c, tokens));
    s.plan_pairs = @intCast(tokens * c.slots);
    s.m = @intCast(@min(moe.tileOf(tokens, false), tokens));
    return s;
}

/// Both sides of every pair of adjacent token counts, up to `sampledMax`, whose routed choice under `reg` differs.
pub fn edges(c: Class, template: registry.Shape, reg: *const affine.Registry, env: registry.Env, list: *std.ArrayList(usize), gpa: std.mem.Allocator) !void {
    var last: ?*const affine.Entry = null;
    for (1..sampledMax(c) + 1) |t| {
        const e = reg.select(env, .mlx, .routed, .prefill, routedShape(c, template, t));
        if (t > 1 and e != last) {
            try put(list, t - 1, gpa);
            try put(list, t, gpa);
        }
        last = e;
    }
}

/// A table row of the prompt path of the plain (dense) or routed op.
pub fn isPrompt(id: []const u8, routed_op: bool) bool {
    return std.mem.startsWith(u8, id, if (routed_op) "mlx.routed.prefill." else "mlx.project.prefill.");
}

fn put(list: *std.ArrayList(usize), v: usize, gpa: std.mem.Allocator) !void {
    if (std.mem.indexOfScalar(usize, list.items, v) == null) try list.append(gpa, v);
}

test "an exact plan's choice can change at every token, and the edges hold each change" {
    const search = @import("../../tuning/search.zig");
    const mlx_entries = @import("../../launches/mlx_entries.zig");
    const flash = routed[4];
    // kp3 at 1,921 items and the least bound that takes 11,264 pairs over them
    var table_rows: [tuning.gfx1201.rows.len]tuning.Row = undefined;
    for (tuning.gfx1201.rows, 0..) |r, i| table_rows[i] = r;
    for (&table_rows) |*r| if (std.mem.eql(u8, r.id, "mlx.routed.prefill.kp3")) {
        r.items = 1921;
        r.pairs_per_item = search.leastAdmitting(11264, 1921);
    };
    const table: tuning.Table = .{ .gfx = "gfx1201", .rows = &table_rows };
    const reg = affine.Registry.init(&table, &mlx_entries.all);
    const env: registry.Env = .{ .bf16 = true, .matrix = false, .matrix_on = false, .stream_on = true, .gemm_on = true };
    const template: registry.Shape = .{ .m = 8, .n = flash.n, .k = flash.k, .bits = flash.bits, .group = flash.group, .fp16 = false };
    var list: std.ArrayList(usize) = .empty;
    defer list.deinit(std.testing.allocator);
    try edges(flash, template, &reg, env, &list, std.testing.allocator);
    for ([_]usize{ 1022, 1023, 1024, 1025 }) |n| try std.testing.expect(std.mem.indexOfScalar(usize, list.items, n) != null);
    // the choices alternate there: kp3, GEMM, kp3, GEMM
    for ([_]usize{ 1022, 1023, 1024, 1025 }, [_][]const u8{ "kp3", "gemm", "kp3", "gemm" }) |n, id| {
        try std.testing.expect(std.mem.endsWith(u8, reg.select(env, .mlx, .routed, .prefill, routedShape(flash, template, n)).?.id, id));
    }
}
