//! The prompt cells the tuner times: dense rows and served routed tokens.

const std = @import("std");
const tuning = @import("../../tuning/tuning.zig");
const moe = @import("../../moe.zig");
const cases = @import("cases.zig");

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
    std.mem.sort(usize, list.items, {}, std.sort.asc(usize));
}

/// A table row of the prompt path of the plain (dense) or routed op.
pub fn isPrompt(id: []const u8, routed_op: bool) bool {
    return std.mem.startsWith(u8, id, if (routed_op) "mlx.routed.prefill." else "mlx.project.prefill.");
}

fn put(list: *std.ArrayList(usize), v: usize, gpa: std.mem.Allocator) !void {
    if (std.mem.indexOfScalar(usize, list.items, v) == null) try list.append(gpa, v);
}
