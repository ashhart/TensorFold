//! The prompt cells the tuner times: dense rows and served routed tokens.

const std = @import("std");
const tuning = @import("../../tuning/tuning.zig");
const moe = @import("../../moe.zig");
const cases = @import("cases.zig");

/// A product geometry on the prompt path: dense, or the served routed gate and up or down.
pub const Class = struct {
    name: []const u8,
    n: usize,
    k: usize,
    bits: u8,
    group: u16,
    routed: bool = false,
    down: bool = false,

    /// The case of `count` rows (dense) or tokens (routed) of this class.
    pub fn case(c: Class, count: usize) cases.Case {
        if (!c.routed) return .{ .name = c.name, .path = .prefill, .rows = count, .n = c.n, .k = c.k, .bits = c.bits, .group = c.group };
        return .{ .name = c.name, .path = .prefill, .rows = count, .n = c.n, .k = c.k, .bits = c.bits, .group = c.group, .experts = experts, .shared = true, .served = true, .down = c.down };
    }
};

/// Qwen3.6-35B-A3B's routed experts: 256 and the shared one, 8 picks and the shared slot a token.
const experts = 256;
const slots = 9;

/// The served routed classes: the experts' gate and up, and their down.
pub const routed = [_]Class{
    .{ .name = "35b served gate_up", .n = 1024, .k = 2048, .bits = 4, .group = 64, .routed = true },
    .{ .name = "35b served down", .n = 2048, .k = 512, .bits = 4, .group = 64, .routed = true, .down = true },
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

/// The item count serving passes the registry for `tokens` tokens (`moe.run`, `expertParts`).
pub fn selectorCount(tokens: usize) usize {
    if (tokens == 1) return slots;
    return moe.capacity(moe.bucket(tokens) * slots, experts + 1);
}

/// The cells on both sides of a bound `b` (rows b and b + 1, or tokens at or below and just above it), both sampled.
pub fn sides(c: Class, b: u32, out: *[2]usize) bool {
    if (b == 0) return false;
    if (!c.routed) {
        out.* = .{ b, b + 1 };
    } else {
        var below: usize = 0;
        var t: usize = 1;
        while (t <= sampledMax(c) and selectorCount(t) <= b) : (t += 1) below = t;
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
