//! The native branch has an optional owner and a separate observer active only while a trainer records a pass.
const std = @import("std");
pub const Run = struct { first: usize, rows: usize, position: usize = 0, untaught: bool = false };
pub const Layout = enum { rows, heads, interleaved };
pub const Node = enum { input, normed, q_raw, k_raw, v_raw, query, key, value, attention, attn_output, mid, mlp_input, gate, up, act, mlp_base, mlp_output, residual, next, head_raw, head };
pub fn View(comptime Buffer: type) type {
    return struct {
        buffer: Buffer,
        offset: usize = 0,
        len: usize,
        rows: usize,
        width: usize,
        stride: usize = 0,
        layout: Layout = .rows,
        heads: usize = 1,
        dim: usize = 0,
    };
}
pub fn Hook(comptime Buffer: type, comptime Pass: type) type {
    return struct {
        ctx: *anyopaque,
        active: *const fn (*anyopaque) bool,
        begin: *const fn (*anyopaque, []const Run, usize) anyerror!void,
        dense: *const fn (*anyopaque, Pass, usize, View(Buffer), View(Buffer)) anyerror!void,
        observe: ?*const fn (*anyopaque, Pass, ?usize, Node, View(Buffer)) anyerror!void = null,
        pub fn note(h: @This(), pass: Pass, layer: ?usize, node: Node, v: View(Buffer)) !void {
            if (h.observe) |f| try f(h.ctx, pass, layer, node, v);
        }
    };
}
pub const Range = struct { first: usize, rows: usize };
pub const Selection = struct {
    ranges: [128]Range = undefined,
    count: usize = 0,
    pub fn of(runs: []const Run, rows: usize) !Selection {
        var out: Selection = .{};
        var end: usize = 0;
        for (runs) |r| {
            if (r.first != end or r.rows == 0 or r.rows > rows -| end) return error.GemmaLearningRows;
            end = try std.math.add(usize, end, r.rows);
            if (r.untaught) continue;
            if (out.count > 0 and out.ranges[out.count - 1].first + out.ranges[out.count - 1].rows == r.first) out.ranges[out.count - 1].rows += r.rows else {
                if (out.count == out.ranges.len) return error.GemmaLearningRows;
                out.ranges[out.count] = .{ .first = r.first, .rows = r.rows };
                out.count += 1;
            }
        }
        if (end != rows) return error.GemmaLearningRows;
        return out;
    }
};
