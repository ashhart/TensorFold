//! GLM window projections: independent linear oracles, predecessor equality and dispatched source controls.
const std = @import("std");
const c = @import("contract").spec;
pub const bits = [_]u8{ 2, 3, 4, 5, 6, 8 };
pub const Kind = enum { narrow, wide, down, glu };
pub const Mode = enum { oracle, replacement, control, up_oracle };
pub const Entry = struct {
    kind: Kind,
    bits: u8,
    mode: Mode,
    pub fn nonlinear(v: Entry) bool {
        return v.kind == .glu and (v.mode == .replacement or v.mode == .control);
    }
    pub fn symbol(v: Entry) [:0]const u8 {
        return switch (v.kind) {
            .down => "glm_gather_down",
            .glu => "glm_gather_glu",
            .narrow, .wide => blk: {
                inline for (bits) |b| if (v.bits == b) break :blk if (v.kind == .narrow)
                    std.fmt.comptimePrint("glm_rows_q{d}", .{b})
                else
                    std.fmt.comptimePrint("glm_rows_wide_q{d}", .{b});
                unreachable;
            },
        };
    }
};
pub const entries = blk: {
    var out: [78]Entry = undefined;
    var n: usize = 0;
    for (std.enums.values(Kind)) |kind| for (bits) |b| {
        for ([_]Mode{ .oracle, .replacement, .control }) |mode| {
            out[n] = .{ .kind = kind, .bits = b, .mode = mode };
            n += 1;
        }
        if (kind == .glu) {
            out[n] = .{ .kind = kind, .bits = b, .mode = .up_oracle };
            n += 1;
        }
    };
    break :blk out;
};
pub fn name(a: std.mem.Allocator, v: Entry) ![]const u8 {
    return std.fmt.allocPrint(a, "metal glm window {s} q{d} {s} {s}", .{ @tagName(v.kind), v.bits, @tagName(v.mode), v.symbol() });
}
pub fn visit(visitor: anytype, only: ?[]const u8) !void {
    for (entries) |v| {
        const text = try name(visitor.a, v);
        defer visitor.a.free(text);
        if (only) |s| if (std.mem.indexOf(u8, text, s) == null) continue;
        try visitor.add(v, text);
    }
}
pub fn depth(s: c.Shape) usize {
    return 16 + 3 * ((s.k + 511) / 512) + 8;
}
pub fn served(kind: Kind) [23]c.Shape {
    var out: [23]c.Shape = undefined;
    const gather = kind == .down or kind == .glu;
    out[0] = .{ .rows = 4, .n = 12, .k = 576 };
    out[1] = switch (kind) {
        .narrow, .wide => .{ .rows = 4, .n = 4096, .k = 16384 },
        .down => .{ .rows = 8, .n = 4096, .k = 2048 },
        .glu => .{ .rows = 8, .n = 2048, .k = 4096 },
    };
    out[2] = .{ .rows = 8, .n = 4096, .k = 4096 };
    for (out[3..20], 1..) |*s, m| s.* = .{ .rows = m, .n = 12, .k = 576 };
    out[20] = .{ .rows = 3, .n = 12, .k = if (kind == .glu) 576 else 544 };
    out[21] = .{ .rows = if (gather) 128 else 17, .n = if (gather) 12 else 14, .k = 576 };
    out[22] = .{ .rows = if (gather) 129 else 16, .n = if (gather) 12 else 10, .k = 512 };
    return out;
}
pub fn spec(v: Entry, text: []const u8, shapes: []const c.Shape) c.Spec {
    const gather = v.kind == .down or v.kind == .glu;
    const final_glu = v.kind == .glu and v.mode == .replacement;
    return .{
        .name = text,
        .accum = .f32,
        .depth = depth,
        // Two six-step BF16 scalar paths can straddle a rounding midpoint after a one-ULP FP32 exp change.
        .ops = if (final_glu) 12 else 0,
        .mids = if (final_glu) 12 else 0,
        .out = .bf16,
        .replaces = if (v.mode == .replacement or v.mode == .control) "the frozen pre-1130 GLM kernel" else null,
        .nan = if (v.nonlinear()) .{ .stated = "unchanged clipped BF16 SwiGLU has the predecessor's nonfinite classes" } else .propagate,
        .tiles = .{ .{ .size = if (v.kind == .narrow) 3 else if (v.kind == .wide) 8 else 4 }, .{ .size = if (gather) 8 else 4, .step = if (gather) 4 else 2, .min = if (gather) 4 else 2 }, .{ .size = 512, .step = if (v.kind == .glu) 64 else 32, .min = if (v.kind == .glu) 64 else 32 } },
        .served = shapes,
        .sums = if (v.nonlinear()) .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "clipped BF16 SwiGLU masks dot errors; both preactivation taps separately prove FP32 accumulation" } else .{},
    };
}
const rows_from = "x.x[r] = x0 + size_t(m + r) * d.ldx;";
const rows_to = "x.x[r] = x0 + size_t(m) * d.ldx;";
const gather_from = "const size_t at = size_t(expert) * d.n + col;";
const gather_to = "const size_t at = col;";
pub const glu_store = "glm_rows_glu(acc[b][j], up[b][j], d.lim)";
pub fn source(a: std.mem.Allocator, original: []const u8, v: Entry) ![]u8 {
    const from = if (v.mode == .control) (if (v.kind == .narrow or v.kind == .wide) rows_from else gather_from) else glu_store;
    const to = if (v.mode == .control) (if (v.kind == .narrow or v.kind == .wide) rows_to else gather_to) else if (v.mode == .up_oracle) "bfloat(up[b][j])" else "bfloat(acc[b][j])";
    if (v.mode != .control and (v.kind != .glu or v.mode == .replacement)) return a.dupe(u8, original);
    if (std.mem.count(u8, original, from) != 1) return error.SourceWitnessMissing;
    return std.mem.replaceOwned(u8, a, original, from, to);
}
