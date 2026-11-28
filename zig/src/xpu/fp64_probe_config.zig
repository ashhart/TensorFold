//! FP64 probe row selectors; probe tooling may be split into a separate PR if upstream prefers.
const std = @import("std");

fn parseList(v: ?[*:0]const u8, default: []const usize, out: *[64]bool) !void {
    out.* = @splat(false);
    if (v) |p| {
        const text = std.mem.span(p);
        if (text.len == 0) return error.BadFp64Selector;
        var it = std.mem.splitScalar(u8, text, ',');
        while (it.next()) |word| {
            const trimmed = std.mem.trim(u8, word, " \t\n");
            if (trimmed.len == 0) return error.BadFp64Selector;
            const n = std.fmt.parseInt(usize, trimmed, 10) catch return error.BadFp64Selector;
            if (n >= 64) return error.BadFp64Selector;
            out[n] = true;
        }
    } else for (default) |n| out[n] = true;
}

pub fn gemmLayer(layer: usize) !bool {
    var on: [64]bool = undefined;
    try parseList(std.c.getenv("XPU_FP64_LAYERS"), &.{ 0, 1, 31, 63 }, &on);
    return layer < 64 and on[layer];
}

pub fn gdnLayer(layer: usize) !bool {
    var on: [64]bool = undefined;
    try parseList(std.c.getenv("XPU_FP64_GDN_LAYERS"), &.{ 0, 1, 30, 62 }, &on);
    return layer < 64 and on[layer];
}

pub fn head() bool {
    const v = std.c.getenv("XPU_FP64_HEAD") orelse return true;
    const s = std.mem.span(v);
    return !(std.mem.eql(u8, s, "0") or std.mem.eql(u8, s, "false") or std.mem.eql(u8, s, "no"));
}

pub fn threads(fallback: usize) !usize {
    const v = std.c.getenv("XPU_FP64_THREADS") orelse return fallback;
    const n = std.fmt.parseInt(usize, std.mem.span(v), 10) catch return error.BadFp64Selector;
    if (n == 0) return error.BadFp64Selector;
    return @min(n, 64);
}
