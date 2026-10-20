//! GLM profiler lifetime operation used by unwind and close.
const std = @import("std");
const Engine = @import("cuda_engine.zig").Engine;
pub fn freeProfs(e: *Engine) void {
    for (&e.profs) |*p| {
        if (p.*) |v| v.deinit();
        p.* = null;
    }
    for (&e.prompt_profs) |*p| {
        if (p.*) |v| v.deinit();
        p.* = null;
    }
}

pub fn heatRoot() []const u8 {
    const p = std.c.getenv("TF_HEAT_ROOT") orelse return @import("heat").default_root;
    const s = std.mem.span(p);
    return if (s.len == 0) @import("heat").default_root else s;
}

pub fn verifyLabel(rows: usize) []const u8 {
    const l = [_][]const u8{ "verify 1 row", "verify 2 rows", "verify 3 rows", "verify 4 rows", "verify 5 rows", "verify 6 rows", "verify 7 rows", "verify 8 rows" };
    return l[@min(rows, l.len) - 1];
}
