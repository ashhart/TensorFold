//! CPU scripted timing and actual Metal measurement wiring without importing a device module.
const std = @import("std");

test {
    _ = @import("src/core/lanes/cost_rule.zig");
    _ = @import("src/core/lanes/cost_cache.zig");
}

test "Nemotron initial and drift passes share interleaved head sampling; the 27B remains headless" {
    const source = @embedFile("src/families/nemotron/timing.zig");
    const start = std.mem.indexOf(u8, source, "pub fn measure(").?;
    const end = std.mem.indexOfPos(u8, source, start, "/// Windows past").?;
    const body = source[start..end];
    try std.testing.expect(std.mem.indexOf(u8, body, "measure(windows, heads, &ms, head_ms, false)") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "measure(windows, heads, &ms, head_ms, true)") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "if (b.head != null) &head[0] else null") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "drifted(&head, &r.head)") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "fastest(heads") == null);
    const qwen = @embedFile("src/native/qwen27_lanes.zig");
    try std.testing.expect(std.mem.indexOf(u8, qwen, "cost_rule.fastest(timer, i)") != null);
    try std.testing.expect(std.mem.indexOf(u8, qwen, "cost_rule.measure(") == null);
}
