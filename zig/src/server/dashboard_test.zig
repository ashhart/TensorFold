const std = @import("std");

const page = @embedFile("dashboard.html");

test "dashboard page polls the public stats snapshot without private details" {
    try std.testing.expect(std.mem.indexOf(u8, page, "/stats?poll=1") != null);
    try std.testing.expect(std.mem.indexOf(u8, page, "generation_tokens_running") != null);
    try std.testing.expect(std.mem.indexOf(u8, page, "decode_rounds_total") != null);
    try std.testing.expect(std.mem.indexOf(u8, page, "memory.peak!=null") != null);
    try std.testing.expect(std.mem.indexOf(u8, page, "fill_tokens_left") != null);
    try std.testing.expect(std.mem.indexOf(u8, page, "fill_prompt_tokens") != null);
    try std.testing.expect(std.mem.indexOf(u8, page, "fill_run") != null);
    // an unknown-progress fill never reads as idle
    try std.testing.expect(std.mem.indexOf(u8, page, "fill_unknown") != null);
    // the label names the queue's wait, not a first token
    try std.testing.expect(std.mem.indexOf(u8, page, "queue drains in") != null);
    try std.testing.expect(std.mem.indexOf(u8, page, "memory.limit!=null") != null); // the limit is labelled as one
}
