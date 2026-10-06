const std = @import("std");

const page = @embedFile("dashboard.html");

test "dashboard page polls the public stats snapshot without private details" {
    try std.testing.expect(std.mem.indexOf(u8, page, "/stats?poll=1") != null);
    try std.testing.expect(std.mem.indexOf(u8, page, "generation_tokens_running") != null);
    try std.testing.expect(std.mem.indexOf(u8, page, "decode_rounds_total") != null);
    try std.testing.expect(std.mem.indexOf(u8, page, "memory.peak!=null") != null);
}
