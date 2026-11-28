//! Prompt dispatch policy and row padding for Qwen27 quantized kernels.
const std = @import("std");
const lane = @import("core").lane_projection;

/// Whether prompt chunks take one dispatch a projection: Ultras with tensor units only.
pub fn longPrompt(name: []const u8, tensor_units: bool) bool {
    return tensor_units and std.mem.endsWith(u8, name, " Ultra");
}

test "long prompt dispatches only on an Ultra with tensor units" {
    try std.testing.expect(longPrompt("Apple M5 Ultra", true));
    try std.testing.expect(!longPrompt("Apple M5 Max", true));
    try std.testing.expect(!longPrompt("Apple M5 Pro", true));
    try std.testing.expect(!longPrompt("Apple M3 Ultra", false));
}

/// Whether a reg projection's rows take its long key: prompt rows past lane.reg_rows on a longPrompt chip.
pub fn longKey(long_prompt: bool, prompt: bool, rows: u32) bool {
    return long_prompt and prompt and rows > lane.reg_rows;
}

test "only prompt chunks past 128 rows take the long key" {
    try std.testing.expect(longKey(true, true, 129));
    try std.testing.expect(longKey(true, true, 2048));
    try std.testing.expect(!longKey(true, true, 128));
    try std.testing.expect(!longKey(true, false, 2048));
    try std.testing.expect(!longKey(false, true, 2048));
}

pub fn padded(rows: u32) u32 {
    return 16 * ((rows + 15) / 16);
}
