//! --prefill-from and hand-off rows split from environment_test.zig: valid forms, bounds and refusals.
const std = @import("std");
const e = @import("environment.zig");
const t = std.testing;

pub const Case = struct { name: []const u8, raw: []const u8, n: u64 };
/// Joined to environment_test.zig's valid forms, which every registered name needs one of.
pub const valid = [_]Case{
    .{ .name = "TF_PREFILL_MIN", .raw = "4096", .n = 4096 },
    .{ .name = "TF_PREFILL_CHUNK", .raw = "512", .n = 512 },
    .{ .name = "TF_PREFILL_KEEP", .raw = "1", .n = 1 },
    .{ .name = "TF_PREFILL_AHEAD", .raw = "1", .n = 1 },
    .{ .name = "TF_PREFILL_TIMEOUT_S", .raw = "86400", .n = 86400 },
    .{ .name = "TF_PREFILL_API_KEY_FILE", .raw = "/etc/tensorfold/gb10.key", .n = 0 },
    .{ .name = "TF_PRODUCE_HELD_GIB", .raw = "32", .n = 32 },
    .{ .name = "TF_DSV41_HANDOFF_DUMP", .raw = "/tmp/handoff exports", .n = 0 },
};

test "environment: TF_PREFILL_* take their bounds and refuse values outside them" {
    try t.expectEqual(@as(?u64, 0), try e.parse("TF_PREFILL_MIN", "0"));
    try t.expectEqual(@as(?u64, 16), try e.parse("TF_PREFILL_CHUNK", "16"));
    for ([_][2][]const u8{
        .{ "TF_PREFILL_CHUNK", "0" },     .{ "TF_PREFILL_CHUNK", "520" },
        .{ "TF_PREFILL_KEEP", "true" },   .{ "TF_PREFILL_AHEAD", "0" },
        .{ "TF_PREFILL_TIMEOUT_S", "0" }, .{ "TF_PREFILL_TIMEOUT_S", "86401" },
    }) |c| try t.expectError(error.InvalidEnvironment, e.parse(c[0], c[1]));
}
