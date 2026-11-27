//! --prefill-from rows split from environment_test.zig: the TF_PREFILL_* bounds and their refusals.
const std = @import("std");
const e = @import("environment.zig");
const t = std.testing;

test "environment: TF_PREFILL_* take their bounds and refuse values outside them" {
    try t.expectEqual(@as(?u64, 0), try e.parse("TF_PREFILL_MIN", "0"));
    try t.expectEqual(@as(?u64, 16), try e.parse("TF_PREFILL_CHUNK", "16"));
    for ([_][2][]const u8{
        .{ "TF_PREFILL_CHUNK", "0" },     .{ "TF_PREFILL_CHUNK", "520" },
        .{ "TF_PREFILL_KEEP", "true" },   .{ "TF_PREFILL_AHEAD", "0" },
        .{ "TF_PREFILL_TIMEOUT_S", "0" }, .{ "TF_PREFILL_TIMEOUT_S", "86401" },
    }) |c| try t.expectError(error.InvalidEnvironment, e.parse(c[0], c[1]));
}
