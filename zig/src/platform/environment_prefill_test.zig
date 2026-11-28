//! The cold-prefill diagnostic is exactly one new strict flag, never enabled by its absence.
const std = @import("std");
const env = @import("environment.zig");
test "CUDA prefill ledger: inventory adds exactly one default-off flag" {
    var count: usize = 0;
    for (env.rules) |rule| if (std.mem.eql(u8, rule.name, "TF_CUDA_PREFILL_TRACE")) {
        count += 1;
    };
    try std.testing.expectEqual(@as(usize, 1), count);
    try std.testing.expect(!try env.value(bool, "TF_CUDA_PREFILL_TRACE", null, false));
    try std.testing.expect(!try env.value(bool, "TF_CUDA_PREFILL_TRACE", "0", false));
    try std.testing.expect(try env.value(bool, "TF_CUDA_PREFILL_TRACE", "1", false));
    try std.testing.expectError(error.InvalidEnvironment, env.value(bool, "TF_CUDA_PREFILL_TRACE", "2", false));
}
