//! The CUDA engine's checkpoint format gate: its kernels run EXL3 trellises, so a release FP8 pack is refused early.

const std = @import("std");
const Config = @import("config.zig").Config;

/// Why a release FP8-block checkpoint cannot run on the CUDA engine, and what to feed the engine instead.
pub const release_fp8_problem = "DeepSeek V4.1 CUDA: this checkpoint stores the release's own FP8 blocks and MXFP4 experts; the CUDA engine executes EXL3 trellises, so pack the checkpoint to EXL3 (or serve it on Metal) first";

/// The refusal's text for `c`, null when the checkpoint's format is one this engine runs.
pub fn problem(c: *const Config) ?[]const u8 {
    return if (c.format == .release_fp8) release_fp8_problem else null;
}

test "the CUDA gate refuses a release checkpoint by name and admits the EXL3 one" {
    const a = std.testing.allocator;
    const exl3 = try Config.parse(a, @embedFile("fixtures/config.json"));
    try std.testing.expect(problem(&exl3) == null);
    const release = try Config.parse(a, @embedFile("fixtures/flash1017-config.json"));
    const text = problem(&release).?;
    try std.testing.expect(std.mem.indexOf(u8, text, "FP8") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "EXL3") != null);
}
