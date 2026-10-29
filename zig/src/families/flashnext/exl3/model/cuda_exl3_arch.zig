//! EXL3 Flash Next runs on sm_120 only; other compute capabilities are refused until their gates have been run on them.
const std = @import("std");
const cuda = @import("cuda");

/// The one compute capability (10 * major + minor) this family's EXL3 path is qualified on: Blackwell sm_120.
pub const qualified: u32 = 120;

pub const message = "Flash Next EXL3 (expert offload, TP=2, graphs) is qualified on sm_120 only; this device is sm_{d}: " ++
    "serve an NVFP4 or BF16 Flash Next checkpoint, or set TF_FLASHNEXT_EXL3_ARCH=unqualified to try it without a claim";

/// Whether a device of `capability` may serve an EXL3 pack; `override` is the TF_FLASHNEXT_EXL3_ARCH value for testers.
pub fn admits(capability: u32, override: ?[]const u8) bool {
    if (capability == qualified) return true;
    const v = override orelse return false;
    return std.mem.eql(u8, v, "unqualified");
}

/// Refuses an EXL3 pack on an unqualified device before any weight loads.
pub fn check(driver: *const cuda.Driver) !void {
    const capability = try driver.capability(0);
    const v = std.c.getenv("TF_FLASHNEXT_EXL3_ARCH");
    if (admits(capability, if (v) |p| std.mem.span(p) else null)) return;
    std.log.err(message, .{capability});
    return error.Exl3ArchUnqualified;
}

test "only sm_120 admits an EXL3 pack unless a tester overrides the gate" {
    try std.testing.expect(admits(120, null));
    for ([_]u32{ 80, 86, 89, 90, 100, 121 }) |c| try std.testing.expect(!admits(c, null));
    try std.testing.expect(!admits(121, "1"));
    try std.testing.expect(admits(121, "unqualified"));
}
