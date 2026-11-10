//! EXL3 Flash Next runs on sm_120 only; other compute capabilities are refused until their gates have been run on them.
const std = @import("std");
const cuda = @import("cuda");

/// The one compute capability (10 * major + minor) this family's EXL3 path is qualified on: Blackwell sm_120.
pub const qualified: u32 = 120;

pub const message = "Flash Next EXL3 (expert offload, one rank) is qualified on sm_120 only; this device is sm_{d}: " ++
    "serve an NVFP4 or BF16 Flash Next checkpoint, or set TF_FLASHNEXT_EXL3_ARCH=unqualified to try it without a claim";

/// Whether a device of `capability` may serve an EXL3 pack; `override` is TF_FLASHNEXT_EXL3_ARCH=unqualified.
pub fn admits(capability: u32, override: bool) bool {
    return capability == qualified or override;
}

/// Refuses an EXL3 pack on an unqualified device before any weight loads.
pub fn check(driver: *const cuda.Driver) !void {
    const capability = try driver.capability(0);
    // the rule's only word is `unqualified`, so a value that parses at all is the override
    if (admits(capability, (try @import("sys").environment.read(?u8, "TF_FLASHNEXT_EXL3_ARCH", null)) != null)) return;
    std.log.err(message, .{capability});
    return error.Exl3ArchUnqualified;
}

test "only sm_120 admits an EXL3 pack unless a tester overrides the gate" {
    try std.testing.expect(admits(120, false));
    for ([_]u32{ 80, 86, 89, 90, 100, 121 }) |c| try std.testing.expect(!admits(c, false));
    try std.testing.expect(admits(121, true));
}
