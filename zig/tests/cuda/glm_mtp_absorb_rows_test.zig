//! Host arithmetic for the CUDA MTP trace's consumed-row selection and planted byte control.
const std = @import("std");
const trace = @import("glm_mtp_absorb_tests.zig");

test "MTP trace: consumed offset follows full-row and absorbed routes" {
    try std.testing.expectEqual(0, trace.consumedOffset(1, 4096, false));
    try std.testing.expectEqual(0, trace.consumedOffset(1, 4096, true));
    try std.testing.expectEqual(4096 * 4, trace.consumedOffset(2, 4096, false));
    try std.testing.expectEqual(0, trace.consumedOffset(2, 4096, true));
    try std.testing.expectEqual(31 * 4096 * 4, trace.consumedOffset(32, 4096, false));
    try std.testing.expectEqual(0, trace.consumedOffset(32, 4096, true));
}

test "MTP trace: a planted consumed-row byte differs between full and absorbed routes" {
    const full = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    const absorbed = [_]u8{ 9, 10, 11, 12, 13, 14, 15, 16 };
    const full_offset = trace.consumedOffset(2, 2, false);
    const absorbed_offset = trace.consumedOffset(2, 2, true);
    try std.testing.expectEqualSlices(u8, &absorbed, full[full_offset..]);
    var changed = full;
    changed[full_offset] ^= 1;
    const first = std.mem.indexOfDiff(u8, changed[full_offset..], absorbed[absorbed_offset..]) orelse return error.ExpectedPlantedDifference;
    try std.testing.expectEqual(0, first);
}
