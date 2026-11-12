//! Shared KDA's widths in Zig against kda.cuh's, so a kernel change cannot leave kda.geometry's refusal behind.
const std = @import("std");
const kda = @import("cuda").kda;

test "shared KDA: kda.zig's dk, dv and taps are kda.cuh's DK, DV and TAPS" {
    const line = std.fmt.comptimePrint("constexpr int DK = {d}, DV = {d}, TAPS = {d};", .{ kda.dk, kda.dv, kda.taps });
    try std.testing.expect(std.mem.indexOf(u8, @embedFile("kernels/cuda/common/kda.cuh"), line) != null);
}
