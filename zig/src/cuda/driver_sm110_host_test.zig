//! Device matching for the SM110 product without opening the CUDA driver.
const std = @import("std");
const gpuRefusal = @import("driver.zig").gpuRefusal;

test "SM110 CUDA driver: a mismatched SM names both targets, while SM110 is admitted" {
    var buf: [256]u8 = undefined;
    const expected = "this server was built for sm_110 (-Dgpu=sm_110), and CUDA device 0 is sm_121: " ++
        "build it with -Dgpu=sm_121, or use the release server";
    try std.testing.expectEqualStrings(expected, gpuRefusal(110, 121, 0, &buf).?);
    try std.testing.expect(gpuRefusal(110, 110, 0, &buf) == null);
}
