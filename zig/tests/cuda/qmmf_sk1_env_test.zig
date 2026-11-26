//! Registration fails before a Qwen lane or model is opened.
const std = @import("std");
const e = @import("sys").environment;

test "QMMF SK1: registered flag is present before model open" {
    const name = "TF_QWEN_NVFP4_SK1";
    var found = false;
    for (e.rules) |rule| if (std.mem.eql(u8, name, rule.name)) {
        found = true;
    };
    try std.testing.expect(found);
}

test "QMMF SK1: strict flag defaults off" {
    const name = "TF_QWEN_NVFP4_SK1";
    try std.testing.expectEqual(false, try e.value(bool, name, null, false));
    try std.testing.expectEqual(false, try e.value(bool, name, "0", false));
    try std.testing.expectEqual(true, try e.value(bool, name, "1", false));
    for ([_][]const u8{ "2", "true", "on", "01", "-1" }) |raw| try std.testing.expectError(error.InvalidEnvironment, e.parse(name, raw));
}
