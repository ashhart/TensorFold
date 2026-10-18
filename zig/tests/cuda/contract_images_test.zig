//! The live resolver with distinct inert image bytes and no CUDA calls.
const std = @import("std");
const live = @import("contract_tests.zig");

fn resolve(name: []const u8) ![]const u8 {
    return live.image(name);
}

test "contract image: GLM top1 selects its own image" {
    try std.testing.expectEqualStrings("glm_glue", try resolve("glm_glue"));
}

test "contract image: only explicit controls select the controls image" {
    try std.testing.expectEqualStrings("contract_controls", try resolve("contract_controls"));
    try std.testing.expectEqualStrings("h3", try resolve("h3"));
    try std.testing.expectEqualStrings("sample", try resolve("sample"));
    for ([_][]const u8{ "unknown_image", "glm_glu", "", "contract_control", "GLM_GLUE" }) |name|
        try std.testing.expectError(error.UnknownContractImage, resolve(name));
}
