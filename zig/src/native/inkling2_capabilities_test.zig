//! Inkling's checkpoint-format offer in the production native Metal registry.
const std = @import("std");
const native_metal = @import("native_metal");

test "Metal capabilities: inkling2_loop serves the checkpoint's ModelOpt NVFP4 weights" {
    var found = false;
    for (native_metal.families) |family| if (std.mem.eql(u8, family.model_type, "inkling2_loop")) {
        found = true;
        try std.testing.expectEqual(@as(usize, 1), family.formats.len);
        try std.testing.expectEqualStrings("modelopt-nvfp4", family.formats[0]);
    };
    try std.testing.expect(found);
}
