//! Inkling's checkpoint-format offer in the production native Metal registry.
const std = @import("std");
const native_metal = @import("native_metal");

test "Metal capabilities: inkling2_loop serves ModelOpt NVFP4 and MLX 4-bit affine g64 weights" {
    var found = false;
    for (native_metal.families) |family| if (std.mem.eql(u8, family.model_type, "inkling2_loop")) {
        found = true;
        try std.testing.expectEqual(@as(usize, 2), family.formats.len);
        try std.testing.expectEqualStrings("modelopt-nvfp4", family.formats[0]);
        try std.testing.expectEqualStrings("mlx-q4g64", family.formats[1]);
    };
    try std.testing.expect(found);
}
