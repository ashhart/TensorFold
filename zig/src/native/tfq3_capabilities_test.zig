//! TFQ3 in the production native Metal registry: only families whose loaders read .tffpw records offer it.
const std = @import("std");
const native_metal = @import("native_metal");

test "Metal capabilities: tfq3 is offered for glm5_next and qwen4_exp, beside their MLX widths, and for no other family" {
    var offered: std.ArrayList([]const u8) = .empty;
    defer offered.deinit(std.testing.allocator);
    for (native_metal.families) |family| for (family.formats) |format| if (std.mem.eql(u8, format, "tfq3"))
        try offered.append(std.testing.allocator, family.model_type);
    try std.testing.expectEqual(@as(usize, 2), offered.items.len);
    try std.testing.expectEqualStrings("qwen4_exp", offered.items[0]);
    try std.testing.expectEqualStrings("glm5_next", offered.items[1]);
    for (native_metal.families) |family| {
        const mlx = if (std.mem.eql(u8, family.model_type, "glm5_next")) "mlx-q4g64" else if (std.mem.eql(u8, family.model_type, "qwen4_exp")) "mlx-q8g32" else continue;
        var kept = false;
        for (family.formats) |format| kept = kept or std.mem.eql(u8, format, mlx);
        try std.testing.expect(kept);
    }
}
