//! Prompt kernels with a new shared-id guard ABI: historical compiler rows do not certify this source.
const std = @import("std");
const registry = @import("registry.zig");
const limits = @import("limits.zig");
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "exl3_experts tf_exl3_prompt_gu", .symbol = "tf_exl3_prompt_gu", .image = "exl3_experts", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "exl3_experts tf_exl3_prompt_down", .symbol = "tf_exl3_prompt_down", .image = "exl3_experts", .threads = 256 }, .usage = registry.none },
};

pub fn owns(name: []const u8) bool {
    for (records) |r| if (std.mem.eql(u8, r.launch.name, name)) return true;
    return false;
}

test "EXL3 prompt: changed source cannot inherit historical compiler usage" {
    for (records) |r| for ([_]?limits.Nvcc{ null, .{ .major = 13, .minor = 0 }, .{ .major = 13, .minor = 1 }, .{ .major = 13, .minor = 2 }, .{ .major = 13, .minor = 3 } }) |version| {
        try std.testing.expectEqualDeep(registry.none, registry.usageFor(r, version));
    };
    try std.testing.expectEqual(@as(usize, 10), try registry.checkCuda(std.testing.allocator, &records, true));
}
