//! Main's grouped EXL3 loop, kept in contract_controls as the baseline tf_exl3_grouped_* must equal (nvcc 13.0).
const std = @import("std");
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "exl3_grouped tf_ctl_exl3_grouped_old_0", .symbol = "tf_ctl_exl3_grouped_old_0", .image = "contract_controls", .threads = 128 }, .usage = .{ .{ .regs = 128, .static = 32832, .bound = 0 }, .{ .regs = 128, .static = 32832, .bound = 0 }, .{ .regs = 128, .static = 32832, .bound = 0 }, .{ .regs = 127, .static = 32832, .bound = 0 }, .{ .regs = 127, .static = 32832, .bound = 0 } }, .later_pending = true },
    .{ .launch = .{ .name = "exl3_grouped tf_ctl_exl3_grouped_old_1", .symbol = "tf_ctl_exl3_grouped_old_1", .image = "contract_controls", .threads = 128 }, .usage = .{ .{ .regs = 128, .static = 32832, .bound = 0 }, .{ .regs = 127, .static = 32832, .bound = 0 }, .{ .regs = 127, .static = 32832, .bound = 0 }, .{ .regs = 128, .static = 32832, .bound = 0 }, .{ .regs = 128, .static = 32832, .bound = 0 } }, .later_pending = true },
    .{ .launch = .{ .name = "exl3_grouped tf_ctl_exl3_grouped_old_2", .symbol = "tf_ctl_exl3_grouped_old_2", .image = "contract_controls", .threads = 128 }, .usage = .{ .{ .regs = 128, .static = 32832, .bound = 0 }, .{ .regs = 128, .static = 32832, .bound = 0 }, .{ .regs = 128, .static = 32832, .bound = 0 }, .{ .regs = 128, .static = 32832, .bound = 0 }, .{ .regs = 128, .static = 32832, .bound = 0 } }, .later_pending = true },
};

fn regs(name: []const u8, release: @import("limits.zig").Nvcc) ![5]u32 {
    for (registry.all_cuda) |r| if (std.mem.eql(u8, r.launch.name, name)) {
        var out: [5]u32 = undefined;
        for (&out, registry.usageFor(r, release)) |*o, u| o.* = u.regs;
        return out;
    };
    return error.MissingRecord;
}

test "exl3 grouped records: sm_80 to sm_89 keep main's under every nvcc, sm_120 and sm_121 the load-ahead's" {
    const names = [3][]const u8{ "exl3_experts tf_exl3_grouped_0", "exl3_experts tf_exl3_grouped_1", "exl3_experts tf_exl3_grouped_2" };
    // main's 13.0 records; the load-ahead's 13.0 and 13.3 on sm_12x; 13.2: main's 128 below, pending on sm_12x
    const main_130 = [3][3]u32{ .{ 128, 128, 128 }, .{ 128, 127, 127 }, .{ 128, 128, 128 } };
    const ahead = [2][3]u32{ .{ 149, 157, 145 }, .{ 158, 158, 148 } };
    for (names, main_130, 0..) |name, old, k| {
        const v130 = try regs(name, .{ .major = 13, .minor = 0 });
        try std.testing.expectEqualSlices(u32, &old, v130[0..3]);
        try std.testing.expectEqualSlices(u32, &.{ ahead[0][k], ahead[0][k] }, v130[3..]);
        try std.testing.expectEqualSlices(u32, &.{ 128, 128, 128, 0, 0 }, &try regs(name, .{ .major = 13, .minor = 2 }));
        try std.testing.expectEqualSlices(u32, &.{ 128, 128, 128, ahead[1][k], ahead[1][k] }, &try regs(name, .{ .major = 13, .minor = 3 }));
    }
}
