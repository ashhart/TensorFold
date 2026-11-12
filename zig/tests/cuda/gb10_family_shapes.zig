//! Additional member geometries for the shared GB10 decode and verify replacement contract.
const c = @import("core").contract.spec;
pub const Case = struct { n: usize, k: usize, widths: []const usize, padded_n: usize = 0 };
pub const cases = [_]Case{
    .{ .n = 4096, .k = 5120, .widths = &.{ 4096, 1024, 1024 } },
    .{ .n = 1024, .k = 5120, .widths = &.{ 1024, 1024 } },
    .{ .n = 17408, .k = 5120, .widths = &.{ 17408, 17408 } },
    .{ .n = 98304, .k = 5120, .widths = &.{ 98304, 320 }, .padded_n = 248320 },
    .{ .n = 8192, .k = 5376, .widths = &.{ 8192, 4096, 4096 } },
    .{ .n = 16384, .k = 5376, .widths = &.{ 16384, 2048 } },
    .{ .n = 21504, .k = 5376, .widths = &.{ 21504, 21504 } },
    .{ .n = 4096, .k = 3840, .widths = &.{ 4096, 2048, 2048 } },
    .{ .n = 8192, .k = 3840, .widths = &.{ 8192, 512 } },
    .{ .n = 15360, .k = 3840, .widths = &.{ 15360, 15360 } },
    .{ .n = 2048, .k = 2560, .widths = &.{ 2048, 512, 512 } },
    .{ .n = 4096, .k = 2560, .widths = &.{ 4096, 1024, 1024 } },
    .{ .n = 10240, .k = 2560, .widths = &.{ 10240, 10240 } },
    .{ .n = 8192, .k = 1024, .widths = &.{ 8192, 8192 } },
    .{ .n = 8192, .k = 5376, .widths = &.{ 8192, 1024, 1024 } },
    .{ .n = 10752, .k = 5376, .widths = &.{ 10752, 10752 } },
    .{ .n = 1024, .k = 5376, .widths = &.{ 1024, 1024 } },
};
pub const shapes = blk: {
    var out: [cases.len]c.Shape = undefined;
    for (cases, 0..) |v, i| out[i] = .{ .rows = 128, .n = v.n, .k = v.k };
    break :blk out;
};
