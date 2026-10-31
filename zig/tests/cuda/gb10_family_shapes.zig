//! Additional member geometries for the shared GB10 decode and verify replacement contract.
const c = @import("core").contract.spec;
pub const Case = struct { n: usize, k: usize, widths: []const usize, padded_n: usize = 0 };
pub const cases = [_]Case{
    .{ .n = 4096, .k = 5120, .widths = &.{ 4096, 1024, 1024 } },
    .{ .n = 1024, .k = 5120, .widths = &.{ 1024, 1024 } },
    .{ .n = 17408, .k = 5120, .widths = &.{ 17408, 17408 } },
    .{ .n = 98304, .k = 5120, .widths = &.{ 98304, 320 }, .padded_n = 248320 },
};
pub const shapes = blk: {
    var out: [cases.len]c.Shape = undefined;
    for (cases, 0..) |v, i| out[i] = .{ .rows = 128, .n = v.n, .k = v.k };
    break :blk out;
};
