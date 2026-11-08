//! The M5 lane formats and the 27B’s group-64 8-bit K splits under the affine contract.
const affine = @import("metal_affine.zig");
const Reg = affine.Reg;
const Shape = @import("contract").spec.Shape;
pub const m5 = [_]Reg{
    .{ .v = .{ .kind = .lane, .bits = 4, .group = 64, .sk = 2 }, .name = "metal lane projection q4 g64 sk2" },
    .{ .v = .{ .kind = .lane, .bits = 4, .group = 64, .coop = true }, .name = "metal lane projection q4 g64 cooperative" },
    .{ .v = .{ .kind = .lane_reg, .bits = 4, .group = 64 }, .name = "metal lane projection q4 g64 reg" },
    .{ .v = .{ .kind = .lane, .bits = 8, .group = 128, .sk = 4 }, .name = "metal lane projection q8 g128 sk4" },
} ++ q8;

const head = [_]Shape{ .{ .rows = 1, .n = 248320, .k = 5120 }, .{ .rows = 16, .n = 64, .k = 5120 } };
const mlp = [_]Shape{.{ .rows = 16, .n = 34816, .k = 5120 }};
const qkv = [_]Shape{ .{ .rows = 16, .n = 10240, .k = 5120 }, .{ .rows = 5, .n = 12288, .k = 5120 } };
const out = [_]Shape{ .{ .rows = 16, .n = 5120, .k = 17408 }, .{ .rows = 5, .n = 5120, .k = 6144 }, .{ .rows = 16, .n = 2048, .k = 5120 }, .{ .rows = 5, .n = 6240, .k = 5120 } };
pub const q8 = [_]Reg{
    .{ .v = .{ .kind = .lane, .bits = 8, .group = 64, .sk = 1, .served = &head }, .name = "metal lane projection q8 g64 sk1" },
    .{ .v = .{ .kind = .lane, .bits = 8, .group = 64, .sk = 2, .served = &mlp }, .name = "metal lane projection q8 g64 sk2" },
    .{ .v = .{ .kind = .lane, .bits = 8, .group = 64, .sk = 4, .served = &qkv }, .name = "metal lane projection q8 g64 sk4" },
    .{ .v = .{ .kind = .lane, .bits = 8, .group = 64, .sk = 8, .served = &out }, .name = "metal lane projection q8 g64 sk8" },
};
