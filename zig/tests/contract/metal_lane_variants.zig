//! The M5 lane formats and the 27B’s group-64 8-bit K splits under the affine contract.
const std = @import("std");
const affine = @import("metal_affine.zig");
const Reg = affine.Reg;
const Shape = @import("contract").spec.Shape;
pub const m5 = [_]Reg{
    .{ .v = .{ .kind = .lane, .bits = 4, .group = 64, .sk = 2 }, .name = "metal lane projection q4 g64 sk2" },
    .{ .v = .{ .kind = .lane, .bits = 4, .group = 64, .coop = true }, .name = "metal lane projection q4 g64 cooperative" },
    .{ .v = .{ .kind = .lane_reg, .bits = 4, .group = 64 }, .name = "metal lane projection q4 g64 reg" },
    .{ .v = .{ .kind = .lane_reg, .bits = 5, .group = 64 }, .name = "metal lane projection q5 g64 reg" },
    .{ .v = .{ .kind = .lane_reg, .bits = 5, .group = 64, .control = true }, .name = "control: metal lane projection q5 g64 reg, unreordered tiles" },
    .{ .v = .{ .kind = .lane_reg, .bits = 6, .group = 64 }, .name = "metal lane projection q6 g64 reg" },
    .{ .v = .{ .kind = .lane_reg, .bits = 6, .group = 64, .control = true }, .name = "control: metal lane projection q6 g64 reg, unreordered tiles" },
    .{ .v = .{ .kind = .lane, .bits = 8, .group = 128, .sk = 4 }, .name = "metal lane projection q8 g128 sk4" },
} ++ q8 ++ hy4 ++ q27;

const q27 = [_]Reg{
    q27Inline(1, &.{ .{ .rows = 16, .n = 98592, .k = 5120 }, .{ .rows = 16, .n = 17408, .k = 5120 } }),
    q27Inline(2, &.{ .{ .rows = 16, .n = 17408, .k = 5120 }, .{ .rows = 5, .n = 10240, .k = 5120 } }),
    q27Inline(4, &.{ .{ .rows = 5, .n = 5120, .k = 6144 }, .{ .rows = 16, .n = 5120, .k = 17408 } }),
    q27Inline(8, &.{ .{ .rows = 16, .n = 4096, .k = 5120 }, .{ .rows = 16, .n = 5120, .k = 4096 }, .{ .rows = 16, .n = 5120, .k = 17408 }, .{ .rows = 16, .n = 1280, .k = 5120 }, .{ .rows = 16, .n = 256, .k = 5120 }, .{ .rows = 16, .n = 1024, .k = 5120 } }),
};

fn q27Inline(comptime sk: usize, served: []const Shape) Reg {
    return .{ .v = .{ .kind = .lane_reg, .sk = sk, .inline_sums = true, .served = served }, .name = std.fmt.comptimePrint("qwen27 inline sums sk{d}", .{sk}) };
}

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

// Hy4-flash's widths at its K splits: a decode row, DFly2's window of 6 and the 8-row inline-sums limit.
const hy4_k2 = [_]Shape{ .{ .rows = 6, .n = 3072, .k = 1536 }, .{ .rows = 1, .n = 3072, .k = 1536 } };
// the first shape of each is the edges' base, narrow so the float64 references stay quick
const hy4_k4 = [_]Shape{ .{ .rows = 6, .n = 1024, .k = 3072 }, .{ .rows = 8, .n = 13824, .k = 3072 }, .{ .rows = 1, .n = 4096, .k = 3072 } };
const hy4_k8 = [_]Shape{ .{ .rows = 6, .n = 512, .k = 13824 }, .{ .rows = 1, .n = 3072, .k = 13824 } };
fn hy4Reg(comptime bits: u8, comptime sk: usize, comptime sums: bool) Reg {
    const served = switch (sk) {
        2 => &hy4_k2,
        4 => &hy4_k4,
        else => &hy4_k8,
    };
    const what = if (sums) " inline sums" else "";
    return .{ .v = .{ .kind = .lane_reg, .bits = bits, .group = 64, .sk = sk, .inline_sums = sums, .served = served }, .name = std.fmt.comptimePrint("metal lane projection q{d} g64 reg sk{d}{s} (Hy4-flash)", .{ bits, sk, what }) };
}
pub const hy4 = blk: {
    var list: [12]Reg = undefined;
    var at = 0;
    for ([_]u8{ 4, 8 }) |bits| for ([_]usize{ 2, 4, 8 }) |sk| for ([_]bool{ false, true }) |sums| {
        list[at] = hy4Reg(bits, sk, sums);
        at += 1;
    };
    break :blk list;
};
