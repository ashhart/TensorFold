//! qlinear's refusals before any launch: a format not loaded, a lane mode not its format's, an EXL3 row stride.
const std = @import("std");
const qmmf = @import("qmmf.zig");
const exl3 = @import("exl3.zig");
const Weight = @import("qlinear.zig").Weight;
const Linear = @import("qlinear.zig").Linear;
const Affine4 = @import("qlinear.zig").Affine4;
const Stream = @import("stream.zig").Stream;

test "a weight view reports its shape in every format" {
    const a: Weight = .{ .affine4 = .{ .w = 0, .s = 0, .b = 0, .n = 10304, .k = 2688, .npad = 10304 } };
    const f: Weight = .{ .fp8g = .{ .mode = .fp8g, .codes = 0, .scales = 0, .n = 7168, .k = 2560, .npad = 7168 } };
    try std.testing.expectEqual(@as(usize, 10304), a.outputs());
    try std.testing.expectEqual(@as(usize, 2560), f.inputs());
}

test "a format that is not loaded is refused before any launch" {
    const l: Linear = .{};
    const w: Weight = .{ .affine4 = .{ .w = 0, .s = 0, .b = 0, .n = 64, .k = 64, .npad = 64 } };
    const s: Stream = undefined; // never reached: the refusal comes first
    try std.testing.expectError(error.FormatNotLoaded, l.decode(s, w, .{ .x = 0 }, 0, 1));
    try std.testing.expectError(error.FormatNotLoaded, l.prompt(s, w, .{ .x = 0 }, 0, 1));
}

test "the affine-4 tiles' sizes follow qmm_fast.tile" {
    const l = Affine4.layout(10304, 2688); // Nemotron's in_proj: rows to whole 128s, 4 bits a weight, 2 bytes a group
    try std.testing.expectEqual(@as(usize, 10368), l.npad);
    try std.testing.expectEqual(@as(usize, 10368 * 2688 / 2), l.words);
    try std.testing.expectEqual(@as(usize, 42 * 10368 * 2), l.scales);
}

test "a lane weight whose mode is not its format's is refused before any launch" {
    const lane: qmmf.Lane = undefined; // never reached: the refusal comes first
    const l: Linear = .{ .lane = &lane };
    const fp4: qmmf.Weight = .{ .mode = .fp4, .codes = 0, .scales = 0, .n = 64, .k = 64, .npad = 128 };
    const s: Stream = undefined;
    try std.testing.expectError(error.FormatMismatch, l.decode(s, .{ .fp8g = fp4 }, .{ .x = 0 }, 0, 1));
    try std.testing.expectError(error.FormatMismatch, l.decode(s, .{ .mxfp8 = fp4 }, .{ .x = 0 }, 0, 1));
    try std.testing.expectError(error.FormatMismatch, l.prompt(s, .{ .mxfp8 = fp4 }, .{ .x = 0 }, 0, 1));
    try std.testing.expectError(error.FormatNotLoaded, l.prompt(s, .{ .nvfp4 = fp4 }, .{ .x = 0 }, 0, 1));
    const f8: qmmf.Weight = .{ .mode = .fp8g, .codes = 0, .scales = 0, .n = 64, .k = 64, .npad = 128 };
    try std.testing.expectError(error.EpilogueNotBuilt, l.decodeF32(s, .{ .mxfp8 = f8 }, .{ .x = 0 }, 0, 1));
    try std.testing.expectError(error.EpilogueNotBuilt, l.promptF32(s, .{ .bf16 = f8 }, .{ .x = 0 }, 0, 1));
    try std.testing.expectError(error.FormatMismatch, l.decodeF32(s, .{ .fp8g = fp4 }, .{ .x = 0 }, 0, 1));
    try std.testing.expectError(error.FormatMismatch, l.promptF32(s, .{ .fp8g = fp4 }, .{ .x = 0 }, 0, 1));
}

test "EXL3 FP32 entries check loading and stride before launching" {
    const w: Weight = .{ .exl3 = .{ .trellis = 0, .suh = 0, .svh = 0, .k = 128, .n = 128, .halves = 8, .codebook = .mul1, .stride_k = 512, .stride_nb = 256 } };
    const unloaded: Linear = .{};
    try std.testing.expectError(error.FormatNotLoaded, unloaded.decodeF32(undefined, w, .{ .x = 0 }, 0, 1));
    try std.testing.expectError(error.FormatNotLoaded, unloaded.promptF32(undefined, w, .{ .x = 0 }, 0, 1));
    var dense: exl3.Dense = undefined;
    const loaded: Linear = .{ .exl3 = &dense };
    try std.testing.expectError(error.RowStride, loaded.decodeF32(undefined, w, .{ .x = 0, .ldx = 127 }, 0, 1));
    try std.testing.expectError(error.RowStride, loaded.promptF32(undefined, w, .{ .x = 0, .ldx = 129 }, 0, 1));
}
