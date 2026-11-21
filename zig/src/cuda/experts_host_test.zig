//! Grouped's refusals before any launch, and the NVFP4 and FP8 block packings' layouts.
const std = @import("std");
const experts = @import("experts.zig");

const Grouped = experts.Grouped;
const Layer = experts.Layer;
const Plan = @import("grouped.zig").Plan;
const Stream = @import("stream.zig").Stream;
const Nvfp4Experts = experts.Nvfp4Experts;
const Fp8Experts = experts.Fp8Experts;

test "a format that is not loaded, or an option it was not built with, is refused before any launch" {
    const g: Grouped = .{};
    const l: Layer = .{ .format = .fp8g, .up = 0, .down = 0, .width = 64, .dims = 64, .experts = 2 };
    const s: Stream = undefined; // never reached: the refusal comes first
    const p: Plan = .{ .members = 0, .items = 0, .counts = 0, .rank = 0, .hist = 0 };
    try std.testing.expectError(error.FormatNotLoaded, g.decode(s, .up, .{ .x = 0, .stride = 64 }, l, p, 0, 1, 1, 16, -1));
    const a4: experts.Affine4Experts = undefined;
    const h: Grouped = .{ .affine4 = &a4 };
    var la = l;
    la.format = .affine4;
    try std.testing.expectError(error.EpilogueNotBuilt, h.decode(s, .down_bf16, .{ .x = 0, .stride = 64 }, la, p, 0, 1, 1, 16, -1));
    try std.testing.expectError(error.EpilogueNotBuilt, h.prompt(s, .down_f32, .{ .x = 0, .stride = 64 }, la, p, 0, 1, 1, 64, -1));
    try std.testing.expectError(error.OptionNotBuilt, h.decode(s, .up, .{ .x = 0, .stride = 64 }, la, p, 0, 1, 1, 16, 3));
    try std.testing.expectError(error.OptionNotBuilt, h.prompt(s, .up, .{ .x = 0, .stride = 64 }, .{ .format = .affine4, .up = 0, .down = 0, .width = 64, .dims = 64, .experts = 2, .after = true }, p, 0, 1, 1, 64, -1));
}

test "the clamp-after order: exl3 takes it on both paths, every other format refuses it on both" {
    const g: Grouped = .{};
    const s: Stream = undefined; // never reached: the refusal comes first
    const p: Plan = .{ .members = 0, .items = 0, .counts = 0, .rank = 0, .hist = 0 };
    const in: experts.Rows = .{ .x = 0, .stride = 64 };
    var l: Layer = .{ .format = .exl3, .up = 0, .down = 0, .width = 64, .dims = 64, .experts = 2, .limit = 10, .after = true };
    // exl3 accepts the order, then finds its kernels not loaded
    try std.testing.expectError(error.FormatNotLoaded, g.decode(s, .up, in, l, p, 0, 1, 1, 16, -1));
    try std.testing.expectError(error.FormatNotLoaded, g.prompt(s, .up, in, l, p, 0, 1, 1, 64, -1));
    for (std.enums.values(experts.Format)) |f| if (f != .exl3) {
        l.format = f; // nvfp4 among them; not loaded either, so the refusal comes before the format's
        try std.testing.expectError(error.OptionNotBuilt, g.decode(s, .up, in, l, p, 0, 1, 1, 16, -1));
        try std.testing.expectError(error.OptionNotBuilt, g.prompt(s, .up, in, l, p, 0, 1, 1, 64, -1));
    };
}

test "bf16 experts refuse dims or widths off the 32-wide blocks before any launch" {
    const b: experts.Bf16Experts = undefined; // never reached: the refusal comes first
    const g: Grouped = .{ .bf16 = &b };
    const s: Stream = undefined;
    const p: Plan = .{ .members = 0, .items = 0, .counts = 0, .rank = 0, .hist = 0 };
    const in: experts.Rows = .{ .x = 0, .stride = 64 };
    for ([_][2]usize{ .{ 48, 64 }, .{ 64, 48 }, .{ 2688, 1880 } }) |wd| {
        const l: Layer = .{ .format = .bf16, .up = 0, .down = 0, .width = wd[0], .dims = wd[1], .experts = 2, .relu2 = true };
        for ([_]experts.Epi{ .up, .down_f32, .down_bf16 }) |epi| {
            try std.testing.expectError(error.UnexpectedShape, g.decode(s, epi, in, l, p, 0, 1, 1, 16, -1));
            try std.testing.expectError(error.UnexpectedShape, g.prompt(s, epi, in, l, p, 0, 1, 1, 64, -1));
        }
    }
}

test "a packed nvfp4 block puts each nibble and scale where nvfp4/experts.py's _pack does" {
    const gpa = std.testing.allocator;
    const n = 64;
    const k = 64;
    const codes = try gpa.alloc(u8, n * k / 2);
    defer gpa.free(codes);
    for (codes, 0..) |*c, i| c.* = @truncate(i * 29 + 3);
    var scales: [n * k / 16]u8 = undefined;
    for (&scales, 0..) |*c, i| c.* = @truncate(i * 7 + 1);
    var out: [n / 32 * (k / 32) * Nvfp4Experts.words]u32 = undefined;
    Nvfp4Experts.packOne(&out, codes, &scales, n, k);
    const words = Nvfp4Experts.words;
    // block (cb 1, g 1), lane gq 3, t 2, tile j 1: row 32 + 8 + 3 = 43; input 32 + 16 + 8 + 3 (h 1, q 3) -> slot 7
    const w = out[(1 * 2 + 1) * words + (3 * 4 + 2) * 4 + 1];
    try std.testing.expectEqual(@as(u32, codes[43 * 32 + 59 / 2] >> 4), (w >> Nvfp4Experts.slot(1, 3)) & 0xF);
    // its scale byte (t 2, h 1, j 1, c 0): row 32 + 8 + 4 = 44, k16 group 3
    const sc = std.mem.sliceAsBytes(out[(1 * 2 + 1) * words + 128 ..][0..16]);
    try std.testing.expectEqual(scales[44 * 4 + 3], sc[((2 * 2 + 1) * 4 + 1) * 2]);
}

test "packGateUp stacks packOne's gate and up blocks per (column block, k32 group)" {
    const n = 64;
    const k = 64;
    const words = Nvfp4Experts.words;
    var codes: [2][n * k / 2]u8 = undefined;
    var scales: [2][n * k / 16]u8 = undefined;
    for (0..2) |m| {
        for (&codes[m], 0..) |*c, i| c.* = @truncate(i * 29 + 3 + m * 101);
        for (&scales[m], 0..) |*c, i| c.* = @truncate(i * 7 + 1 + m * 13);
    }
    var one: [2][n / 32 * (k / 32) * words]u32 = undefined;
    for (0..2) |m| Nvfp4Experts.packOne(&one[m], &codes[m], &scales[m], n, k);
    var both: [2 * one[0].len]u32 = undefined;
    Nvfp4Experts.packGateUp(&both, .{ &codes[0], &scales[0] }, .{ &codes[1], &scales[1] }, n, k);
    for (0..one[0].len / words) |b| for (0..2) |m| {
        try std.testing.expectEqualSlices(u32, one[m][b * words ..][0..words], both[(2 * b + m) * words ..][0..words]);
    };
}

test "nvfp4 rows of either kind are refused when the format is not loaded" {
    const g: Grouped = .{};
    const l: Layer = .{ .format = .nvfp4, .up = 0, .down = 0, .width = 64, .dims = 64, .experts = 2, .relu2 = true };
    const s: Stream = undefined; // never reached: the refusal comes first
    const p: Plan = .{ .members = 0, .items = 0, .counts = 0, .rank = 0, .hist = 0 };
    try std.testing.expectError(error.FormatNotLoaded, g.prompt(s, .up, .{ .x = 0, .stride = 64 }, l, p, 0, 1, 1, 64, -1));
    try std.testing.expectError(error.FormatNotLoaded, g.decode(s, .down_bf16, .{ .x = 0, .stride = 64 }, l, p, 0, 1, 1, 16, -1));
    try std.testing.expectEqual(@as(usize, 2 * 2 * 3 * 144 * 4), Nvfp4Experts.bytes(2, 64, 96, 1));
}

test "a packed fp8 expert puts each word where fp8/experts.py's pack does" {
    const gpa = std.testing.allocator;
    const n = 128;
    const k = 128;
    const w = try gpa.alloc(u8, n * k);
    defer gpa.free(w);
    for (w, 0..) |*b, i| b.* = @truncate(i * 31 + 7);
    const out = try gpa.alloc(u32, n * k / 4);
    defer gpa.free(out);
    Fp8Experts.packOne(out, w, n, k);
    // (cb 1, g 2, gq 5, t 3, h 1, j 2): row 32 + 16 + 5 = 53, word 2 * 8 + 4 + 3 = 23
    const idx = (((((1 * 4 + 2) * 8 + 5) * 4 + 3) * 2 + 1) * 4) + 2;
    const words = std.mem.bytesAsSlice(u32, @as([]align(1) const u8, w));
    try std.testing.expectEqual(words[53 * (k / 4) + 23], out[idx]);
}
