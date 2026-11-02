//! Grouped experts refuse a format that is not loaded, an option it was not built with, or a shape, before any launch.
const std = @import("std");
const experts = @import("experts.zig");
const Grouped = experts.Grouped;
const Layer = experts.Layer;
const Plan = @import("grouped.zig").Plan;
const Stream = @import("stream.zig").Stream;
const Affine4Experts = experts.Affine4Experts;

test "a format that is not loaded, or an option it was not built with, is refused before any launch" {
    const g: Grouped = .{};
    const l: Layer = .{ .format = .fp8g, .up = 0, .down = 0, .width = 64, .dims = 64, .experts = 2 };
    const s: Stream = undefined; // never reached: the refusal comes first
    const p: Plan = .{ .members = 0, .items = 0, .counts = 0, .rank = 0, .hist = 0 };
    try std.testing.expectError(error.FormatNotLoaded, g.decode(s, .up, .{ .x = 0, .stride = 64 }, l, p, 0, 1, -1));
    const a4: Affine4Experts = undefined;
    const h: Grouped = .{ .affine4 = &a4 };
    var la = l;
    la.format = .affine4;
    try std.testing.expectError(error.EpilogueNotBuilt, h.decode(s, .down_bf16, .{ .x = 0, .stride = 64 }, la, p, 0, 1, -1));
    try std.testing.expectError(error.EpilogueNotBuilt, h.prompt(s, .down_f32, .{ .x = 0, .stride = 64 }, la, p, 0, 1, -1));
    try std.testing.expectError(error.OptionNotBuilt, h.decode(s, .up, .{ .x = 0, .stride = 64 }, la, p, 0, 1, 3));
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
            try std.testing.expectError(error.UnexpectedShape, g.decode(s, epi, in, l, p, 0, 1, -1));
            try std.testing.expectError(error.UnexpectedShape, g.prompt(s, epi, in, l, p, 0, 1, -1));
        }
    }
}
