//! Grouped's refusals before any launch: a format not loaded, an epilogue or an option it was not built with.
const std = @import("std");
const experts = @import("experts.zig");

const Grouped = experts.Grouped;
const Layer = experts.Layer;
const Plan = @import("grouped.zig").Plan;
const Stream = @import("stream.zig").Stream;

test "a format that is not loaded, or an option it was not built with, is refused before any launch" {
    const g: Grouped = .{};
    const l: Layer = .{ .format = .fp8g, .up = 0, .down = 0, .width = 64, .dims = 64, .experts = 2 };
    const s: Stream = undefined; // never reached: the refusal comes first
    const p: Plan = .{ .members = 0, .items = 0, .counts = 0, .rank = 0, .hist = 0 };
    try std.testing.expectError(error.FormatNotLoaded, g.decode(s, .up, .{ .x = 0, .stride = 64 }, l, p, 0, 1, -1));
    const a4: experts.Affine4Experts = undefined;
    const h: Grouped = .{ .affine4 = &a4 };
    var la = l;
    la.format = .affine4;
    try std.testing.expectError(error.EpilogueNotBuilt, h.decode(s, .down_bf16, .{ .x = 0, .stride = 64 }, la, p, 0, 1, -1));
    try std.testing.expectError(error.EpilogueNotBuilt, h.prompt(s, .down_f32, .{ .x = 0, .stride = 64 }, la, p, 0, 1, -1));
    try std.testing.expectError(error.OptionNotBuilt, h.decode(s, .up, .{ .x = 0, .stride = 64 }, la, p, 0, 1, 3));
    try std.testing.expectError(error.OptionNotBuilt, h.prompt(s, .up, .{ .x = 0, .stride = 64 }, .{ .format = .affine4, .up = 0, .down = 0, .width = 64, .dims = 64, .experts = 2, .after = true }, p, 0, 1, -1));
}

test "the clamp-after order: exl3 takes it on both paths, every other format refuses it on both" {
    const g: Grouped = .{};
    const s: Stream = undefined; // never reached: the refusal comes first
    const p: Plan = .{ .members = 0, .items = 0, .counts = 0, .rank = 0, .hist = 0 };
    const in: experts.Rows = .{ .x = 0, .stride = 64 };
    var l: Layer = .{ .format = .exl3, .up = 0, .down = 0, .width = 64, .dims = 64, .experts = 2, .limit = 10, .after = true };
    // exl3 accepts the order, then finds its kernels not loaded
    try std.testing.expectError(error.FormatNotLoaded, g.decode(s, .up, in, l, p, 0, 1, -1));
    try std.testing.expectError(error.FormatNotLoaded, g.prompt(s, .up, in, l, p, 0, 1, -1));
    for (std.enums.values(experts.Format)) |f| if (f != .exl3) {
        l.format = f; // nvfp4 among them; not loaded either, so the refusal comes before the format's
        try std.testing.expectError(error.OptionNotBuilt, g.decode(s, .up, in, l, p, 0, 1, -1));
        try std.testing.expectError(error.OptionNotBuilt, g.prompt(s, .up, in, l, p, 0, 1, -1));
    };
}
