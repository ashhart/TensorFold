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

test "mxfp4 experts take SiTU's up and the two downs, on whole 32-value groups with scales, and E8M0 codes 2 to 252" {
    const m: experts.Mxfp4Experts = undefined; // never reached: the refusal comes first
    const g: Grouped = .{ .mxfp4 = &m };
    const s: Stream = undefined;
    const p: Plan = .{ .members = 0, .items = 0, .counts = 0, .rank = 0, .hist = 0 };
    const in: experts.Rows = .{ .x = 0, .stride = 64, .slots = 10 };
    const l: Layer = .{ .format = .mxfp4, .up = 1, .up_scale = 1, .down = 1, .down_scale = 1, .width = 64, .dims = 64, .experts = 2 };
    // an up without both of SiTU's betas is not built
    for ([_][2]f32{ .{ 0, 0 }, .{ 4, 0 }, .{ 0, 25 } }) |b| {
        var o = l;
        o.situ = b;
        try std.testing.expectError(error.EpilogueNotBuilt, g.decode(s, .up, in, o, p, 0, 1, 1, 16, -1));
        try std.testing.expectError(error.EpilogueNotBuilt, g.prompt(s, .up, in, o, p, 0, 1, 1, 64, -1));
    }
    var situ = l;
    situ.situ = .{ 4, 25 };
    for ([_][2]usize{ .{ 48, 64 }, .{ 64, 48 }, .{ 2048, 2040 } }) |wd| {
        var o = situ;
        o.width = wd[0];
        o.dims = wd[1];
        for ([_]experts.Epi{ .up, .down_f32, .down_bf16 }) |epi| {
            try std.testing.expectError(error.UnexpectedShape, g.decode(s, epi, in, o, p, 0, 1, 1, 16, -1));
            try std.testing.expectError(error.UnexpectedShape, g.prompt(s, epi, in, o, p, 0, 1, 1, 64, -1));
        }
    }
    var bare = situ;
    bare.down_scale = 0;
    try std.testing.expectError(error.MissingScales, g.decode(s, .down_f32, in, bare, p, 0, 1, 1, 16, -1));
    bare = situ;
    bare.up_scale = 0;
    try std.testing.expectError(error.MissingScales, g.decode(s, .up, in, bare, p, 0, 1, 1, 16, -1));
    try experts.Mxfp4Experts.checkScales(&.{ 2, 113, 122, 252 });
    for ([_]u8{ 0, 1, 253, 254, 255 }) |c| try std.testing.expectError(error.Mxfp4ScaleRange, experts.Mxfp4Experts.checkScales(&.{ 120, c }));
}
