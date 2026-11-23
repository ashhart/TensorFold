//! The row projection's narrow/wide cutover by chip: generation and tier, a window width per code width (#3396).
const std = @import("std");

/// The code widths the row kernels unpack.
pub const bit_widths = [_]u8{ 2, 3, 4, 5, 6, 8 };

/// A chip's tier, from the Metal device's name ("Apple M5 Max"): GPU cores, not the generation, decide the cutover.
pub const Tier = enum { base, pro, max, ultra };

/// What the cutover is keyed on: the GPU generation (applegpu_gN) and the tier its name gives (null: none read).
pub const Chip = struct { generation: u32, tier: ?Tier = null };

/// The chip of a device's generation and name: "Max", "Ultra" or "Pro" as its name's last word, else base.
pub fn chipOf(generation: u32, name: []const u8) Chip {
    const last = name[if (std.mem.lastIndexOfScalar(u8, name, ' ')) |i| i + 1 else 0..];
    const tiers = [_]struct { []const u8, Tier }{ .{ "Max", .max }, .{ "Ultra", .ultra }, .{ "Pro", .pro } };
    for (tiers) |t| if (std.mem.eql(u8, last, t[0])) return .{ .generation = generation, .tier = t[1] };
    return .{ .generation = generation, .tier = if (std.mem.startsWith(u8, name, "Apple M")) .base else null };
}

/// Never: the wide kernels lose at every window to 32 rows.
pub const never = std.math.maxInt(usize);

/// The narrowest wide window a code width (tf-row-cutover measures it); a tier's row: its generation only, else on.
pub const Cutover = struct { generation: u32, tier: ?Tier = null, wide_from: [bit_widths.len]usize };

/// Tensor-unit chips, oldest first; earlier generations never take the wide kernels, unlisted chips their generation's.
pub const cutovers = [_]Cutover{
    .{ .generation = 17, .wide_from = @splat(8) }, // M5 Ultra (#3396's measurements), and every other M5 as before
    .{ .generation = 17, .tier = .max, .wide_from = .{ 32, never, never, never, never, never } }, // M5 Max: narrow wins
};

/// The table's cutover for a chip at a code width, or null before the first tensor-unit generation.
pub fn wideFrom(chip: Chip, bits: u8) ?usize {
    const b = std.mem.indexOfScalar(u8, &bit_widths, bits) orelse return null;
    var out: ?usize = null;
    for (cutovers) |c| if (c.tier == null and chip.generation >= c.generation) {
        out = c.wide_from[b];
    };
    for (cutovers) |c| if (c.tier != null and c.tier == chip.tier and c.generation == chip.generation) {
        out = c.wide_from[b];
    };
    return out;
}

/// tf-row-cutover's key line: the chip it keys on and the table's cutover at each width.
pub fn keyLine(buf: []u8, chip: Chip) ![]const u8 {
    var w: std.Io.Writer = .fixed(buf);
    try w.print("generation {d}, tier {s}, table cutover", .{ chip.generation, if (chip.tier) |t| @tagName(t) else "unknown" });
    for (bit_widths) |bits| {
        const from = wideFrom(chip, bits) orelse never;
        if (from == never) try w.print(" q{d}=never", .{bits}) else try w.print(" q{d}={d}", .{ bits, from });
    }
    return w.buffered();
}

test "the cutover table covers tensor-unit generations only, an unlisted chip at its generation's as before" {
    for (bit_widths) |bits| {
        for ([_]u32{ 0, 13, 14, 15, 16 }) |generation| try std.testing.expectEqual(@as(?usize, null), wideFrom(.{ .generation = generation, .tier = .max }, bits));
        for ([_]?Tier{ null, .base, .pro, .ultra }) |tier| { // unlisted tiers and unread names keep generation 17's
            try std.testing.expectEqual(@as(?usize, 8), wideFrom(.{ .generation = 17, .tier = tier }, bits));
            try std.testing.expectEqual(@as(?usize, 8), wideFrom(.{ .generation = 18, .tier = tier }, bits));
        }
        try std.testing.expectEqual(@as(?usize, 8), wideFrom(.{ .generation = 18, .tier = .max }, bits)); // a later Max: unmeasured
    }
}

test "an M5 Max keeps the narrow kernel to 32 rows, at 2 bits wide only at 32 (#3396's measurements)" {
    const max = chipOf(17, "Apple M5 Max");
    try std.testing.expectEqual(@as(?Tier, .max), max.tier);
    try std.testing.expectEqual(@as(?usize, 32), wideFrom(max, 2));
    for ([_]u8{ 3, 4, 5, 6, 8 }) |bits| try std.testing.expectEqual(@as(?usize, never), wideFrom(max, bits));
    try std.testing.expectEqual(@as(?usize, 8), wideFrom(chipOf(17, "Apple M5 Ultra"), 4)); // the Ultra's row as it was
}

test "the chip key reads the tier from the device name, and nothing it does not know" {
    try std.testing.expectEqual(Chip{ .generation = 17, .tier = .ultra }, chipOf(17, "Apple M5 Ultra"));
    try std.testing.expectEqual(Chip{ .generation = 17, .tier = .pro }, chipOf(17, "Apple M5 Pro"));
    try std.testing.expectEqual(Chip{ .generation = 17, .tier = .base }, chipOf(17, "Apple M5"));
    try std.testing.expectEqual(Chip{ .generation = 17, .tier = null }, chipOf(17, "Some GPU"));
    try std.testing.expectEqual(Chip{ .generation = 17, .tier = null }, chipOf(17, ""));
}

test "tf-row-cutover prints the key it uses and the table's cutover at each width" {
    var buf: [256]u8 = undefined;
    try std.testing.expectEqualStrings("generation 17, tier max, table cutover q2=32 q3=never q4=never q5=never q6=never q8=never", try keyLine(&buf, chipOf(17, "Apple M5 Max")));
    try std.testing.expectEqualStrings("generation 17, tier ultra, table cutover q2=8 q3=8 q4=8 q5=8 q6=8 q8=8", try keyLine(&buf, chipOf(17, "Apple M5 Ultra")));
    try std.testing.expectEqualStrings("generation 16, tier unknown, table cutover q2=never q3=never q4=never q5=never q6=never q8=never", try keyLine(&buf, chipOf(16, "")));
}
