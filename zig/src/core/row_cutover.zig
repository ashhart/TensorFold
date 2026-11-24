//! The row projection's narrow/wide cutover by chip: generation and tier, a window width per code width (#3396).
const std = @import("std");

/// The code widths the row kernels unpack.
pub const bit_widths = [_]u8{ 2, 3, 4, 5, 6, 8 };

/// A chip's tier, from the Metal device's name ("Apple M5 Max"): GPU cores, not the generation, decide the cutover.
pub const Tier = enum { base, pro, max, ultra };

/// What the cutover is keyed on: the GPU generation (applegpu_gN) and the tier its name gives (null: none read).
pub const Chip = struct { generation: u32, tier: ?Tier = null };

/// A device's chip: "Apple M<n>" is base, "Apple M<n> Pro|Max|Ultra" that tier; any other name keys no tier.
pub fn chipOf(generation: u32, name: []const u8) Chip {
    var words = std.mem.tokenizeScalar(u8, name, ' ');
    const apple = words.next() orelse "";
    const model = words.next() orelse "";
    const tier = words.next();
    const unknown: Chip = .{ .generation = generation };
    if (!std.mem.eql(u8, apple, "Apple") or model.len < 2 or model[0] != 'M' or words.next() != null) return unknown;
    for (model[1..]) |ch| if (!std.ascii.isDigit(ch)) return unknown;
    const t = tier orelse return .{ .generation = generation, .tier = .base };
    const tiers = [_]struct { []const u8, Tier }{ .{ "Pro", .pro }, .{ "Max", .max }, .{ "Ultra", .ultra } };
    for (tiers) |x| if (std.mem.eql(u8, t, x[0])) return .{ .generation = generation, .tier = x[1] };
    return unknown;
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

/// Whether a window of `rows` at code width `bits` takes the wide kernels, by a table of a cutover each width.
pub fn takes(wide_from: *const [bit_widths.len]usize, bits: u8, rows: usize) bool {
    const b = std.mem.indexOfScalar(u8, &bit_widths, bits) orelse return false;
    return rows >= wide_from[b];
}

/// tf-row-cutover's key line: the device name it read, the chip it keys on and the table's cutover at each width.
pub fn keyLine(buf: []u8, name: []const u8, chip: Chip) ![]const u8 {
    var w: std.Io.Writer = .fixed(buf);
    try w.print("device \"{s}\", generation {d}, tier {s}, table cutover", .{ name, chip.generation, if (chip.tier) |t| @tagName(t) else "unknown" });
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
    try std.testing.expectEqual(Chip{ .generation = 17, .tier = .max }, chipOf(17, "Apple M5 Max "));
    try std.testing.expectEqual(Chip{ .generation = 17, .tier = .base }, chipOf(17, "Apple M5"));
    // names it does not know key no tier (REV75 2): a new tier word, a suffix, a non-M chip, another vendor
    for ([_][]const u8{ "Apple M5 Extreme", "Apple M5 Max (40-core)", "Apple Mx", "Apple A19 Pro", "AMD Radeon Pro", "Foo Max", "Apple Paravirtual device", "Some GPU", "" }) |name|
        try std.testing.expectEqual(Chip{ .generation = 17, .tier = null }, chipOf(17, name));
}

test "tf-row-cutover prints the key it uses and the table's cutover at each width" {
    var buf: [256]u8 = undefined;
    try std.testing.expectEqualStrings("device \"Apple M5 Max\", generation 17, tier max, table cutover q2=32 q3=never q4=never q5=never q6=never q8=never", try keyLine(&buf, "Apple M5 Max", chipOf(17, "Apple M5 Max")));
    try std.testing.expectEqualStrings("device \"Apple M5 Ultra\", generation 17, tier ultra, table cutover q2=8 q3=8 q4=8 q5=8 q6=8 q8=8", try keyLine(&buf, "Apple M5 Ultra", chipOf(17, "Apple M5 Ultra")));
    try std.testing.expectEqualStrings("device \"Apple M5 Max (40-core)\", generation 16, tier unknown, table cutover q2=never q3=never q4=never q5=never q6=never q8=never", try keyLine(&buf, "Apple M5 Max (40-core)", chipOf(16, "Apple M5 Max (40-core)")));
}

test "a per-width table routes each code width by its own cutover, as an M5 Max's does (REV75 3)" {
    var max: [bit_widths.len]usize = undefined;
    for (&max, bit_widths) |*from, bits| from.* = wideFrom(chipOf(17, "Apple M5 Max"), bits).?;
    try std.testing.expect(takes(&max, 2, 32)); // 2 bits: wide at 32
    try std.testing.expect(!takes(&max, 2, 31));
    for ([_]u8{ 3, 4, 5, 6, 8 }) |bits| try std.testing.expect(!takes(&max, bits, 32)); // the rest narrow at every width
    var ultra: [bit_widths.len]usize = undefined;
    for (&ultra, bit_widths) |*from, bits| from.* = wideFrom(chipOf(17, "Apple M5 Ultra"), bits).?;
    for (bit_widths) |bits| {
        try std.testing.expect(takes(&ultra, bits, 8) and !takes(&ultra, bits, 7)); // the Ultra's 8, as before
    }
    try std.testing.expect(!takes(&ultra, 7, 32)); // a width the kernels do not unpack takes none
}
