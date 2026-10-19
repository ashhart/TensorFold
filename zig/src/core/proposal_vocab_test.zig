const std = @import("std");
const p = @import("proposal_vocab.zig");
const t = std.testing;
const fixture = @embedFile("testdata/proposal-vocab.json");
fn bind(name: []const u8, hash: []const u8) p.Binding {
    return .{ .name = name, .revision = "r1", .sha256 = hash };
}
const expected: p.Identity = .{
    .target = bind("synthetic-target", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
    .drafter = bind("synthetic-drafter", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
    .head = bind("proposal-head", "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"),
    .tokenizer = bind("synthetic-tokenizer", "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"),
    .recipe = bind("synthetic-recipe", "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"),
    .vocab = 11,
    .hidden = 8,
    .format = "bf16",
    .group = 0,
    .scale_recipe = "none",
};
fn changed(from: []const u8, to: []const u8) ![]u8 {
    try t.expect(std.mem.count(u8, fixture, from) == 1);
    return std.mem.replaceOwned(u8, t.allocator, fixture, from, to);
}
fn refuses(bytes: []const u8) !void {
    if (p.load(t.allocator, bytes, expected, &.{0})) |ok| {
        var a = ok;
        defer a.deinit();
        return error.BadArtifactAccepted;
    } else |_| {}
}

test "proposal vocabulary: owned immutable IDs survive input destruction and map every logical column" {
    const input = try t.allocator.dupe(u8, fixture);
    defer t.allocator.free(input);
    var a = try p.load(t.allocator, input, expected, &.{0});
    defer a.deinit();
    @memset(input, '?');
    try t.expectEqualSlices(u32, &.{ 0, 3, 7 }, a.ids());
    try t.expectEqualStrings("synthetic-tokenizer", a.identity().tokenizer.name);
    for (a.ids(), 0..) |id, n| {
        try t.expectEqual(@as(?u32, id), a.global(n));
        try t.expectEqual(@as(?usize, n), a.column(id));
    }
    try t.expectEqual(@as(?usize, null), a.column(2));
    try t.expectEqual(@as(usize, 3), a.countCovered(&.{ 0, 7, 2, 3, 10 }));
    try t.expectEqual(@as(usize, 0), a.countCovered(&.{}));
}

test "proposal vocabulary: non-tiled IDs never turn padding or a padded winner into a token" {
    var a = try p.load(t.allocator, fixture, expected, &.{0});
    defer a.deinit();
    try t.expectEqual(@as(u32, 64), try a.physicalCount(64));
    try t.expectEqual(@as(u32, 3), try a.physicalCount(1));
    for (3..64) |column| try t.expectEqual(@as(?u32, null), a.global(column));
    try t.expectEqual(@as(?u32, null), a.global(63));
    try t.expectError(error.InvalidTile, a.physicalCount(0));
    try t.expectEqual(std.math.maxInt(u32), try a.physicalCount(std.math.maxInt(u32)));
}

test "proposal vocabulary: same vocabulary size cannot substitute any checkpoint tokenizer head or recipe" {
    inline for (.{ "target", "drafter", "head", "tokenizer", "recipe" }) |field| {
        var other = expected;
        @field(other, field).sha256 = "1111111111111111111111111111111111111111111111111111111111111111";
        try t.expectError(error.IdentityMismatch, p.load(t.allocator, fixture, other, &.{0}));
        other = expected;
        @field(other, field).revision = "stale";
        try t.expectError(error.IdentityMismatch, p.load(t.allocator, fixture, other, &.{0}));
    }
    var other = expected;
    other.hidden = 9;
    try t.expectError(error.IdentityMismatch, p.load(t.allocator, fixture, other, &.{0}));
    other = expected;
    other.format = "affine4";
    try t.expectError(error.IdentityMismatch, p.load(t.allocator, fixture, other, &.{0}));
    other = expected;
    other.group = 64;
    try t.expectError(error.IdentityMismatch, p.load(t.allocator, fixture, other, &.{0}));
    other = expected;
    other.scale_recipe = "different-scales";
    try t.expectError(error.IdentityMismatch, p.load(t.allocator, fixture, other, &.{0}));
}

test "proposal vocabulary: malformed stale duplicate empty fractional string and out-of-range artifacts refuse" {
    for ([_][2][]const u8{
        .{ "\"version\": 1", "\"version\": 2" },
        .{ "\"version\": 1", "\"version\": 1, \"unknown\": true" },
        .{ "\"version\": 1", "\"version\": 1, \"version\": 1" },
        .{ "\"version\": 1", "\"version\": \"1\"" },
        .{ "\"version\": 1", "\"version\": 1.0" },
        .{ "\"candidate_size\": 3", "\"candidate_size\": 4" },
        .{ "    3,", "    3.0," },
        .{ "    3,", "    \"3\"," },
        .{ "    3,", "    true," },
        .{ "    3,", "    -1," },
        .{ "    3,", "    0," },
        .{ "    3,", "    8," },
        .{ "    7\n", "    11\n" },
        .{ "    7\n", "    4294967296\n" },
        .{ "    0,\n    3,\n    7\n", "" },
        .{ "\"policy_version\": 1", "\"policy_version\": 2" },
    }) |pair| {
        const bad = try changed(pair[0], pair[1]);
        defer t.allocator.free(bad);
        try refuses(bad);
    }
    try refuses(fixture[0 .. fixture.len - 4]);
    try t.expectError(error.ArtifactSize, p.load(t.allocator, "", expected, &.{0}));
}

test "proposal vocabulary: ID checksums mandatory controls and provenance cannot silently drift" {
    const bad = try changed("    3,", "    2,");
    defer t.allocator.free(bad);
    try t.expectError(error.ChecksumMismatch, p.load(t.allocator, bad, expected, &.{0}));
    try t.expectError(error.MandatoryMismatch, p.load(t.allocator, fixture, expected, &.{ 0, 1 }));
    const missing = try changed("    0,\n    3,", "    1,\n    3,");
    defer t.allocator.free(missing);
    try t.expectError(error.MandatoryMissing, p.load(t.allocator, missing, expected, &.{0}));
    const source = try changed("\"mandatory_source\": \"synthetic-recipe\"", "\"mandatory_source\": \"other\"");
    defer t.allocator.free(source);
    try t.expectError(error.InvalidProvenance, p.load(t.allocator, source, expected, &.{0}));
}

test "proposal vocabulary: split bucket coverage is complete checked integer evidence" {
    for ([_][2][]const u8{
        .{ "\"bucket\": \"toy\"", "\"bucket\": \"*\"" },
        .{ "\"covered\": 3", "\"covered\": 5" },
        .{ "\"tokens\": 4", "\"tokens\": 0" },
        .{ "\"tokens\": 4", "\"tokens\": 4.0" },
        .{ "\"split\": \"heldout\"", "\"split\": \"calibration\"" },
    }) |pair| {
        const bad = try std.mem.replaceOwned(u8, t.allocator, fixture, pair[0], pair[1]);
        defer t.allocator.free(bad);
        try refuses(bad);
    }
}

fn allocationTrial(gpa: std.mem.Allocator) !void {
    var a = try p.load(gpa, fixture, expected, &.{0});
    defer a.deinit();
    try t.expectEqual(@as(usize, 3), a.ids().len);
}
test "proposal vocabulary: parser allocation failures release every owned artifact byte" {
    try t.checkAllAllocationFailures(t.allocator, allocationTrial, .{});
}

test "proposal vocabulary: planted same-size wrong-tokenizer artifact refuses before use" {
    if (p.load(t.allocator, @embedFile("testdata/proposal-vocab-bad.json"), expected, &.{0})) |loaded| {
        var bad = loaded;
        defer bad.deinit();
        return error.BadArtifactAccepted;
    } else |err| try t.expectEqual(error.IdentityMismatch, err);
}

test "proposal vocabulary: offline response artifact shares the parser ID hash and split coverage contract" {
    var a = try p.load(t.allocator, @embedFile("testdata/proposal-vocab-responses.json"), expected, &.{0});
    defer a.deinit();
    try t.expectEqualSlices(u32, &.{ 0, 3, 7 }, a.ids());
    try t.expectEqual(@as(u64, 0), a.owned.value.coverage[4].covered);
    try t.expectEqual(@as(u64, 4), a.owned.value.coverage[4].tokens);
    try t.expectEqual(@as(u32, 64), try a.physicalCount(64));
}
