//! Refusal text and kept forms for every Flash Next setting checked before its device or checkpoint work.
const std = @import("std");
const e = @import("environment.zig");
const t = std.testing;

const Fixture = struct {
    name: []const u8 = "",
    raw: ?[]const u8 = null,
    pub fn get(f: Fixture, key: []const u8) ?[]const u8 {
        return if (std.mem.eql(u8, key, f.name)) f.raw else null;
    }
};

const Group = enum { switches, integers, reals, structured };

fn group(r: e.Rule) Group {
    if (r.kind == .flag or r.kind == .toggle) return .switches;
    if (e.isReal(r.kind)) return .reals;
    return if (r.kind == .word or r.kind == .range or r.only.len > 0) .structured else .integers;
}

fn check(name: []const u8, raw: []const u8, valid: bool) !void {
    const fx: Fixture = .{ .name = name, .raw = raw };
    const why = try e.refusal(t.allocator, fx, .{ .flashnext = true });
    defer if (why) |w| t.allocator.free(w);
    if (valid) {
        try e.checkScope(.flashnext, fx);
        return t.expect(why == null);
    }
    try t.expectError(error.InvalidEnvironment, e.checkScope(.flashnext, fx));
    try t.expect(std.mem.startsWith(u8, why.?, name));
    try t.expect(std.mem.indexOf(u8, why.?, raw) != null);
    try t.expect(std.mem.endsWith(u8, why.?, e.rule(name).forms));
}

test "Flash Next strict env: the table holds all 29 settings: 20 switches, 7 integers, 1 real, 1 structured" {
    var counts: [4]usize = @splat(0);
    for (e.rules) |r| if (r.scope == .flashnext) {
        try t.expect(std.mem.startsWith(u8, r.name, "TF_FLASHNEXT_"));
        counts[@backingInt(group(r))] += 1;
    };
    try t.expectEqual([4]usize{ 20, 7, 1, 1 }, counts);
}

test "Flash Next strict env: unset and blank settings both read as the caller's default" {
    for (e.rules) |r| if (r.scope == .flashnext) for ([_]?[]const u8{ null, "", " ", "\t", " \t\r\n" }) |raw| {
        if (raw) |v| try check(r.name, v, true);
        switch (group(r)) {
            .switches => for ([_]bool{ false, true }) |d| try t.expectEqual(d, try e.value(bool, r.name, raw, d)),
            .reals => try t.expectEqual(@as(f64, 2), try e.value(f64, r.name, raw, 2)),
            .structured => try t.expect(try e.value(?u64, r.name, raw, null) == null),
            .integers => try t.expectEqual(@as(u64, 2), try e.value(u64, r.name, raw, 2)),
        }
    };
}

test "Flash Next strict env: every switch keeps exactly 0 and 1 and refuses the word forms" {
    for (e.rules) |r| if (r.scope == .flashnext and group(r) == .switches) {
        try t.expect(!try e.value(bool, r.name, "0", true));
        try t.expect(try e.value(bool, r.name, "1", false));
        for ([_][]const u8{ "2", "yes", "on", "true", "x", "00", "-1", "1 " }) |raw| try check(r.name, raw, false);
    };
}

test "Flash Next strict env: every count keeps its whole numbers and refuses junk, negatives, overflow" {
    for (e.rules) |r| if (r.scope == .flashnext and group(r) == .integers) {
        for ([_][]const u8{ "0", "7", "007" }) |raw| try check(r.name, raw, true);
        try t.expectEqual(@as(u64, 7), try e.value(u64, r.name, "007", 0));
        for ([_][]const u8{ "-1", "1.5", "abc", "18446744073709551616" }) |raw| try check(r.name, raw, false);
    };
}

test "Flash Next strict env: the page test keeps 0 in the table, which its reader reads as off" {
    const name = "TF_FLASHNEXT_KV_PAGE_TEST";
    try check(name, "0", true);
    try check(name, "48", true);
    try t.expectEqual(@as(?usize, 0), try e.value(?usize, name, "0", null));
    try t.expect((try e.value(?usize, name, null, null)) == null);
}

test "Flash Next strict env: the chain's confidence is any finite number, as its reader takes it" {
    const name = "TF_FLASHNEXT_CONFIDENCE";
    const Kept = struct { raw: []const u8, want: f64 };
    for ([_]Kept{
        .{ .raw = "-0.4", .want = -0.4 },
        .{ .raw = "-1", .want = -1 },
        .{ .raw = "0", .want = 0 },
        .{ .raw = "0.6", .want = 0.6 },
        .{ .raw = "1e-3", .want = 0.001 },
    }) |kept| {
        try check(name, kept.raw, true);
        try t.expectEqual(kept.want, try e.value(f64, name, kept.raw, 2));
    }
    for ([_][]const u8{ "nan", "inf", "-inf", "x", "0.4.1" }) |raw| try check(name, raw, false);
}

test "Flash Next strict env: the prompt word keeps rows as a present index and refuses every other text" {
    const name = "TF_FLASHNEXT_PROMPT";
    try check(name, "rows", true);
    for ([_][]const u8{ "ROWS", "row", "1" }) |raw| try check(name, raw, false);
    // rows is the only word, so it decodes to index 0: present, which the reader reads as the rows path
    try t.expect((try e.value(?u64, name, "rows", null)) != null);
    try t.expect((try e.value(?u64, name, null, null)) == null);
}

test "Flash Next strict env: the late keep reads unset and blank as null, and 0 and 1 as false and true" {
    const name = "TF_FLASHNEXT_KEEP_LATE";
    for ([_]?[]const u8{ null, "", " ", "\t", " \t\r\n" }) |raw|
        try t.expectEqual(@as(?bool, null), try e.value(?bool, name, raw, null));
    try t.expectEqual(@as(?bool, false), try e.value(?bool, name, "0", null));
    try t.expectEqual(@as(?bool, true), try e.value(?bool, name, "1", null));
    try check(name, "0", true);
    try check(name, "1", true);
}

test "Flash Next strict env: a refusal names the setting, quotes its text, and ends with the accepted forms" {
    const name = "TF_FLASHNEXT_DRAFTS";
    const fx: Fixture = .{ .name = name, .raw = "abc" };
    try t.expectError(error.InvalidEnvironment, e.checkScope(.flashnext, fx));
    const why = (try e.refusal(t.allocator, fx, .{ .flashnext = true })).?;
    defer t.allocator.free(why);
    try t.expect(std.mem.startsWith(u8, why, name));
    try t.expect(std.mem.indexOf(u8, why, "abc") != null);
    try t.expect(std.mem.endsWith(u8, why, e.rule(name).forms));
}

test "Flash Next strict env: the scope refuses only its own settings" {
    const fx: Fixture = .{ .name = "TF_FLASHNEXT_GPU_TOPK", .raw = "yes" };
    const why = try e.refusal(t.allocator, fx, .{ .flashnext = false });
    defer if (why) |w| t.allocator.free(w);
    try t.expect(why == null);
    try t.expectError(error.InvalidEnvironment, e.checkScope(.flashnext, fx));
}
