//! Refusal text and canonical/legacy forms for every setting validated before DeepSeek startup.
const std = @import("std");
const e = @import("sys").environment;
const env_validate = @import("env_validate.zig");
const t = std.testing;

const Fixture = struct {
    name: []const u8 = "",
    raw: ?[]const u8 = null,
    canonical: ?[]const u8 = null,
    pub fn get(f: Fixture, key: []const u8) ?[]const u8 {
        if (std.mem.eql(u8, key, "TENSORFOLD_TOOL_GRAMMAR")) return f.canonical;
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
    const why = try env_validate.refusal(t.allocator, fx);
    defer if (why) |w| t.allocator.free(w);
    if (valid) {
        try env_validate.check(fx);
        return t.expect(why == null);
    }
    try t.expectError(error.InvalidEnvironment, env_validate.check(fx));
    try t.expect(std.mem.startsWith(u8, why.?, name));
    try t.expect(std.mem.indexOf(u8, why.?, raw) != null);
    try t.expect(std.mem.endsWith(u8, why.?, e.rule(name).forms));
}

test "DeepSeek strict env: the shared table holds all 82 settings: 49 switches, 14 integers, 16 reals and 3 structured" {
    var counts: [4]usize = @splat(0);
    for (e.rules) |r| if (r.scope == .dsv41) {
        try t.expect(std.mem.startsWith(u8, r.name, "TF_DSV41_"));
        counts[@backingInt(group(r))] += 1;
    };
    try t.expectEqual([4]usize{ 49, 14, 16, 3 }, counts);
}

test "DeepSeek strict env: unset and blank settings both read as the caller's default" {
    for (e.rules) |r| if (r.scope == .dsv41) for ([_]?[]const u8{ null, "", " ", "\t", " \t\r\n" }) |raw| {
        if (raw) |v| try check(r.name, v, true);
        switch (group(r)) {
            .switches => for ([_]bool{ false, true }) |d| try t.expectEqual(d, try e.value(bool, r.name, raw, d)),
            .reals => try t.expectEqual(@as(f64, 2), try e.value(f64, r.name, raw, 2)),
            else => if (r.kind == .range) try t.expect(try e.range(r.name, raw) == null) else try t.expectEqual(@as(u64, 2), try e.value(u64, r.name, raw, 2)),
        }
    };
}

test "DeepSeek strict env: every switch refuses garbage and keeps 0, 1 and its existing word forms" {
    for (e.rules) |r| if (r.scope == .dsv41 and group(r) == .switches) {
        try t.expect(!try e.value(bool, r.name, "0", true));
        try t.expect(try e.value(bool, r.name, "1", false));
        for ([_][]const u8{ "bad", "2", "00", "0 trailing", "truee", "-1" }) |raw| try check(r.name, raw, false);
        for ([_][]const u8{ "true", "on", "yes", "TRUE", "On", "YES" }) |raw| {
            try check(r.name, raw, r.kind == .toggle);
            if (r.kind == .toggle) try t.expect(try e.value(bool, r.name, raw, false));
        }
        for ([_][]const u8{ "false", "off", "no", "FALSE", "Off", "NO" }) |raw| {
            const off = r.kind == .toggle and (r.no_word or !std.ascii.eqlIgnoreCase(raw, "no"));
            try check(r.name, raw, off);
            if (off) try t.expect(!try e.value(bool, r.name, raw, true));
        }
    };
}

test "DeepSeek strict env: the word forms belong to exactly the switches that read words on main" {
    const words = [_][]const u8{ "MHC_DEFER", "COEF_LATE", "LOOKUP", "LOOKUP_PLAN" };
    var toggles: usize = 0;
    for (e.rules) |r| if (r.scope == .dsv41 and r.kind == .toggle) {
        toggles += 1;
    };
    try t.expectEqual(words.len, toggles);
    for (words) |suffix| {
        var buf: [64]u8 = undefined;
        try t.expectEqual(e.Kind.toggle, e.rule(try std.fmt.bufPrint(&buf, "TF_DSV41_{s}", .{suffix})).kind);
    }
}

test "DeepSeek strict env: every numeric refusal is named, not defaulted or clamped" {
    for (e.rules) |r| if (r.scope == .dsv41) switch (group(r)) {
        .integers => {
            try check(r.name, if (r.min == 16) "16" else "1", true);
            for ([_][]const u8{ "nan", "inf", "-1", "1.5", "1x", "18446744073709551616", " bad " }) |raw| try check(r.name, raw, false);
        },
        .reals => {
            try check(r.name, "1", true);
            for ([_][]const u8{ "nan", "-nan", "inf", "-inf", "1e999", "text", " bad " }) |raw| try check(r.name, raw, false);
        },
        else => {},
    };
    try check("TF_DSV41_ENGRAM_READERS", "32", true);
    try check("TF_DSV41_ENGRAM_READERS", "33", false);
    try check("TF_DSV41_GEN_STREAMS", "0", false);
    try check("TF_DSV41_FLOOR_HARD_GIB", "0", false);
    try check("TF_DSV41_PF_WS_FLOOR_GIB", "0", true);
    try check("TF_DSV41_GRAPH_COST_MIB", "0", true);
    try check("TF_DSV41_GRAPH_FLOOR_GIB", "3.25", true);
    try check("TF_DSV41_PREFILL_SHARE", "1.1", false);
    try check("TF_DSV41_EXPERT_TOPP", "-0.5", true);
}

test "DeepSeek strict env: layer ranges, row buckets and image modes refuse ignored tails" {
    for ([_][]const u8{ "0-0", "0-24", "1-2" }) |raw| try check("TF_DSV41_LAYERS", raw, true);
    for ([_][]const u8{ "0", "0-1-tail", "0--1", "2-1", "a-b" }) |raw| try check("TF_DSV41_LAYERS", raw, false);
    for ([_][]const u8{ "1", "2", "3", "4", "5", "6", "7", "8", "12", "16", "24", "32", "48", "64" }) |raw| try check("TF_DSV41_ROWS_CAP", raw, true);
    for ([_][]const u8{ "0", "9", "11", "17", "25", "49", "65", "bad" }) |raw| try check("TF_DSV41_ROWS_CAP", raw, false);
    for ([_][]const u8{ "native", " native ", "0" }) |raw| try check("TF_DSV41_IMAGES", raw, true);
    for ([_][]const u8{ "nativee", "1" }) |raw| try check("TF_DSV41_IMAGES", raw, false);
}

test "DeepSeek strict env: the row-cap list is exactly the batch's buckets" {
    const only = e.rule("TF_DSV41_ROWS_CAP").only;
    for (1..65) |n| {
        const rows: u32 = @intCast(n);
        try t.expectEqual(@import("rowtab.zig").bucketRows(rows, 64) == rows, std.mem.indexOfScalar(u64, only, n) != null);
    }
}

test "DeepSeek strict env: the canonical tool grammar masks its legacy alias" {
    const fx: Fixture = .{ .name = "TF_DSV41_TOOL_GRAMMAR", .raw = "garbage", .canonical = "0" };
    try e.checkScope(.dsv41, fx);
    try t.expect(try e.refusal(t.allocator, fx, .{ .dsv41 = true }) == null);
    try t.expectError(error.InvalidEnvironment, e.checkScope(.dsv41, Fixture{ .name = fx.name, .raw = fx.raw }));
}

test "DeepSeek strict env: copy-drafter lookup-off refusal names TF_DSV41_COPY_DRAFTER and TF_DSV41_LOOKUP" {
    const Pair = struct {
        lookup: []const u8,
        pub fn get(p: @This(), key: []const u8) ?[]const u8 {
            if (std.mem.eql(u8, key, "TF_DSV41_COPY_DRAFTER")) return "1";
            if (std.mem.eql(u8, key, "TF_DSV41_LOOKUP")) return p.lookup;
            return null;
        }
    };
    for ([_][]const u8{ "0", "off", "false", "", " " }) |off| {
        const pair: Pair = .{ .lookup = off };
        try t.expectError(error.InvalidEnvironment, env_validate.check(pair));
        const why = (try env_validate.refusal(t.allocator, pair)).?;
        defer t.allocator.free(why);
        try t.expect(std.mem.indexOf(u8, why, "TF_DSV41_COPY_DRAFTER") != null);
        try t.expect(std.mem.indexOf(u8, why, "TF_DSV41_LOOKUP") != null);
    }
    try env_validate.check(Pair{ .lookup = "1" });
    try t.expect(try env_validate.refusal(t.allocator, Pair{ .lookup = "1" }) == null);
}
