//! Refusal text and canonical/legacy forms for every setting validated before DeepSeek startup.
const std = @import("std");
const v = @import("env_validate.zig");
const strict = @import("sys").strict_env;
const t = std.testing;
const Fixture = struct {
    var name: []const u8 = "";
    var value: ?[]const u8 = null;
    var canonical: ?[]const u8 = null;
    fn get(key: [:0]const u8) ?[]const u8 {
        if (std.mem.eql(u8, key, "TENSORFOLD_TOOL_GRAMMAR")) return canonical;
        return if (std.mem.eql(u8, key, Fixture.name)) value else null;
    }
};

fn check(name: []const u8, raw: []const u8, valid: bool) !void {
    Fixture.name = name;
    Fixture.value = raw;
    var p: strict.Parser = .{};
    if (valid) {
        try v.validate(&p, Fixture.get);
        try t.expect(p.problem == null);
    } else {
        try t.expectError(error.BadEnvironmentValue, v.validate(&p, Fixture.get));
        try t.expectEqualStrings(name, p.problem.?.name);
        try t.expectEqualStrings(raw, p.problem.?.value);
        var buf: [512]u8 = undefined;
        const text = try p.problem.?.message(&buf);
        try t.expect(std.mem.startsWith(u8, text, name));
        try t.expect(std.mem.indexOf(u8, text, raw) != null);
        try t.expect(p.problem.?.accepted.len > 0);
    }
}

test "DeepSeek strict env: all absent settings preserve the caller defaults" {
    Fixture.name = "";
    Fixture.value = null;
    var p: strict.Parser = .{};
    try v.validate(&p, Fixture.get);
    for (v.flags) |f| for ([_]bool{ false, true }) |fallback|
        try t.expectEqual(fallback, try p.flag(f.name, null, fallback, f.mode));
    for (v.reals) |n| try t.expectEqual(@as(f64, 2), try p.real(n.name, null, 2, n.min, n.max, "range"));
    for (v.integers) |n| try t.expectEqual(@as(u64, 2), try p.integer(u64, n.name, null, 2, n.min, n.max, "range"));
}

test "DeepSeek strict env: every boolean rejects garbage and retains canonical values and existing aliases" {
    for (v.flags) |f| {
        for ([_][]const u8{ "0", "1" }) |raw| try check(f.name, raw, true);
        for ([_][]const u8{ "bad", "2", "00", "0 trailing", "truee", "-1" }) |raw| try check(f.name, raw, false);
        try check(f.name, "", f.mode.empty_off);
        if (f.mode.words) {
            var p: strict.Parser = .{};
            for ([_][]const u8{ "true", "on", "TRUE", "On" }) |raw| {
                try check(f.name, raw, true);
                try t.expect(try p.flag(f.name, raw, false, f.mode));
            }
            for ([_][]const u8{ "false", "off", "FALSE", "Off" }) |raw| {
                try check(f.name, raw, true);
                try t.expect(!try p.flag(f.name, raw, true, f.mode));
            }
        } else for ([_][]const u8{ "true", "false", "on", "off" }) |raw| try check(f.name, raw, false);
        if (f.mode.no) {
            try check(f.name, "yes", true);
            try check(f.name, "NO", true);
        }
    }
}

test "DeepSeek strict env: every numeric refusal is named, not defaulted or clamped" {
    for (v.integers) |n| {
        try check(n.name, n.sample, true);
        for ([_][]const u8{ "", "nan", "inf", "-1", "1.5", "1x", "18446744073709551616" }) |raw| try check(n.name, raw, false);
    }
    try check("TF_DSV41_ENGRAM_READERS", "32", true);
    try check("TF_DSV41_ENGRAM_READERS", "33", false);
    try check("TF_DSV41_GEN_STREAMS", "0", false);
    for (v.reals) |n| {
        try check(n.name, n.sample, true);
        for ([_][]const u8{ "", "nan", "-nan", "inf", "-inf", "1e999", "text" }) |raw| try check(n.name, raw, false);
    }
    try check("TF_DSV41_FLOOR_HARD_GIB", "0", false);
    try check("TF_DSV41_PF_WS_FLOOR_GIB", "0", true);
    try check("TF_DSV41_GRAPH_COST_MIB", "0", true);
    try check("TF_DSV41_GRAPH_FLOOR_GIB", "3.25", true);
    try check("TF_DSV41_PREFILL_SHARE", "1.1", false);
}

test "DeepSeek strict env: layer ranges, row buckets and image modes refuse ignored tails" {
    for ([_][]const u8{ "0-0", "0-24", "1-2" }) |raw| try check("TF_DSV41_LAYERS", raw, true);
    for ([_][]const u8{ "", "0", "0-1-tail", "0--1", "2-1", "a-b" }) |raw| try check("TF_DSV41_LAYERS", raw, false);
    for ([_][]const u8{ "1", "2", "3", "4", "5", "6", "7", "8", "12", "16", "24", "32", "48", "64" }) |raw| try check("TF_DSV41_ROWS_CAP", raw, true);
    for ([_][]const u8{ "0", "9", "11", "17", "25", "49", "65", "bad" }) |raw| try check("TF_DSV41_ROWS_CAP", raw, false);
    for ([_][]const u8{ "native", " native ", "0" }) |raw| try check("TF_DSV41_IMAGES", raw, true);
    for ([_][]const u8{ "", "nativee", "1" }) |raw| try check("TF_DSV41_IMAGES", raw, false);
}

test "DeepSeek strict env: nonzero flags keep on words and recognize off words" {
    for ([_][]const u8{ "MHC_DEFER", "COEF_LATE", "GM_V3", "OWN_PREFILL", "DRAFTS", "GRAPHS", "AOT_GUARD", "ENGRAM_PREFETCH", "FLOOR_PRICE", "SESSIONS", "PREFILL_ADAPT", "ENGRAM_URING", "PREFETCH_AHEAD", "SCHEMA_PROMPT", "PIECE_RUNS", "REPLAY_RUNS", "KV_SPLIT" }) |suffix| {
        var buf: [64]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, "TF_DSV41_{s}", .{suffix});
        const flag = for (v.flags) |f| {
            if (std.mem.eql(u8, name, f.name)) break f;
        } else return error.MissingFlag;
        var p: strict.Parser = .{};
        for ([_][]const u8{ "1", "true", "on", "yes", "TRUE", "On", "YES" }) |raw| {
            try check(name, raw, true);
            try t.expect(try p.flag(name, raw, false, flag.mode));
        }
        for ([_][]const u8{ "0", "false", "off", "no", "FALSE", "Off", "NO" }) |raw| {
            try check(name, raw, true);
            try t.expect(!try p.flag(name, raw, true, flag.mode));
        }
        try check(name, "mistyped", false);
    }
}

test "DeepSeek strict env: trimmed numeric refusals retain raw text and canonical tool grammar masks its alias" {
    for (v.integers) |n| try check(n.name, " bad ", false);
    for (v.reals) |n| try check(n.name, " bad ", false);
    Fixture.name = "TF_DSV41_TOOL_GRAMMAR";
    Fixture.value = "garbage";
    Fixture.canonical = "0";
    defer Fixture.canonical = null;
    var p: strict.Parser = .{};
    try v.validate(&p, Fixture.get);
    try t.expect(p.problem == null);
}
