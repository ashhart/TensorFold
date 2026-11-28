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
    return if (r.kind == .word or r.kind == .path or r.kind == .range or r.only.len > 0) .structured else .integers;
}

fn check(name: []const u8, raw: []const u8, valid: bool) !void {
    const fx: Fixture = .{ .name = name, .raw = raw };
    const why = try e.refusal(t.allocator, fx, .{ .flashnext = true });
    defer if (why) |w| t.allocator.free(w);
    if (valid) {
        try e.checkScope(.flashnext, fx);
        return t.expect(why == null);
    }
    try t.expectError(error.InvalidEnvironment, e.parse(name, raw));
    try t.expect(std.mem.startsWith(u8, why.?, name));
    try t.expect(std.mem.indexOf(u8, why.?, raw) != null);
    try t.expect(std.mem.endsWith(u8, why.?, e.rule(name).forms));
}

test "Flash Next strict env: the table holds all 55 settings: 34 switches, 13 integers, 3 reals, 5 structured" {
    var counts: [4]usize = @splat(0);
    for (e.rules) |r| if (r.scope == .flashnext) {
        try t.expect(std.mem.startsWith(u8, r.name, "TF_FLASHNEXT_") or std.mem.startsWith(u8, r.name, "FZ_") or std.mem.eql(u8, r.name, "TENSORFOLD_CHECKPOINT_SHA256"));
        counts[@backingInt(group(r))] += 1;
    };
    try t.expectEqual([4]usize{ 34, 13, 3, 5 }, counts);
}

test "Flash Next strict env: unset reads the caller default and every row keeps its blank contract" {
    for (e.rules) |r| if (r.scope == .flashnext) for ([_]?[]const u8{ null, "", " ", "\t", " \t\r\n" }) |raw| {
        const blank_ok = raw == null or r.blank_default;
        if (raw) |v| {
            if (blank_ok) {
                try check(r.name, v, true);
            } else {
                const fx: Fixture = .{ .name = r.name, .raw = v };
                try t.expectError(error.InvalidEnvironment, e.checkScope(.flashnext, fx));
            }
        }
        if (!blank_ok) {
            try t.expectError(error.InvalidEnvironment, e.parse(r.name, raw));
            continue;
        }
        switch (group(r)) {
            .switches => for ([_]bool{ false, true }) |d| try t.expectEqual(d, try e.value(bool, r.name, raw, d)),
            .reals => try t.expectEqual(@as(f64, 2), try e.value(f64, r.name, raw, 2)),
            .structured => if (r.kind == .path) try t.expect(try e.path(r.name, raw) == null) else try t.expect(try e.value(?u64, r.name, raw, null) == null),
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

test "Flash Next strict env: the dense split, head shard and fused read-out switches keep their own defaults, blank as unset" {
    const Switch = struct { name: [:0]const u8, default: bool };
    for ([_]Switch{
        .{ .name = "TF_FLASHNEXT_SPLIT_DENSE", .default = true },
        .{ .name = "TF_FLASHNEXT_SHARD_HEAD", .default = false },
        .{ .name = "TF_FLASHNEXT_READOUT_FUSED", .default = true },
    }) |s| {
        // unset, empty and whitespace-only all read as unset, so each keeps the default its reader passes
        for ([_]?[]const u8{ null, "", " ", "\t", " \t\r\n" }) |raw| {
            if (raw) |v| try check(s.name, v, true);
            try t.expectEqual(s.default, try e.value(bool, s.name, raw, s.default));
        }
        try t.expect(!try e.value(bool, s.name, "0", true));
        try t.expect(try e.value(bool, s.name, "1", false));
        for ([_][]const u8{ "2", "00", "1 ", "-1", "1.0", "yes", "on", "true", "x" }) |raw| try check(s.name, raw, false);
    }
    try t.expectEqualStrings("0 or 1 (default 1)", e.rule("TF_FLASHNEXT_READOUT_FUSED").forms);
}

test "Flash Next strict env: every count keeps its whole numbers and refuses junk, negatives, overflow" {
    for (e.rules) |r| if (r.scope == .flashnext and group(r) == .integers) {
        try check(r.name, "0", r.min == 0);
        for ([_][]const u8{ "7", "007" }) |raw| try check(r.name, raw, true);
        try t.expectEqual(@as(u64, 7), try e.value(u64, r.name, "007", 0));
        for ([_][]const u8{ "-1", "1.5", "abc", "18446744073709551616" }) |raw| try check(r.name, raw, false);
    };
}

test "Flash Next strict env: chunk rows take 1 to 16384 and refuse the rest by name" {
    const chunk = "TF_FLASHNEXT_CHUNK";
    const solo = "TF_FLASHNEXT_CHUNK_SOLO";
    for ([_][]const u8{ "1", "1536", "16384" }) |raw| try check(chunk, raw, true);
    for ([_][]const u8{ "1", "8192", "16384" }) |raw| try check(solo, raw, true);
    for ([_][]const u8{ "0", "16385", "4k", "-1" }) |raw| try check(chunk, raw, false);
    for ([_][]const u8{ "16385", "4k", "-1" }) |raw| try check(solo, raw, false);
}

test "Flash Next strict env: the page test keeps 0 in the table while its reader refuses it, blank reads unset" {
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

test "Flash Next strict env: the draft head keeps bf16 and affine4 in the reader's order and refuses other text" {
    const name = "TF_FLASHNEXT_DRAFT_HEAD";
    try check(name, "bf16", true);
    try check(name, "affine4", true);
    // the words' order is the reader's: bf16 is 0 and affine4 is 1, its default
    try t.expectEqual(@as(u1, 0), try e.value(u1, name, "bf16", 1));
    try t.expectEqual(@as(u1, 1), try e.value(u1, name, "affine4", 0));
    for ([_][]const u8{ "BF16", "affine", "fp8", "1", "affine4 " }) |raw| try check(name, raw, false);
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
    for ([_][:0]const u8{ "TF_FLASHNEXT_GPU_TOPK", "TF_FLASHNEXT_SPLIT_DENSE", "TF_FLASHNEXT_SHARD_HEAD" }) |name| {
        const fx: Fixture = .{ .name = name, .raw = "yes" };
        const why = try e.refusal(t.allocator, fx, .{ .flashnext = false });
        defer if (why) |w| t.allocator.free(w);
        try t.expect(why == null);
        try t.expectError(error.InvalidEnvironment, e.checkScope(.flashnext, fx));
    }
}
fn checkPath(name: []const u8, raw: []const u8, valid: bool) !void {
    const fx: Fixture = .{ .name = name, .raw = raw };
    const why = try e.refusal(t.allocator, fx, .{ .flashnext = true });
    defer if (why) |w| t.allocator.free(w);
    if (valid) {
        try e.checkScope(.flashnext, fx);
        return t.expect(why == null);
    }
    try t.expectError(error.InvalidEnvironment, e.parse(name, raw));
    try t.expect(std.mem.startsWith(u8, why.?, name));
    try t.expect(std.mem.endsWith(u8, why.?, e.rule(name).forms));
}

const Moved = struct { name: []const u8, good: []const u8, bad: []const u8 };
const moved = [_]Moved{
    .{ .name = "FZ_BATCH_MTP_CHAIN", .good = "1", .bad = "2" },
    .{ .name = "FZ_BATCH_MTP_SERIAL", .good = "1", .bad = "2" },
    .{ .name = "FZ_VERIFY_OVERLAP", .good = "0", .bad = "2" },
    .{ .name = "FZ_SETTLE_SYNC", .good = "1", .bad = "2" },
    .{ .name = "TF_FLASHNEXT_PROMPT_EXPERT_TILE", .good = "2", .bad = "3" },
    .{ .name = "TF_FLASHNEXT_SPLIT_DENSE", .good = "0", .bad = "2" },
    .{ .name = "TF_FLASHNEXT_SHARD_HEAD", .good = "1", .bad = "2" },
    .{ .name = "TF_FLASHNEXT_LOAD_THREADS", .good = "32", .bad = "33" },
    .{ .name = "TF_FLASHNEXT_TEMPERATURE", .good = "0.8", .bad = "nan" },
    .{ .name = "TF_FLASHNEXT_TOP_P", .good = "0.9", .bad = "inf" },
    .{ .name = "TF_FLASHNEXT_TOP_K", .good = "20", .bad = "-1" },
    .{ .name = "TF_FLASHNEXT_REPEAT", .good = "2", .bad = "abc" },
    .{ .name = "TF_FLASHNEXT_SEED", .good = "7", .bad = "x" },
    .{ .name = "TF_FLASHNEXT_DRAFT_HEAD", .good = "affine4", .bad = "fp8" },
    .{ .name = "TF_FLASHNEXT_PLE_IO", .good = "pread", .bad = "direct" },
    .{ .name = "TENSORFOLD_CHECKPOINT_SHA256", .good = "abababababababababababababababababababababababababababababababab", .bad = "ab\nab" },
    .{ .name = "TENSORFOLD_PREFILL_SHARE", .good = "0.5", .bad = "1.5" },
};

test "Flash Next migrated settings: unset, one valid value and one malformed value per row" {
    for (moved) |m| {
        const unset: Fixture = .{ .name = m.name };
        try e.checkScope(.flashnext, unset);
        try t.expect(try e.refusal(t.allocator, unset, .{ .flashnext = true }) == null);
        switch (e.rule(m.name).kind) {
            .path => {
                try checkPath(m.name, m.good, true);
                try checkPath(m.name, m.bad, false);
            },
            else => {
                try check(m.name, m.good, true);
                try check(m.name, m.bad, false);
            },
        }
    }
}

test "Flash Next migrated shared names keep checkpoint scope and common prefill validation" {
    const other: e.Flags = .{ .cuda = true, .metal = true, .xpu = true, .dsv41 = true, .glm = true };
    const checkpoint: Fixture = .{ .name = "TENSORFOLD_CHECKPOINT_SHA256", .raw = "bad\npath" };
    const outside = try e.refusal(t.allocator, checkpoint, other);
    defer if (outside) |why| t.allocator.free(why);
    try t.expect(outside == null);
    const inside = (try e.refusal(t.allocator, checkpoint, .{ .flashnext = true })).?;
    defer t.allocator.free(inside);
    try t.expect(std.mem.startsWith(u8, inside, checkpoint.name));

    const share: Fixture = .{ .name = "TENSORFOLD_PREFILL_SHARE", .raw = "0" };
    const common = (try e.refusal(t.allocator, share, other)).?;
    defer t.allocator.free(common);
    try t.expect(std.mem.startsWith(u8, common, share.name));
    const flash = (try e.refusal(t.allocator, share, .{ .flashnext = true })).?;
    defer t.allocator.free(flash);
    try t.expect(std.mem.startsWith(u8, flash, share.name));
}
