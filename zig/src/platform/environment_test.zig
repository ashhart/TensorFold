//! Startup refusals carry names, values and accepted forms; valid settings preserve defaults and clamps.
const std = @import("std");
const e = @import("environment.zig");
const t = std.testing;
const a = t.allocator;
const all: e.Flags = .{ .cuda = true, .metal = true, .xpu = true, .slide = true, .hooks = true };
const Env = struct {
    name: []const u8,
    raw: ?[]const u8,
    pub fn get(v: Env, name: []const u8) ?[]const u8 {
        if (std.mem.eql(u8, name, v.name)) return v.raw;
        if (std.mem.eql(u8, name, "TF_CUDA_CARVEOUT")) return "1";
        return null;
    }
};

test "environment: every strict reader refuses bad text before opening a model, naming value and accepted forms" {
    for (e.rules) |r| {
        try t.expectEqual(@as(?u64, null), try e.parse(r.name, null));
        for ([_][]const u8{ "typo", "-1", "18446744073709551616", "1\nwrong" }) |raw| {
            try t.expectError(error.InvalidEnvironment, e.parse(r.name, raw));
            const why = (try e.refusal(a, Env{ .name = r.name, .raw = raw }, all)).?;
            defer a.free(why);
            try t.expect(std.mem.indexOf(u8, why, r.name) != null);
            try t.expect(std.mem.indexOf(u8, why, r.forms) != null);
            if (std.mem.indexOfScalar(u8, raw, '\n') == null) try t.expect(std.mem.indexOf(u8, why, raw) != null);
            for (why) |byte| try t.expect(byte >= 0x20 and byte <= 0x7e);
        }
        const untouched = try e.refusal(a, Env{ .name = r.name, .raw = null }, all);
        try t.expect(untouched == null);
    }
}

const Case = struct { name: []const u8, raw: []const u8, n: u64 };
const valid = [_]Case{
    .{ .name = "TENSORFOLD_VIDEO_TOKENS", .raw = "0", .n = 0 },
    .{ .name = "TENSORFOLD_VIDEO_TOKENS", .raw = "18446744073709551615", .n = std.math.maxInt(u64) },
    .{ .name = "TENSORFOLD_MAX_PDF_PAGES", .raw = "4294967295", .n = std.math.maxInt(u32) },
    .{ .name = "TENSORFOLD_PRESSURE_CACHE_PERCENT", .raw = "\t100 ", .n = 100 },
    .{ .name = "TENSORFOLD_PRESSURE_CACHE_PERCENT", .raw = "0", .n = 0 },
    .{ .name = "TENSORFOLD_PREFIX_WAIT_MS", .raw = "\t4294967295 ", .n = std.math.maxInt(u32) },
    .{ .name = "TENSORFOLD_RESTORE_PREFETCH", .raw = " 0\t", .n = 0 },
    .{ .name = "TENSORFOLD_RESTORE_PREFETCH", .raw = "65537", .n = 65537 },
    .{ .name = "TENSORFOLD_RESTORE_PREFETCH_GIB", .raw = "16777217", .n = 16777217 },
    .{ .name = "TENSORFOLD_RESTORE_PREFETCH_MS", .raw = "1099511627777", .n = 1099511627777 },
    .{ .name = "TF_CUDA_DEVICE", .raw = " 3 ", .n = 3 },
    .{ .name = "TF_CUDA_DEVICE", .raw = "4294967295", .n = std.math.maxInt(u32) },
    .{ .name = "TF_CUDA_SEGMENTS", .raw = " 1 ", .n = 1 },
    .{ .name = "TF_CUDA_SEGMENTS", .raw = "4", .n = 4 },
    .{ .name = "TF_CUDA_CARVEOUT", .raw = "0", .n = 0 },
    .{ .name = "TF_CUDA_CARVEOUT", .raw = "1", .n = 1 },
    .{ .name = "TF_CUDA_CARVEOUT_MIB", .raw = "1792", .n = 1792 },
    .{ .name = "TF_MTP_DRAFTS", .raw = "0", .n = 0 },
    .{ .name = "TF_MTP_DRAFTS", .raw = "4294967295", .n = std.math.maxInt(u32) },
    .{ .name = "TF_DRAFT_NODES", .raw = "0", .n = 0 },
    .{ .name = "TF_DRAFT_NODES", .raw = "64", .n = 64 },
    .{ .name = "TF_QMMF_HALF_BELOW", .raw = "0", .n = 0 },
    .{ .name = "TF_ROCE_GID_INDEX", .raw = "2147483647", .n = 2147483647 },
    .{ .name = "TF_Q27_COPY_ROWS", .raw = "0", .n = 0 },
    .{ .name = "TF_Q27_COPY_ROWS", .raw = "48", .n = 48 },
    .{ .name = "TF_Q27_DRAFT_SLOTS", .raw = "0", .n = 0 },
    .{ .name = "TF_Q27_DRAFT_SLOTS", .raw = "4294967295", .n = std.math.maxInt(u32) },
    .{ .name = "TF_HEAD_GEOMETRY", .raw = "0", .n = 0 },
    .{ .name = "TF_PROMPT_ROWS", .raw = "128", .n = 128 },
    .{ .name = "TF_PROMPT_ROWS", .raw = "2048", .n = 2048 },
    .{ .name = "TF_ANE_COLUMNS", .raw = "0", .n = 0 },
    .{ .name = "TF_ANE_COLUMNS", .raw = "3072", .n = 3072 },
    .{ .name = "XPU_A8_BITS", .raw = "4", .n = 4 },
    .{ .name = "XPU_A8_BITS", .raw = "8", .n = 8 },
    .{ .name = "LFM3_FUSE", .raw = "7", .n = 7 },
    .{ .name = "TF_SLIDE_POCKET_KEY", .raw = "auto", .n = 0 },
    .{ .name = "TF_SLIDE_POCKET_KEY", .raw = "reply", .n = 1 },
    .{ .name = "TF_SLIDE_POCKET_KEY", .raw = "subject", .n = 2 },
    .{ .name = "TF_SLIDE_POCKET_KEY", .raw = "both", .n = 3 },
    .{ .name = "TF_SLIDE_FIT_GIB", .raw = "4", .n = 4 },
    .{ .name = "TF_SLIDE_FORGET_FAIL", .raw = " 0 ", .n = 0 },
};

test "environment: valid forms remain raw numbers before existing consumer clamps and defaults" {
    for (valid) |c| {
        try t.expectEqual(@as(?u64, c.n), try e.parse(c.name, c.raw));
        const why = try e.refusal(a, Env{ .name = c.name, .raw = c.raw }, all);
        defer if (why) |v| a.free(v);
        try t.expect(why == null);
    }
    for (e.rules) |r| {
        var covered = false;
        for (valid) |c| if (std.mem.eql(u8, r.name, c.name)) {
            covered = true;
        };
        try t.expect(covered);
        for ([_][]const u8{ "", " ", "\t \n" }) |raw| try t.expectEqual(@as(?u64, null), try e.parse(r.name, raw));
    }
}

test "environment: range, enum, whitespace and native prompt geometry refusals are explicit" {
    for ([_][2][]const u8{
        .{ "TENSORFOLD_MAX_PDF_PAGES", "0" },   .{ "TENSORFOLD_PRESSURE_CACHE_PERCENT", "101" },
        .{ "TENSORFOLD_PREFIX_WAIT_MS", "+1" }, .{ "TENSORFOLD_RESTORE_PREFETCH", "+1" },
        .{ "TF_CUDA_SEGMENTS", "0" },           .{ "TF_CUDA_SEGMENTS", "5" },
        .{ "TF_CUDA_CARVEOUT", "true" },        .{ "TF_CUDA_CARVEOUT", "01" },
        .{ "TF_CUDA_CARVEOUT_MIB", "0" },       .{ "TF_CUDA_CARVEOUT_MIB", "18446744073709551615" },
        .{ "TF_ROCE_GID_INDEX", "2147483648" }, .{ "TF_PROMPT_ROWS", "16" },
        .{ "TF_PROMPT_ROWS", "129" },           .{ "TF_PROMPT_ROWS", "2064" },
        .{ "TF_PROMPT_ROWS", " 512" },          .{ "XPU_A8_BITS", "04" },
        .{ "XPU_A8_BITS", "16" },               .{ "TF_SLIDE_POCKET_KEY", "AUTO" },
    }) |c| try t.expectError(error.InvalidEnvironment, e.parse(c[0], c[1]));
    try t.expectEqual(@as(?u64, null), try e.parse("TF_CUDA_DEVICE", "   "));
    try t.expectEqual(@as(?u64, 0), try e.parse("TF_Q27_COPY_ROWS", "+0"));
}

test "environment: explicit flags and disabled features retain their old precedence" {
    try t.expectEqual(@as(u32, 7), try e.device(7, "bad"));
    try t.expectEqual(@as(u32, 3), try e.device(null, " 3 "));
    try t.expectEqual(@as(u32, 0), try e.device(null, null));
    try t.expectEqual(@as(u32, 0), try e.device(null, ""));
    var f = all;
    f.device = 3;
    try t.expect((try e.refusal(a, Env{ .name = "TF_CUDA_DEVICE", .raw = "bad" }, f)) == null);
    f.segments = 2;
    try t.expect((try e.refusal(a, Env{ .name = "TF_CUDA_SEGMENTS", .raw = "bad" }, f)) == null);
    try t.expect((try e.refusal(a, Env{ .name = "TF_CUDA_DEVICE", .raw = "bad" }, .{})) == null);
    try t.expect((try e.refusal(a, Env{ .name = "TF_SLIDE_FORGET_FAIL", .raw = "bad" }, .{})) == null);
}

const Inactive = struct {
    pub fn get(_: Inactive, name: []const u8) ?[]const u8 {
        if (std.mem.eql(u8, name, "TF_CUDA_CARVEOUT")) return "0";
        if (std.mem.eql(u8, name, "TF_CUDA_CARVEOUT_MIB")) return "bad-size";
        if (std.mem.eql(u8, name, "TF_NO_TREES")) return "";
        if (std.mem.eql(u8, name, "TF_DRAFT_NODES")) return "bad-nodes";
        return null;
    }
};

test "environment: inactive carveout size and disabled trees are still ignored" {
    try t.expect((try e.refusal(a, Inactive{}, all)) == null);
    var enabled = all;
    enabled.carveout = true;
    const why = (try e.refusal(a, Inactive{}, enabled)).?;
    defer a.free(why);
    try t.expect(std.mem.indexOf(u8, why, "TF_CUDA_CARVEOUT_MIB=\"bad-size\"") != null);
}

test "environment: video refusal describes the token budget without advertising a storage maximum" {
    const why = try e.describe(a, "TENSORFOLD_VIDEO_TOKENS", "bad-value");
    defer a.free(why);
    try t.expectEqualStrings(
        "TENSORFOLD_VIDEO_TOKENS=\"bad-value\": expected a nonnegative whole video-token budget (default 16384; unsigned 64-bit count)",
        why,
    );
    try t.expect(std.mem.indexOf(u8, why, "18446744073709551615") == null);
}
