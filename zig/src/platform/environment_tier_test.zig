//! Tier settings: the keep-others switch and LFM2's page switch take exact words and are refused by name.
const std = @import("std");
const e = @import("environment.zig");
const t = std.testing;
const a = t.allocator;
const Env = struct {
    name: []const u8,
    raw: ?[]const u8,
    pub fn get(v: Env, name: []const u8) ?[]const u8 {
        if (std.mem.eql(u8, name, v.name)) return v.raw;
        if (std.mem.eql(u8, name, "TF_CUDA_CARVEOUT")) return "1";
        return null;
    }
};

test "environment: the keep-others switch is exact 0 or 1, checked when the CUDA backend starts" {
    const name = "TENSORFOLD_PROMPT_CACHE_KEEP_OTHERS";
    for ([_][]const u8{ "true", "on", "yes", "01", " 1", "1 ", "2" }) |raw| {
        try t.expectError(error.InvalidEnvironment, e.parse(name, raw));
        const why = (try e.refusal(a, Env{ .name = name, .raw = raw }, .{ .cuda = true })).?;
        a.free(why);
        try t.expect((try e.refusal(a, Env{ .name = name, .raw = raw }, .{ .metal = true })) == null);
    }
    try t.expect(!try e.value(bool, name, null, false));
    try t.expect(try e.value(bool, name, "1", false));
}

test "environment: LFM2's page switch is exact 0 or 1, refused by name, and unset keeps pages on" {
    const name = "TENSORFOLD_LFM2_KV_PAGES";
    for ([_][]const u8{ "off", "false", "no", "on", "2", " 0" }) |raw| {
        try t.expectError(error.InvalidEnvironment, e.value(bool, name, raw, true));
        const why = (try e.refusal(a, Env{ .name = name, .raw = raw }, .{ .cuda = true })).?;
        defer a.free(why);
        try t.expect(std.mem.indexOf(u8, why, name) != null);
    }
    try t.expect(try e.value(bool, name, null, true));
    try t.expect(try e.value(bool, name, "1", true));
    try t.expect(!try e.value(bool, name, "0", true));
}
