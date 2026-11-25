//! Qwen rows split from environment_test.zig: K/V grow and prefix snapshots, and TF_MTP_DRAFTS's two scopes.
const std = @import("std");
const e = @import("environment.zig");
const t = std.testing;

const One = struct {
    name: []const u8,
    raw: ?[]const u8,
    pub fn get(v: One, name: []const u8) ?[]const u8 {
        return if (std.mem.eql(u8, name, v.name)) v.raw else null;
    }
};

test "environment: TF_Q27_KV_GROW takes 0 or a multiple of 16 that fits u32, and refuses blank, others and padding" {
    for ([_][]const u8{ "0", "16", "4096", "4294967280", "+32", "1_6" }) |raw| _ = try e.parse("TF_Q27_KV_GROW", raw);
    try t.expectEqual(@as(?u64, 4096), try e.parse("TF_Q27_KV_GROW", "4096"));
    try t.expectEqual(@as(?u64, null), try e.parse("TF_Q27_KV_GROW", null));
    for ([_][]const u8{ "", " ", "8", "24", "15", " 16", "16 ", "4294967296", "0x10", "-16", "off" }) |raw|
        try t.expectError(error.InvalidEnvironment, e.parse("TF_Q27_KV_GROW", raw));
    const why = (try e.refusal(t.allocator, One{ .name = "TF_Q27_KV_GROW", .raw = "24" }, .{ .metal = true })).?;
    defer t.allocator.free(why);
    try t.expect(std.mem.indexOf(u8, why, "TF_Q27_KV_GROW=\"24\"") != null);
    try t.expect(try e.refusal(t.allocator, One{ .name = "TF_Q27_KV_GROW", .raw = "24" }, .{ .cuda = true }) == null);
}

test "environment: TF_QWEN_KV_SNAPSHOT_GPU is exact 0 or 1 on CUDA, blank as unset" {
    try t.expectEqual(@as(?u64, 1), try e.parse("TF_QWEN_KV_SNAPSHOT_GPU", "1"));
    try t.expectEqual(@as(?u64, 0), try e.parse("TF_QWEN_KV_SNAPSHOT_GPU", "0"));
    for ([_]?[]const u8{ null, "", " \t" }) |raw| try t.expectEqual(@as(?u64, null), try e.parse("TF_QWEN_KV_SNAPSHOT_GPU", raw));
    // texts the reader took as off (only exact 1 is on) are refused at a CUDA start now
    for ([_][]const u8{ "true", "on", "2", " 1", "1 ", "yes" }) |raw|
        try t.expectError(error.InvalidEnvironment, e.parse("TF_QWEN_KV_SNAPSHOT_GPU", raw));
    const why = (try e.refusal(t.allocator, One{ .name = "TF_QWEN_KV_SNAPSHOT_GPU", .raw = "true" }, .{ .cuda = true })).?;
    defer t.allocator.free(why);
    try t.expect(std.mem.indexOf(u8, why, e.rule("TF_QWEN_KV_SNAPSHOT_GPU").forms) != null);
    try t.expect(try e.refusal(t.allocator, One{ .name = "TF_QWEN_KV_SNAPSHOT_GPU", .raw = "true" }, .{ .metal = true }) == null);
}

test "environment: TF_MTP_DRAFTS is checked on CUDA and Metal and ignored on XPU" {
    const env: One = .{ .name = "TF_MTP_DRAFTS", .raw = "bad" };
    for ([_]e.Flags{ .{ .cuda = true }, .{ .metal = true } }) |flags| {
        const why = (try e.refusal(t.allocator, env, flags)).?;
        defer t.allocator.free(why);
        try t.expect(std.mem.indexOf(u8, why, "TF_MTP_DRAFTS") != null);
    }
    try t.expect((try e.refusal(t.allocator, env, .{ .xpu = true })) == null);
    try t.expect((try e.refusal(t.allocator, env, .{})) == null);
    try t.expectError(error.InvalidEnvironment, e.checkScope(.cuda, env));
    try t.expectError(error.InvalidEnvironment, e.checkScope(.metal, env));
    try e.checkScope(.xpu, env);
}

test "environment: Qwen host GiB cap is strict, bounded and blank as unset" {
    const name = "TF_QWEN_KV_HOST_GIB";
    for ([_]?[]const u8{ null, "", " \t\r\n" }) |raw| try t.expectEqual(@as(u64, 0), try e.value(u64, name, raw, 0));
    try t.expectEqual(@as(u64, 8), try e.value(u64, name, "8", 0));
    try t.expectEqual(@as(u64, std.math.maxInt(u64) >> 30), try e.value(u64, name, "17179869183", 0));
    for ([_][]const u8{ "0", "-1", "bad", "1.5", "17179869184" }) |raw| try t.expectError(error.InvalidEnvironment, e.value(u64, name, raw, 0));
}
