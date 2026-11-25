//! GLM_POOL_TOKENS: the planning functions with no device, and the setting's refusals.
const std = @import("std");
const cfg = @import("config.zig");
const load_plan = @import("load_plan.zig");
const state = @import("state.zig");
const pool_size = @import("pool_size.zig");
const environment = @import("sys").environment;

const window: u32 = 1_048_576;

test "glm pool tokens: unset sizes the engine as the window did, a set value sizes it by the pool" {
    try std.testing.expectEqual(window + 64, pool_size.cacheTokens(window, null));
    try std.testing.expectEqual(@as(u32, 2_000_064), pool_size.cacheTokens(window, 2_000_000));
    // the window's pool is the one a 1M --context always had; the set pool keeps whole blocks (the 2M pin's 1,999,868)
    try std.testing.expectEqual(@as(u32, 1_048_572), pool_size.poolTokens(pool_size.cacheTokens(window, null), 4));
    try std.testing.expectEqual(@as(u32, 1_999_868), pool_size.poolTokens(pool_size.cacheTokens(window, 2_000_000), 4));
    try std.testing.expectEqual(window, pool_size.streamWindow(window, 1_999_868));
    try std.testing.expectEqual(@as(u32, 1_048_555), pool_size.streamWindow(window, 1_048_572));
    try std.testing.expectEqual(@as(u32, 3_000), pool_size.streamWindow(window, 3_017)); // a pool under the window cuts it
    try std.testing.expectEqual(@as(u32, 0), pool_size.streamWindow(window, 8));
}

test "glm pool tokens: the memory plan counts the larger pool's caches and the scratch that grows with it" {
    const c: cfg.Config = .{};
    const small = pool_size.cacheTokens(window, null);
    const big = pool_size.cacheTokens(window, 2_000_000);
    const gpa = std.testing.allocator;
    const grew = try load_plan.leastArena(gpa, &c, big, true) - try load_plan.leastArena(gpa, &c, small, true);
    const caches = state.stateBytes(&c, big, true) - state.stateBytes(&c, small, true);
    const sscore = state.max_rows * @as(usize, (big - small) / c.kpool) * 4; // the window scratch's indexer scores
    try std.testing.expect(grew >= caches + sscore);
    try std.testing.expect(caches > 17_000_000_000);
}

test "glm pool tokens: a pool past the load limit is refused by name, one inside it is not" {
    var buf: [400]u8 = undefined;
    const c: cfg.Config = .{};
    const need = try load_plan.leastArena(std.testing.allocator, &c, pool_size.cacheTokens(window, 2_000_000), true);
    const weights: usize = 180_000_000_000;
    try std.testing.expect(pool_size.problem(&buf, 2_000_000, need, weights, 300_000_000_000, "GLM_LOAD_LIMIT_GB") == null);
    const why = pool_size.problem(&buf, 2_000_000, need, weights, 200_000_000_000, "GLM_LOAD_LIMIT_GB").?;
    try std.testing.expect(std.mem.startsWith(u8, why, "GLM_POOL_TOKENS=2000000 needs "));
    try std.testing.expect(std.mem.indexOf(u8, why, "GLM_LOAD_LIMIT_GB") != null and std.mem.endsWith(u8, why, "lower GLM_POOL_TOKENS or raise the limit"));
    const cap = pool_size.cacheTokens(window, 2_000_000);
    const refused = (try pool_size.check(&buf, std.testing.allocator, &c, 2_000_000, cap, true, weights, 200_000_000_000, "70% of RAM")).?;
    try std.testing.expect(std.mem.startsWith(u8, refused, "GLM_POOL_TOKENS="));
    try std.testing.expect(try pool_size.check(&buf, std.testing.allocator, &c, 2_000_000, cap, true, weights, 300_000_000_000, "70% of RAM") == null);
}

test "glm pool tokens: malformed or out-of-range text is refused naming the setting, blank is unset" {
    for ([_][]const u8{ "lots", "2k", "0", "1023", "-1", "+2000000", "1e6", "4294967232", "2 000" }) |raw| {
        try std.testing.expectError(error.InvalidEnvironment, environment.value(?u32, pool_size.name, raw, null));
        const why = try environment.describe(std.testing.allocator, pool_size.name, raw);
        defer std.testing.allocator.free(why);
        try std.testing.expect(std.mem.startsWith(u8, why, "GLM_POOL_TOKENS=\""));
    }
    for ([_]?[]const u8{ null, "", "  " }) |raw| try std.testing.expectEqual(@as(?u32, null), try environment.value(?u32, pool_size.name, raw, null));
    try std.testing.expectEqual(@as(?u32, 2_000_000), try environment.value(?u32, pool_size.name, " 2000000\t", null));
}

test "glm pool tokens: the startup line states the window and the pool, and is today's line when unset" {
    var buf: [160]u8 = undefined;
    try std.testing.expectEqualStrings("context 1048576 tokens", pool_size.contextWords(&buf, window, null, 1_048_572));
    try std.testing.expectEqualStrings(
        "context 1048576 tokens a stream in a shared pool of 1999868 tokens (GLM_POOL_TOKENS=2000000)",
        pool_size.contextWords(&buf, window, 2_000_000, 1_999_868),
    );
}

test "glm pool tokens: explicit pool preserves named refusal when MTP puts load over limit; unset drops MTP" {
    const gpa = std.testing.allocator;
    const cap = pool_size.cacheTokens(window, 65_536);
    var lean_c: cfg.Config = .{ .mtp = 0 };
    var mtp_c: cfg.Config = .{ .mtp = 1 };
    const lean_arena = try load_plan.leastArena(gpa, &lean_c, cap, true);
    const full_arena = try load_plan.leastArena(gpa, &mtp_c, cap, true);
    const full_weights: usize = 180_000_000_000;
    const lean_weights: usize = 175_000_000_000;
    // Budget fits after dropping MTP (lean_weights + lean_arena <= limit), fails with MTP enabled.
    const limit = full_weights + full_arena - 1;
    try std.testing.expect(lean_weights + lean_arena <= limit);

    const LeanPlan = struct {
        bytes: usize,
        pub fn planBytes(self: @This(), _: *const cfg.Config) !usize {
            return self.bytes;
        }
    };
    const lean_ctx: LeanPlan = .{ .bytes = lean_weights };
    var buf: [400]u8 = undefined;

    // Explicit GLM_POOL_TOKENS keeps MTP enabled and returns the named refusal instead of starting without MTP.
    var explicit_bytes = full_weights;
    var explicit_c = mtp_c;
    const out_explicit = try pool_size.planFit(&buf, gpa, &explicit_c, 65_536, cap, true, false, 0, limit, "70% of RAM", &explicit_bytes, lean_ctx);
    switch (out_explicit) {
        .refused => |why| try std.testing.expect(std.mem.startsWith(u8, why, "GLM_POOL_TOKENS=65536 needs ")),
        else => return error.TestUnexpectedResult,
    }
    try std.testing.expectEqual(@as(u32, 1), explicit_c.mtp);
    try std.testing.expectEqual(full_weights, explicit_bytes);

    // Unset GLM_POOL_TOKENS keeps the existing MTP-drop fallback.
    var unset_bytes = full_weights;
    var unset_c = mtp_c;
    const out_unset = try pool_size.planFit(&buf, gpa, &unset_c, null, cap, true, false, 0, limit, "70% of RAM", &unset_bytes, lean_ctx);
    try std.testing.expectEqual(pool_size.StartupOutcome.dropped_mtp, out_unset);
    try std.testing.expectEqual(@as(u32, 0), unset_c.mtp);
    try std.testing.expectEqual(lean_weights, unset_bytes);
}

test "glm pool tokens: fixed stream refusal blames GLM_POOL_TOKENS instead of --context when pool is set" {
    var buf: [320]u8 = undefined;
    const set_msg = pool_size.fitProblem(&buf, 4, 2_000_064, 200_000_000_000, "70% of RAM", 2, 2_000_000);
    try std.testing.expect(std.mem.indexOf(u8, set_msg, "GLM_POOL_TOKENS") != null);
    try std.testing.expect(std.mem.indexOf(u8, set_msg, "--parallel") != null);
    try std.testing.expect(std.mem.indexOf(u8, set_msg, "--context") == null);

    var unset_buf: [320]u8 = undefined;
    const unset_msg = pool_size.fitProblem(&unset_buf, 4, 1_048_640, 200_000_000_000, "70% of RAM", 2, null);
    try std.testing.expect(std.mem.indexOf(u8, unset_msg, "--context") != null);
    try std.testing.expect(std.mem.indexOf(u8, unset_msg, "--parallel") != null);
}
