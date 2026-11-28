//! Native Inkling 2 DFlash 2 Q4 setup tests: switch 0/1, exact & 1-byte-short budgets, and refusal cleanup.
const std = @import("std");
const mtl = @import("metal");
const sys = @import("sys");
const native = @import("inkling2.zig");
const dflash2 = @import("../families/inkling2/metal/dflash2.zig");
const w = @import("../families/inkling2/metal/dflash2_weights.zig");
const q4 = @import("../families/inkling2/metal/draft_head_q4.zig");
const batch_test = @import("../families/inkling2/metal/dflash2_batch_test.zig");

const gib: u64 = 1 << 30;
const ram_bytes: u64 = 64 * gib;
const limit_bytes: u64 = 48 * gib;
const margin_bytes: u64 = 2 * gib;
const max_context: u32 = 262144;

fn saveEnv(gpa: std.mem.Allocator) !?[:0]u8 {
    const saved = sys.environment.rawText(q4.env_name) orelse return null;
    return try std.fmt.allocPrintSentinel(gpa, "{s}", .{saved}, 0);
}

fn restoreEnv(gpa: std.mem.Allocator, saved: ?[:0]u8) void {
    if (saved) |val| {
        _ = sys.env.setenv(q4.env_name, val.ptr, 1);
        gpa.free(val);
    } else {
        _ = sys.env.unsetenv(q4.env_name);
    }
}

test "Inkling native DFlash2 setup honors switch 0/1, exact and 1-byte-short budgets, and unwinds refusals" {
    if (comptime @import("builtin").os.tag != .macos) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const gpa = std.testing.allocator;
    const saved = try saveEnv(gpa);
    defer restoreEnv(gpa, saved);

    var rig = try batch_test.SyntheticRig.init(gpa, device, queue);
    defer rig.deinit();

    const q4_need = try q4.bytesNeeded(w.valid_vocab, w.hidden);
    const window: usize = 64;

    // 1. Switch unset and "0" keep Q4 draft head off and fast pipelines unloaded even with abundant budget.
    for ([_]?[:0]const u8{ null, "0" }) |val| {
        if (val) |v| _ = sys.env.setenv(q4.env_name, v.ptr, 1) else _ = sys.env.unsetenv(q4.env_name);
        const df = try rig.makeDrafter(4, device.tensorUnits(), 1);
        defer df.deinit();
        try std.testing.expectEqual(df.loaded.fixedBytes() + dflash2.scratchBytes(1), df.fixedBytes());
        var problem: []const u8 = "";
        const setup = (try native.setupDFlash2DraftHeadAndContext(df, ram_bytes, ram_bytes, null, 4 * gib, 1, window, max_context, &problem)).?;
        try std.testing.expectEqual(@as(i64, window), setup.window);
        try std.testing.expect(df.loaded.draft_head_q4 == null);
        try std.testing.expect(!dflash2.Drafter.testing.hasFast(df));
        try std.testing.expectEqual(mtl.DispatchType.serial, dflash2.Drafter.testing.dispatchMode(df));
    }

    // 2. Switch "1" with exact and 1-byte-short budgets across 1-stream (16-byte dummy taps) and 4-stream setups.
    _ = sys.env.setenv(q4.env_name, "1", 1);
    for ([_]usize{ 1, 4 }) |streams| {
        const min_need = native.contextStateBytes(window, streams, dflash2.streamBytes(), 0, 0);

        // Exact budget: enables Q4 draft head, loads fast pipelines, switches to concurrent encoder, and fits window.
        {
            const df = try rig.makeDrafter(4, device.tensorUnits(), streams);
            defer df.deinit();
            const base_df = df.fixedBytes();
            try std.testing.expectEqual(df.loaded.fixedBytes() + dflash2.scratchBytes(streams), base_df);
            const other = limit_bytes - margin_bytes - base_df - (min_need + q4_need);
            var problem: []const u8 = "";
            const setup = (try native.setupDFlash2DraftHeadAndContext(df, ram_bytes, ram_bytes, null, other, streams, window, max_context, &problem)).?;
            try std.testing.expect(df.loaded.draft_head_q4 != null);
            try std.testing.expect(dflash2.Drafter.testing.hasFast(df));
            try std.testing.expectEqual(mtl.DispatchType.concurrent, dflash2.Drafter.testing.dispatchMode(df));
            try std.testing.expectEqual(base_df + q4_need, df.fixedBytes());
            try std.testing.expectEqual(min_need, setup.budget);
            try std.testing.expectEqual(window, setup.fits);
            try std.testing.expectEqual(@as(i64, window), setup.window);
        }

        // One-byte-short budget: refuses Q4 draft head cleanly, stays on serial path, and still admits the window.
        {
            const df = try rig.makeDrafter(4, device.tensorUnits(), streams);
            defer df.deinit();
            const base_df = df.fixedBytes();
            const other = limit_bytes - margin_bytes - base_df - (min_need + q4_need - 1);
            var problem: []const u8 = "";
            const setup = (try native.setupDFlash2DraftHeadAndContext(df, ram_bytes, ram_bytes, null, other, streams, window, max_context, &problem)).?;
            try std.testing.expect(df.loaded.draft_head_q4 == null);
            try std.testing.expect(!dflash2.Drafter.testing.hasFast(df));
            try std.testing.expectEqual(mtl.DispatchType.serial, dflash2.Drafter.testing.dispatchMode(df));
            try std.testing.expectEqual(base_df, df.fixedBytes());
            try std.testing.expectEqual(@as(i64, window), setup.window);
        }
    }

    // 3. Cleanup after later refusal once Q4 draft head and fast pipelines are already allocated.
    for ([_]struct { ctx: ?i64, limit: ?[]const u8, needle: []const u8 }{
        .{ .ctx = 0, .limit = null, .needle = "--context must be at least one token" },
        .{ .ctx = @as(i64, max_context) + 1, .limit = null, .needle = "--context exceeds" },
        .{ .ctx = window, .limit = "not-a-number", .needle = "INKLING2_LOAD_LIMIT_GB" },
    }) |case| {
        const df = try rig.makeDrafter(4, device.tensorUnits(), 1);
        defer df.deinit();
        var problem: []const u8 = "";
        const setup = try native.setupDFlash2DraftHeadAndContext(df, ram_bytes, ram_bytes, case.limit, 4 * gib, 1, case.ctx, max_context, &problem);
        try std.testing.expect(setup == null);
        try std.testing.expect(std.mem.indexOf(u8, problem, case.needle) != null);
        if (case.limit == null) {
            // Confirm Q4 head and fast pipelines were actually allocated before the later context refusal unwound them.
            try std.testing.expect(df.loaded.draft_head_q4 != null);
            try std.testing.expect(dflash2.Drafter.testing.hasFast(df));
        }
    }
}
