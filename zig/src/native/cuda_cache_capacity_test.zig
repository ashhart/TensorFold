//! Startup diagnostics exercise an enabled cache that cannot hold its first legal state, without a device.
const std = @import("std");
const pool = @import("cuda_kv_pool.zig");
const api = @import("engine_api");
const t = std.testing;
const Fake = struct {
    pub const snap_grid = 2048;
    pub const snap_save = 1;
    var calls: usize = 0;
    pub fn snap_min_bytes(target: *anyopaque) u64 {
        const bytes: *u64 = @ptrCast(@alignCast(target));
        calls += 1;
        return bytes.*;
    }
};
const Log = struct {
    var calls: usize = 0;
    var text: [1024]u8 = undefined;
    var len: usize = 0;
    pub fn warn(comptime format: []const u8, args: anytype) void {
        calls += 1;
        len = (std.fmt.bufPrint(&text, format, args) catch unreachable).len;
    }
};
test "cache capacity: 256K context, four streams, 0.3 GiB cannot retain one planned state" {
    const bytes: u64 = 322122547;
    const o: api.Open = .{ .dir = "", .model_type = "glm5_next", .context = 262144, .lanes = 4 };
    var minimum: u64 = 384 << 20;
    const loaded = .{ .target = @as(*anyopaque, &minimum) };
    const kept = pool.kept(Fake, loaded, null, o, 10 << 30, (10 << 30) - bytes, 0, false);
    try t.expectEqual(bytes, kept.copied);
    const before = kept;
    Fake.calls = 0;
    const facts = pool.capacity(Fake, loaded, kept, 262144).?;
    try t.expectEqual(@as(usize, 1), Fake.calls);
    try t.expectEqual(bytes, facts.budget_bytes);
    try t.expectEqual(minimum, facts.minimum_state_bytes);
    Log.calls = 0;
    pool.warnCapacity(facts, false, Log);
    try t.expectEqual(@as(usize, 1), Log.calls);
    for ([_][]const u8{ "322122547", "402653184", "--context", "--parallel", "--prompt-cache-gib", "TENSORFOLD_MEMORY_RESERVE_GIB" }) |word| try t.expect(std.mem.indexOf(u8, Log.text[0..Log.len], word) != null);
    try t.expectEqualDeep(before, kept);
    Log.calls = 0;
    pool.warnCapacity(.{ .budget_bytes = minimum, .minimum_state_bytes = minimum }, false, Log);
    pool.warnCapacity(.{ .budget_bytes = minimum + 1, .minimum_state_bytes = minimum }, false, Log);
    pool.warnCapacity(facts, true, Log);
    pool.warnCapacity(null, false, Log);
    try t.expectEqual(@as(usize, 0), Log.calls);
    pool.warnCapacity(.{ .budget_bytes = minimum - 1, .minimum_state_bytes = minimum }, false, Log);
    try t.expectEqual(@as(usize, 1), Log.calls);
    try t.expectEqual(@as(?api.PromptCacheCapacity, null), pool.capacity(struct {}, loaded, kept, 262144));
    try t.expectEqual(@as(?api.PromptCacheCapacity, null), pool.capacity(Fake, loaded, kept, 2048));
}
