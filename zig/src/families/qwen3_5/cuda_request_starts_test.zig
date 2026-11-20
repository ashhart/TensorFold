//! Startup compatibility for the explicit message-cut opt-in, through the actual dense and MoE entry points.
const std = @import("std");
const cuda = @import("cuda");
const native = @import("cuda_native.zig");
const moe = @import("cuda_native_moe.zig");
const prefix = @import("cuda_prefix.zig");
const env = @import("sys").env;
const t = std.testing;

const Saved = struct {
    const names = [_][:0]const u8{ "TF_QWEN_REQUEST_STARTS", "TF_QWEN_KV_SHARE", "TF_QWEN_KV_PAGES" };
    values: [names.len]?[:0]u8 = @splat(null),
    fn take() !Saved {
        var s: Saved = .{};
        errdefer for (s.values) |v| if (v) |old| t.allocator.free(old);
        for (names, 0..) |n, i| if (@import("sys").getenv(n)) |v| {
            s.values[i] = try t.allocator.dupeSentinel(u8, v, 0);
        };
        return s;
    }
    fn restore(s: *Saved) void {
        for (names, s.values) |n, v| if (v) |old| {
            _ = env.setenv(n, old, 1);
            t.allocator.free(old);
        } else {
            _ = env.unsetenv(n);
        };
    }
};

fn put(name: [:0]const u8, value: ?[:0]const u8) !void {
    try t.expectEqual(@as(c_int, 0), if (value) |v| env.setenv(name, v, 1) else env.unsetenv(name));
}

test "request starts: shared KV refuses at actual dense and MoE startup before loading" {
    var saved = try Saved.take();
    defer saved.restore();
    try put("TF_QWEN_REQUEST_STARTS", "1");
    try put("TF_QWEN_KV_SHARE", "32");
    try put("TF_QWEN_KV_PAGES", "256");
    var d = @import("cuda_sm110_open_test.zig").driver(121);
    const ctx: cuda.Context = .{ .d = &d, .device = 0, .handle = null };
    inline for (.{ native, moe }) |F| {
        try t.expectError(error.QwenRequestStartsWithSharedKv, F.open(t.allocator, t.io, &ctx, "missing-request-starts-checkpoint", null, .{ .context = 8192, .drafts = false }));
        const reason = F.explain(null, error.QwenRequestStartsWithSharedKv) orelse return error.MissingNamedRefusal;
        try t.expect(std.mem.indexOf(u8, reason, "TF_QWEN_REQUEST_STARTS=1") != null);
        try t.expect(std.mem.indexOf(u8, reason, "TF_QWEN_KV_SHARE") != null);
    }
}

test "request starts: shared-KV parser keeps off values and its existing count errors" {
    var saved = try Saved.take();
    defer saved.restore();
    try put("TF_QWEN_KV_SHARE", "32");
    for ([_]?[:0]const u8{ null, "", "0", "off", "on", "true", "2" }) |v| {
        try put("TF_QWEN_REQUEST_STARTS", v);
        try t.expectEqual(@as(usize, 32), try prefix.sharing());
    }
    try put("TF_QWEN_REQUEST_STARTS", "1");
    for ([_]?[:0]const u8{ null, "0", "00" }) |v| {
        try put("TF_QWEN_KV_SHARE", v);
        try t.expectEqual(@as(usize, 0), try prefix.sharing());
    }
    for ([_][:0]const u8{ "1", "32", "00032" }) |v| {
        try put("TF_QWEN_KV_SHARE", v);
        try t.expectError(error.QwenRequestStartsWithSharedKv, prefix.sharing());
    }
    try put("TF_QWEN_KV_SHARE", "bad");
    try t.expectError(error.BadQwenKvShare, prefix.sharing());
}
