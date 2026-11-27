//! TF_GLM_REQUEST_STARTS: shared page snapshots retain exact tails; learning still needs the fixed grid.
const std = @import("std");
const cuda = @import("cuda");
const native = @import("cuda_native.zig");
const glm_env = @import("cuda_env.zig");
const env = @import("sys").env;
const t = std.testing;

const Saved = struct {
    const names = [_][:0]const u8{ "TF_GLM_REQUEST_STARTS", "TF_KV_SHARE" };
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

test "glm request starts: shared KV accepts the existing opt-in at startup" {
    var saved = try Saved.take();
    defer saved.restore();
    try put("TF_GLM_REQUEST_STARTS", "1");
    try put("TF_KV_SHARE", "32");
    try glm_env.requestStartsAlone(false);
}

test "glm request starts: --learn-conversations refuses at startup before any rank opens, by name" {
    var saved = try Saved.take();
    defer saved.restore();
    try put("TF_GLM_REQUEST_STARTS", "1");
    try put("TF_KV_SHARE", null);
    var ctx: cuda.Context = undefined;
    const o: native.Options = .{ .context = 8192, .drafts = false, .learn_conversations = true };
    try t.expectError(error.GlmRequestStartsWithLearnConversations, native.open(t.allocator, t.io, &ctx, "missing-request-starts-checkpoint", null, o));
    const reason = native.explain(null, error.GlmRequestStartsWithLearnConversations) orelse return error.MissingNamedRefusal;
    try t.expect(std.mem.indexOf(u8, reason, "TF_GLM_REQUEST_STARTS=1") != null);
    try t.expect(std.mem.indexOf(u8, reason, "--learn-conversations") != null);
    try put("TF_GLM_REQUEST_STARTS", null);
    try glm_env.requestStartsAlone(true); // the switch off: --learn-conversations starts as today
}

test "glm request starts: shared counts keep the opt-in; malformed sharing still fails its own parser" {
    var saved = try Saved.take();
    defer saved.restore();
    try put("TF_KV_SHARE", "32");
    for ([_]?[:0]const u8{ null, "", "0", "off", "on", "true", "2" }) |v| {
        try put("TF_GLM_REQUEST_STARTS", v);
        try glm_env.requestStartsAlone(false);
    }
    try put("TF_GLM_REQUEST_STARTS", "1");
    for ([_]?[:0]const u8{ null, "0", "00" }) |v| {
        try put("TF_KV_SHARE", v);
        try glm_env.requestStartsAlone(false);
    }
    for ([_][:0]const u8{ "1", "32", "00032" }) |v| {
        try put("TF_KV_SHARE", v);
        try glm_env.requestStartsAlone(false);
    }
    try put("TF_KV_SHARE", "bad");
    try t.expectError(error.BadKvShare, @import("core").kv_prompts.sharing());
}
