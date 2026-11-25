//! The real Qwen exports guard all geometry reads; host-only admission stays outside the unchanged plan.
const std = @import("std");
const api = @import("engine_api");
const qwen = @import("qwen35");
const kv = @import("core").kv_pool;
const pages = @import("cuda_kv_pool.zig");
const streams = @import("cuda_streams.zig");
const host_plan = @import("cuda_kv_host.zig");
const refusal = @import("cuda_refusal.zig");
const env = @import("sys").env;
const t = std.testing;

const Dense = struct {
    pub const Options = struct {};
    pub const kvHostRoom = qwen.native.kvHostRoom;
};
const Moe = struct {
    pub const Options = struct {};
    pub const kvHostRoom = qwen.native_moe.kvHostRoom;
};

const Loop = struct {
    pub const Options = struct {};
    pub const kvHostRoom = qwen.native_loop.kvHostRoom;
};

test "kv safety: Qwen host KV unset preserves dense, MoE and Loop plans with pages off or on, one and eight streams" {
    _ = env.unsetenv("TF_QWEN_KV_HOST");
    _ = env.setenv("TF_QWEN_KV_HOST_GIB", "bad", 1);
    defer _ = env.unsetenv("TF_QWEN_KV_HOST_GIB");
    const l: kv.Layout = .{ .fixed_bytes = 32, .token_bytes = 2, .page_tokens = 16, .snap_bytes = 0, .run_bytes = 64 };
    inline for (.{ Dense, Moe, Loop }) |F| {
        try t.expectEqual(@as(u64, 0), try F.kvHostRoom(@ptrFromInt(0x1000)));
        for ([_]u32{ 1, 8 }) |lanes| for ([_]bool{ false, true }) |on_pages| {
            var problem: []const u8 = "";
            const o: api.Open = .{ .dir = "", .model_type = "qwen3_5", .lanes = lanes, .lanes_fixed = true };
            const loaded = .{ .ctx = @as(*anyopaque, @ptrFromInt(0x1000)), .layout = if (on_pages) @as(?kv.Layout, l) else null, .engine_paged = on_pages, .free_streams = @as(u32, 1), .stream_bytes = @as(u64, 64) };
            const got = (try streams.count(F, t.allocator, loaded, o, .{ .room = 10000, .model = 0, .reserve = 0, .low = "", .window = 128, .stream_bytes = 64, .host_cache = true, .host_budget = 0, .share = false }, &problem)).?;
            try t.expectEqual(lanes, got.streams);
            try t.expectEqual(@as(u64, 0), got.extra_device);
            if (on_pages) {
                const want = (try pages.plan(F, t.allocator, l, 10000, o, 128, &problem, true, true, 0, null)).?;
                try t.expectEqual(want, got.paged.?);
                try t.expectEqual(@as(usize, if (lanes == 1) 8 else 78), got.paged.?.pages);
            } else try t.expect(got.paged == null);
        };
    }
}

test "kv safety: Qwen host KV refuses parallel and disk before device work for dense, MoE and Loop" {
    _ = env.setenv("TF_QWEN_KV_HOST", "1", 1);
    defer _ = env.unsetenv("TF_QWEN_KV_HOST");
    inline for (.{ qwen.native, qwen.native_moe, qwen.native_loop }) |F| {
        var o: api.Open = .{ .dir = "", .model_type = F.model_type, .lanes = 8 };
        try t.expectStringStartsWith(refusal.family(F, o).?, "TF_QWEN_KV_HOST supports one stream");
        o.lanes = 1;
        o.spill = .evicted;
        try t.expectStringStartsWith(refusal.family(F, o).?, "TF_QWEN_KV_HOST cannot use --spill-gib");
        o.spill = .off;
        o.learn_conversations = true;
        try t.expectStringStartsWith(refusal.family(F, o).?, "TF_QWEN_KV_HOST cannot use disk snapshots");
        o.learn_conversations = false;
        try t.expect(refusal.family(F, o) == null);
    }
}

test "kv safety: Loop host KV validates its cap before reading lane geometry" {
    _ = env.setenv("TF_QWEN_KV_HOST", "1", 1);
    _ = env.setenv("TF_QWEN_KV_HOST_GIB", "bad", 1);
    defer _ = env.unsetenv("TF_QWEN_KV_HOST");
    defer _ = env.unsetenv("TF_QWEN_KV_HOST_GIB");
    try t.expectError(error.InvalidEnvironment, qwen.native_loop.kvHostRoom(@ptrFromInt(0x1000)));
}

test "kv safety: Qwen host KV charges one device run and table before the host prompt cache" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const l: kv.Layout = .{ .fixed_bytes = 32, .token_bytes = 2, .page_tokens = 16, .snap_bytes = 0, .run_bytes = 64 };
    const o: api.Open = .{ .dir = "", .model_type = "qwen3_5", .lanes = 1 };
    var problem: []const u8 = "";
    try t.expectEqual(@as(u64, 96), try host_plan.deviceBytes(l));
    try t.expect((try host_plan.plan(arena.allocator(), l, 95, o, 256, &problem)) == null);
    try t.expect(std.mem.indexOf(u8, problem, "TF_QWEN_KV_HOST") != null);
    const p = (try host_plan.plan(arena.allocator(), l, 96, o, 256, &problem)).?;
    try t.expectEqual(kv.Plan{ .lanes = 1, .pages = 8, .snaps = 0, .copied = 0 }, p);
    const c = streams.cacheRoom(1000, 64, 1, .{ .streams = 1, .paged = p, .layout = l, .extra_device = 96 });
    try t.expectEqual(@as(u64, 904), c.room);
    try t.expectEqual(@as(u64, 0), c.streams);
}
