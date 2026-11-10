const std = @import("std");
const api = @import("engine_api");
const kv_pool = @import("cuda_pool_plan.zig");

pub fn Checks(comptime native: type) type {
    return struct {
        const wrapped = native.wrapped;
        const plan = native.plan;
        const Plan = native.Plan;

        test "pooled first run: native wrapper charges external fixed bytes without a duplicate window" {
            const l: kv_pool.Layout = .{ .fixed_bytes = 10, .token_bytes = 3, .page_tokens = 4, .snap_bytes = 0, .run_bytes = 13 };
            const loaded = .{ .layout = @as(?kv_pool.Layout, l), .stream_bytes = @as(u64, 1000), .engine_paged = @as(bool, true), .free_streams = @as(u32, 0) };
            const wrapped_plan = wrapped(struct {}, loaded, 0, 1000, 70);
            try std.testing.expectEqual(@as(u64, 60), wrapped_plan.room);
            try std.testing.expect(wrapped_plan.layout.?.engine_run);
            const first_loaded = wrapped(struct {}, loaded, 1, 1000, 70);
            try std.testing.expectEqual(@as(u64, 70), first_loaded.room);
            try std.testing.expect(!first_loaded.layout.?.engine_run);
            const unpaged = .{ .layout = loaded.layout, .stream_bytes = loaded.stream_bytes, .engine_paged = false, .free_streams = loaded.free_streams };
            const previous = wrapped(struct {}, unpaged, 0, 1000, 1070);
            try std.testing.expectEqual(@as(u64, 70), previous.room);
            try std.testing.expect(!previous.layout.?.engine_run);
        }

        test "pooled first run: native admission fits exact single and C4 boundaries and rejects one byte less" {
            var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
            defer arena.deinit();
            const a = arena.allocator();
            const F = struct {};
            const l: kv_pool.Layout = .{ .fixed_bytes = 10, .token_bytes = 3, .page_tokens = 4, .snap_bytes = 0, .run_bytes = 13 };
            const loaded = .{ .layout = @as(?kv_pool.Layout, l), .stream_bytes = @as(u64, 1000), .engine_paged = true, .free_streams = @as(u32, 0) };
            for ([_]struct { lanes: u32, room: u64, pages: usize }{
                .{ .lanes = 1, .room = 70, .pages = 5 },
                .{ .lanes = 4, .room = 208, .pages = 14 },
            }) |case| {
                const o: api.Open = .{ .dir = "", .model_type = "q", .lanes = case.lanes, .lanes_fixed = true, .prompt_cache_gib = 0 };
                var problem: []const u8 = "";
                const full = wrapped(F, loaded, 0, 1000, case.room);
                const accepted = try plan(F, a, full.layout.?, full.room, o, 9, &problem, false, true, 0, null);
                try std.testing.expect(accepted != null);
                try std.testing.expectEqual(case.lanes, accepted.?.lanes);
                try std.testing.expectEqual(case.pages, accepted.?.pages);
                const short = wrapped(F, loaded, 0, 1000, case.room - 1);
                try std.testing.expectEqual(@as(?Plan, null), try plan(F, a, short.layout.?, short.room, o, 9, &problem, false, true, 0, null));
            }
        }
    };
}
