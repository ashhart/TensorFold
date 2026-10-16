//! A CUDA family's prompt cache in the native server: rules, budget and copies from the family's `snap_*` decls.
const std = @import("std");
const api = @import("engine_api");

/// Any row, the head one token on (Nemotron's); `snap_grid`'s chunk starts, `snap_lookahead` and `warm_quiet` if declared.
pub fn cacheRules(comptime F: type) api.prompt_cache.Rules {
    var rules: api.prompt_cache.Rules = .{ .lookahead = 1, .min_prompt = 0 };
    if (@hasDecl(F, "snap_lookahead")) rules.lookahead = F.snap_lookahead;
    if (@hasDecl(F, "warm_quiet")) rules.warm = .waits;
    if (@hasDecl(F, "snap_grid")) {
        rules.planned = true;
        rules.grid = F.snap_grid;
    }
    return rules;
}

/// The cache's bytes less what the family sets aside for its states (`snap_ready`, as Nemotron's spare buffers).
pub fn cacheBudget(comptime F: type, target: anytype, cache: u64) !u64 {
    return if (@hasDecl(F, "snap_ready")) cache - try F.snap_ready(target, cache) else cache;
}

/// The family's `snap_*` copies and `image`, its disk rows (null: none); a container-level value, so the pointer holds.
pub fn Snaps(comptime F: type, comptime image: ?*const api.prompt_cache.rows.Image) type {
    return struct {
        pub const vtable: api.prompt_cache.Snapshots.VTable = .{
            .bytes = F.snap_bytes,
            .save = F.snap_save,
            .restore = F.snap_restore,
            .drop = F.snap_drop,
            .rows = image,
        };
    };
}

test "a family with snap_grid keeps on its chunk starts, any other at any row as Nemotron does" {
    const Glm = struct {
        pub const snap_grid: u32 = 2048;
    };
    const Other = struct {};
    const Warm = struct {
        pub const warm_quiet: u64 = 1;
    };
    const any = cacheRules(Other);
    try std.testing.expect(!any.planned and any.min_prompt == 0 and any.lookahead == 1 and any.warm == .off);
    try std.testing.expect(cacheRules(Warm).warm == .waits and cacheRules(Glm).warm == .off);
    const Stub = struct {
        fn bytes(_: *anyopaque, _: u32) u64 {
            return 0;
        }
        fn save(_: *anyopaque, _: ?*anyopaque, _: u32) anyerror!api.prompt_cache.Saved {
            return error.Unused;
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: api.prompt_cache.Saved) anyerror!void {}
        fn drop(_: *anyopaque, _: api.prompt_cache.Saved) void {}
    };
    var unused: u8 = 0;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const prompt = try arena.allocator().alloc(u32, 3547);
    for (prompt, 0..) |*t, i| t.* = @intCast(i);
    const vtable: api.prompt_cache.Snapshots.VTable = .{
        .bytes = Stub.bytes,
        .save = Stub.save,
        .restore = Stub.restore,
        .drop = Stub.drop,
    };
    const copies: api.prompt_cache.Snapshots = .{ .ptr = &unused, .vtable = &vtable };
    var store = api.prompt_cache.Store.init(std.testing.allocator, copies, cacheRules(Glm), 1 << 30);
    defer store.deinit();
    const marks = try store.marks(arena.allocator(), prompt, 0, 3540, &.{}, &.{}, &.{});
    try std.testing.expectEqualSlices(u32, &[_]u32{2048}, marks);
}
