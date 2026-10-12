//! A CUDA family's prompt cache in the native server: rules, budget and copies from the family's `snap_*` decls.
const std = @import("std");
const api = @import("engine_api");

/// Any row, the head one token on (Nemotron's); `snap_grid`'s chunk starts and `warm_quiet`'s warm turns if declared.
pub fn cacheRules(comptime F: type) api.prompt_cache.Rules {
    var rules: api.prompt_cache.Rules = .{ .lookahead = 1, .min_prompt = 0 };
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

/// The family's `snap_*` copies (none: no prompt cache), a container-level value so the store's pointer stays valid.
pub fn Snaps(comptime F: type) type {
    return struct {
        pub const vtable: api.prompt_cache.Snapshots.VTable = .{
            .bytes = F.snap_bytes,
            .save = F.snap_save,
            .restore = F.snap_restore,
            .drop = F.snap_drop,
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
    var store = api.prompt_cache.Store.init(std.testing.allocator, .{ .ptr = &unused, .vtable = &.{ .bytes = Stub.bytes, .save = Stub.save, .restore = Stub.restore, .drop = Stub.drop } }, cacheRules(Glm), 1 << 30);
    defer store.deinit();
    try std.testing.expectEqualSlices(u32, &[_]u32{2048}, try store.marks(arena.allocator(), prompt, 0, 3540, &.{}, &.{}, &.{}));
}
