//! A CUDA family's prompt cache in the native server: rules, budget and copies from the family's `snap_*` decls.
const std = @import("std");
const api = @import("engine_api");

/// Any row, the head one token on (Nemotron's); a family's snap_grid, snap_lookahead, warm_quiet and snap_fork if set.
pub fn cacheRules(comptime F: type) api.prompt_cache.Rules {
    var rules: api.prompt_cache.Rules = .{ .lookahead = 1, .min_prompt = 0 };
    if (@hasDecl(F, "snap_lookahead")) rules.lookahead = F.snap_lookahead;
    if (@hasDecl(F, "warm_quiet")) rules.warm = .waits;
    if (@hasDecl(F, "snap_fork")) rules.fork = F.snap_fork; // opt-in, a family at a time after its device check
    if (@hasDecl(F, "snap_grid")) {
        rules.planned = true;
        rules.grid = F.snap_grid;
    }
    return rules;
}

/// Native CUDA families own their grid; a loaded `snap_grid` (the prefill step) keeps states on chunk starts only.
pub fn passRules(comptime F: type, prompt_rows: u32, snap_grid: u32) api.prompt_cache.Rules {
    var rules = cacheRules(F);
    rules.chunk = prompt_rows;
    if (@hasDecl(F, "snap_repeat_min")) rules.repeat_min = if (std.c.getenv("TF_FLASHNEXT_REPEAT_CACHE")) |v| (if (std.mem.eql(u8, std.mem.span(v), "0")) 0 else F.snap_repeat_min) else F.snap_repeat_min;
    rules.chunk_starts = false;
    if (snap_grid > 0) {
        rules.planned = true;
        rules.grid = snap_grid;
    }
    return rules;
}

/// The grid a loaded family declares (0: any row).
pub fn loadedGrid(loaded: anytype) u32 {
    return if (@hasField(@TypeOf(loaded), "snap_grid")) loaded.snap_grid else 0;
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
            .charged = if (@hasDecl(F, "snap_charged")) F.snap_charged else null,
            .spare = if (@hasDecl(F, "snap_spare")) F.snap_spare else null,
            .reuses = if (@hasDecl(F, "snap_reuses")) F.snap_reuses else null,
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
    const Headless = struct {
        pub const snap_lookahead: u32 = 0;
    };
    try std.testing.expectEqual(@as(u32, 0), cacheRules(Headless).lookahead);
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

test {
    _ = @import("cuda_snaps_test.zig");
}

test "a family that names its prefill step keeps marks on chunk starts, so a resumed pass cuts as a fresh one" {
    const Exl3 = struct {};
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
    const prompt = try arena.allocator().alloc(u32, 12780);
    for (prompt, 0..) |*t, i| t.* = @intCast(i);
    const vtable: api.prompt_cache.Snapshots.VTable = .{ .bytes = Stub.bytes, .save = Stub.save, .restore = Stub.restore, .drop = Stub.drop };
    const copies: api.prompt_cache.Snapshots = .{ .ptr = &unused, .vtable = &vtable };
    // any row: the history mark cuts the cold pass's last chunk to 7 rows, a different kernel path than the fresh pass
    var any = api.prompt_cache.Store.init(std.testing.allocator, copies, passRules(Exl3, 4096, 0), 1 << 30);
    defer any.deinit();
    const loose = try any.marks(arena.allocator(), prompt, 0, 12773, &.{}, &.{}, &.{});
    try std.testing.expect(std.mem.indexOfScalar(u32, loose, 12773) != null);
    var grid = api.prompt_cache.Store.init(std.testing.allocator, copies, passRules(Exl3, 4096, 4096), 1 << 30);
    defer grid.deinit();
    const exact = try grid.marks(arena.allocator(), prompt, 0, 12773, &.{}, &.{}, &.{});
    try std.testing.expect(exact.len > 0);
    for (exact) |m| try std.testing.expectEqual(@as(u32, 0), m % 4096);
}
