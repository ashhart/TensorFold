const std = @import("std");
const mx = @import("mlx.zig");
const states = @import("request_state.zig");

fn fill(cache: anytype, value: mx.Array) !void {
    inline for (comptime std.meta.fieldNames(@TypeOf(cache.*))) |field| if (@FieldType(@TypeOf(cache.*), field) == mx.Array) {
        @field(cache, field) = try mx.retain(value);
    };
}

fn equal(cache: anytype, value: mx.Array) !void {
    inline for (comptime std.meta.fieldNames(@TypeOf(cache))) |field| if (@FieldType(@TypeOf(cache), field) == mx.Array) {
        try equalArray(@field(cache, field), value);
    };
}

fn equalArray(actual: mx.Array, expected: mx.Array) !void {
    if (expected.ctx == null) return std.testing.expect(actual.ctx == null);
    try mx.evalMany(&.{ actual, expected }, false);
    try std.testing.expectEqualSlices(i32, mx.c.mlx_array_data_int32(expected)[0..2], mx.c.mlx_array_data_int32(actual)[0..2]);
}

fn checkModel(comptime M: type) !void {
    var scope = mx.Scope{};
    defer scope.deinit();
    const original = try scope.ints(&.{ 17, 31 });
    const other = try scope.ints(&.{ 67, 89 });
    var m: M = undefined;
    if (@typeInfo(@FieldType(M, "cache")) == .array) {
        m.cache = @splat(.{});
    } else {
        m.cache = try mx.allocator.alloc(@typeInfo(@FieldType(M, "cache")).pointer.child, 3);
        @memset(m.cache, .{});
    }
    defer {
        for (m.cache[0..]) |*cache| cache.deinit();
        if (@typeInfo(@FieldType(M, "cache")) == .pointer) mx.allocator.free(m.cache);
    }
    for (m.cache[0..]) |*cache| try fill(cache, original);
    m.position = 73;
    inline for (.{ "rope_delta", "generation", "mtp_position", "mtp_generation" }) |field| if (@hasField(M, field)) {
        @field(m, field) = 19;
    };
    if (@hasField(M, "mtp_cache")) {
        m.mtp_cache = .{};
        try fill(&m.mtp_cache, original);
    }
    defer if (@hasField(M, "mtp_cache")) m.mtp_cache.deinit();
    if (@hasField(M, "draft")) {
        var draft: @import("dflash.zig").Draft = undefined;
        draft.cache = try mx.allocator.alloc(@import("dflash.zig").Cache, 2);
        @memset(draft.cache, .{});
        for (draft.cache) |*cache| try fill(cache, original);
        draft.position = 73;
        draft.projected_position = 71;
        draft.pending = try mx.retain(original);
        draft.started = true;
        m.draft = draft;
    }
    defer if (@hasField(M, "draft")) {
        for (m.draft.?.cache) |cache| {
            mx.free(cache.keys);
            mx.free(cache.values);
        }
        mx.allocator.free(m.draft.?.cache);
        mx.free(m.draft.?.pending);
    };
    if (@hasField(M, "dspark")) {
        var draft: @import("deepseek_dspark.zig").Draft = undefined;
        draft.keys = try mx.allocator.alloc(mx.Array, 3);
        for (draft.keys) |*key| key.* = try mx.retain(original);
        draft.position = 73;
        m.dspark = draft;
    }
    defer if (@hasField(M, "dspark")) {
        for (m.dspark.?.keys) |key| mx.free(key);
        mx.allocator.free(m.dspark.?.keys);
    };
    var saved = try states.State(M).init(&m);
    defer saved.deinit();
    saved.draft_hidden = try mx.retain(other);
    if (@hasDecl(M, "DraftCache")) try fill(&saved.head_cache, other);
    var tree: @import("drafter.zig").Drafter = undefined;
    tree.cache = @splat(.{});
    tree.offset = 73;
    for (&tree.cache) |*cache| try fill(cache, original);
    defer for (&tree.cache) |*cache| cache.deinit();
    saved.swapDFlash(&tree);
    try std.testing.expectEqual(@as(i32, 0), tree.offset);
    try std.testing.expectEqual(@as(i32, 73), saved.dflash_offset);
    saved.swap(&m);
    try std.testing.expectEqual(@as(i32, 0), m.position);
    for (m.cache[0..]) |*cache| {
        try equal(cache.*, mx.empty);
        try fill(cache, other);
    }
    m.position = 91;
    saved.swap(&m);
    try std.testing.expectEqual(@as(i32, 73), m.position);
    try std.testing.expectEqual(@as(i32, 91), saved.position);
    for (m.cache) |cache| try equal(cache, original);
    for (saved.cache) |cache| try equal(cache, other);
    var snapshot = try saved.clone();
    defer snapshot.deinit();
    try std.testing.expectEqual(saved.position, snapshot.position);
    try std.testing.expectEqual(saved.nbytes(), snapshot.nbytes());
    try std.testing.expect(snapshot.nbytes() > 0);
    try equalArray(snapshot.draft_hidden, other);
    if (@hasDecl(M, "DraftCache")) try equal(snapshot.head_cache, other);
    try std.testing.expectEqual(@as(i32, 73), snapshot.dflash_offset);
    for (snapshot.dflash_cache) |cache| try equal(cache, original);
    try mx.replace(&saved.draft_hidden, original);
    saved.swapDFlash(&tree);
    try std.testing.expectEqual(@as(i32, 73), tree.offset);
    for (snapshot.dflash_cache) |cache| try equal(cache, original);
    try equalArray(snapshot.draft_hidden, other);
    for (saved.cache) |*cache| {
        cache.deinit();
        try fill(cache, original);
    }
    for (snapshot.cache) |cache| try equal(cache, other);
    // Clone the populated attached draft state as well, after swapping it out.
    saved.swap(&m);
    var full = try saved.clone();
    defer full.deinit();
    try std.testing.expectEqual(saved.nbytes(), full.nbytes());
    if (@hasField(M, "mtp_cache")) try equal(full.mtp_cache, original);
    if (full.draft) |draft| {
        try equalArray(draft.pending, original);
        for (draft.cache) |cache| try equal(cache, original);
    }
    if (full.dspark) |draft| for (draft.keys) |key| try equalArray(key, original);
    saved.swap(&m);
    inline for (.{ "rope_delta", "generation", "mtp_position", "mtp_generation" }) |field| if (@hasField(M, field)) {
        try std.testing.expectEqual(19, @field(m, field));
    };
    if (@hasField(M, "mtp_cache")) try equal(m.mtp_cache, original);
    if (@hasField(M, "draft")) {
        for (m.draft.?.cache) |cache| try equal(cache, original);
        try equalArray(m.draft.?.pending, original);
        try std.testing.expectEqual(@as(i32, 73), m.draft.?.position);
        try std.testing.expectEqual(@as(i32, 71), m.draft.?.projected_position);
        try std.testing.expect(m.draft.?.started);
    }
    if (@hasField(M, "dspark")) {
        for (m.dspark.?.keys) |key| try equalArray(key, original);
        try std.testing.expectEqual(@as(i32, 73), m.dspark.?.position);
    }
}

pub fn check() !void {
    try mx.init();
    defer mx.shutdown();
    inline for (.{ @import("model.zig").Model, @import("gemma.zig").Model, @import("nemotron.zig").Model, @import("flash.zig").Model, @import("glm.zig").Model, @import("deepseek.zig").Model }) |M| try checkModel(M);
    try mx.check(mx.c.mlx_synchronize(mx.stream));
    var active: usize = 0;
    try mx.check(mx.c.mlx_get_active_memory(&active));
    try std.testing.expectEqual(@as(usize, 0), active);
    std.debug.print("PASS: all six backend cache layouts, MTP, DFlash and DSpark state preserve ownership across request switches\n", .{});
}
