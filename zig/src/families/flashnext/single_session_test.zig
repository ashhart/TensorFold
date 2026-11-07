//! A constrained request borrows the original slot and unwraps the single host's existing snapshot owner.
const std = @import("std");
const Backend = @import("backend.zig").Backend;
const Engine = @import("engine.zig").Engine;
const fz = @import("replay.zig");
const snap = @import("snapshot.zig");

test "Flash Next constraint: one slot reaches allocation and an empty slot list is refused" {
    var run: fz.Run = undefined;
    run.tp = null;
    var engine: Engine = undefined;
    engine.r = &run;
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    try std.testing.expectError(error.BadBatchSlots, Backend.init(failing.allocator(), &engine, &.{}));
    try std.testing.expectError(error.OutOfMemory, Backend.init(failing.allocator(), &engine, &.{1024}));
}

test "Flash Next constraint: the actual restore entry unwraps owned states and retains raw batch states" {
    const Owner = struct {
        key: u64,
        state: *snap.State,
        fn unwrap(ptr: *anyopaque) *snap.State {
            return @as(*@This(), @ptrCast(@alignCast(ptr))).state;
        }
    };
    var state: snap.State = undefined;
    var owner: Owner = .{ .key = 91, .state = &state };
    var backend: Backend = undefined;
    backend.unwrap_snapshot = Owner.unwrap;
    try std.testing.expectEqual(&state, @import("prefill_metal.zig").savedState(&backend, &owner));
    try std.testing.expectEqual(@as(u64, 91), owner.key);
    backend.unwrap_snapshot = null;
    try std.testing.expectEqual(&state, @import("prefill_metal.zig").savedState(&backend, &state));
}

test "Flash Next constraint: temporary packed MTP scratch releases only its own engine buffers" {
    const metal = @import("metal");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    metal.allocator = arena.allocator();
    var run: fz.Run = undefined;
    run.arena = arena.allocator();
    run.device = .{};
    run.load_limit = 0;
    run.held = .empty;
    run.follow = null;
    const original = try run.buffer(128);
    const drafts = @import("batch_draft.zig");
    var state: drafts.State = .{};
    for (0..5) |_| {
        _ = try drafts.scratch(&state, &run, 7);
        try std.testing.expectEqual(@as(usize, 11), run.held.items.len);
        _ = try drafts.scratch(&state, &run, 7);
        try std.testing.expectEqual(@as(usize, 11), run.held.items.len);
        state.deinit(&run);
        try std.testing.expectEqual(@as(usize, 1), run.held.items.len);
        try std.testing.expectEqual(original.id, run.held.items[0].id);
    }
    state.deinit(&run);
    for (0..10) |index| {
        var failing = std.testing.FailingAllocator.init(arena.allocator(), .{ .fail_index = index });
        metal.allocator = failing.allocator();
        try std.testing.expectError(error.OutOfMemory, drafts.scratch(&state, &run, 7));
        try std.testing.expectEqual(@as(usize, 1), run.held.items.len);
        try std.testing.expectEqual(original.id, run.held.items[0].id);
        try std.testing.expect(state.x == null);
    }
}
