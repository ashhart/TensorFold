//! The Metal backend's Mamba slots: a verify refused after a window took its fresh slot gives it back (backend.zig).
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const cfg = @import("config.zig");
const st = @import("state.zig");
const Metal = @import("backend.zig").Metal;
const Masks = @import("../../core/grammar_mask.zig").Masks;

const be = lanes.backend;
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

test "a verify refused after a window took its Mamba slot gives the slot back to the pool" {
    const gpa = std.testing.allocator;
    var why: cfg.Why = .{};
    const c = try cfg.parse(gpa, cfg.lightning, &why);
    var device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    // only what a verify reads before its forward: the pool, the token ring, the caches and grammar bits (no weights)
    var b: Metal = .{
        .gpa = gpa,
        .m = undefined,
        .o = .{ .capacity = 4096, .streams = 2 },
        .pool = try st.Pool.init(gpa, device, c, st.poolSlots(2), 4),
        .scratch = undefined,
        .tokens = try device.buffer((1 << 16) * 4, opts),
        .prompt = undefined,
        .fence = undefined,
        .masks = try Masks.init(device, st.max_rows, c.vocab),
    };
    defer {
        b.masks.?.deinit();
        b.caches.deinit(gpa);
        b.tokens.deinit();
        b.pool.deinit();
    }
    const back = b.backend();
    var streams: [2]lanes.Stream = undefined;
    for (&streams) |*s| {
        s.* = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = &.{ 1, 2 }, .max_new = 8 });
        const cache = try gpa.create(st.Cache);
        cache.* = try st.Cache.init(device, c, 4096, false);
        cache.replay = 1; // the last verify's kept row replays first, so the next verify stores to a fresh slot
        try b.caches.put(gpa, s, cache);
    }
    defer for (&streams) |*s| {
        back.release(s);
        s.deinit(gpa);
    };
    const free = b.pool.free.items.len;
    var sampled: [2]u32 = undefined;
    var drafts: [0]u32 = .{};
    var out = [_]be.Verified{ .{ .sampled = sampled[0..1], .drafts = &drafts }, .{ .sampled = sampled[1..2], .drafts = &drafts } };
    const first: be.Window = .{ .stream = &streams[0], .pending = 1, .held = 0, .tokens = &.{}, .parents = null, .positions = &.{2} };

    // a later window is refused: the first one's slot is already taken
    const broken: be.Window = .{ .stream = &streams[1], .pending = 1, .held = 0, .tokens = &.{5}, .parents = null, .positions = &.{ 2, 9 } };
    try std.testing.expectError(error.PositionsNotAChain, back.verify(&.{ first, broken }, &out));
    try std.testing.expectEqual(free, b.pool.free.items.len);

    // the window's own grammar bits are refused
    var masked = first;
    masked.masks = &.{0};
    try std.testing.expectError(error.InvalidGrammarMasks, back.verify(&.{masked}, out[0..1]));
    try std.testing.expectEqual(free, b.pool.free.items.len);
}
