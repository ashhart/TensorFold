//! The actual shared pocket round/save gate grows a native checkpoint and a fresh native owner reads identical rows.
const std = @import("std");
const living = @import("living");
const core = @import("core");
const native = @import("train_test.zig");
const cfg = @import("config.zig");
const E = @import("learning_native_fixture.zig").E;
const a = std.testing.allocator;
const Host = struct {
    r: *native.R,
    path: []const u8,
    fail_finish: *bool,
    pub fn layout(h: @This()) living.Layout {
        return @import("learning_profile.zig").layout(&h.r.e.c);
    }
    pub fn pocketReadSwiglu(h: @This(), l: usize, g: []f32, u: []f32, d: []f32) !usize {
        return h.r.pockets.read(l, g, u, d);
    }
    pub fn pocketSwiglu(h: @This(), l: usize, g: []const f32, u: []const f32, d: []const f32, used: usize) !void {
        try h.r.pockets.set(l, g, u, d, used);
    }
    pub fn saveBegin(h: @This()) !void {
        try @import("save_guard").begin(std.testing.io, h.path);
    }
    pub fn saveFinish(h: @This()) !void {
        if (h.fail_finish.*) return error.PlantedSaveFinish;
        try @import("save_guard").finish(std.testing.io, h.path);
    }
    pub fn pocketSaveSwiglu(h: @This(), layers: []const usize, neurons: usize, g: []f32, u: []f32, d: []f32) !usize {
        return @import("pocket_storage.zig").save(a, std.testing.io, h.path, h.r.e.reader.namespace, false, h.layout(), layers, neurons, g, u, d);
    }
};
fn digest(bytes: []const u8) [32]u8 {
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &out, .{});
    return out;
}
test "Qwen4 learning: real shared save failure stays pending, retries and reopens native original columns exactly" {
    var f = try @import("learning_native_fixture.zig").Fixture.open();
    defer f.deinit();
    const r = try native.R.init(a, f.e);
    defer r.close() catch unreachable;
    const trainer = try native.T.init(a, std.testing.io, f.e, &r.pockets, r);
    defer trainer.deinit();
    var fail_finish = false;
    const host: Host = .{ .r = r, .path = f.path, .fail_finish = &fail_finish };
    const Pocket = living.pocket.Pocket(native.T, living.families.qwen4);
    var p = Pocket.init(a);
    defer p.deinit();
    try p.load(trainer, host);
    p.used.items[0] = 1;
    p.gate.items[0] = 1;
    p.up.items[1] = 1;
    p.down.items[2] = 0.25;
    try p.asWritten(trainer, host);
    const original = try a.alloc([32]u8, f.e.reader.header.count());
    defer a.free(original);
    for (f.e.reader.header.keys(), original) |name, *hash| hash.* = digest((try f.e.reader.get(name)).bytes);
    fail_finish = true;
    try std.testing.expectError(error.PlantedSaveFinish, p.save(trainer, host));
    try std.testing.expect(p.save_pending);
    try std.testing.expectError(error.PocketUnresolved, p.mark());
    const preserved = try a.dupe(f32, r.pockets.layers[0].?.values);
    defer a.free(preserved);
    try std.testing.expectEqualSlices(f32, preserved, r.pockets.layers[0].?.values);
    fail_finish = false;
    _ = try p.save(trainer, host);
    try std.testing.expect(!p.save_pending and !p.saved_mismatch);
    var problem: cfg.Problem = .{};
    const reopened = try E.open(a, std.testing.io, f.path, f.b, .{ .capacity = 32, .window = 16, .streams = 2, .budget = std.math.maxInt(u64), .learning = true }, &problem, @import("../../../tests/qwen4/numeric_gpu.zig").Loader{});
    defer reopened.close() catch unreachable;
    const r2 = try native.R.init(a, reopened);
    defer r2.close() catch unreachable;
    try std.testing.expectEqual(@as(usize, 64), reopened.c.learned);
    for (r.pockets.layers, r2.pockets.layers) |old, new| {
        if (old) |o| {
            const got = new orelse return error.MissingSavedPocket;
            try std.testing.expectEqualSlices(f32, o.values, got.values);
            try std.testing.expectEqual(o.used, got.used);
        } else try std.testing.expect(new == null);
    }
    var temporary: std.heap.ArenaAllocator = .init(a);
    defer temporary.deinit();
    for (f.e.reader.header.keys(), original) |name, hash| {
        const got = try reopened.reader.get(name);
        const last = std.mem.lastIndexOfScalar(u8, name, '.');
        const stem = if (last) |at| name[0..at] else name;
        const base = try @import("pocket_matrix.zig").base(temporary.allocator(), &reopened.c, stem, got);
        try std.testing.expectEqual(hash, digest(base));
    }
    // An ordinary serving owner uses the exact same loaded pocket, while untaught streams still skip it.
    reopened.learning = r2.listener();
    const t2 = try native.T.init(a, std.testing.io, reopened, &r2.pockets, r2);
    defer t2.deinit();
    const ids = [_]u32{ 3, 5, 7, 2, 9 };
    const turn: living.lesson_route.Turn = .{ .decide = 0, .whole = true };
    const before = try trainer.step(&ids, 3, .loss, turn);
    const after = try t2.step(&ids, 3, .loss, turn);
    try std.testing.expectEqual(before.loss, after.loss);
    try std.testing.expectEqual(before.recalled, after.recalled);
}
