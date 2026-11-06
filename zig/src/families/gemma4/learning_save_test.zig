//! The real GeGLU pocket round/save gate rewrites disposable shards, retries a pending transaction and reopens rows.
const std = @import("std");
const living = @import("living");
const core = @import("core");
const n = @import("learning_numeric.zig");
const a = std.testing.allocator;
const T = @import("learning_train.zig").Trainer(n.E);
const Host = struct {
    e: *n.E,
    fail_finish: *bool,
    pub fn layout(h: @This()) living.Layout {
        return @import("learning_profile.zig").layout(h.e.r.shape);
    }
    pub fn pocketReadSwiglu(h: @This(), l: usize, g: []f32, u: []f32, d: []f32) !usize {
        return h.e.r.pockets.read(l, g, u, d);
    }
    pub fn pocketSwiglu(h: @This(), l: usize, g: []f32, u: []f32, d: []f32, used: usize) !void {
        try h.e.r.pockets.set(l, g, u, d, used);
    }
    pub fn saveBegin(h: @This()) !void {
        try @import("save_guard").begin(std.testing.io, h.e.dir);
    }
    pub fn saveFinish(h: @This()) !void {
        if (h.fail_finish.*) return error.PlantedSaveFinish;
        try @import("save_guard").finish(std.testing.io, h.e.dir);
    }
    pub fn pocketSaveSwiglu(h: @This(), layers: []const usize, neurons: usize, g: []f32, u: []f32, d: []f32) !usize {
        return living.shards.Io(core).writePending(a, std.testing.io, h.e.dir, living.families.gemma4, h.layout(), layers, neurons, &.{ g, u }, d);
    }
};
fn digest(bytes: []const u8) [32]u8 {
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &out, .{});
    return out;
}
test "Gemma4 learning: actual shared save gate retries, preserves base bytes and reopens portable GeGLU rows" {
    var f = try @import("learning_fixture.zig").Fixture.open();
    defer f.deinit();
    const t = try T.init(a, std.testing.io, f.e, &f.e.r.pockets, f.e);
    defer t.deinit();
    var fail = false;
    const host: Host = .{ .e = f.e, .fail_finish = &fail };
    var p = living.pocket.Pocket(T, living.families.gemma4).init(a);
    defer p.deinit();
    try p.load(t, host);
    p.used.items[0] = 1;
    p.gate.items[0] = 1;
    p.up.items[1] = 1;
    p.down.items[2] = 0.25;
    try p.asWritten(t, host);
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const mem = arena.allocator();
    var old = try core.Checkpoint.openModel(a, std.testing.io, f.path);
    defer old.close();
    const Item = struct { name: []const u8, hash: [32]u8 };
    var hashes: std.ArrayList(Item) = .empty;
    for (old.files.items) |file| for (file.names.keys()) |name| try hashes.append(mem, .{ .name = try mem.dupe(u8, name), .hash = digest((try old.get(name)).bytes) });
    fail = true;
    try std.testing.expectError(error.PlantedSaveFinish, p.save(t, host));
    try std.testing.expect(p.save_pending);
    try std.testing.expectError(error.PocketUnresolved, p.mark());
    fail = false;
    _ = try p.save(t, host);
    var reopened = try core.Checkpoint.openModel(a, std.testing.io, f.path);
    defer reopened.close();
    const config = .{ .learned = @as(usize, 64), .geo = .{ .hidden = @as(usize, 64), .intermediate = @as(usize, 64), .layers = @as(usize, 6) } };
    const r2 = try n.E.Runtime.init(a, f.b, f.e.r.shape, std.math.maxInt(u64));
    defer r2.close() catch unreachable;
    try r2.pockets.load(&reopened, config);
    for (f.e.r.pockets.layers, r2.pockets.layers) |left, right| if (left) |l| {
        const r = right orelse return error.MissingPocket;
        try std.testing.expectEqualSlices(f32, l.values, r.values);
        try std.testing.expectEqual(l.used, r.used);
    };
    const matrix = living.dense_matrix;
    for (hashes.items) |item| {
        const tensor = try reopened.get(item.name);
        const last = std.mem.lastIndexOfScalar(u8, item.name, '.') orelse item.name.len;
        const r = matrix.role(item.name[0..last]);
        const bytes = if (r) |role| try matrix.base(mem, 64, 64, 64, role, std.meta.stringToEnum(matrix.Part, item.name[last + 1 ..]).?, matrix.View.of(tensor)) else tensor.bytes;
        try std.testing.expectEqual(item.hash, digest(bytes));
    }
}
