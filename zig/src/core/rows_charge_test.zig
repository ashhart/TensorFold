//! A family can save a small resident record yet need a whole image when restoring from disk.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const rows = @import("rows_store.zig");
const fake = @import("rows_fake.zig");
const t = std.testing;
const a = t.allocator;
const rules: pc.Rules = .{ .min_prompt = 0, .min_gap = 1, .lookahead = 1 };
fn options(root: []const u8) rows.Options {
    return .{ .sync_writes = true, .root = root, .identity = 1572, .cap = 1 << 26, .floor = 0, .daily = 1 << 28, .lookahead = 1 };
}
fn small(_: *anyopaque, _: u32) u64 {
    return 16;
}
fn actual(_: *anyopaque, saved: pc.Saved) u64 {
    const k: *fake.Kept = @ptrCast(@alignCast(saved));
    return k.img.len;
}
fn seed(root: []const u8, prompt: []const u32) !void {
    var f: fake.Fake = .{ .gpa = a, .prompt = prompt, .at = 1400 };
    const r = try rows.Rows.open(a, f.snapshots(), options(root));
    defer r.deinit();
    var s = pc.Store.init(a, f.snapshots(), rules, 1 << 30);
    defer s.deinit();
    s.disk = r;
    try t.expect(s.keep(prompt, f.at, null, &.{}, &.{}));
    while (rows.write.step(r)) {}
    try t.expectEqual(@as(usize, 1), r.states.items.len);
}

test "disk charge: actual blank size gates selection before allocation and is charged until eviction" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    defer a.free(root);
    var prompt: [1500]u32 = undefined;
    for (&prompt, 0..) |*token, i| token.* = @intCast(13 + i * 7);
    try seed(root, &prompt);
    for ([_]bool{ true, false }) |tight| {
        var f: fake.Fake = .{ .gpa = a, .prompt = &prompt };
        const original = f.snapshots();
        var img = original.vtable.rows.?.*;
        img.blank_bytes = original.vtable.bytes;
        var vtable = original.vtable.*;
        vtable.bytes = small;
        vtable.charged = actual;
        vtable.rows = &img;
        const family: pc.Snapshots = .{ .ptr = &f, .vtable = &vtable };
        const needed = original.vtable.bytes(&f, 1400);
        const r = try rows.Rows.open(a, family, options(root));
        defer r.deinit();
        var s = pc.Store.init(a, family, rules, if (tight) needed - 1 else needed);
        defer s.deinit();
        s.disk = r;
        const chosen = @import("rows_select.zig").choose(&s, &prompt, &.{}, &.{}, null);
        try t.expectEqual(!tight, chosen.restore != null);
        const plan = try s.begin(a, &prompt, 0, &.{}, &.{}, null, &.{});
        defer a.free(plan.marks);
        try t.expectEqual(@as(u32, if (tight) 0 else 1400), plan.from);
        try t.expectEqual(@as(usize, if (tight) 0 else 1), f.blanks);
        try t.expectEqual(@as(u64, if (tight) 0 else needed), s.held);
        if (!tight) {
            const entry = s.entries.items[0];
            try t.expectEqual(actual(&f, entry.saved), entry.bytes);
            try t.expectEqual(fake.fresh(&prompt), f.pass(&s, plan));
            try t.expect(f.restored_ok);
            s.remove(0);
            try t.expectEqual(@as(usize, 0), f.live);
            try t.expectEqual(@as(u64, 0), s.held);
        }
    }
}
