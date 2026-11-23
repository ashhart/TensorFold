//! A salted state never reaches disk, and a salted lookup never reads a plain disk state; plain ones restore as before.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const rows = @import("rows_store.zig");
const fake = @import("rows_fake.zig");
const salt_key = @import("cache_salt_key.zig");
const t = std.testing;
const a = t.allocator;
const rules: pc.Rules = .{ .min_prompt = 0, .min_gap = 1, .lookahead = 1 };

fn options(root: []const u8) rows.Options {
    return .{ .sync_writes = true, .root = root, .identity = 853, .cap = 1 << 26, .floor = 0, .daily = 1 << 28, .lookahead = 1 };
}

test "the disk store keeps salted and plain apart: a salted state is not written, a salted lookup reads no plain one" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    defer a.free(root);
    var prompt: [3000]u32 = undefined;
    for (&prompt, 0..) |*token, i| token.* = @intCast(29 + i % 983);
    const key = try salt_key.salted(a, &prompt, 0x5a17);
    defer a.free(key);
    {
        var f: fake.Fake = .{ .gpa = a, .prompt = &prompt };
        const r = try rows.Rows.open(a, f.snapshots(), options(root));
        defer r.deinit();
        var s = pc.Store.init(a, f.snapshots(), rules, 1 << 30);
        defer s.deinit();
        s.disk = r;
        f.at = 2000;
        try t.expect(s.keep(&prompt, 2000, null, &.{}, &.{}));
        while (rows.write.step(r)) {}
        f.prompt = key;
        f.at = 2500;
        try t.expect(s.keep(key, 2500, null, &.{}, &.{})); // kept in RAM like any state
        while (rows.write.step(r)) {}
        try t.expectEqual(@as(usize, 1), r.states.items.len); // only the plain state went to disk
        try t.expectEqual(@as(u32, 2000), r.states.items[0].at);
        for (r.states.items[0].tokens) |w| try t.expect(w & 0x8000_0000 == 0); // no salted word, nor the salt, on disk
    }
    // a new store over the same folder: the plain prompt restores from disk, the salted key reads nothing
    var f: fake.Fake = .{ .gpa = a, .prompt = &prompt };
    const r = try rows.Rows.open(a, f.snapshots(), options(root));
    defer r.deinit();
    var s = pc.Store.init(a, f.snapshots(), rules, 1 << 30);
    defer s.deinit();
    s.disk = r;
    const plain = try s.begin(a, &prompt, 2900, &.{}, &.{}, null, &.{});
    defer a.free(plain.marks);
    try t.expectEqual(@as(u32, 2000), plain.from);
    try t.expectEqual(fake.fresh(&prompt), f.pass(&s, plain));
    const reads = r.counts.reads;
    for (s.entries.items) |e| e.expired = true;
    s.collectExpired();
    f.prompt = key;
    const salted = try s.begin(a, key, 2900, &.{}, &.{}, null, &.{});
    defer a.free(salted.marks);
    try t.expectEqual(@as(u32, 0), salted.from); // the whole prompt prefilled
    try t.expectEqual(fake.fresh(key), f.pass(&s, salted));
    try t.expectEqual(reads, r.counts.reads); // no disk read for it
}
