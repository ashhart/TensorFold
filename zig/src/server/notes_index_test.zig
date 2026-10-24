//! Index build and storage rebuild: no lookup, decode, live fetch or promotion.
const std = @import("std");
const testing = std.testing;
const index = @import("notes/index.zig");
const types = @import("notes/types.zig");
const model_mod = @import("notes/model.zig");
const sa = @import("notes/sa.zig");
const Storage = @import("notes/storage.zig").Storage;
const snapshot = @embedFile("notes/testdata/l6_bytes_snapshot.json");

test {
    testing.refAllDecls(@import("notes/index.zig"));
    testing.refAllDecls(@import("notes/sa.zig"));
    testing.refAllDecls(@import("notes/exact_alloc.zig"));
    testing.refAllDecls(@import("notes/alias.zig"));
    testing.refAllDecls(@import("notes/bounded.zig"));
    testing.refAllDecls(@import("notes/value.zig"));
    testing.refAllDecls(@import("notes/layer.zig"));
    testing.refAllDecls(@import("notes/merge.zig"));
    testing.refAllDecls(@import("notes/core_build.zig"));
    testing.refAllDecls(@import("notes/index_build.zig"));
}

test "index re-exports the snapshot types storage already uses" {
    comptime {
        if (types.State != index.State) @compileError("index.State must be types.State");
        if (types.Hop != index.Hop) @compileError("index.Hop must be types.Hop");
        if (types.Settings != index.Settings) @compileError("index.Settings must be types.Settings");
        if (types.LoadError != index.LoadError) @compileError("index.LoadError must be types.LoadError");
        if (types.none_key != index.none_key) @compileError("index.none_key must be types.none_key");
        if (types.max_n != index.max_n) @compileError("index.max_n must be types.max_n");
    }
}

/// One view's notes holding `g` by the suffix array (each once, by place), against the postings' list.
fn notesOf(a: std.mem.Allocator, b: *const sa.Base, arr: sa.Packed, g: index.Gram, punct: bool) ![]u32 {
    var keys: [index.max_n]u32 = undefined;
    var len: usize = 0;
    while (len < index.max_n) : (len += 1) {
        keys[len] = @truncate(g >> @intCast(32 * len));
        if (keys[len] == index.none_key) break;
    }
    var out: std.ArrayList(u32) = .empty;
    var buf: [index.max_n]u32 = undefined;
    const pat = b.codes(keys[0..len], &buf) orelse return out.items;
    if (len == 1 and !punct and !b.wordyCode(pat[0])) return out.items;
    const r = b.range(arr, pat);
    for (r[0]..r[1]) |k| {
        const p = b.placeOf(arr.get(k));
        if (std.mem.indexOfScalar(u32, out.items, p) == null) try out.append(a, p);
    }
    std.sort.pdq(u32, out.items, {}, std.sort.asc(u32));
    return out.items;
}

fn sameAsPostings(a: std.mem.Allocator, post: *const index.Index, sx: *const index.Index) !usize {
    const b = sx.base.?;
    var bad: usize = 0;
    var it = post.grams.iterator();
    while (it.next()) |kv| {
        const p = kv.value_ptr.*;
        const views = [2][]const u32{ post.post[p.cur_off..][0..p.cur_len], post.post[p.hist_off..][0..p.hist_len] };
        for (views, [2]sa.Packed{ b.sa_cur, b.sa_hist }) |want, arr| {
            const sorted = try a.dupe(u32, want);
            std.sort.pdq(u32, sorted, {}, std.sort.asc(u32));
            if (!std.mem.eql(u32, sorted, try notesOf(a, b, arr, kv.key_ptr.*, post.settings.punct))) bad += 1;
        }
    }
    return bad;
}

test "every n-gram's notes by the suffix array are its postings, and the snapshot's distinct counts" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const m = try model_mod.parse(testing.allocator, snapshot);
    defer m.deinit();
    const post = try index.Index.build(testing.allocator, m);
    defer post.deinit();
    const sx = try index.Index.buildKind(testing.allocator, m, .sa);
    defer sx.deinit();
    try testing.expectEqual(m.notes.items.len, post.notes());
    try testing.expectEqual(post.notes(), sx.notes());
    try testing.expect(sx.base != null);
    try testing.expectEqual(@as(usize, 0), try sameAsPostings(a, post, sx));
    const b = sx.base.?;
    const Keep = struct {
        fn wordy(idx: *const index.Index, key: u32) bool {
            for (idx.vkey, 0..) |k, t| if (k == key) return idx.vwordy[t];
            return false;
        }
    };
    try testing.expectEqual(@as(usize, m.grams_current), sa.distinctGrams(b, b.sa_cur, 1, 3, sx, Keep.wordy));
    try testing.expectEqual(@as(usize, m.grams_history), sa.distinctGrams(b, b.sa_hist, 1, 3, sx, Keep.wordy));
    const All = struct {
        fn any(_: *const index.Index, _: u32) bool {
            return true;
        }
    };
    try testing.expect(sa.distinctGrams(b, b.sa_cur, 1, 3, sx, All.any) > m.grams_current);
}

test "a vocabulary the suffix array cannot match as the postings do is refused" {
    const at = std.mem.indexOf(u8, snapshot, "\"wordy\":\"").? + "\"wordy\":\"".len;
    const bad = try testing.allocator.dupe(u8, snapshot);
    defer testing.allocator.free(bad);
    try testing.expect(bad[at + 'a'] == '1');
    bad[at + 'a'] = '0';
    const m = try model_mod.parse(testing.allocator, bad);
    defer m.deinit();
    try testing.expectError(error.Unsupported, index.Index.buildKind(testing.allocator, m, .sa));
    const post = try index.Index.buildKind(testing.allocator, m, .postings);
    defer post.deinit();
}

test "an unsupported suffix-array load with a journal path closes the journal" {
    var bytes: @import("notes_storage_bytes_test.zig").Bytes = .{};
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
    const path = try std.fs.path.join(a, &.{ dir, "notes.journal" });
    const at = std.mem.indexOf(u8, snapshot, "\"wordy\":\"").? + "\"wordy\":\"".len;
    const bad = try testing.allocator.dupe(u8, snapshot);
    defer testing.allocator.free(bad);
    try testing.expect(bad[at + 'a'] == '1');
    bad[at + 'a'] = '0';
    var problem: []const u8 = "";
    try testing.expectError(error.Unsupported, Storage.loadKind(testing.allocator, bad, bytes.text(), path, .sa, &problem));
    const post = try Storage.loadKind(testing.allocator, bad, bytes.text(), path, .postings, &problem);
    defer post.deinit();
    try testing.expectEqual(index.Kind.postings, post.kind);
}

test "storage load builds the default suffix-array index, postings on request" {
    var bytes: @import("notes_storage_bytes_test.zig").Bytes = .{};
    var problem: []const u8 = "";
    {
        const s = try Storage.load(testing.allocator, snapshot, bytes.text(), null, &problem);
        defer s.deinit();
        try testing.expectEqual(index.Kind.sa, s.kind);
        try testing.expect(s.idx != null);
        try testing.expectEqual(s.model.notes.items.len, s.idx.?.notes());
        try testing.expect(s.idx.?.base != null);
        try testing.expectEqual(s.model.entities.items.len, s.idx.?.entity_names.len);
        try testing.expectEqualStrings(s.model.entities.items[0], s.idx.?.entity_names[0]);
        const ids = try bytes.text().encode(testing.allocator, " Ana", false);
        defer testing.allocator.free(ids);
        var keys: index.Phrase = @splat(index.none_key);
        for (ids, keys[0..ids.len]) |id, *key| key.* = s.idx.?.keyOf(id);
        try testing.expect(s.idx.?.aliasOf(keys) != null);
    }
    const s = try Storage.loadKind(testing.allocator, snapshot, bytes.text(), null, .postings, &problem);
    defer s.deinit();
    try testing.expectEqual(index.Kind.postings, s.kind);
    try testing.expect(s.idx != null);
    try testing.expectEqual(s.model.notes.items.len, s.idx.?.notes());
    try testing.expect(s.idx.?.base == null);
}

test "a committed write rebuilds the index; abandoning prepare leaves it" {
    var bytes: @import("notes_storage_bytes_test.zig").Bytes = .{};
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
    const path = try std.fs.path.join(a, &.{ dir, "notes.journal" });
    var problem: []const u8 = "";
    const s = try Storage.load(testing.allocator, snapshot, bytes.text(), path, &problem);
    defer s.deinit();
    try s.beginRun(a, .{});
    const n = s.idx.?.notes();
    {
        var abandoned = try s.prepare(a, .{
            .op = .add,
            .source = "w1",
            .told = 739900,
            .subject = "me",
            .attribute = "note:w1",
            .text = "My favourite colour is green.",
        });
        defer abandoned.deinit();
        try testing.expectEqual(n, s.idx.?.notes());
        try testing.expectEqual(n + 1, abandoned.idx.notes());
    }
    try testing.expectEqual(n, s.idx.?.notes());
    _ = try s.apply(a, .{
        .op = .add,
        .source = "w1",
        .told = 739900,
        .subject = "me",
        .attribute = "note:w1",
        .text = "My favourite colour is green.",
    });
    try testing.expectEqual(n + 1, s.idx.?.notes());
    try testing.expectEqual(s.model.notes.items.len, s.idx.?.notes());
}

test "erase rebuilds the index from the kept notes" {
    const erase = @import("notes/erase_storage.zig");
    var bytes: @import("notes_storage_bytes_test.zig").Bytes = .{};
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
    const path = try std.fs.path.join(a, &.{ dir, "notes.journal" });
    var problem: []const u8 = "";
    const s = try Storage.load(testing.allocator, snapshot, bytes.text(), path, &problem);
    defer s.deinit();
    try s.beginRun(a, .{});
    const n = s.idx.?.notes();
    _ = try s.apply(a, .{
        .op = .add,
        .source = "w1",
        .told = 739900,
        .subject = "me",
        .attribute = "note:w1",
        .text = "My favourite colour is green.",
    });
    try testing.expectEqual(n + 1, s.idx.?.notes());
    const id = s.model.notes.items[n].id;
    var pending = try erase.prepare(s, a, id);
    defer pending.deinit();
    try pending.commit(a);
    try testing.expectEqual(n, s.idx.?.notes());
    try testing.expectEqual(s.model.notes.items.len, s.idx.?.notes());
}
