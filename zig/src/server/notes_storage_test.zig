//! Storage-only dependency boundary: no search index, lookup, inference or promotion.
test "storage erase prepares without mutation and survives journal replay" {
    const std = @import("std");
    const testing = std.testing;
    const Storage = @import("notes/storage.zig").Storage;
    const erase = @import("notes/erase_storage.zig");
    var bytes: @import("notes_storage_bytes_test.zig").Bytes = .{};
    const snapshot = @embedFile("notes/testdata/l6_bytes_snapshot.json");
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
    const path = try std.fs.path.join(a, &.{ dir, "notes.journal" });
    var problem: []const u8 = "";
    var after: usize = 0;
    var next_source: u64 = 0;
    {
        const s = try Storage.load(testing.allocator, snapshot, bytes.text(), path, &problem);
        defer s.deinit();
        try s.beginRun(a, .{});
        const n = s.model.notes.items.len;
        _ = try s.apply(a, .{
            .op = .add,
            .source = "w1",
            .told = 739900,
            .subject = "me",
            .attribute = "note:w1",
            .text = "My favourite colour is green.",
        });
        const id = s.model.notes.items[n].id;
        try testing.expectError(error.SnapshotNote, erase.prepare(s, a, s.model.notes.items[0].id));
        {
            var abandoned = try erase.prepare(s, a, id);
            defer abandoned.deinit();
            try testing.expectEqual(n + 1, s.model.notes.items.len);
            try testing.expectEqual(@as(usize, 1), s.log.items.items.len);
        }
        var pending = try erase.prepare(s, a, id);
        defer pending.deinit();
        try testing.expectEqualStrings("w1", pending.removed[0]);
        try testing.expectEqualStrings("My favourite colour is green.", pending.texts[0]);
        try pending.commit(a);
        try testing.expect(!pending.unsynced);
        try testing.expectError(error.AlreadyCommitted, pending.commit(a));
        try testing.expectEqual(n, s.model.notes.items.len);
        try testing.expectEqual(@as(usize, 0), s.log.items.items.len);
        try testing.expect(s.model.placeOfSource("w1") == null);
        after = n;
        next_source = s.log.next_source;
    }
    const s = try Storage.load(testing.allocator, snapshot, bytes.text(), path, &problem);
    defer s.deinit();
    try testing.expectEqual(after, s.model.notes.items.len);
    try testing.expectEqual(next_source, s.log.next_source);
    try testing.expect(s.model.placeOfSource("w1") == null);
}

test {
    _ = @import("notes_journal_oom_test.zig");
    const testing = @import("std").testing;
    testing.refAllDecls(@import("notes/model.zig"));
    testing.refAllDecls(@import("notes/write.zig"));
    testing.refAllDecls(@import("notes/journal.zig"));
    testing.refAllDecls(@import("notes/storage.zig"));
}

test "notes HTTP write parser keeps route, field and Unicode refusals without an index" {
    const std = @import("std");
    const testing = std.testing;
    const request = @import("notes/http_request.zig");
    const ops = @import("notes/operations.zig");
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: ops.Refusal = .{};
    try testing.expectEqual(@as(?request.Route, null), request.route("/v1/notebooks"));
    const r = request.route("/v1/notes").?;
    const add = try request.parse(a, "POST", r, "{\"text\":\"My colour is green.\",\"subject\":\"me\"}", &why);
    try testing.expectEqual(@import("notes/write.zig").Op.add, add.op);
    const gone = try request.parse(a, "DELETE", request.route("/v1/notes/30").?, "", &why);
    try testing.expectEqual(@as(u32, 30), gone.target);
    try testing.expectEqualStrings(@import("notes/write.zig").default_takeback, gone.text);
    const Cell = struct { method: []const u8 = "POST", path: []const u8 = "/v1/notes", body: []const u8, status: u16 };
    for ([_]Cell{
        .{ .body = "{\"text\":\"x\",\"unknown\":1}", .status = 400 },
        .{ .body = "{\"text\":\" \"}", .status = 400 },
        .{ .body = "{\"text\":\"x\",\"subject\":\" bad\"}", .status = 400 },
        .{ .body = "{\"text\":\"x\",\"replaces\":-1}", .status = 400 },
        .{ .method = "PUT", .body = "{\"text\":\"x\"}", .status = 405 },
        .{ .method = "DELETE", .body = "", .status = 405 },
        .{ .path = "/v1/notes/no", .body = "{}", .status = 404 },
        .{ .body = "[]", .status = 400 },
    }) |c| {
        try testing.expectError(error.Refused, request.parse(a, c.method, request.route(c.path).?, c.body, &why));
        try testing.expectEqual(c.status, why.status);
    }
}

test "journaled storage applies only durable records and replays without search" {
    const std = @import("std");
    const testing = std.testing;
    const storage = @import("notes/storage.zig");
    const journal = @import("notes/journal.zig");
    var bytes: @import("notes_storage_bytes_test.zig").Bytes = .{};
    const snapshot = @embedFile("notes/testdata/l6_bytes_snapshot.json");
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
    const path = try std.fs.path.join(a, &.{ dir, "notes.journal" });
    var problem: []const u8 = "";
    {
        const s = try storage.Storage.load(testing.allocator, snapshot, bytes.text(), path, &problem);
        defer s.deinit();
        try testing.expectError(error.MissingRun, s.apply(a, .{
            .op = .add,
            .source = "w1",
            .told = 739900,
            .subject = "me",
            .attribute = "note:w1",
            .text = "My favourite colour is green.",
        }));
        try s.beginRun(a, .{});
        try testing.expectError(error.AlreadyStarted, s.beginRun(a, .{}));
        const n = s.model.notes.items.len;
        const ops = @import("notes/operations.zig");
        var why: ops.Refusal = .{};
        var prepared = try ops.prepare(s, a, .{
            .op = .add,
            .subject = "me",
            .text = "My favourite colour is green.",
        }, 739900, &why);
        defer prepared.deinit();
        const json = @import("json");
        const values = @import("notes/http_values.zig");
        const body = try json.stringify(a, try values.writtenValue(a, .add, prepared.reply), .{});
        try testing.expectEqual(@as(usize, 0), s.log.items.items.len);
        try testing.expect(std.mem.indexOf(u8, body, "My favourite colour is green.") != null);
        try prepared.commit(a, &why);
        const out = prepared.reply;
        try testing.expect(out.note != null);
        try testing.expectEqualStrings("w1", out.source);
        try testing.expectEqualStrings("My favourite colour is green.", out.note.?.text);
        try testing.expectError(error.Refused, ops.perform(s, a, .{
            .op = .change,
            .target = out.note.?.id,
            .subject = "another owner",
            .text = "My favourite colour is blue.",
        }, 739901, &why));
        try testing.expectEqual(@as(u16, 400), why.status);
        try testing.expectEqual(n + 1, s.model.notes.items.len);
        var faults: journal.Faults = .{ .write = 0 };
        s.journal.?.faults = &faults;
        try testing.expectError(error.JournalIo, s.apply(a, .{
            .op = .add,
            .source = "w2",
            .told = 739901,
            .subject = "me",
            .attribute = "note:w2",
            .text = "My favourite colour is blue.",
        }));
        try testing.expectEqual(n + 1, s.model.notes.items.len);
        try testing.expectEqual(@as(usize, 1), s.log.items.items.len);
        try testing.expectEqual(@as(u64, 2), s.log.next_source);
        {
            var pending = try s.prepare(a, .{
                .op = .add,
                .source = "w2",
                .told = 739901,
                .subject = "me",
                .attribute = "note:w2",
                .text = "My favourite colour is blue.",
            });
            pending.deinit();
            try testing.expectEqual(n + 1, s.model.notes.items.len);
            try testing.expectEqual(@as(usize, 1), s.log.items.items.len);
        }
    }
    const s = try storage.Storage.load(testing.allocator, snapshot, bytes.text(), path, &problem);
    defer s.deinit();
    try testing.expectEqual(@as(usize, 1), s.replayed);
    const p = s.model.placeOfSource("w1").?;
    try testing.expectEqualStrings("My favourite colour is green.", s.model.notes.items[p].text());
    try testing.expectEqual(@as(?u32, null), s.model.placeOfSource("w2"));
    s.journal.?.poisoned = true;
    try testing.expectError(error.JournalPoisoned, s.apply(a, .{
        .op = .add,
        .source = "w2",
        .told = 739901,
        .subject = "me",
        .attribute = "note:w2",
        .text = "My favourite colour is blue.",
    }));
    try testing.expectEqual(@as(?u32, null), s.model.placeOfSource("w2"));
    const readonly = try storage.Storage.load(testing.allocator, snapshot, bytes.text(), null, &problem);
    defer readonly.deinit();
    try testing.expectError(error.ReadOnly, readonly.apply(a, .{
        .op = .add,
        .source = "w1",
        .told = 739900,
        .subject = "me",
        .attribute = "note:w1",
        .text = "My favourite colour is green.",
    }));
    try testing.expectEqual(@as(?u32, null), readonly.model.placeOfSource("w1"));
}

test "storage file startup requires the named tokenizer hash before journal access" {
    const std = @import("std");
    const testing = std.testing;
    const storage = @import("notes/storage.zig");
    const check = @import("notes/tokenizer_check.zig");
    var bytes: @import("notes_storage_bytes_test.zig").Bytes = .{};
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
    const path = try std.fs.path.join(a, &.{ dir, "snap.json" });
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "tokenizer.json", .data = "{}" });
    var problem: []const u8 = "";
    const sha = try check.tokenizerSha(a, testing.io, dir, 256, &problem);
    const tokenizer = try std.fmt.allocPrint(a, "\"tokenizer\":{{\"name\":\"bytes\",\"sha256\":\"{s}\"}}", .{sha});
    const snapshot = try std.mem.replaceOwned(u8, a, @embedFile("notes/testdata/l6_bytes_snapshot.json"), "\"tokenizer\":{\"name\":\"bytes\"}", tokenizer);
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "snap.json", .data = snapshot });
    try testing.expectError(error.NoModelDir, storage.Storage.open(
        testing.allocator,
        testing.io,
        path,
        bytes.text(),
        "",
        &problem,
    ));
    {
        const s = try storage.Storage.open(testing.allocator, testing.io, path, bytes.text(), dir, &problem);
        defer s.deinit();
        try testing.expectEqual(storage.Check.sha256, s.check);
    }
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "tokenizer.json", .data = "different" });
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "snap.json.journal", .data = "invalid journal\n" });
    try testing.expectError(error.TokenizerMismatch, storage.Storage.open(
        testing.allocator,
        testing.io,
        path,
        bytes.text(),
        dir,
        &problem,
    ));
    try testing.expectError(error.StreamTooLong, check.tokenizerSha(a, testing.io, dir, 1, &problem));
    try tmp.dir.deleteFile(testing.io, "tokenizer.json");
    try testing.expectError(error.FileNotFound, storage.Storage.open(
        testing.allocator,
        testing.io,
        path,
        bytes.text(),
        dir,
        &problem,
    ));
}

test "storage journal commits, replays and rejects another snapshot without a search index" {
    const std = @import("std");
    const testing = std.testing;
    const journal = @import("notes/journal.zig");
    const write = @import("notes/write.zig");
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
    const path = try std.fs.path.join(a, &.{ dir, "notes.journal" });
    const sha: [64]u8 = @splat('a');
    var records: std.ArrayList(write.Record) = .empty;
    var problem: []const u8 = "";
    {
        var j = try journal.Journal.open(testing.allocator, a, path, sha, &records, &problem);
        defer j.deinit();
        j.addRun(a, .{});
        try j.append(a, .{
            .op = .add,
            .source = "w1",
            .told = 739900,
            .subject = "owner",
            .attribute = "note:w1",
            .text = "My favourite colour is green.",
        });
        try testing.expect(!j.poisoned);
    }
    {
        var j = try journal.Journal.open(testing.allocator, a, path, sha, &records, &problem);
        defer j.deinit();
        try testing.expectEqual(@as(usize, 1), records.items.len);
        try testing.expectEqualStrings("w1", records.items[0].source);
        try testing.expectEqualStrings("My favourite colour is green.", records.items[0].text);
        try testing.expectEqual(@as(usize, 1), j.runs.items.len);
        var other: std.ArrayList(write.Record) = .empty;
        try testing.expectError(error.Locked, journal.Journal.open(
            testing.allocator,
            a,
            path,
            sha,
            &other,
            &problem,
        ));
        var faults: journal.Faults = .{ .write = 0 };
        j.faults = &faults;
        try testing.expectError(error.JournalIo, j.append(a, .{
            .op = .add,
            .source = "w2",
            .told = 739901,
            .subject = "owner",
            .attribute = "note:w2",
            .text = "My favourite colour is blue.",
        }));
        try testing.expect(!j.poisoned);
    }
    records.clearRetainingCapacity();
    {
        var j = try journal.Journal.open(testing.allocator, a, path, sha, &records, &problem);
        defer j.deinit();
        try testing.expectEqual(@as(usize, 1), records.items.len);
        try testing.expectEqualStrings("w1", records.items[0].source);
    }
    try testing.expectError(error.OtherSnapshot, journal.Journal.open(
        testing.allocator,
        a,
        path,
        @splat('b'),
        &records,
        &problem,
    ));
}
