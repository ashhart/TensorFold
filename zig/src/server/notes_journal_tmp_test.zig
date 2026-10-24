//! A journal rewrite's temporary file: a second server never touches it, and a planted link never redirects it.
const std = @import("std");
const rig = @import("notes_storage_rig_test.zig");
const ops = @import("notes/operations.zig");
const journal_mod = @import("notes/journal.zig");
const tmp_file = @import("notes/journal_tmp.zig");
const write = @import("notes/write.zig");
const erase = @import("notes/erase_storage.zig");
const testing = std.testing;

fn add(r: *rig.Rig, a: std.mem.Allocator, text: []const u8) !u32 {
    var why: ops.Refusal = .{};
    var pending = try ops.prepare(r.store(), a, .{ .op = .add, .text = text }, 739900000000, &why);
    defer pending.deinit();
    const id = pending.reply.note.?.id;
    try pending.commit(a, &why);
    return id;
}

/// What the second process saw, written by the first once the child exited.
const Second = struct {
    journal: [:0]const u8,
    status: c_int = -1, // the child's exit code: 10 refused as Locked with our temp intact, 11 Locked but the
    // temp gone, 12 opened (no lock), 13 another error
    temp_seen: bool = false,

    /// A second server starting while the first is between its rewrite's sync and its rename: another process.
    fn hook(ctx: *anyopaque, tmp: [:0]const u8) void {
        const s: *Second = @ptrCast(@alignCast(ctx));
        s.temp_seen = tmp_file.idOf(-1, tmp) != null;
        const pid = std.c.fork();
        if (pid == 0) child(s.journal, tmp);
        var status: c_int = 0;
        _ = std.c.waitpid(pid, &status, 0);
        s.status = if (status & 0x7f == 0) (status >> 8) & 0xff else -1;
    }

    /// The child: the journal opened as a starting server opens it (page memory only, no threads), then exit.
    fn child(path: [:0]const u8, tmp: [:0]const u8) noreturn {
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        var records: std.ArrayList(write.Record) = .empty;
        var problem: []const u8 = "";
        const sha: [64]u8 = @splat('0');
        var code: c_int = 13;
        if (journal_mod.Journal.open(std.heap.page_allocator, arena.allocator(), path, sha, &records, &problem)) |_| {
            code = 12;
        } else |e| {
            if (e == error.Locked) code = if (tmp_file.idOf(-1, tmp) != null) 10 else 11;
        }
        std.c._exit(code);
    }
};

test "a second server (another process) starting mid-rewrite is refused as Locked and leaves the first's temp" {
    const r = try rig.Rig.init(testing.allocator);
    defer r.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const keep = try add(r, a, "My favourite colour is green.");
    const gone = try add(r, a, "I live in Oslo.");
    _ = keep;
    const path = try std.fmt.allocPrintSentinel(a, "{s}.journal", .{r.snap}, 0);
    var second: Second = .{ .journal = path };
    r.store().journal.?.hook = Second.hook;
    r.store().journal.?.hook_ctx = &second;
    try eraseOne(r, a, gone); // fails (a 500) if its rename finds its temp gone
    r.store().journal.?.hook = null;
    try testing.expect(second.temp_seen);
    try testing.expectEqual(@as(c_int, 10), second.status);
    const j = try r.journal(a);
    try testing.expect(std.mem.indexOf(u8, j, "Oslo") == null and std.mem.indexOf(u8, j, "green") != null);
    var it = r.tmp.dir.iterate();
    while (try it.next(testing.io)) |e| try testing.expect(std.mem.indexOf(u8, e.name, ".tmp") == null); // no temp left
}

test "a symlink or hardlink planted at the old temp name and at every name a rewrite tries changes no other file" {
    const r = try rig.Rig.init(testing.allocator);
    defer r.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    _ = try add(r, a, "My favourite colour is green.");
    const gone = try add(r, a, "I live in Oslo.");
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{r.tmp.sub_path});
    const journal = try std.fmt.allocPrint(a, "{s}.journal", .{r.snap});
    try r.tmp.dir.writeFile(testing.io, .{ .sub_path = "sentinel-a", .data = "SENTINEL A: must not change" });
    try r.tmp.dir.writeFile(testing.io, .{ .sub_path = "sentinel-b", .data = "SENTINEL B: must not change" });
    const sa = try std.fmt.allocPrintSentinel(a, "{s}/sentinel-a", .{dir}, 0);
    const sb = try std.fmt.allocPrintSentinel(a, "{s}/sentinel-b", .{dir}, 0);
    // the names the rewrite will try (the random part fixed for the test), a symlink and a hardlink in turn
    tmp_file.fixed_random = 0;
    defer tmp_file.fixed_random = null;
    const first = tmp_file.next();
    const legacy = try std.fmt.allocPrintSentinel(a, "{s}.tmp", .{journal}, 0);
    try testing.expectEqual(@as(c_int, 0), std.c.symlink(sa, legacy));
    for (0..8) |k| {
        const name = try std.fmt.allocPrintSentinel(a, "{s}.tmp.{d}.{d}.{x:0>16}", .{ journal, std.c.getpid(), first + k, @as(u64, 0) }, 0);
        try testing.expectEqual(@as(c_int, 0), if (k % 2 == 0) std.c.symlink(sb, name) else std.c.link(sb, name));
    }
    try eraseOne(r, a, gone);
    try testing.expect(std.mem.indexOf(u8, try r.journal(a), "Oslo") == null);
    try testing.expectEqualStrings("SENTINEL A: must not change", try r.tmp.dir.readFileAlloc(testing.io, "sentinel-a", a, .unlimited));
    try testing.expectEqualStrings("SENTINEL B: must not change", try r.tmp.dir.readFileAlloc(testing.io, "sentinel-b", a, .unlimited));
    // a restart removes the stale names (the links themselves, never what they point to)
    try r.restart();
    try testing.expectEqual(@as(usize, 9), r.store().journal.?.stale);
    try testing.expectEqualStrings("SENTINEL A: must not change", try r.tmp.dir.readFileAlloc(testing.io, "sentinel-a", a, .unlimited));
    try testing.expectEqualStrings("SENTINEL B: must not change", try r.tmp.dir.readFileAlloc(testing.io, "sentinel-b", a, .unlimited));
}

test "a rewrite that fails before its rename removes its own temp file (known by device and inode), nothing else" {
    try rewriteFails();
}

fn eraseOne(r: *rig.Rig, a: std.mem.Allocator, id: u32) !void {
    var p = try erase.prepare(r.store(), a, id);
    defer p.deinit();
    try p.commit(a);
}

fn rewriteFails() !void {
    const r = try rig.Rig.init(testing.allocator);
    defer r.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    _ = try add(r, a, "My favourite colour is green.");
    const gone = try add(r, a, "I live in Oslo.");
    const before = try r.journal(a);
    const before_model = r.store().model;
    var faults: journal_mod.Faults = .{};
    r.store().journal.?.faults = &faults;
    faults.sync = faults.syncs; // the new file's sync, before the rename
    try testing.expectError(error.JournalIo, eraseOne(r, a, gone));
    try testing.expectEqualStrings(before, try r.journal(a));
    try testing.expect(r.store().model == before_model and r.store().model.placeOf(gone) != null); // nothing swapped in
    var it = r.tmp.dir.iterate();
    while (try it.next(testing.io)) |e| try testing.expect(std.mem.indexOf(u8, e.name, ".tmp") == null); // no temp left
}
