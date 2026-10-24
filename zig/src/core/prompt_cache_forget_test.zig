//! Store.forgetSpan regressions plant kept and learned states, erase matching copies, and preserve unrelated ones.
const std = @import("std");
const libc = @import("sys").c;
const pc = @import("prompt_cache.zig");
const imprint = @import("prompt_imprint.zig");
const rmTree = @import("prompt_imprint_test.zig").rmTree;
const Fake = @import("prompt_cache_fake.zig").Fake;
const fresh = @import("prompt_cache_test.zig").fresh;
const Store = pc.Store;
const testing = std.testing;

const note = [_]u32{ 50, 51, 52, 53 }; // the erased note's tokens
const system = [_]u32{ 1, 2, 3 };
const with_note = system ++ note ++ [_]u32{ 9, 9, 4 }; // a block holding the note after the client's system prompt
const without = system ++ [_]u32{ 60, 61, 62, 63 } ++ [_]u32{ 9, 9, 4 }; // an unrelated block

test "forgetSpan cannot clear an unacknowledged peer even after local entries are gone" {
    var f: Fake = .{ .gpa = testing.allocator };
    var family = f.snapshots();
    var vtable = family.vtable.*;
    vtable.forget_peers_clear = struct {
        fn clear(_: *anyopaque) bool {
            return false;
        }
    }.clear;
    family.vtable = &vtable;
    var store = Store.init(testing.allocator, family, .{}, 1 << 20);
    defer store.deinit();
    for (0..2) |_| {
        const out = store.forgetSpan(&note, &.{});
        try testing.expectEqual(@as(u32, 1), out.held);
        try testing.expect(!out.complete());
    }
    vtable.forget_peers_clear = struct {
        fn clear(_: *anyopaque) bool {
            return true;
        }
    }.clear;
    try testing.expectEqual(@as(u32, 0), store.forgetSpan(&note, &.{}).held);
}

test "forgetSpan folder diagnostics are owned bounded deduplicated and preserve overflow" {
    var out: pc.Forgotten = .{};
    out.folders.add(42, .unindexed);
    out.folders.add(42, .unindexed);
    try testing.expectEqual(@as(u8, 1), out.folders.len);
    var copied = out;
    out.folders.items[0].identity = 99;
    try testing.expectEqual(@as(u64, 42), copied.folders.items[0].identity);
    for (0..10) |i| copied.folders.add(i, .appending);
    try testing.expectEqual(@as(u8, 8), copied.folders.len);
    try testing.expectEqual(@as(u32, 3), copied.folders.omitted);
    var merged: pc.Forgotten = .{};
    merged.folders.merge(copied.folders);
    merged.folders.merge(copied.folders);
    try testing.expectEqual(@as(u8, 8), merged.folders.len);
    try testing.expectEqual(@as(u32, 6), merged.folders.omitted);
}

test "forgetSpan searches sibling identities created after the current index opened" {
    var buf: [128]u8 = undefined;
    const root = try std.fmt.bufPrint(&buf, "/tmp/tf-learn-sibling-{d}", .{libc.getpid()});
    defer rmTree(root);
    var im = try imprint.Imprint.open(testing.allocator, root, 3, 1 << 20);
    defer im.deinit();
    try testing.expectEqual(@as(u64, 0), im.others);
    const key = imprint.Imprint.keyOf(&with_note);
    {
        var other = try imprint.Imprint.open(testing.allocator, root, 4, 1 << 20);
        defer other.deinit();
        try other.add(key, with_note.len, &with_note, &.{}, 0);
        try other.add(imprint.Imprint.keyOf(&without), without.len, &without, &.{}, 0);
    }
    var f: Fake = .{ .gpa = testing.allocator };
    var s = Store.init(testing.allocator, f.learned(), .{}, 1 << 20);
    defer s.deinit();
    s.imprint = &im;
    const out = s.forgetSpan(&note, &.{});
    try testing.expectEqual(@as(u32, 0), out.learned);
    try testing.expectEqual(@as(u8, 1), out.folders.len);
    try testing.expectEqual(@as(u64, 4), out.folders.items[0].identity);
    try testing.expectEqual(.unowned_match, out.folders.items[0].reason);
    try testing.expectEqual(@as(u32, 0), out.held);
    try testing.expectEqual(@as(u32, 1), out.unmanaged);
    try testing.expect(!out.complete());
    var other = try imprint.Imprint.inspect(testing.allocator, root, 4);
    defer other.deinit();
    try testing.expect(other.has(key)); // do not rewrite a potentially active sibling server's index
    try testing.expect(other.has(imprint.Imprint.keyOf(&without)));
}

test "learned sibling inspection refuses a torn index without repairing it" {
    var buf: [128]u8 = undefined;
    const root = try std.fmt.bufPrint(&buf, "/tmp/tf-learn-inspect-{d}", .{libc.getpid()});
    defer rmTree(root);
    var im = try imprint.Imprint.open(testing.allocator, root, 3, 1 << 20);
    defer im.deinit();
    try im.add(imprint.Imprint.keyOf(&with_note), with_note.len, &with_note, &.{}, 0);
    {
        var scanned = try imprint.Imprint.inspect(testing.allocator, root, 3);
        defer scanned.deinit();
        try testing.expectEqual(@as(usize, 1), scanned.metas.items.len);
    }
    var path: [1100]u8 = undefined;
    const fd = libc.open(try std.fmt.bufPrintSentinel(&path, "{s}/index", .{im.dir}, 0), .{ .ACCMODE = .WRONLY, .APPEND = true }, @as(libc.mode_t, 0));
    if (fd < 0) return error.TestFile;
    defer _ = libc.close(fd);
    try testing.expectEqual(@as(isize, 3), libc.write(fd, "bad", 3));
    const before = try im.indexScratchBytes();
    try testing.expectError(error.ImprintAppending, imprint.Imprint.inspect(testing.allocator, root, 3));
    try testing.expectEqual(before, try im.indexScratchBytes());
    var current = try imprint.Imprint.open(testing.allocator, root, 4, 1 << 20);
    defer current.deinit();
    var f: Fake = .{ .gpa = testing.allocator };
    var store = Store.init(testing.allocator, f.learned(), .{}, 1 << 20);
    defer store.deinit();
    store.imprint = &current;
    const held = store.forgetSpan(&note, &.{});
    try testing.expectEqual(@as(u32, 0), held.held);
    try testing.expectEqual(@as(u32, 1), held.unmanaged);
    try testing.expectEqual(@as(u32, 0), held.failed);
    try testing.expect(!current.forget_pending);
    const junk = std.mem.zeroes([33]u8);
    try testing.expectEqual(@as(isize, junk.len), libc.write(fd, &junk, junk.len));
    try testing.expectError(error.ImprintRead, imprint.Imprint.inspect(testing.allocator, root, 3));
    const corrupt = store.forgetSpan(&note, &.{});
    try testing.expectEqual(@as(u32, 0), corrupt.failed);
    try testing.expectEqual(@as(u32, 1), corrupt.unmanaged);
    try testing.expect(!current.forget_pending);
}

test "forgetSpan names an unindexed sibling without retaining retry work" {
    var buf: [128]u8 = undefined;
    const root = try std.fmt.bufPrint(&buf, "/tmp/tf-learn-sibling-missing-{d}", .{libc.getpid()});
    defer rmTree(root);
    var im = try imprint.Imprint.open(testing.allocator, root, 3, 1 << 20);
    defer im.deinit();
    var other = try imprint.Imprint.open(testing.allocator, root, 4, 1 << 20);
    defer other.deinit(); // identity exists but has never written an index
    var f: Fake = .{ .gpa = testing.allocator };
    var s = Store.init(testing.allocator, f.learned(), .{}, 1 << 20);
    defer s.deinit();
    s.imprint = &im;
    const out = s.forgetSpan(&note, &.{});
    try testing.expectEqual(@as(u32, 0), out.failed);
    try testing.expectEqual(@as(u32, 0), out.held);
    try testing.expectEqual(@as(u32, 1), out.unmanaged);
    try testing.expect(!out.complete());
    try testing.expect(!im.forget_pending);
    try other.beginWrite(42); // the first payload may precede the first index
    const again = s.forgetSpan(&note, &.{});
    try testing.expectEqual(@as(u32, 0), again.failed);
    try testing.expectEqual(@as(u32, 0), again.held);
    try testing.expectEqual(@as(u32, 1), again.unmanaged);
    try testing.expect(!im.forget_pending);
}

test "learned batch removal preserves unrelated records across restart" {
    var buf: [128]u8 = undefined;
    const root = try std.fmt.bufPrint(&buf, "/tmp/tf-learn-batch-{d}", .{libc.getpid()});
    defer rmTree(root);
    const a = [_]u32{ 71, 72 };
    const b = [_]u32{ 81, 82 };
    const c = [_]u32{ 91, 92 };
    const ka = imprint.Imprint.keyOf(&a);
    const kb = imprint.Imprint.keyOf(&b);
    const kc = imprint.Imprint.keyOf(&c);
    {
        var im = try imprint.Imprint.open(testing.allocator, root, 3, 1 << 20);
        defer im.deinit();
        for ([_][]const u32{ &a, &b, &c }) |tokens| try im.add(imprint.Imprint.keyOf(tokens), 2, tokens, &.{}, 0);
        try im.removeMany(&.{ ka, kc, ka }); // duplicate keys must not double-free
        try testing.expect(!im.has(ka) and im.has(kb) and !im.has(kc));
        try testing.expectEqual(@as(usize, 1), im.metas.items.len);
        var orphan: [1100]u8 = undefined;
        const fd = libc.open(try Fake.file(&orphan, im.dir, ka), .{ .ACCMODE = .WRONLY, .CREAT = true }, @as(libc.mode_t, 0o600));
        if (fd < 0) return error.TestFile;
        _ = libc.close(fd); // simulate the crash/failure window between index commit and file unlink
        try testing.expect(exists(im.dir, ka));
    }
    var im = try imprint.Imprint.open(testing.allocator, root, 3, 1 << 20);
    defer im.deinit();
    try testing.expect(!im.has(ka) and im.has(kb) and !im.has(kc));
    try testing.expectEqual(@as(usize, 1), im.metas.items.len);
    try testing.expect(!exists(im.dir, ka)); // startup cleanup removes the unindexed state
}

test "forgetSpan removes the learned index before a failing file unlink" {
    var buf: [128]u8 = undefined;
    const root = try std.fmt.bufPrint(&buf, "/tmp/tf-learn-order-{d}", .{libc.getpid()});
    defer rmTree(root);
    var im = try imprint.Imprint.open(testing.allocator, root, 3, 1 << 20);
    defer im.deinit();
    const key = imprint.Imprint.keyOf(&with_note);
    try im.add(key, with_note.len, &with_note, &.{}, 0);
    const Check = struct {
        im: *imprint.Imprint,
        observed: bool = false,
        fn unlink(ptr: *anyopaque, dir: [:0]const u8, k: u64) !void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.observed = !self.im.has(k) and !try fileHolds(dir, "index", &note);
            return error.LearnedUnlink;
        }
    };
    var check: Check = .{ .im = &im };
    var f: Fake = .{ .gpa = testing.allocator };
    var vt = f.learned().vtable.*;
    vt.forget_checked = Check.unlink;
    var s = Store.init(testing.allocator, .{ .ptr = &check, .vtable = &vt }, .{}, 1 << 20);
    defer s.deinit();
    s.imprint = &im;
    const out = s.forgetSpan(&note, &.{});
    try testing.expect(check.observed);
    try testing.expectEqual(@as(u32, 1), out.failed);
    try testing.expectEqual(@as(u32, 0), out.learned);
    try testing.expect(!out.complete());
    try testing.expect(!im.has(key));
    const again = s.forgetSpan(&note, &.{});
    try testing.expect(again.failed > 0); // missing index records must not make leftover files look clear
    try testing.expect(!again.complete());
}

test "forgetSpan preserves learned files when the batch index cannot be rewritten" {
    var buf: [128]u8 = undefined;
    const root = try std.fmt.bufPrint(&buf, "/tmp/tf-learn-batch-fail-{d}", .{libc.getpid()});
    defer rmTree(root);
    var im = try imprint.Imprint.open(testing.allocator, root, 3, 1 << 20);
    defer im.deinit();
    const key = imprint.Imprint.keyOf(&with_note);
    try im.add(key, with_note.len, &with_note, &.{}, 0);
    var path: [1100]u8 = undefined;
    const file = try Fake.file(&path, im.dir, key);
    const fd = libc.open(file, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(libc.mode_t, 0o600));
    if (fd < 0) return error.TestFile;
    _ = libc.close(fd);
    var part: [1100]u8 = undefined;
    const blocked = try std.fmt.bufPrintSentinel(&part, "{s}/index.part", .{im.dir}, 0);
    if (libc.mkdir(blocked, 0o700) != 0) return error.TestDirectory;
    var f: Fake = .{ .gpa = testing.allocator };
    var s = Store.init(testing.allocator, f.learned(), .{}, 1 << 20);
    defer s.deinit();
    s.imprint = &im;
    const out = s.forgetSpan(&note, &.{});
    try testing.expectEqual(@as(u32, 1), out.failed);
    try testing.expectEqual(@as(u32, 0), out.learned);
    try testing.expect(!out.complete());
    try testing.expect(im.has(key));
    try testing.expect(exists(im.dir, key));
    try testing.expect(try fileHolds(im.dir, "index", &note));
}

test "forgetSpan preserves an ambiguous one-token state but reports it held" {
    var f: Fake = .{ .gpa = testing.allocator };
    var s = Store.init(testing.allocator, f.snapshots(), .{ .min_gap = 1, .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    const unrelated = system ++ [_]u32{ 60, 50, 99, 98 };
    _ = f.pass(&s, &unrelated, .{ .marks = &.{5} });
    try testing.expectEqual(@as(usize, 1), s.entries.items.len);
    const out = s.forgetSpan(&note, &.{});
    try testing.expectEqual(@as(u32, 0), out.states);
    try testing.expectEqual(@as(u32, 1), out.held);
    try testing.expect(!out.complete());
    try testing.expect(keptAt(&s, &unrelated, 5));
}

test "forgetSpan scrubs a last prompt ending inside an erased span" {
    for ([_]usize{ 4, 5, 6 }) |cut| {
        var f: Fake = .{ .gpa = testing.allocator };
        var s = Store.init(testing.allocator, f.snapshots(), .{ .min_gap = 1, .min_prompt = 0 }, 1 << 20);
        defer s.deinit();
        _ = f.pass(&s, with_note[0..cut], .{ .marks = &.{3} });
        try testing.expectEqual(@as(usize, 1), s.entries.items.len);
        try testing.expectEqualSlices(u32, with_note[0..cut], s.entries.items[0].last);
        const out = s.forgetSpan(&note, &.{});
        try testing.expectEqual(@as(u32, 0), out.states);
        try testing.expectEqual(@as(u32, 1), out.scrubbed);
        try testing.expect(out.complete());
        try testing.expectEqualSlices(u32, &system, s.entries.items[0].last);
    }
}

test "forgetSpan drops states cut inside an erased span" {
    var f: Fake = .{ .gpa = testing.allocator };
    var s = Store.init(testing.allocator, f.snapshots(), .{ .min_gap = 1, .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    _ = f.pass(&s, &with_note, .{ .marks = &.{ 4, 5, 6 } });
    try testing.expect(s.entries.items.len > 0);
    const out = s.forgetSpan(&note, &.{});
    try testing.expect(out.states > 0);
    try testing.expectEqual(@as(u32, 1), out.held);
    try testing.expectEqual(@as(usize, 1), s.entries.items.len);
}

test "forgetSpan removes learned index records cut inside an erased span" {
    var root_buf: [128]u8 = undefined;
    const root = try std.fmt.bufPrint(&root_buf, "/tmp/tf-learn-cut-{d}", .{libc.getpid()});
    defer rmTree(root);
    var im = try imprint.Imprint.open(testing.allocator, root, 3, 1 << 20);
    defer im.deinit();
    for ([_]usize{ 4, 5, 6 }) |cut| {
        const tokens = with_note[0..cut];
        try im.add(imprint.Imprint.keyOf(tokens), @intCast(cut), tokens, &.{}, 0);
    }
    try testing.expectEqual(@as(usize, 3), im.metas.items.len);
    var f: Fake = .{ .gpa = testing.allocator };
    var s = Store.init(testing.allocator, f.learned(), .{}, 1 << 20);
    defer s.deinit();
    s.imprint = &im;
    const out = s.forgetSpan(&note, &.{});
    try testing.expectEqual(@as(u32, 2), out.learned);
    try testing.expectEqual(@as(u32, 1), out.held);
    try testing.expectEqual(@as(usize, 1), im.metas.items.len);
}

/// Whether any kept state's tokens or last prompt still hold `span`.
fn holds(s: *const Store, span: []const u32) bool {
    for (s.entries.items) |e| if (std.mem.indexOf(u32, e.tokens, span) != null or std.mem.indexOf(u32, e.last, span) != null) return true;
    return false;
}

fn keptAt(s: *const Store, prompt: []const u32, at: u32) bool {
    for (s.entries.items) |e| if (e.at == at and std.mem.eql(u32, e.tokens, prompt[0..e.tokens.len])) return true;
    return false;
}

test "forgetSpan drops the kept state holding the note, keeps the state before it and an unrelated conversation's" {
    const gpa = testing.allocator;
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .min_gap = 1, .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    // Keep states before (3) and after (9) the note block, plus an unrelated conversation's state (9).
    _ = f.pass(&s, &with_note, .{ .marks = &.{ 3, 9 } });
    _ = f.pass(&s, &without, .{ .marks = &.{9} });
    try testing.expectEqual(@as(usize, 3), s.entries.items.len);
    try testing.expect(holds(&s, &note)); // the planted copy is there: the checks below can fail
    try testing.expectEqual(@as(usize, 9), s.find(&with_note, &.{}, &.{}).?.at);
    const live = f.live;

    const out = s.forgetSpan(&note, &.{});
    try testing.expectEqual(@as(u32, 1), out.states); // the state past the block
    try testing.expectEqual(@as(u32, 1), out.scrubbed); // the state before it: its last prompt held the note
    try testing.expect(out.complete());
    try testing.expect(!holds(&s, &note));
    try testing.expectEqual(live - 1, f.live); // dropped through the family: its storage freed
    try testing.expectEqual(@as(usize, 2), s.entries.items.len);
    // the note's conversation resumes at most at its block's start; the unrelated one where it did
    try testing.expectEqual(@as(usize, 3), s.find(&with_note, &.{}, &.{}).?.at);
    try testing.expect(keptAt(&s, &without, 9));
    try testing.expectEqual(@as(usize, 9), s.find(&without, &.{}, &.{}).?.at);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const next = without ++ [_]u32{ 5, 5 };
    const plan = try s.begin(arena.allocator(), &next, @intCast(next.len - 1), &.{}, &.{}, null, &.{});
    try testing.expectEqual(@as(u32, 9), plan.from);
    try testing.expectEqual(fresh(&next), f.pass(&s, &next, plan)); // what it resumed is still right
    // a second forget finds nothing more
    try testing.expectEqual(pc.Forgotten{}, s.forgetSpan(&note, &.{}));
}

test "forgetSpan leaves a state a prompt pass is restoring, counted, and takes it on the next call; an empty span takes nothing" {
    const gpa = testing.allocator;
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .min_gap = 1, .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    _ = f.pass(&s, &with_note, .{ .marks = &.{9} });
    try testing.expectEqual(pc.Forgotten{}, s.forgetSpan(&.{}, &.{}));
    try testing.expect(holds(&s, &note));
    const e = s.entries.items[0];
    const out = s.forgetSpan(&note, &.{e});
    try testing.expectEqual(@as(u32, 1), out.held);
    try testing.expect(!out.complete());
    try testing.expect(holds(&s, &note));
    try testing.expectEqual(@as(u32, 1), s.forgetSpan(&note, &.{}).states);
    try testing.expect(!holds(&s, &note));
}

/// Whether the file at `dir`/`name` holds `span`'s bytes.
fn fileHolds(dir: []const u8, name: []const u8, span: []const u32) !bool {
    var buf: [512]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ dir, name }, 0);
    const fd = libc.open(path, .{ .ACCMODE = .RDONLY }, @as(libc.mode_t, 0));
    if (fd < 0) return false;
    defer _ = libc.close(fd);
    var bytes: [1 << 16]u8 = undefined;
    var n: usize = 0;
    while (n < bytes.len) {
        const got = libc.read(fd, bytes[n..].ptr, bytes.len - n);
        if (got <= 0) break;
        n += @intCast(got);
    }
    return std.mem.indexOf(u8, bytes[0..n], std.mem.sliceAsBytes(span)) != null;
}

fn exists(dir: []const u8, key: u64) bool {
    var buf: [512]u8 = undefined;
    const path = Fake.file(&buf, dir, key) catch return false;
    return libc.access(path, libc.F_OK) == 0;
}

test "forgetSpan removes learned states on disk that hold the note, also ones an earlier server learned, and keeps the rest" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var root_buf: [128]u8 = undefined;
    const root = try std.fmt.bufPrint(&root_buf, "/tmp/tf-learn-forget-{d}", .{libc.getpid()});
    defer rmTree(root);
    const rules: pc.Rules = .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 };
    const tainted = [_]u32{ 7, 50, 51, 52, 53, 7 }; // a shared block holding the note: its state at 5 reads it all
    const clean = [_]u32{ 8, 8, 8, 8, 8, 8 };
    const tainted_key = imprint.Imprint.keyOf(&tainted);
    const clean_key = imprint.Imprint.keyOf(&clean);
    var dir: []const u8 = undefined;
    { // an earlier server learns both shared blocks to disk
        var im = try imprint.Imprint.open(gpa, root, 3, 1 << 30);
        defer im.deinit();
        var f: Fake = .{ .gpa = gpa };
        var s = Store.init(gpa, f.learned(), rules, 1 << 20);
        defer s.deinit();
        s.imprint = &im;
        for ([_][]const u32{ &(tainted ++ [_]u32{ 1, 2, 9 }), &(clean ++ [_]u32{ 1, 2, 9 }) }) |p| {
            try testing.expectEqual(fresh(p), f.pass(&s, p, try s.begin(a, p, 8, &.{5}, &.{}, null, &.{})));
        }
        try testing.expect(im.has(tainted_key) and im.has(clean_key));
        dir = try a.dupe(u8, im.dir);
    }
    // the planted copies are there: the state's file, and the note's tokens in the index
    try testing.expect(exists(dir, tainted_key) and exists(dir, clean_key));
    try testing.expect(try fileHolds(dir, "index", &note));
    { // this server holds nothing in memory: the forget finds the learned state by the index's tokens
        var im = try imprint.Imprint.open(gpa, root, 3, 1 << 30);
        defer im.deinit();
        var f: Fake = .{ .gpa = gpa };
        var s = Store.init(gpa, f.learned(), rules, 1 << 20);
        defer s.deinit();
        s.imprint = &im;
        try testing.expectEqual(@as(usize, 0), s.entries.items.len);
        const out = s.forgetSpan(&note, &.{});
        try testing.expectEqual(@as(u32, 1), out.learned);
        try testing.expectEqual(@as(u32, 0), out.states);
        try testing.expect(out.complete());
        try testing.expect(!im.has(tainted_key) and im.has(clean_key));
    }
    try testing.expect(!exists(dir, tainted_key)); // its file is gone
    try testing.expect(exists(dir, clean_key)); // the other's stays
    try testing.expect(!try fileHolds(dir, "index", &note)); // and the index holds none of its tokens
    try testing.expect(try fileHolds(dir, "index", &clean));
    // a server started after: the note's block is read again from 0; the other resumes from disk
    var im = try imprint.Imprint.open(gpa, root, 3, 1 << 30);
    defer im.deinit();
    try testing.expect(!im.has(tainted_key) and im.has(clean_key));
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.learned(), rules, 1 << 20);
    defer s.deinit();
    s.imprint = &im;
    const again = tainted ++ [_]u32{ 3, 4, 9 };
    const p1 = try s.begin(a, &again, 8, &.{5}, &.{}, null, &.{});
    try testing.expectEqual(@as(u32, 0), p1.from);
    try testing.expectEqual(fresh(&again), f.pass(&s, &again, p1));
    const other = clean ++ [_]u32{ 3, 4, 9 };
    const p2 = try s.begin(a, &other, 8, &.{5}, &.{}, null, &.{});
    try testing.expectEqual(@as(u32, 5), p2.from);
    try testing.expectEqual(fresh(&other), f.pass(&s, &other, p2));
}

test "forgetSpan on a store that read a learned state back drops it from memory and from disk" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var root_buf: [128]u8 = undefined;
    const root = try std.fmt.bufPrint(&root_buf, "/tmp/tf-learn-forget-mem-{d}", .{libc.getpid()});
    defer rmTree(root);
    const rules: pc.Rules = .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 };
    const tainted = [_]u32{ 7, 50, 51, 52, 53, 7 };
    const key = imprint.Imprint.keyOf(&tainted);
    var im = try imprint.Imprint.open(gpa, root, 3, 1 << 30);
    defer im.deinit();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.learned(), rules, 1 << 20);
    defer s.deinit();
    s.imprint = &im;
    const p = tainted ++ [_]u32{ 1, 2, 9 };
    _ = f.pass(&s, &p, try s.begin(a, &p, 8, &.{5}, &.{}, null, &.{}));
    try testing.expect(im.has(key) and holds(&s, &note));
    const out = s.forgetSpan(&note, &.{});
    try testing.expectEqual(@as(u32, 1), out.learned);
    try testing.expect(out.states >= 1);
    try testing.expect(!im.has(key) and !holds(&s, &note) and !exists(im.dir, key));
}
