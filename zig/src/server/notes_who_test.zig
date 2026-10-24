//! Who first: a chat request's notes are the subjects' notes and the notes they link to, or none when nobody is known.
const std = @import("std");
const server = @import("root.zig");
const json = server.json;
const rig = @import("notes_lookup_rig_test.zig");
const index = @import("notes/index.zig");
const retrieve = @import("notes/retrieve.zig");
const Storage = @import("notes/storage.zig").Storage;
const Bytes = @import("notes_storage_bytes_test.zig").Bytes;
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// A made-up store under the byte tokenizer, and Python's answers (tools/notes/who_fixture.py writes them).
const snapshot = @embedFile("notes/testdata/who_bytes_snapshot.json");
const cases_json = @embedFile("notes/testdata/who_bytes_cases.json");
const l6_who_json = @embedFile("notes/testdata/l6_bytes_who.json");

/// One chat request's notes field, the reply checked to be a 200.
fn notesOf(r: *rig.Rig, a: Allocator, question: []const u8, extra: []const u8) !json.Value {
    const body = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"max_tokens\":4,\"notes\":true{s},\"messages\":[{{\"role\":\"user\",\"content\":{s}}}]}}", .{ extra, try json.quote(a, question, .{}) });
    const got = try rig.request(r.srv, a, "POST", "/v1/chat/completions", body);
    errdefer std.debug.print("{d} {s}\n", .{ got.status, got.body });
    try testing.expectEqual(@as(u16, 200), got.status);
    return got.value.get("notes").?;
}

/// The note ids of a chat request's block.
fn idsOf(r: *rig.Rig, a: Allocator, question: []const u8, extra: []const u8) ![]const u32 {
    const notes = try notesOf(r, a, question, extra);
    var out: std.ArrayList(u32) = .empty;
    for (notes.get("ids").?.array) |id| try out.append(a, @intCast(id.int64().?));
    return out.items;
}

fn has(ids: []const u32, id: u32) bool {
    return std.mem.indexOfScalar(u32, ids, id) != null;
}

test "who first: a question that names nobody the store knows gets no block, today's lookup sends notes" {
    const r = try rig.Rig.initWithSnap(testing.allocator, .{}, snapshot);
    defer r.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ "What colour are my cousin's eyes?", "Tell me about Rafa.", "What is the weather like?" }) |q| {
        const notes = try notesOf(r, a, q, "");
        try testing.expectEqualStrings("who", notes.strField("lookup").?);
        try testing.expectEqual(@as(usize, 0), notes.get("ids").?.array.len);
        try testing.expect(notes.get("block_sha256").? == .null);
        // negative control: retrieve() sends notes for each, the eye notes for the cousin among them
        const plain = try idsOf(r, a, q, ",\"notes_lookup\":\"plain\"");
        try testing.expect(plain.len > 0);
    }
    const cousin = try idsOf(r, a, "What colour are my cousin's eyes?", ",\"notes_lookup\":\"plain\"");
    try testing.expect(has(cousin, 2) and has(cousin, 10));
}

test "who first: a known subject's notes and the notes of the entities they name, no one else's" {
    const r = try rig.Rig.initWithSnap(testing.allocator, .{}, snapshot);
    defer r.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try testing.expectEqualSlices(u32, &.{ 1, 2, 3, 5, 15 }, try idsOf(r, a, "What does my son like?", ""));
    const plain = try idsOf(r, a, "What does my son like?", ",\"notes_lookup\":\"plain\"");
    try testing.expect(has(plain, 9) and has(plain, 14)); // "My kid likes figs." and "Mira's cat likes fish."
    // a two-step question: the son's link to the aunt, and the aunt's current town (not the one it replaced)
    const home = try idsOf(r, a, "Where is my son's home?", "");
    try testing.expectEqualSlices(u32, &.{ 1, 2, 3, 5, 15 }, home);
    try testing.expect(has(try idsOf(r, a, "Where is my son's home?", ",\"notes_lookup\":\"plain\""), 6));
    try testing.expectEqualSlices(u32, &.{ 1, 2, 5, 6, 7 }, try idsOf(r, a, "Where does the owner of the van live?", ""));
}

test "who first: an excluded note is never shown and the link it states is never followed" {
    const r = try rig.Rig.initWithSnap(testing.allocator, .{}, snapshot);
    defer r.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try testing.expectEqualSlices(u32, &.{ 1, 2 }, try idsOf(r, a, "Where is my son's home?", ",\"notes_exclude\":[5]"));
    try testing.expectEqualSlices(u32, &.{3}, try idsOf(r, a, "Where does my aunt live?", ",\"notes_exclude\":[15]"));
    const plain = try idsOf(r, a, "Where is my son's home?", ",\"notes_exclude\":[5],\"notes_lookup\":\"plain\"");
    try testing.expect(!has(plain, 5) and has(plain, 15)); // retrieve() still finds the aunt's town
}

test "who first: my X is the user when X is a word of the user's notes or of a stored attribute, not of a name" {
    const r = try rig.Rig.initWithSnap(testing.allocator, .{}, snapshot);
    defer r.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ "What colour are my eyes?", "Tell me the colour of my eyes.", "What is my town?", "Do I like figs?" }) |q| {
        try testing.expectEqualSlices(u32, &.{ 8, 10 }, try idsOf(r, a, q, ""));
    }
    try testing.expectEqualSlices(u32, &.{ 1, 2, 5, 6, 7, 8, 10 }, try idsOf(r, a, "Is the van mine?", ""));
    // a name several entities hold: the user's circle's (my son, not Rafa's dog), or all of them weighed down
    try testing.expectEqualSlices(u32, &.{ 1, 2, 3, 5, 15 }, try idsOf(r, a, "What colour are Ivo's eyes?", ""));
    try testing.expectEqualSlices(u32, &.{ 12, 13, 14 }, try idsOf(r, a, "What does Pip like?", ""));
}

test "who first: other languages and notes_lookup plain keep today's lookup; the field's refusals" {
    const r = try rig.Rig.initWithSnap(testing.allocator, .{}, snapshot);
    defer r.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const q = "Welche Farbe haben die Augen meines Cousins?";
    const de = try notesOf(r, a, q, ",\"notes_language\":\"de\"");
    try testing.expectEqualStrings("plain", de.strField("lookup").?);
    try testing.expect(de.get("ids").?.array.len > 0);
    try testing.expectEqual(@as(usize, 0), (try idsOf(r, a, q, "")).len); // English: nobody known
    r.store().lookup = .plain;
    try testing.expectEqualStrings("plain", (try notesOf(r, a, "What does my son like?", "")).strField("lookup").?);
    try testing.expect(has(try idsOf(r, a, "What does my son like?", ""), 9));
    try testing.expectEqualSlices(u32, &.{ 1, 2, 3, 5, 15 }, try idsOf(r, a, "What does my son like?", ",\"notes_lookup\":\"who\""));
    const Bad = struct { body: []const u8, why: []const u8 };
    for ([_]Bad{
        .{ .body = "{\"model\":\"m\",\"notes\":true,\"notes_lookup\":\"all\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", .why = "notes_lookup must be \"who\" or \"plain\"" },
        .{ .body = "{\"model\":\"m\",\"notes\":true,\"notes_lookup\":1,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", .why = "notes_lookup must be \"who\" or \"plain\"" },
    }) |bad| {
        const got = try rig.request(r.srv, a, "POST", "/v1/chat/completions", bad.body);
        try testing.expectEqual(@as(u16, 400), got.status);
        try testing.expectEqualStrings(bad.why, rig.message(got));
    }
    // a request without notes is answered as before: the field is not read
    for ([_][]const u8{ "\"who\"", "\"all\"", "1" }) |v| {
        const body = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"max_tokens\":4,\"notes_lookup\":{s},\"messages\":[{{\"role\":\"user\",\"content\":\"x\"}}]}}", .{v});
        const got = try rig.request(r.srv, a, "POST", "/v1/chat/completions", body);
        try testing.expectEqual(@as(u16, 200), got.status);
        try testing.expect(got.value.get("notes") == null);
    }
}

/// How many cases the store's who-first lookup answers otherwise than Python's (or, with `plain`, retrieve()'s).
fn differing(store: *Storage, text: anytype, cases: []const u8, plain: bool) !usize {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var bad: usize = 0;
    for ((try json.parse(a, cases)).ok.array) |c| {
        const q = c.strField("question").?;
        var ex: std.ArrayList(u32) = .empty;
        for (c.get("exclude").?.array) |x| try ex.append(a, @intCast(x.int64().?));
        const cap: u32 = if (c.field("cap")) |x| @intCast(x.int64().?) else store.model.settings.cap;
        const qids = try text.encode(a, q, false);
        const past = c.get("past_read").?.bool;
        const got = try retrieve.retrieve(store, testing.io, a, text, qids, past, cap, ex.items, .{ .who = if (plain) null else q });
        var same = got.tokens == @as(usize, @intCast(c.get("tokens").?.int64().?)) and got.candidates == c.get("candidates").?.int64().?;
        const ids = c.get("ids").?.array;
        same = same and ids.len == got.ids.len;
        if (same) for (ids, got.ids) |want, have| {
            same = same and want.int64().? == have;
        };
        var sha: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(got.block, &sha, .{});
        const hex = std.fmt.bytesToHex(sha, .lower);
        if (c.field("block_sha256")) |h| {
            same = same and std.mem.eql(u8, h.string, &hex);
        } else same = same and got.block.len == 0;
        if (!same) bad += 1;
    }
    return bad;
}

test "who first gives Python's ids, tokens, candidates and block for every case, with either index" {
    var text: Bytes = .{};
    for ([_]index.Kind{ .postings, .sa }) |kind| {
        for ([_][2][]const u8{ .{ snapshot, cases_json }, .{ rig.snapshot, l6_who_json } }) |pair| {
            var why: []const u8 = "";
            const store = try Storage.loadKind(testing.allocator, pair[0], text.text(), null, kind, &why);
            defer store.deinit();
            try testing.expectEqual(@as(usize, 0), try differing(store, text.text(), pair[1], false));
            // negative control: retrieve() answers most of them otherwise
            try testing.expect(try differing(store, text.text(), pair[1], true) > 10);
        }
    }
}
