//! Who first: a chat request's notes are the subjects' notes and the notes they link to, or none when nobody is known.
const std = @import("std");
const server = @import("root.zig");
const json = server.json;
const rig = @import("notes_lookup_rig_test.zig");
const index = @import("notes/index.zig");
const retrieve = @import("notes/retrieve.zig");
const Storage = @import("notes/storage.zig").Storage;
const cli = @import("cli.zig");
const Bytes = @import("notes_storage_bytes_test.zig").Bytes;
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// A made-up store under the byte tokenizer, and Python's answers (tools/notes/who_fixture.py writes them).
const snapshot = @embedFile("notes/testdata/who_bytes_snapshot.json");
const cases_json = @embedFile("notes/testdata/who_bytes_cases.json");
const live_json = @embedFile("notes/testdata/who_bytes_live.json");
const l6_who_json = @embedFile("notes/testdata/l6_bytes_who.json");
const empty_snapshot = @embedFile("notes/testdata/tb_empty_bytes_snapshot.json");

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

/// A note written over /v1/notes (`subject`: "" for the user's own); its id.
fn written(r: *rig.Rig, a: Allocator, text: []const u8, subject: []const u8) !u32 {
    const about = if (subject.len > 0) try std.fmt.allocPrint(a, ",\"subject\":{s}", .{try json.quote(a, subject, .{})}) else "";
    const got = try rig.request(r.srv, a, "POST", "/v1/notes", try std.fmt.allocPrint(a, "{{\"text\":{s}{s}}}", .{ try json.quote(a, text, .{}), about }));
    try testing.expectEqual(@as(u16, 201), got.status);
    return @intCast(got.value.get("id").?.int64().?);
}

test "who first: a question that names nobody the store knows gets no block, today's lookup sends notes" {
    const r = try rig.Rig.initWithSnap(testing.allocator, .{}, snapshot);
    defer r.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ "What colour are my cousin's eyes?", "Tell me about Rafa.", "What is the weather like?", "Is my van taxed?" }) |q| {
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
    try testing.expectEqualSlices(u32, &.{ 1, 2, 5, 6, 7, 16 }, try idsOf(r, a, "Where does the owner of the van live?", ""));
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
    try testing.expect(!has(plain, 5) and has(plain, 3)); // retrieve() still finds the aunt
}

test "who first: my X is the user when X is a word of the user's notes or of a stored attribute, not of a name" {
    const r = try rig.Rig.initWithSnap(testing.allocator, .{}, snapshot);
    defer r.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // eyes and town: attribute parts (town cut at its apostrophe); bees: a word of "I keep bees."; van: of "the van"
    for ([_][]const u8{ "What colour are my eyes?", "Tell me the colour of my eyes.", "What is my town?", "Do I like figs?", "Where are my bees?", "What is my town's name?", "\u{e9}my eyes" }) |q| {
        try testing.expectEqualSlices(u32, &.{ 8, 10, 17 }, try idsOf(r, a, q, ""));
    }
    for ([_][]const u8{ "Is my van taxed?", "What colour are my\u{a0}eyes?" }) |q| try testing.expectEqual(@as(usize, 0), (try idsOf(r, a, q, "")).len);
    try testing.expectEqualSlices(u32, &.{ 1, 2, 5, 6, 7, 8, 10, 16, 17 }, try idsOf(r, a, "Is the van mine?", ""));
    try testing.expectEqualSlices(u32, &.{24}, try idsOf(r, a, "What do my kids like?", "")); // not "my kid" inside it
}

test "who first: a name several entities hold names the user's circle's holder, else all of them weighed down" {
    const r = try rig.Rig.initWithSnap(testing.allocator, .{}, snapshot);
    defer r.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // my son (not Rafa's dog), the hen my kid feeds (not Ada's), the cow (not Ada's dog)
    try testing.expectEqualSlices(u32, &.{ 1, 2, 3, 5, 15 }, try idsOf(r, a, "What colour are Ivo's eyes?", ""));
    try testing.expectEqualSlices(u32, &.{19}, try idsOf(r, a, "What does Bo eat?", ""));
    try testing.expectEqualSlices(u32, &.{22}, try idsOf(r, a, "What does Max eat?", ""));
    // neither cat is the user's: both, and the dog one's note links to, weighed below my kid's notes under the cap
    try testing.expectEqualSlices(u32, &.{ 11, 12, 13, 14, 18 }, try idsOf(r, a, "What does Pip like?", ""));
    try testing.expectEqualSlices(u32, &.{ 9, 13, 14, 18, 21 }, try idsOf(r, a, "Does Pip play with Rafa's dog, and what does my kid like?", ",\"notes_cap\":404"));
}

test "who first: a name of Latin letters with marks, with a hyphen, or of several words one space apart" {
    const r = try rig.Rig.initWithSnap(testing.allocator, .{}, snapshot);
    defer r.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try testing.expectEqualSlices(u32, &.{25}, try idsOf(r, a, "What does Zo\u{eb} like?", ""));
    try testing.expectEqualSlices(u32, &.{26}, try idsOf(r, a, "Is Ea-Li big?", ""));
    try testing.expectEqualSlices(u32, &.{27}, try idsOf(r, a, "Is Al Ko big?", ""));
    try testing.expectEqual(@as(usize, 0), (try idsOf(r, a, "Is Al  Ko big?", "")).len);
}

test "who first: on notes the server wrote, a question that names someone gets the notes today's lookup finds" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (rig.kinds) |kind| {
        const r = try rig.Rig.initWithSnap(testing.allocator, kind, empty_snapshot);
        defer r.deinit();
        for ([_][]const u8{ "My dog's name is Rex.", "Rex sleeps in the barn.", "Anna's birthday is in May.", "My sister's name is Ana." }) |t| _ = try written(r, a, t, "");
        _ = try written(r, a, "Lumi likes the red ball.", ""); // the chat write is a later slice's
        for ([_][]const u8{ "Where does Rex sleep?", "When is Anna's birthday?", "Who is Ana?", "What does Lumi like?", "Where does my dog sleep?" }) |q| {
            const who = try idsOf(r, a, q, "");
            try testing.expect(who.len > 0);
            try testing.expectEqualSlices(u32, try idsOf(r, a, q, ",\"notes_lookup\":\"plain\""), who);
        }
    }
    // a store that has an entity named "Rafa's dog": the user's own note on "my dog" still comes in
    const r = try rig.Rig.initWithSnap(testing.allocator, .{}, snapshot);
    defer r.deinit();
    const id = try written(r, a, "My dog sleeps in the barn.", "");
    try testing.expect(has(try idsOf(r, a, "Where does my dog sleep?", ""), id));
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

test "who first after /v1/notes writes gives Python's answers: the user's unsorted notes are candidates" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const doc = (try json.parse(a, live_json)).ok;
    const cases = try json.stringify(a, doc.get("cases").?, .{});
    for (rig.kinds) |kind| {
        const r = try rig.Rig.initWithSnap(testing.allocator, kind, snapshot);
        defer r.deinit();
        for (doc.get("writes").?.array) |w| {
            const id = try written(r, a, w.strField("text").?, w.strField("subject") orelse "");
            try testing.expectEqual(w.get("id").?.int64().?, id);
        }
        try testing.expectEqual(@as(usize, 0), try differing(r.store(), r.text.text(), cases, false));
        try testing.expect(try differing(r.store(), r.text.text(), cases, true) >= 4); // retrieve() answers otherwise
    }
}

test "who first: a long question takes no more request memory than today's lookup, nor store memory per word" {
    var text: Bytes = .{};
    var why: []const u8 = "";
    var store_gpa = std.testing.FailingAllocator.init(testing.allocator, .{}); // counts what the lookup takes
    const store = try Storage.loadKind(store_gpa.allocator(), snapshot, text.text(), null, .postings, &why);
    defer store.deinit();
    var qa = std.heap.ArenaAllocator.init(testing.allocator);
    defer qa.deinit();
    var q: std.ArrayList(u8) = .empty;
    for (0..1000) |_| try q.appendSlice(qa.allocator(), "Where does my son live and what does Ivo like now? ");
    const qids = try text.text().encode(qa.allocator(), q.items, false);
    var bytes: [2][2]usize = undefined; // request arena, store allocator: plain, then who first
    for (0..2) |round| for ([_]bool{ false, true }, 0..) |who, k| { // the first round grows the kept scratch
        var counted = std.testing.FailingAllocator.init(testing.allocator, .{});
        var arena = std.heap.ArenaAllocator.init(counted.allocator());
        defer arena.deinit();
        const before = store_gpa.allocated_bytes;
        _ = try retrieve.retrieve(store, testing.io, arena.allocator(), text.text(), qids, false, store.model.settings.cap, &.{}, .{ .who = if (who) q.items else null });
        if (round == 1) bytes[k] = .{ counted.allocated_bytes, store_gpa.allocated_bytes - before };
    };
    try testing.expect(bytes[1][0] <= bytes[0][0] + 4096);
    try testing.expect(bytes[1][1] <= bytes[0][1] + q.items.len); // not a run's encode kept for each of the 11,000 runs
}

test "who first: notes_lookup and --notes-lookup choose the lookup; other languages keep today's; the refusals" {
    const r = try rig.Rig.initWithSnap(testing.allocator, .{}, snapshot);
    defer r.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const q = "Welche Farbe haben die Augen meines Cousins?";
    const de = try notesOf(r, a, q, ",\"notes_language\":\"de\"");
    try testing.expectEqualStrings("plain", de.strField("lookup").?);
    try testing.expect(de.get("ids").?.array.len > 0);
    try testing.expectEqual(@as(usize, 0), (try idsOf(r, a, q, "")).len); // not declared: English, nobody known
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
    var u: cli.Usage = .{};
    try testing.expectEqualStrings("who", (try cli.parse(a, &.{ "m", "--notes", "s.json" }, &u)).notes.lookup);
    try testing.expectEqualStrings("plain", (try cli.parse(a, &.{ "m", "--notes", "s.json", "--notes-lookup", "plain" }, &u)).notes.lookup);
    try testing.expectEqualStrings("who", (try cli.parse(a, &.{ "m", "--notes-lookup=who" }, &u)).notes.lookup);
    try testing.expectError(error.Usage, cli.parse(a, &.{ "m", "--notes-lookup", "all" }, &u));
}
