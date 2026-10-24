//! The decode lookup's cheaper paths: (c) notes before the question, the name trigger, the value hop, notes_position.
const std = @import("std");
const server = @import("root.zig");
const json = server.json;
const index = @import("notes/index.zig");
const model_mod = @import("notes/model.zig");
const value = @import("notes/value.zig");
const retrieve = @import("notes/retrieve.zig");
const notes_decode = @import("notes_decode.zig");
const lookup_rig = @import("notes_lookup_rig_test.zig");
const base = @import("notes_decode_test.zig");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const Rig = base.Rig;
const question = base.question;
const answers = base.answers;
const content = base.content;
const decodeOf = base.decodeOf;
const injected = base.injected;

const system = "Be brief.";
const earlier_user = "Recommend a book.";
const earlier_reply = "Try Dune.";
const named_late = [_][]const u8{ "Your sister Anna has a red car.", " drives a red Volvo.", "My sister is nice and kind", "Your sister Anna drives a red Volvo." };
const more = answers ++ named_late ++ [_][]const u8{ system, earlier_user, earlier_reply };

fn conversation(r: *Rig) ![]const u8 {
    return std.fmt.allocPrint(r.arena.allocator(), "[{{\"role\":\"system\",\"content\":\"{s}\"}},{{\"role\":\"user\",\"content\":\"{s}\"}},{{\"role\":\"assistant\",\"content\":\"{s}\"}},{{\"role\":\"user\",\"content\":{f}}}]", .{ system, earlier_user, earlier_reply, std.json.fmt(question, .{}) });
}

test "(c): the new notes spliced in right before the question, everything before it as the first request had it" {
    var r: Rig = .{};
    try r.init(&more);
    defer r.deinit();
    const a = r.arena.allocator();
    const got = (try r.ask(&.{ answers[0], answers[1] }, ",\"notes_decode_place\":\"user\"")).value;
    try testing.expectEqualStrings("Your sister is Anna. She drives a red Volvo.", content(got));
    const late = decodeOf(got).strField("block").?;
    try testing.expectEqualStrings("Current:\n[2] \"Anna drives a red Volvo.\"", late);
    const first = got.get("notes").?.strField("block").?;
    // two system messages that normalize would merge: the splice keeps them apart, so the first one's tokens stay put
    const head = try r.ids(try std.mem.concat(a, u8, &.{ "[system]", first, "\n[system]", late, "\n[user]", question, "\n[assistant]" }));
    try testing.expectEqualSlices(u32, try std.mem.concat(a, u32, &.{ head, try r.ids("Your sister is Anna") }), r.script.prompts.items[1]);
    const before = try r.ids(try std.mem.concat(a, u8, &.{ "[system]", first, "\n" }));
    try testing.expectEqualSlices(u32, before, r.script.prompts.items[0][0..before.len]);
    try testing.expect(before.len < 512);
    try testing.expectEqual(@as(u32, @intCast(before.len - 1)), r.script.retain[0]);
    try testing.expectEqual(r.script.retain[0], r.script.retain[1]);
    try testing.expectEqualSlices(u32, r.script.prompts.items[0][0..before.len], r.script.prompts.items[1][0..before.len]);
}

test "the name trigger: a name the reply just wrote, of someone with a note outside the block, is a check point" {
    var r: Rig = .{};
    try r.init(&more);
    defer r.deinit();
    const on = (try r.ask(&.{ named_late[0], named_late[1] }, ",\"notes_decode_names\":true")).value;
    try testing.expectEqualStrings("Your sister Anna drives a red Volvo.", content(on));
    try testing.expectEqual(@as(i64, @intCast((try r.ids("Your sister Anna")).len)), injected(on)[0].get("at_token").?.int64().?);
    // control: without it there is no check point at all; the reply is the engine's first one
    const off = (try r.ask(&.{ named_late[0], named_late[1] }, ",\"notes_decode_names\":false")).value;
    try testing.expectEqual(@as(usize, 0), injected(off).len);
    try testing.expectEqualStrings(named_late[0], content(off));
    // a name whose notes are all in the block, and words that name nobody, are no check point
    const held = (try r.ask(&.{named_late[2]}, ",\"notes_decode_names\":true")).value;
    try testing.expectEqual(@as(i64, 0), decodeOf(held).get("checks").?.int64().?);
}

test "the value hop: a note in the block that names someone brings that one's notes in before prefill" {
    var r: Rig = .{ .links = true };
    try r.init(&more);
    defer r.deinit();
    const plain = (try r.ask(&.{named_late[3]}, ",\"notes_decode\":false")).value;
    const has = struct {
        fn of(reply: json.Value, id: []const u8) bool {
            for (reply.get("notes").?.get("ids").?.array) |x| if (std.mem.eql(u8, x.int, id)) return true;
            return false;
        }
    }.of;
    try testing.expect(!has(plain, "2"));
    const hop = (try r.ask(&.{named_late[3]}, ",\"notes_value_hop\":true")).value;
    try testing.expect(has(hop, "2")); // "My sister is Anna." names Anna: her note comes in with it
    try testing.expect(hop.get("notes").?.get("tokens").?.int64().? <= 512);
    try testing.expectEqual(@as(usize, 0), injected(hop).len); // nothing left to put in mid-reply
    try testing.expectEqual(@as(usize, 1), r.script.prompts.items.len);
    // control: the hop follows a note's value entity; without one it brings nothing
    var q: Rig = .{};
    try q.init(&more);
    defer q.deinit();
    try testing.expect(!has((try q.ask(&.{named_late[3]}, ",\"notes_value_hop\":true")).value, "2"));
}

test "a written note's value entity is the entity its text names; a snapshot's note keeps its own" {
    var w: base.Words = .{};
    defer w.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const m = try model_mod.parse(testing.allocator, try base.snapshot(a, &w, &more, false));
    defer m.deinit();
    for ([_]usize{ 0, 6 }) |k| {
        const n = &m.notes.items[k];
        const old = n.*;
        try n.makeBody(testing.allocator, old.n_tokens, old.text(), "w1"); // as if written
        @memcpy(n.tokensMut(), old.tokens());
        testing.allocator.free(old.body[0 .. 4 * @as(usize, old.n_tokens) + old.text_len + old.source_len]);
    }
    const idx = try index.Index.build(testing.allocator, m);
    defer idx.deinit();
    try testing.expectEqual(@as(i32, 1), idx.valueEntityAt(0)); // "My sister is Anna." names Anna (entity 1)
    try testing.expectEqual(@as(i32, -1), idx.valueEntityAt(6)); // "I have a red bicycle." names nobody
    try testing.expectEqual(@as(i32, -1), idx.valueEntityAt(2)); // "My brother is Otto." is a snapshot's note
    try testing.expectEqual(@as(?u32, 3), value.named(idx, try w.ids(a, "My brother is Otto."), 2));
}

test "notes_position user: the block right before the question, (a) re-renders it there and (c) splices after it" {
    var r: Rig = .{};
    try r.init(&more);
    defer r.deinit();
    const a = r.arena.allocator();
    const msgs = try conversation(&r);
    const first = (try r.askWith(&.{answers[5]}, msgs, ",\"notes_position\":\"user\"")).value;
    const block = first.get("notes").?.strField("block").?;
    try testing.expectEqualSlices(u32, try r.prompt(&.{ .{ "system", system }, .{ "user", earlier_user }, .{ "assistant", earlier_reply }, .{ "system", block }, .{ "user", question } }), r.script.prompts.items[0]);
    // alone with the question it is the first message, as by default
    _ = try r.ask(&.{answers[5]}, ",\"notes_position\":\"user\"");
    const alone = try a.dupe(u32, r.script.prompts.items[0]);
    _ = try r.ask(&.{answers[5]}, "");
    try testing.expectEqualSlices(u32, r.script.prompts.items[0], alone);
    const said = try r.ids("Your sister is Anna");
    const re = (try r.askWith(&.{ answers[0], answers[1] }, msgs, ",\"notes_position\":\"user\"")).value;
    const enlarged = decodeOf(re).strField("block").?;
    try testing.expect(r.script.retain[0] > 0);
    try testing.expectEqual(r.script.retain[0], r.script.retain[1]);
    try testing.expectEqualSlices(u32, try std.mem.concat(a, u32, &.{ try r.prompt(&.{ .{ "system", system }, .{ "user", earlier_user }, .{ "assistant", earlier_reply }, .{ "system", enlarged }, .{ "user", question } }), said }), r.script.prompts.items[1]);
    const sp = (try r.askWith(&.{ answers[0], answers[1] }, msgs, ",\"notes_position\":\"user\",\"notes_decode_place\":\"user\"")).value;
    const late = decodeOf(sp).strField("block").?;
    try testing.expectEqualSlices(u32, try std.mem.concat(a, u32, &.{ try r.prompt(&.{ .{ "system", system }, .{ "user", earlier_user }, .{ "assistant", earlier_reply }, .{ "system", block }, .{ "system", late }, .{ "user", question } }), said }), r.script.prompts.items[1]);
}

test "turn retention preserves its initial prefix and first-message replacement retains nothing" {
    var r: Rig = .{};
    try r.init(&more);
    defer r.deinit();
    _ = try r.ask(&.{ answers[0], answers[1] }, ",\"notes_decode_place\":\"turn\"");
    try testing.expect(r.script.retain[0] > 0);
    try testing.expectEqual(r.script.retain[0], r.script.retain[1]);
    const len = @as(usize, r.script.retain[0]) + 1;
    try testing.expectEqualSlices(u32, r.script.prompts.items[0][0..len], r.script.prompts.items[1][0..len]);
    _ = try r.ask(&.{ answers[0], answers[1] }, ",\"notes_decode_place\":\"first\"");
    try testing.expectEqual(@as(u32, 0), r.script.retain[0]);
    try testing.expectEqual(@as(u32, 0), r.script.retain[1]);
}

test "the defaults: the name trigger and the notes before the question, with no field sent" {
    var r: Rig = .{ .links = true, .legacy = "" };
    try r.init(&more);
    defer r.deinit();
    const got = (try r.ask(&.{named_late[3]}, "")).value;
    const d = decodeOf(got);
    try testing.expectEqualStrings("user", d.strField("place").?);
    try testing.expect(d.get("names").?.bool);
    try testing.expect(!d.get("value_hop").?.bool);
    try testing.expectEqualStrings("first", d.strField("position").?);
}

const zebra_text = "Zebra eats hay.";
const zebra_reply = "The zebra is here.";
const extra = answers ++ more ++ [_][]const u8{ zebra_text, zebra_reply, "I like blue ink." };

fn journaled(r: *Rig, tmp: *testing.TmpDir) !void {
    const a = r.arena.allocator();
    const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
    r.journal = try std.fs.path.join(a, &.{ dir, "notes.json.journal" });
    try r.init(&extra);
}

fn addNote(r: *Rig, a: Allocator, text: []const u8, subject: []const u8) !u32 {
    const body = try std.fmt.allocPrint(a, "{{\"text\":{s},\"subject\":{s}}}", .{ try json.quote(a, text, .{}), try json.quote(a, subject, .{}) });
    const got = try lookup_rig.request(r.srv, a, "POST", "/v1/notes", body);
    try testing.expectEqual(@as(u16, 201), got.status);
    return @intCast(got.value.get("id").?.int64().?);
}

fn firstBlock(r: *Rig, a: Allocator) !retrieve.Found {
    const store = r.srv.notes.?;
    return retrieve.retrieve(store, r.srv.io, a, r.words.text(), try r.ids(question), false, 512, &.{}, .{});
}

fn liveOf(r: *Rig, a: Allocator, found: retrieve.Found) !notes_decode.Live {
    var l: notes_decode.Live = .{ .a = a, .on = true, .store = r.srv.notes.?, .io = r.srv.io, .ver = found.ver, .question = question, .past = false, .cap = 512, .exclude = &.{}, .place = .first, .echo = true, .block = found.block, .block_tokens = found.tokens };
    try l.held.appendSlice(a, found.ids);
    return l;
}

test "an erase of a note a live fetch would take stops further fetches" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var r: Rig = .{};
    try journaled(&r, &tmp);
    defer r.deinit();
    const a = r.arena.allocator();
    const id = try addNote(&r, a, zebra_text, "zebra");
    const found = try firstBlock(&r, a);
    try testing.expect(std.mem.indexOfScalar(u32, found.ids, id) == null);
    var l = try liveOf(&r, a, found);
    const hit = (try l.check(r.words.text(), a, zebra_reply, 5)).?;
    try testing.expectEqualSlices(u32, &.{id}, hit.ids);
    const erased = try lookup_rig.request(r.srv, a, "DELETE", try std.fmt.allocPrint(a, "/v1/notes/{d}?erase=true", .{id}), "");
    try testing.expectEqual(@as(u16, 200), erased.status);
    try testing.expect(r.srv.notes.?.gen != found.ver);
    var m = try liveOf(&r, a, found);
    try testing.expect(try m.check(r.words.text(), a, zebra_reply, 5) == null);
    try testing.expectEqualStrings("the notes changed during the reply", m.stopped.?);
    try testing.expectEqual(@as(usize, 0), m.injections.items.len);
    try testing.expect(m.nameWritten(try r.ids(" zebra")) == null);
}

test "a write mid-reply stops live fetch even when the named note is still there" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var r: Rig = .{};
    try journaled(&r, &tmp);
    defer r.deinit();
    const a = r.arena.allocator();
    const id = try addNote(&r, a, zebra_text, "zebra");
    const found = try firstBlock(&r, a);
    var l = try liveOf(&r, a, found);
    try testing.expectEqualSlices(u32, &.{id}, (try l.check(r.words.text(), a, zebra_reply, 5)).?.ids);
    _ = try addNote(&r, a, "I like blue ink.", "me");
    try testing.expect(r.srv.notes.?.gen != found.ver);
    var m = try liveOf(&r, a, found);
    try testing.expect(try m.check(r.words.text(), a, zebra_reply, 5) == null);
    try testing.expectEqualStrings("the notes changed during the reply", m.stopped.?);
    try testing.expectEqual(@as(usize, 0), m.injections.items.len);
}
