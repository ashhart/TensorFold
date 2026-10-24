//! Live lookup limits, request validation and off-switch regression cases.
const std = @import("std");
const testing = std.testing;
const base = @import("notes_decode_test.zig");
const Rig = base.Rig;
const answers = base.answers;
const question = base.question;
const long_plain = base.long_plain;
const five_names = base.five_names;
const content = base.content;
const decodeOf = base.decodeOf;
const injected = base.injected;
const notes_decode = @import("notes_decode.zig");
const lookup_rig = @import("notes_lookup_rig_test.zig");

test "the whole notes block stays within the cap: a note that does not fit is not put in" {
    var r: Rig = .{};
    try r.init(&answers);
    defer r.deinit();
    const fits = (try r.ask(&.{ answers[0], answers[1] }, "")).value;
    const enlarged: i64 = injected(fits)[0].get("block_tokens").?.int64().?;
    const a = r.arena.allocator();
    // (a): the enlarged first block, counted by the tokenizer, against the request's cap
    const over = (try r.ask(&.{ answers[0], answers[1] }, try std.fmt.allocPrint(a, ",\"notes_cap\":{d}", .{enlarged - 1}))).value;
    try testing.expectEqual(@as(usize, 0), injected(over).len);
    try testing.expectEqualStrings("Your sister is Anna. She drives a red car.", content(over));
    const at = (try r.ask(&.{ answers[0], answers[1] }, try std.fmt.allocPrint(a, ",\"notes_cap\":{d}", .{enlarged}))).value;
    try testing.expectEqual(enlarged, injected(at)[0].get("block_tokens").?.int64().?);
    // (b): the first block and the late one together
    const first: i64 = fits.get("notes").?.get("tokens").?.int64().?;
    const late: i64 = @intCast((try r.ids("Current:\n[2] \"Anna drives a red Volvo.\"")).len);
    const b_over = (try r.ask(&.{ answers[0], answers[1] }, try std.fmt.allocPrint(a, ",\"notes_decode_place\":\"turn\",\"notes_cap\":{d}", .{first + late - 1}))).value;
    try testing.expectEqual(@as(usize, 0), injected(b_over).len);
    const b_at = (try r.ask(&.{ answers[0], answers[1] }, try std.fmt.allocPrint(a, ",\"notes_decode_place\":\"turn\",\"notes_cap\":{d}", .{first + late}))).value;
    try testing.expectEqual(first + late, injected(b_at)[0].get("block_tokens").?.int64().?);
}

test "notes_decode false: one request, the same first prompt, the reply as the engine made it; a reply naming nothing is alike" {
    var r: Rig = .{};
    try r.init(&answers);
    defer r.deinit();
    const on = (try r.ask(&.{ answers[0], answers[1] }, "")).value;
    const on_first = try r.arena.allocator().dupe(u32, r.script.prompts.items[0]);
    const off = (try r.ask(&.{ answers[0], answers[1] }, ",\"notes_decode\":false")).value;
    try testing.expectEqual(@as(usize, 1), r.script.prompts.items.len);
    try testing.expectEqualSlices(u32, on_first, r.script.prompts.items[0]); // the feature leaves the first prefill alone
    try testing.expectEqualStrings("Your sister is Anna. She drives a red car.", content(off));
    try testing.expect(!decodeOf(off).get("on").?.bool);
    try testing.expect(decodeOf(on).get("on").?.bool);
    // the negative control through the server: a reply that names nobody makes no check and puts nothing in
    const none = (try r.ask(&.{answers[5]}, "")).value;
    try testing.expectEqual(@as(usize, 1), r.script.prompts.items.len);
    try testing.expectEqual(@as(usize, 0), injected(none).len);
    try testing.expectEqual(@as(i64, 0), decodeOf(none).get("checks").?.int64().?);
    // a dry run checks and reports, and puts nothing in
    const dry = (try r.ask(&.{ answers[0], answers[1] }, ",\"notes_decode_dry\":true")).value;
    try testing.expectEqual(@as(usize, 1), r.script.prompts.items.len);
    try testing.expectEqual(@as(usize, 0), injected(dry).len);
    try testing.expectEqualStrings("2", decodeOf(dry).get("check_list").?.array[0].get("ids").?.array[0].int);
}

test "a thinking reply is read whole: a name in its reasoning brings the note in before the answer" {
    var r: Rig = .{};
    try r.init(&.{ "Her name is Anna. What does she drive?</think>A red car.", ". What does she drive? Anna drives a red Volvo.</think>A red Volvo." });
    defer r.deinit();
    const got = (try r.ask(&.{ "Her name is Anna. What does she drive?</think>A red car.", ". What does she drive? Anna drives a red Volvo.</think>A red Volvo." }, ",\"chat_template_kwargs\":{\"enable_thinking\":true}")).value;
    const inj = injected(got);
    try testing.expectEqual(@as(usize, 1), inj.len);
    try testing.expectEqual(@as(i64, @intCast((try r.ids("Her name is Anna")).len)), inj[0].get("at_token").?.int64().?);
    try testing.expectEqualStrings("A red Volvo.", content(got));
    const reasoning = got.get("choices").?.array[0].get("message").?.strField("reasoning_content").?;
    try testing.expectEqualStrings("Her name is Anna. What does she drive? Anna drives a red Volvo.", reasoning);
}

test "the request fields: wrong types and the decode fields without notes are refused" {
    var r: Rig = .{};
    try r.init(&answers);
    defer r.deinit();
    const Case = struct { extra: []const u8, message: []const u8 };
    for ([_]Case{
        .{ .extra = ",\"notes_decode\":\"yes\"", .message = "notes_decode must be a boolean" },
        .{ .extra = ",\"notes_decode_dry\":1", .message = "notes_decode_dry must be a boolean" },
        .{ .extra = ",\"notes_decode_place\":\"middle\"", .message = "notes_decode_place must be \"first\", \"turn\" or \"user\"" },
        .{ .extra = ",\"notes_decode_names\":\"yes\"", .message = "notes_decode_names must be a boolean" },
        .{ .extra = ",\"notes_value_hop\":1", .message = "notes_value_hop must be a boolean" },
        .{ .extra = ",\"notes_position\":\"last\"", .message = "notes_position must be \"first\" or \"user\"" },
        .{ .extra = ",\"notes_decode_checks\":-1", .message = "notes_decode_checks must be a whole number, 0 or more" },
        .{ .extra = ",\"notes_decode_checks\":1.5", .message = "notes_decode_checks must be a whole number, 0 or more" },
        .{ .extra = ",\"notes_decode_gap\":\"4\"", .message = "notes_decode_gap must be a whole number of tokens, 0 or more" },
        .{ .extra = ",\"notes_decode_budget_ms\":-1", .message = "notes_decode_budget_ms must be a number of milliseconds, 0 to 3600000" },
        .{ .extra = ",\"notes_decode_budget_ms\":true", .message = "notes_decode_budget_ms must be a number of milliseconds, 0 to 3600000" },
    }) |c| {
        const got = try r.ask(&.{answers[0]}, c.extra);
        try testing.expectEqual(@as(u16, 400), got.status);
        try testing.expectEqualStrings(c.message, got.value.get("error").?.strField("message").?);
    }
    const a = r.arena.allocator();
    const body = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}],\"notes_decode\":false}}", .{});
    r.script.replies = &.{answers[0]};
    const got = try lookup_rig.request(r.srv, a, "POST", "/v1/chat/completions", body);
    try testing.expectEqual(@as(u16, 400), got.status);
    try testing.expectEqualStrings("the notes_decode fields need \"notes\": true", got.value.get("error").?.strField("message").?);
}

test "no new name, no check: a long reply of sentence ends and line breaks is notes_decode false's reply, token for token" {
    var r: Rig = .{ .legacy = "" }; // the server's defaults
    try r.init(&answers);
    defer r.deinit();
    const a = r.arena.allocator();
    const on = (try r.ask(&.{long_plain}, "")).value;
    try testing.expectEqual(@as(usize, 1), r.script.prompts.items.len);
    const on_prompt = try a.dupe(u32, r.script.prompts.items[0]);
    const d = decodeOf(on);
    try testing.expect(d.get("on").?.bool);
    try testing.expectEqual(@as(i64, 0), d.get("checks").?.int64().?); // the old cadence made one at each of its 21 line and sentence ends
    try testing.expectEqual(@as(i64, 0), d.get("attempts").?.int64().?);
    try testing.expectEqual(@as(i64, 0), d.get("triggers").?.int64().?);
    try testing.expect(d.get("stopped").? == .null);
    const off = (try r.ask(&.{long_plain}, ",\"notes_decode\":false")).value;
    try testing.expectEqualSlices(u32, on_prompt, r.script.prompts.items[0]);
    try testing.expectEqualStrings(long_plain, content(on));
    try testing.expectEqualStrings(content(off), content(on));
    try testing.expectEqualStrings(on.get("usage").?.get("completion_tokens").?.int, off.get("usage").?.get("completion_tokens").?.int);
    try testing.expectEqualStrings(on.get("choices").?.array[0].strField("finish_reason").?, off.get("choices").?.array[0].strField("finish_reason").?);
    try testing.expectEqualStrings(on.get("notes").?.strField("block").?, off.get("notes").?.strField("block").?);
}

test "a new name makes exactly one lookup; a name already looked up never asks again, even when it found nothing" {
    var r: Rig = .{ .legacy = "" };
    try r.init(&answers);
    defer r.deinit();
    const one = (try r.ask(&.{ answers[0], answers[1] }, ",\"notes_decode_place\":\"first\"")).value;
    try testing.expectEqual(@as(i64, 1), decodeOf(one).get("attempts").?.int64().?);
    try testing.expectEqual(@as(i64, 1), decodeOf(one).get("checks").?.int64().?);
    try testing.expectEqual(@as(usize, 1), injected(one).len);
    // dry: Anna's lookup puts nothing in, so her note stays outside the block, and her name comes twice more
    const dry = (try r.ask(&.{answers[11]}, ",\"notes_decode_dry\":true")).value;
    try testing.expectEqual(@as(i64, 1), decodeOf(dry).get("attempts").?.int64().?);
    try testing.expectEqual(@as(i64, 1), decodeOf(dry).get("triggers").?.int64().?);
    try testing.expectEqual(@as(usize, 1), r.script.prompts.items.len);
    // control: the name scan itself still sees the uncovered note at the third "Anna"; only the covered list stops it
    const store = r.srv.notes.?;
    const a = r.arena.allocator();
    var l: notes_decode.Live = .{ .a = a, .on = true, .store = store, .io = r.srv.io, .ver = store.gen, .question = question, .past = false, .cap = 512, .exclude = &.{}, .place = .first, .echo = false, .block = "", .block_tokens = 0 };
    for (decodeOf(dry).get("ids").?.array) |id| try l.held.append(a, @intCast(id.int64().?));
    const upto = try r.ids("Anna is here. Anna is there. Anna");
    const e = l.nameWritten(upto).?;
    try l.covered.append(a, e);
    try testing.expect(l.nameWritten(upto) == null);
}

test "the cap: three lookups a reply by default, a fourth name is told the cap ran out; the budget, the same at 0 ms" {
    var r: Rig = .{ .legacy = "" };
    try r.init(&answers);
    defer r.deinit();
    const capped = decodeOf((try r.ask(&.{five_names}, ",\"notes_decode_dry\":true")).value);
    try testing.expectEqual(@as(i64, 3), capped.get("attempts").?.int64().?);
    try testing.expectEqual(@as(i64, 3), capped.get("triggers").?.int64().?); // the fourth and fifth came after the stop
    try testing.expectEqualStrings("the check cap ran out", capped.strField("stopped").?);
    try testing.expectEqual(@as(i64, 3), capped.get("limits").?.get("checks").?.int64().?);
    // Control: cap 6 leaves room after all five names; cap 5 would be exhausted at the tent.
    const all = decodeOf((try r.ask(&.{five_names}, ",\"notes_decode_dry\":true,\"notes_decode_checks\":6")).value);
    try testing.expectEqual(@as(i64, 5), all.get("attempts").?.int64().?);
    try testing.expect(all.get("stopped").? == .null);
    // the budget: read before each lookup, so with 0 ms the first name finds it spent
    const broke = decodeOf((try r.ask(&.{five_names}, ",\"notes_decode_dry\":true,\"notes_decode_budget_ms\":0")).value);
    try testing.expectEqual(@as(i64, 0), broke.get("attempts").?.int64().?);
    try testing.expectEqualStrings("the check time budget ran out", broke.strField("stopped").?);
    // the gap: a large one holds the second lookup back past the reply's end
    const gapped = decodeOf((try r.ask(&.{five_names}, ",\"notes_decode_dry\":true,\"notes_decode_gap\":1000")).value);
    try testing.expectEqual(@as(i64, 1), gapped.get("attempts").?.int64().?);
    try testing.expect(gapped.get("pending").?.bool);
    try testing.expect(gapped.get("stopped").? == .null); // reply ended before the gap, not a spent cap/budget
    try testing.expect(!all.get("pending").?.bool);
    // no name scan at all: notes_decode_names false makes no check point
    const nameless = decodeOf((try r.ask(&.{five_names}, ",\"notes_decode_names\":false")).value);
    try testing.expectEqual(@as(i64, 0), nameless.get("attempts").?.int64().?);
}

test "spent means spent: after the cap or the budget runs out, the rest of a long reply makes no name scan at all" {
    var r: Rig = .{ .legacy = "" };
    try r.init(&(answers ++ [_][]const u8{ "Anna", "Anna " ++ long_plain }));
    defer r.deinit();
    const reply = "Anna " ++ long_plain;
    // a cap of 1: Anna's lookup at the first token, then the cap is spent and not one more token is scanned
    const cap = decodeOf((try r.ask(&.{reply}, ",\"notes_decode_dry\":true,\"notes_decode_checks\":1")).value);
    try testing.expectEqual(@as(i64, 1), cap.get("attempts").?.int64().?);
    try testing.expectEqual(@as(i64, 1), cap.get("scans").?.int64().?);
    try testing.expectEqualStrings("the check cap ran out", cap.strField("stopped").?);
    // a budget of 0: spent before the first token, so no scan and no lookup
    const budget = decodeOf((try r.ask(&.{reply}, ",\"notes_decode_dry\":true,\"notes_decode_budget_ms\":0")).value);
    try testing.expectEqual(@as(i64, 0), budget.get("scans").?.int64().?);
    try testing.expectEqual(@as(i64, 0), budget.get("attempts").?.int64().?);
    try testing.expectEqualStrings("the check time budget ran out", budget.strField("stopped").?);
    // control: with room left, every token of the same reply is scanned
    const room = decodeOf((try r.ask(&.{reply}, ",\"notes_decode_dry\":true")).value);
    const n: i64 = @intCast((try r.ids(reply)).len);
    try testing.expectEqual(n, room.get("scans").?.int64().?);
    try testing.expectEqual(@as(i64, 1), room.get("attempts").?.int64().?);
}
