//! Supplied questions through the lesson's own question filter, over a fake engine that answers by what it is asked.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const model_text = @import("model_text.zig");
const slide_supplied = @import("slide_supplied.zig");
const slide_subject = @import("slide_subject.zig");
const slide_take = @import("slide_take.zig");
const wording = @import("slide_words.zig");
const Server = @import("server.zig").Server;
const Allocator = std.mem.Allocator;

/// Bytes as tokens; a conversation renders as "Q:" and its last user message, then "\nA:".
const ByteText = struct {
    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }

    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }

    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (ids) |id| if (id < 256) try out.append(a, @intCast(id));
        return out.toOwnedSlice(a);
    }

    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }

    fn tokenString(_: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }

    fn render(_: *anyopaque, a: Allocator, messages: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        var last: []const u8 = "";
        for (messages.items() orelse &.{}) |m| if (std.mem.eql(u8, m.strField("role") orelse "", "user")) {
            last = m.strField("content") orelse "";
        };
        return std.fmt.allocPrint(a, "Q:{s}\nA:", .{last});
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// An engine that knows the warehouse robot and answers the lesson's own prompts the way a model told the fact would.
const Robot = struct {
    asked: u32 = 0,

    fn engine(m: *Robot) api.Engine {
        return .{ .ctx = m, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{};
    }

    fn has(text: []const u8, word: []const u8) bool {
        return std.mem.indexOf(u8, text, word) != null;
    }

    fn answer(q: []const u8) []const u8 {
        // the judge: "Does it answer this question they ask you: ...? Reply yes or no."
        if (has(q, "Reply yes or no.")) return if (has(q, "top speed")) "no" else "yes";
        // the short answer: "...\nAnswer in as few words as you can: just the answer."
        if (has(q, "Answer in as few words")) return "Pebble";
        if (has(q, "called?")) return "Our warehouse robot is called Pebble.";
        if (has(q, "do we call")) return "We call it Pebble.";
        return "I don't know.";
    }

    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const m: *Robot = @ptrCast(@alignCast(ctx));
        m.asked += 1;
        var prompt: [768]u8 = undefined;
        const n = @min(prompt.len, request.prompt.len);
        for (request.prompt[0..n], prompt[0..n]) |x, *b| b.* = @intCast(x);
        const asked = prompt[0..n];
        const q = asked[if (std.mem.indexOf(u8, asked, "Q:")) |i| i + 2 else 0 .. std.mem.indexOf(u8, asked, "\nA:") orelse n];
        const said = answer(q);
        var ids: [64]u32 = undefined;
        for (said, ids[0..said.len]) |b, *x| x.* = b;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = ids[0..said.len] });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

const Stays = struct {
    pub fn check(_: Stays) bool {
        return false;
    }
};

const fact = "Our company's warehouse robot is called Pebble.";

/// The questions `take` keeps for the fact from `qs`, in order, its subject worded and told in chat as a lesson would.
fn kept(a: Allocator, qs: []const []const u8) ![]const []const u8 {
    var text: ByteText = .{};
    var m: Robot = .{};
    const srv = try Server.init(std.testing.allocator, std.testing.io, m.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    var cx: errors.Cx = .{ .a = a };
    const document = !wording.firstPerson(fact); // a chat fact in the first person asks as the user does
    const words = try slide_subject.factSubject(a, "warehouse robot", fact, qs, document);
    var got: slide_take.Got = .{ .entity = true, .words = words, .named = try slide_subject.named(a, words, fact, qs) };
    try slide_take.take(srv, &cx, &got, fact, 0, qs, &.{}, false, document, Stays{});
    var out: std.ArrayList([]const u8) = .empty;
    for (got.refs.items) |r| try out.append(a, r.question);
    return out.items;
}

test "supplied questions that name the subject and the fact answers are kept, the rest dropped by the lesson's own filters" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const supplied = try slide_supplied.read(a, (try json.parse(a,
        \\{"text": "Our company's warehouse robot is called Pebble.", "questions": [
        \\ "What is our warehouse robot called?", "What do we call the warehouse robot?", "Is our warehouse robot called Pebble?",
        \\ "What is the capital of France?", "What is Pebble?", "What is our warehouse robot's top speed?"]}
    )).ok);
    try std.testing.expectEqual(@as(usize, 6), supplied.len);
    const got = try kept(a, try slide_supplied.ahead(a, supplied, &.{}));
    // kept: the two the user asks that the fact answers; the rest are dropped by the lesson's own filters
    try std.testing.expectEqual(@as(usize, 2), got.len);
    try std.testing.expectEqualStrings("What is our warehouse robot called?", got[0]);
    try std.testing.expectEqualStrings("What do we call the warehouse robot?", got[1]);
}

test "a supplied question goes first, and the model's own questions are judged beside it as before" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const written = [_][]const u8{ "What is the capital of France?", "What do we call the warehouse robot?" };
    const got = try kept(a, try slide_supplied.ahead(a, &.{"What is our warehouse robot called?"}, &written));
    try std.testing.expectEqual(@as(usize, 2), got.len);
    try std.testing.expectEqualStrings("What is our warehouse robot called?", got[0]);
    try std.testing.expectEqualStrings("What do we call the warehouse robot?", got[1]);
}

test "without supplied questions the lesson keeps exactly what it kept from the model's list alone" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const written = [_][]const u8{ "What do we call the warehouse robot?", "What is the capital of France?", "What is our warehouse robot called?" };
    const alone = try kept(a, &written);
    const with_none = try kept(a, try slide_supplied.ahead(a, &.{}, &written));
    try std.testing.expectEqual(alone.len, with_none.len);
    for (alone, with_none) |x, y| try std.testing.expectEqualStrings(x, y);
    try std.testing.expectEqual(@as(usize, 2), alone.len);
}
