//! /v1/systemone against llama-server System One goldens: the same fake tokenizer and head logits give the same bytes.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const model_text = @import("model_text.zig");
const systemone = @import("systemone.zig");
const Server = @import("server.zig").Server;
const Allocator = std.mem.Allocator;

// Greedy longest match over `table` (ids from 256), else UTF-8 bytes: the goldens' tokenizer.
const table = [_][]const u8{ "<|startoftext|>", "<|im_start|>", "<|im_end|>", "yes", "Yes", "no", "No", " A", " B", "AB" };

const Text = struct {
    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }

    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        var ids: std.ArrayList(u32) = .empty;
        var i: usize = 0;
        while (i < input.len) {
            var best: ?usize = null;
            for (table, 0..) |t, j| if (std.mem.startsWith(u8, input[i..], t) and (best == null or t.len > table[best.?].len)) {
                best = j;
            };
            if (best) |j| {
                try ids.append(a, @intCast(256 + j));
                i += table[j].len;
            } else {
                try ids.append(a, input[i]);
                i += 1;
            }
        }
        return ids.toOwnedSlice(a);
    }

    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (ids) |id| if (id >= 256) try out.appendSlice(a, table[id - 256]) else try out.append(a, @intCast(id));
        return out.toOwnedSlice(a);
    }

    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }

    fn tokenString(_: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 256 + table.len;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }

    fn render(_: *anyopaque, a: Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        return .{ .text = try (a.dupe(u8, "")) };
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// Scores each label as the goldens' head does and keeps every prompt it was asked for, as text.
const Head = struct {
    gpa: Allocator,
    prompts: std.ArrayList([]u8) = .empty,
    window: u32 = 0,
    hooks: enum { none, score, decide } = .decide,
    pictures: bool = false,
    seen: ?api.Decision = null, // the last decision, its slices valid while the request's arena lives

    fn engine(h: *Head) api.Engine {
        return .{ .ctx = h, .vtable = switch (h.hooks) {
            .none => &bare_vt,
            .score => &score_vt,
            .decide => &decide_vt,
        } };
    }

    fn info(ctx: *anyopaque) api.Info {
        const h: *Head = @ptrCast(@alignCast(ctx));
        return .{ .context_window = h.window, .pictures = h.pictures };
    }

    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {}
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }

    fn score(ctx: *anyopaque, prompt: []const u32, labels: []const u32, logits: []f64, mass: bool) error{Failed}!f64 {
        const h: *Head = @ptrCast(@alignCast(ctx));
        const t = Text.decode(undefined, h.gpa, prompt) catch return error.Failed;
        h.prompts.append(h.gpa, t) catch return error.Failed;
        if (mass) return error.Failed; // System One reads the options alone
        for (labels, logits) |id, *x| x.* = @as(f64, @floatFromInt((id * 37) % 101)) / 10.0;
        return 0;
    }

    fn decide(ctx: *anyopaque, d: *api.Decision) error{ Failed, Refused }!void {
        of(ctx).seen = d.*;
        var at: usize = 0;
        d.read = d.trunk.len;
        for (d.branches, d.counts) |b, n| {
            const prompt = std.mem.concat(of(ctx).gpa, u32, &.{ d.trunk, b }) catch return error.Failed;
            defer of(ctx).gpa.free(prompt);
            _ = try score(ctx, prompt, d.labels[at..][0..n], d.logits[at..][0..n], false);
            at += n;
            d.read += b.len;
        }
    }

    fn of(ctx: *anyopaque) *Head {
        return @ptrCast(@alignCast(ctx));
    }

    fn reset(h: *Head) void {
        for (h.prompts.items) |p| h.gpa.free(p);
        h.prompts.clearRetainingCapacity();
    }
};

const bare_vt: api.Engine.VTable = .{ .info = Head.info, .submit = Head.submit, .cancel = Head.cancel, .status = Head.status, .memory = Head.memory };
const score_vt: api.Engine.VTable = .{ .info = Head.info, .submit = Head.submit, .cancel = Head.cancel, .status = Head.status, .memory = Head.memory, .score = Head.score };
const decide_vt: api.Engine.VTable = .{ .info = Head.info, .submit = Head.submit, .cancel = Head.cancel, .status = Head.status, .memory = Head.memory, .score = Head.score, .decide = Head.decide };

fn serve(text: *Text, head: *Head) !*Server {
    return Server.init(std.testing.allocator, std.testing.io, head.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
}

/// Equal JSON in key order and text; floats within 1e-12 relative (Python's exp and Zig's differ in the last bit).
fn expectSame(want: json.Value, got: json.Value) !void {
    try std.testing.expectEqual(std.meta.activeTag(want), std.meta.activeTag(got));
    switch (want) {
        .float => |w| try std.testing.expectApproxEqRel(w, got.float, 1e-12),
        .object => |o| {
            try std.testing.expectEqual(o.count(), got.object.count());
            for (o.keys(), got.object.keys()) |w, g| try std.testing.expectEqualStrings(w, g);
            for (o.values(), got.object.values()) |w, g| try expectSame(w, g);
        },
        .array => |items| {
            try std.testing.expectEqual(items.len, got.array.len);
            for (items, got.array) |w, g| try expectSame(w, g);
        },
        else => try std.testing.expect(json.equal(want, got)),
    }
}

const Case = struct { body: []const u8, want: []const u8, prompts: []const []const u8 };

test "System One answers and prompts equal the goldens, packed or a pass a question" {
    for ([_]@TypeOf(@as(Head, undefined).hooks){ .decide, .score }) |hooks| {
        var text: Text = .{};
        var head: Head = .{ .gpa = std.testing.allocator, .hooks = hooks };
        defer {
            head.reset();
            head.prompts.deinit(std.testing.allocator);
        }
        const srv = try serve(&text, &head);
        defer srv.deinit();
        for (answered) |case| {
            var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
            defer arena.deinit();
            const a = arena.allocator();
            head.reset();
            var cx: errors.Cx = .{ .a = a };
            const out = try systemone.reply(srv, &cx, (try json.parse(a, case.body)).ok);
            try expectSame((try json.parse(a, case.want)).ok, out);
            try std.testing.expectEqual(case.prompts.len, head.prompts.items.len);
            for (case.prompts, head.prompts.items) |want, got| try std.testing.expectEqualStrings(want, got);
        }
    }
}

test "System One refuses in the goldens' words, before any scoring" {
    var text: Text = .{};
    var head: Head = .{ .gpa = std.testing.allocator };
    defer head.prompts.deinit(std.testing.allocator);
    const srv = try serve(&text, &head);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (refused) |case| {
        var cx: errors.Cx = .{ .a = a };
        try std.testing.expectError(error.Refused, systemone.reply(srv, &cx, (try json.parse(a, case[0])).ok));
        try std.testing.expectEqualStrings(case[1], cx.message);
    }
    // pictures wait for an engine that reads them; a sequence past the engine's context is refused by its length
    const pictured = "{\"state\": \"x\", \"images\": [\"data:image/png;base64,AA\"], \"questions\": {\"q\": {\"type\": \"noul\", \"instructions\": \"i\"}}}";
    var cx: errors.Cx = .{ .a = a };
    try std.testing.expectError(error.Refused, systemone.reply(srv, &cx, (try json.parse(a, pictured)).ok));
    try std.testing.expectEqualStrings("this checkpoint has no vision tower", cx.message);
    head.window = 8;
    try std.testing.expectError(error.Refused, systemone.reply(srv, &cx, (try json.parse(a, answered[0].body)).ok));
    try std.testing.expectEqualStrings("a decision sequence exceeds the 8-token context", cx.message);
    try std.testing.expectEqual(@as(usize, 0), head.prompts.items.len);
}

test "pictures reach the engine in llama-server's order, the state's image parts taken out of its text" {
    var text: Text = .{};
    var head: Head = .{ .gpa = std.testing.allocator, .pictures = true };
    defer {
        head.reset();
        head.prompts.deinit(std.testing.allocator);
    }
    const srv = try serve(&text, &head);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body =
        \\{"state": {"messages": [{"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBB"}}]}, {"role": "assistant", "content": "ok"}], "note": 1},
        \\ "images": ["data:image/png;base64,AAA"], "questions": {"q": {"type": "noul", "instructions": "i"}}}
    ;
    var cx: errors.Cx = .{ .a = a };
    _ = try systemone.reply(srv, &cx, (try json.parse(a, body)).ok);
    const d = head.seen.?;
    try std.testing.expectEqual(@as(usize, 2), d.pictures.len);
    try std.testing.expectEqualStrings("data:image/png;base64,AAA", d.pictures[0]);
    try std.testing.expectEqualStrings("data:image/png;base64,BBB", d.pictures[1]);
    const head_text = try Text.decode(undefined, a, d.trunk[0..d.at]);
    try std.testing.expectEqualStrings("<|startoftext|><|im_start|>user\n", head_text);
    // the System One state block for the same state with its image part taken out
    const want = "{\n  \"messages\": [\n    {\n      \"role\": \"user\",\n      \"content\": [\n        {\n          \"type\": \"text\",\n          \"text\": \"look\"\n        }\n      ]\n    },\n    {\n      \"role\": \"assistant\",\n      \"content\": \"ok\"\n    }\n  ],\n  \"note\": 1\n}\n\n\nQUESTION:\n";
    try std.testing.expectEqualStrings(want, try Text.decode(undefined, a, d.trunk[d.at..]));
    const five = "{\"state\": null, \"images\": [\"a\", \"b\", \"c\", \"d\", \"e\"], \"questions\": {\"q\": {\"type\": \"noul\", \"instructions\": \"i\"}}}";
    try std.testing.expectError(error.Refused, systemone.reply(srv, &cx, (try json.parse(a, five)).ok));
    try std.testing.expectEqualStrings("at most 4 pictures a request", cx.message);
}

// The goldens, reproduced from the d1 recipe's Python reference (github.com/plotarmordev/d1-tensorfold).
pub const answered = [_]Case{
    .{ .body = "{\"state\": \"I was charged twice, refund one.\", \"questions\": {\"refund\": {\"type\": \"noul\", \"instructions\": \"Is this a refund request?\"}}}", .want = "{\"model\": \"m\", \"answers\": {\"refund\": {\"type\": \"noul\", \"noul\": 0.2689414213699951}}, \"usage\": {\"input_tokens\": 115, \"output_tokens\": 0}}", .prompts = &.{
        "<|startoftext|><|im_start|>user\nI was charged twice, refund one.\n\n\nQUESTION:\nIs this a refund request?\n\nReply with yes or no only.<|im_end|>\n<|im_start|>assistant\n",
    } },
    .{ .body = "{\"state\": null, \"questions\": {\"q\": {\"type\": \"noul\", \"instructions\": \"Is the sky green?\", \"criteria\": {\"true\": \"green\", \"false\": null}}}}", .want = "{\"model\": \"m\", \"answers\": {\"q\": {\"type\": \"noul\", \"noul\": 0.2689414213699951}}, \"usage\": {\"input_tokens\": 78, \"output_tokens\": 0}}", .prompts = &.{
        "<|startoftext|><|im_start|>user\nIs the sky green?\nYes: green\nNo: None\n\nReply with yes or no only.<|im_end|>\n<|im_start|>assistant\n",
    } },
    .{ .body = "{\"state\": {\"user\": \"Ana\", \"items\": [1, 2.5, true, null, {}], \"note\": \"caf\xc3\xa9 \xe2\x98\x95\", \"empty\": []}, \"questions\": {\"mood\": {\"instructions\": \"Which mood?\", \"criteria\": {\"happy_now\": null, \"sad\": \"feeling low\", \"n\": {\"x\": 1}, \"zero\": 0}}, \"rate\": {\"type\": \"score\", \"instructions\": \"Rate it.\", \"criteria\": [\"bad\", \"ok\", {\"level\": \"good\"}]}}}", .want = "{\"model\": \"m\", \"answers\": {\"mood\": {\"type\": \"choice\", \"choice\": \"zero\", \"confidence\": 0.6544767103154275, \"probabilities\": {\"happy_now\": 0.24076852645056343, \"sad\": 0.08857379096230497, \"n\": 0.01618097227170413, \"zero\": 0.6544767103154275}}, \"rate\": {\"type\": \"score\", \"score\": 0.9775308804049918, \"confidence\": 0.974293192524506, \"probabilities\": {\"0\": 0.02408796353525111, \"1\": 0.974293192524506, \"2\": 0.0016188439402429462}, \"legend\": {\"0\": \"bad\", \"1\": \"ok\", \"2\": \"{\\\"level\\\": \\\"good\\\"}\"}}}, \"usage\": {\"input_tokens\": 338, \"output_tokens\": 0}}", .prompts = &.{
        "<|startoftext|><|im_start|>user\n{\n  \"user\": \"Ana\",\n  \"items\": [\n    1,\n    2.5,\n    true,\n    null,\n    {}\n  ],\n  \"note\": \"caf\xc3\xa9 \xe2\x98\x95\",\n  \"empty\": []\n}\n\n\nQUESTION:\nWhich mood?\n\nOptions:\nA happy now\nB feeling low\nC {\"x\": 1}\nD zero\n\nReply with the option code only.<|im_end|>\n<|im_start|>assistant\n",
        "<|startoftext|><|im_start|>user\n{\n  \"user\": \"Ana\",\n  \"items\": [\n    1,\n    2.5,\n    true,\n    null,\n    {}\n  ],\n  \"note\": \"caf\xc3\xa9 \xe2\x98\x95\",\n  \"empty\": []\n}\n\n\nQUESTION:\nRate it.\n\n0 bad\n1 ok\n2 {\"level\": \"good\"}\n\nReply with a single digit 0-2 only.<|im_end|>\n<|im_start|>assistant\n",
    } },
    .{ .body = "{\"state\": [{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"hi\"}]}], \"questions\": {\"letters\": {\"type\": \"choice\", \"instructions\": \"Pick.\", \"criteria\": {\"A\": \"first\", \"B\": \"second\", \"c\": null}}, \"dup\": {\"type\": \"choice\", \"instructions\": \"Pick.\", \"criteria\": {\"a\": \"x\", \" a\": \"y\"}}}}", .want = "{\"model\": \"m\", \"answers\": {\"letters\": {\"type\": \"choice\", \"choice\": \"A\", \"confidence\": 0.7288809234577932, \"probabilities\": {\"A\": 0.7288809234577932, \"B\": 0.26814030680217804, \"c\": 0.002978769740028626}}, \"dup\": {\"type\": \"choice\", \"choice\": \" a\", \"confidence\": 0.9426758241011313, \"probabilities\": {\"a\": 0.05732417589886881, \" a\": 0.9426758241011313}}}, \"usage\": {\"input_tokens\": 288, \"output_tokens\": 0}}", .prompts = &.{
        "<|startoftext|><|im_start|>user\n[\n  {\n    \"role\": \"user\",\n    \"content\": [\n      {\n        \"type\": \"text\",\n        \"text\": \"hi\"\n      }\n    ]\n  }\n]\n\n\nQUESTION:\nPick.\n\nOptions:\nA first\nB second\nc c\n\nReply with the option code only.<|im_end|>\n<|im_start|>assistant\n",
        "<|startoftext|><|im_start|>user\n[\n  {\n    \"role\": \"user\",\n    \"content\": [\n      {\n        \"type\": \"text\",\n        \"text\": \"hi\"\n      }\n    ]\n  }\n]\n\n\nQUESTION:\nPick.\n\nOptions:\na x\nA y\n\nReply with the option code only.<|im_end|>\n<|im_start|>assistant\n",
    } },
    .{ .body = "{\"state\": \"many\", \"questions\": {\"big\": {\"type\": \"choice\", \"instructions\": \"Pick one.\", \"criteria\": {\"opt0\": null, \"opt1\": null, \"opt2\": null, \"opt3\": null, \"opt4\": null, \"opt5\": null, \"opt6\": null, \"opt7\": null, \"opt8\": null, \"opt9\": null, \"opt10\": null, \"opt11\": null, \"opt12\": null, \"opt13\": null, \"opt14\": null, \"opt15\": null, \"opt16\": null, \"opt17\": null, \"opt18\": null, \"opt19\": null, \"opt20\": null, \"opt21\": null, \"opt22\": null, \"opt23\": null, \"opt24\": null, \"opt25\": null, \"opt26\": null}}}}", .want = "{\"model\": \"m\", \"answers\": {\"big\": {\"type\": \"choice\", \"choice\": \"opt25\", \"confidence\": 0.27603257025325034, \"probabilities\": {\"opt0\": 0.05573001478724626, \"opt1\": 0.02050192669640839, \"opt2\": 0.003745364218770724, \"opt3\": 0.15148988649592537, \"opt4\": 0.0002517091227195988, \"opt5\": 0.010180955496845168, \"opt6\": 1.6916240653643228e-05, \"opt7\": 0.0006842163343460532, \"opt8\": 0.02767470632342445, \"opt9\": 4.598310957463856e-05, \"opt10\": 0.0018598928283877349, \"opt11\": 0.07522765130690531, \"opt12\": 0.0001249950511727811, \"opt13\": 0.005055712878287677, \"opt14\": 0.20448995754521404, \"opt15\": 0.0003397717762502794, \"opt16\": 0.013742852446955769, \"opt17\": 2.2834536437395574e-05, \"opt18\": 0.000923595445204387, \"opt19\": 0.03735694607775379, \"opt20\": 6.207070545905834e-05, \"opt21\": 0.002510592715546627, \"opt22\": 0.10154670768988264, \"opt23\": 0.00016872567072899193, \"opt24\": 0.006824498557332045, \"opt25\": 0.27603257025325034, \"opt26\": 0.0033889456893167716}}}, \"usage\": {\"input_tokens\": 296, \"output_tokens\": 0}}", .prompts = &.{
        "<|startoftext|><|im_start|>user\nmany\n\n\nQUESTION:\nPick one.\n\nOptions:\nA opt0\nB opt1\nC opt2\nD opt3\nE opt4\nF opt5\nG opt6\nH opt7\nI opt8\nJ opt9\nK opt10\nL opt11\nM opt12\nN opt13\nO opt14\nP opt15\nQ opt16\nR opt17\nS opt18\nT opt19\nU opt20\nV opt21\nW opt22\nX opt23\nY opt24\nZ opt25\na opt26\n\nReply with the option code only.<|im_end|>\n<|im_start|>assistant\n",
    } },
    .{ .body = "{\"state\": 42, \"questions\": {\"n\": {\"type\": \"noul\", \"instructions\": \"Even?\", \"criteria\": {}}}}", .want = "{\"model\": \"m\", \"answers\": {\"n\": {\"type\": \"noul\", \"noul\": 0.2689414213699951}}, \"usage\": {\"input_tokens\": 65, \"output_tokens\": 0}}", .prompts = &.{
        "<|startoftext|><|im_start|>user\n42\n\n\nQUESTION:\nEven?\n\nReply with yes or no only.<|im_end|>\n<|im_start|>assistant\n",
    } },
};

pub const refused = [_][2][]const u8{
    .{ "{\"questions\": {}}", "a request needs a state (null when the question is the whole state)" },
    .{ "{\"state\": \"x\"}", "a request needs questions" },
    .{ "{\"state\": \"x\", \"questions\": []}", "questions is a non-empty object of id: question" },
    .{ "{\"state\": \"x\", \"questions\": {}}", "questions is a non-empty object of id: question" },
    .{ "{\"state\": \"x\", \"questions\": {\"q\": 1}}", "a question is an object with type, instructions and criteria" },
    .{ "{\"state\": \"x\", \"questions\": {\"q\": {\"type\": \"noul\"}}}", "a question needs instructions (llama-server's System One requires them)" },
    .{ "{\"state\": \"x\", \"questions\": {\"q\": {\"type\": \"maybe\", \"instructions\": \"i\"}}}", "unknown question type 'maybe': choose noul, choice or score" },
    .{ "{\"state\": \"x\", \"questions\": {\"q\": {\"type\": null, \"instructions\": \"i\"}}}", "unknown question type None: choose noul, choice or score" },
    .{ "{\"state\": \"x\", \"questions\": {\"q\": {\"type\": \"noul\", \"instructions\": \"i\", \"criteria\": [1]}}}", "noul criteria is an object with true and false" },
    .{ "{\"state\": \"x\", \"questions\": {\"q\": {\"type\": \"score\", \"instructions\": \"i\", \"criteria\": [\"one\"]}}}", "score criteria is a list of 2 to 10 levels, lowest first" },
    .{ "{\"state\": \"x\", \"questions\": {\"q\": {\"type\": \"choice\", \"instructions\": \"i\", \"criteria\": {}}}}", "choice criteria is a non-empty object of option: description" },
    .{ "{\"state\": \"x\", \"images\": \"data:x\", \"questions\": {}}", "images must be an array of data URLs" },
    .{ "{\"state\": \"x\", \"videos\": [\"v\"], \"questions\": {}}", "videos are not supported" },
    .{ "{\"state\": [{\"role\": \"user\", \"content\": [{\"type\": \"input_audio\"}]}], \"questions\": {\"q\": {\"type\": \"noul\", \"instructions\": \"i\"}}}", "input_audio is not supported: a decision state takes images" },
};
