//! oMLX's /v1/decisions and /v1/systemone shapes over the qev prompt: rendered prompts, the prefix split, answers.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const model_text = @import("model_text.zig");
const qev = @import("qev.zig");
const Server = @import("server.zig").Server;
const Allocator = std.mem.Allocator;

// Bytes as ids, these specials from 256: a Qwen-like template's markers.
const table = [_][]const u8{ "<|im_start|>", "<|im_end|>", "<think>", "</think>" };

const Text = struct {
    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }
    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        var ids: std.ArrayList(u32) = .empty;
        var i: usize = 0;
        outer: while (i < input.len) {
            for (table, 0..) |t, j| if (std.mem.startsWith(u8, input[i..], t)) {
                try ids.append(a, @intCast(256 + j));
                i += t.len;
                continue :outer;
            };
            try ids.append(a, input[i]);
            i += 1;
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
    /// Qwen's template with thinking off: each message in its turn, then the empty think block.
    fn render(_: *anyopaque, a: Allocator, messages: json.Value, o: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        var out: std.ArrayList(u8) = .empty;
        for (messages.array) |m| try out.print(a, "<|im_start|>{s}\n{s}<|im_end|>\n", .{ json.strOr(m.get("role")), json.strOr(m.get("content")) });
        if (o.add_generation_prompt and !o.enable_thinking) try out.appendSlice(a, "<|im_start|>assistant\n<think>\n\n</think>\n\n");
        return .{ .text = out.items };
    }
    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// Label j of a branch scores j / 2; keeps the last decision's trunk and branches as text.
const Head = struct {
    gpa: Allocator,
    trunk: []u8 = &.{},
    branches: std.ArrayList([]u8) = .empty,
    fn engine(h: *Head) api.Engine {
        return .{ .ctx = h, .vtable = &vt };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .decides = true };
    }
    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {}
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
    fn decide(ctx: *anyopaque, d: *api.Decision) error{ Failed, Refused }!void {
        const h: *Head = @ptrCast(@alignCast(ctx));
        h.reset();
        h.trunk = Text.decode(undefined, h.gpa, d.trunk) catch return error.Failed;
        var at: usize = 0;
        d.read = d.trunk.len;
        for (d.branches, d.counts) |b, n| {
            h.branches.append(h.gpa, Text.decode(undefined, h.gpa, b) catch return error.Failed) catch return error.Failed;
            for (d.logits[at..][0..n], 0..) |*x, j| x.* = @as(f64, @floatFromInt(j)) / 2;
            at += n;
            d.read += b.len;
        }
    }
    fn reset(h: *Head) void {
        h.gpa.free(h.trunk);
        h.trunk = &.{};
        for (h.branches.items) |b| h.gpa.free(b);
        h.branches.clearRetainingCapacity();
    }
};

const vt: api.Engine.VTable = .{ .info = Head.info, .submit = Head.submit, .cancel = Head.cancel, .status = Head.status, .memory = Head.memory, .decide = Head.decide };

const tail = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
const head_text = "<|im_start|>system\nApply the question to the state. Choose exactly one of the listed options. Respond with only its uppercase letter, with no explanation or reasoning.<|im_end|>\n<|im_start|>user\n<state>\nUser: add a zig test\n</state>\n\n";

test "qev: a skill-routing decisions body renders the qev prompt, splits at the state and answers as oMLX" {
    var text: Text = .{};
    var head: Head = .{ .gpa = std.testing.allocator };
    defer {
        head.reset();
        head.branches.deinit(std.testing.allocator);
    }
    const srv = try Server.init(std.testing.allocator, std.testing.io, head.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body =
        \\{"model": "decision", "input": "User: add a zig test", "questions": [{"type": "choice", "name": "pick", "instructions": "  Which skill fits?  ", "choices": ["none", "test-first", {"value": "zig", "description": " the Zig build "}]}, {"type": "predicate", "name": "fits", "instructions": "a listed skill fits"}]}
    ;
    var cx: errors.Cx = .{ .a = a };
    const out = try qev.reply(srv, &cx, (try json.parse(a, body)).ok, false);
    try std.testing.expectEqualStrings(head_text, head.trunk);
    try std.testing.expectEqual(@as(usize, 2), head.branches.items.len);
    try std.testing.expectEqualStrings("Question: Which skill fits?\n\nOptions:\nA. none\nB. test-first\nC. zig: the Zig build\n\nAnswer with one letter: A, B, C." ++ tail, head.branches.items[0]);
    try std.testing.expectEqualStrings("Question: a listed skill fits\n\nOptions:\nA. yes: the statement is true / the answer is yes\nB. no: the statement is false / the answer is no\n\nAnswer with one letter: A, B." ++ tail, head.branches.items[1]);
    // softmax of (0, 0.5, 1) and (0, 0.5), rounded to 6 places as Python's round does
    const want =
        \\{"object": "decision", "model": "decision", "answers": [{"type": "choice", "name": "pick", "choice": "zig", "probabilities": [{"value": "none", "probability": 0.186324}, {"value": "test-first", "probability": 0.307196}, {"value": "zig", "probability": 0.50648}], "confidence": 0.259721}, {"type": "predicate", "name": "fits", "probability": 0.377541}], "usage": {"input_tokens": 0, "output_tokens": 0}}
    ;
    const expected = (try json.parse(a, want)).ok;
    var tokens: usize = (try Text.encode(undefined, a, head_text, false)).len;
    for (head.branches.items) |b| tokens += (try Text.encode(undefined, a, b, false)).len;
    try expected.get("usage").?.object.put(a, "input_tokens", try json.intValue(a, tokens));
    try std.testing.expect(json.equal(expected, out));
}

test "qev: System One's shape keys answers by id, noul criteria replace the defaults, a score keeps its legend" {
    var text: Text = .{};
    var head: Head = .{ .gpa = std.testing.allocator };
    defer {
        head.reset();
        head.branches.deinit(std.testing.allocator);
    }
    const srv = try Server.init(std.testing.allocator, std.testing.io, head.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body =
        \\{"model": "decision", "state": {"command": "ls"}, "questions": {"safe": {"type": "noul", "instructions": "the command is safe", "criteria": {"true": "safe", "false": null}}, "harm": {"type": "score", "instructions": "", "criteria": ["none", "some"]}}}
    ;
    var cx: errors.Cx = .{ .a = a };
    const out = try qev.reply(srv, &cx, (try json.parse(a, body)).ok, true);
    try std.testing.expect(std.mem.indexOf(u8, head.trunk, "<state>\n{\n  \"command\": \"ls\"\n}\n</state>\n\n") != null);
    try std.testing.expect(std.mem.startsWith(u8, head.branches.items[0], "Question: the command is safe\n\nOptions:\nA. yes: safe\nB. no: the statement is false / the answer is no\n\n"));
    try std.testing.expect(std.mem.startsWith(u8, head.branches.items[1], "Rate the state: Which level describes the state?\n\nOptions:\nA. level 0: none\nB. level 1: some\n\n"));
    const answers = out.get("answers").?;
    try std.testing.expectEqualStrings("safe", answers.object.keys()[0]);
    try std.testing.expectEqual(@as(f64, 0.377541), answers.get("safe").?.get("noul").?.float);
    try std.testing.expectEqual(@as(f64, 0.622459), answers.get("harm").?.get("score").?.float);
    try std.testing.expectEqualStrings("some", answers.get("harm").?.get("legend").?.get("1").?.string);
    try std.testing.expect(out.get("object") == null);
}

test "qev: duplicate names, one-choice questions and pictures are refused before any pass" {
    var text: Text = .{};
    var head: Head = .{ .gpa = std.testing.allocator };
    defer head.branches.deinit(std.testing.allocator);
    const srv = try Server.init(std.testing.allocator, std.testing.io, head.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{
        \\{"model": "m", "input": "x", "questions": [{"type": "predicate", "name": "a", "instructions": "i"}, {"type": "predicate", "name": "a", "instructions": "j"}]}
        ,
        \\{"model": "m", "input": "x", "questions": [{"type": "choice", "name": "a", "instructions": "i", "choices": ["only"]}]}
        ,
        \\{"model": "m", "input": [{"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,AA"}]}], "questions": [{"type": "predicate", "name": "a", "instructions": "i"}]}
        ,
        \\{"model": "m", "input": "x", "questions": []}
    }) |body| {
        var cx: errors.Cx = .{ .a = a };
        try std.testing.expectError(error.Refused, qev.reply(srv, &cx, (try json.parse(a, body)).ok, false));
    }
    try std.testing.expectEqual(@as(usize, 0), head.branches.items.len);
}
