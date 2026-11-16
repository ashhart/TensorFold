//! ``suffix`` on /v1/completions through the real HfText: the FIM prompt's ids, its end tokens and the refusals.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const HfText = @import("hf_text.zig").HfText;
const eos_fixture = @import("hf_eos_fixture.zig");
const fim = @import("completion_fim.zig");
const openai = @import("openai.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;
const T = std.testing;

const prefix_id = 100;
const middle_id = 101;
const suffix_id = 102;
const pad_id = 103;
const text_end_id = 104;
const im_end_id = 105;

/// Qwen's FIM tokens added but not special, as its tokenizers hold them; every other token is one character.
const tokenizer_json =
    \\{"model":{"type":"BPE","vocab":{"H":0,"O":1,"K":2,"<":3,"|":4,"f":5,"i":6,"m":7,"_":8,"s":9,"u":10,"x":11,">":12,
    \\ "<|fim_prefix|>":100,"<|fim_middle|>":101,"<|fim_suffix|>":102,"<|fim_pad|>":103,"<|endoftext|>":104,"<|im_end|>":105},
    \\ "merges":[]},"decoder":{"type":"Fuse"},"added_tokens":[
    \\ {"id":100,"content":"<|fim_prefix|>","special":false,"normalized":false},
    \\ {"id":101,"content":"<|fim_middle|>","special":false,"normalized":false},
    \\ {"id":102,"content":"<|fim_suffix|>","special":false,"normalized":false},
    \\ {"id":103,"content":"<|fim_pad|>","special":false,"normalized":false},
    \\ {"id":104,"content":"<|endoftext|>","special":true},{"id":105,"content":"<|im_end|>","special":true}]}
;

/// A checkpoint folder with that tokenizer, `<|im_end|>` its one end token.
const Folder = struct {
    tmp: T.TmpDir,
    path: [:0]u8,

    fn init() !Folder {
        var tmp = T.tmpDir(.{});
        errdefer tmp.cleanup();
        try tmp.dir.writeFile(T.io, .{ .sub_path = "tokenizer.json", .data = tokenizer_json });
        try tmp.dir.writeFile(T.io, .{ .sub_path = "config.json", .data = "{\"eos_token_id\":105,\"vocab_size\":106}" });
        try tmp.dir.writeFile(T.io, .{ .sub_path = "tokenizer_config.json", .data = "{\"eos_token\":\"<|im_end|>\"}" });
        return .{ .tmp = tmp, .path = try tmp.dir.realPathFileAlloc(T.io, ".", T.allocator) };
    }

    fn deinit(f: *Folder) void {
        T.allocator.free(f.path);
        f.tmp.cleanup();
    }
};

/// Keeps the prompt and end tokens it is given, then sends `reply` whole; the server ends the reply at an end token.
const Script = struct {
    a: Allocator,
    reply: []const u32,
    prompt: []const u32 = &.{},
    eos: []const u32 = &.{},
    submits: usize = 0,

    fn engine(s: *Script) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 4096, .reasoning = false };
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        s.submits += 1;
        s.prompt = s.a.dupe(u32, request.prompt) catch return error.Busy;
        s.eos = s.a.dupe(u32, request.eos) catch return error.Busy;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = s.reply });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .length } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// The route's answer: its status and body, or a stream's text and finish reason.
const Got = struct {
    a: Allocator,
    status: u16 = 0,
    body: json.Value = .null,
    text: std.ArrayList(u8) = .empty,
    finish: []const u8 = "",

    fn out(g: *Got) openai.Out {
        return .{ .ctx = g, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }
    fn open(ctx: *anyopaque, _: usize) error{Closed}!void {
        const g: *Got = @ptrCast(@alignCast(ctx));
        g.status = 200;
    }
    fn event(ctx: *anyopaque, value: ?json.Value) error{Closed}!void {
        const g: *Got = @ptrCast(@alignCast(ctx));
        const choices = (value orelse return).get("choices") orelse return;
        if (choices.array.len == 0) return;
        g.text.appendSlice(g.a, json.strOr(choices.array[0].get("text"))) catch return error.Closed;
        if (choices.array[0].get("finish_reason")) |f| if (f == .string) {
            g.finish = f.string;
        };
    }
    fn reply(ctx: *anyopaque, status_code: u16, value: json.Value) void {
        const g: *Got = @ptrCast(@alignCast(ctx));
        g.status = status_code;
        g.body = value;
        const choices = value.get("choices") orelse return;
        g.text.appendSlice(g.a, json.strOr(choices.array[0].get("text"))) catch {};
        g.finish = json.strOr(choices.array[0].get("finish_reason"));
    }
};

/// `body` through the completions route over the tokenizer in `dir` and `script`'s engine.
fn send(a: Allocator, dir: []const u8, script: *Script, body: []const u8) !Got {
    var problem: []const u8 = "";
    const text = try HfText.load(T.allocator, T.io, dir, null, a, &problem);
    defer text.deinit();
    const srv = try Server.init(T.allocator, T.io, script.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var got: Got = .{ .a = a };
    openai.run(srv, a, got.out(), .{ .conn = &conn }, false, (try json.parse(a, body)).ok);
    return got;
}

test "completions suffix: the marks go around the texts, and the reply ends at the vocabulary's FIM end tokens" {
    var arena = std.heap.ArenaAllocator.init(T.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var folder = try Folder.init();
    defer folder.deinit();
    for ([_]u32{ pad_id, text_end_id, im_end_id }) |end| for ([_][]const u8{ "}", ",\"stream\":true}" }) |tail| {
        var script: Script = .{ .a = a, .reply = &.{ 1, end, 2 } };
        const body = try std.mem.concat(a, u8, &.{ "{\"model\":\"m\",\"prompt\":\"HO\",\"suffix\":\"K\",\"max_tokens\":4", tail });
        const got = try send(a, folder.path, &script, body);
        try T.expectEqual(@as(u16, 200), got.status);
        // prefix, suffix, middle: the reply is what goes between "HO" and "K"
        try T.expectEqualSlices(u32, &.{ prefix_id, 0, 1, suffix_id, 2, middle_id }, script.prompt);
        try T.expectEqualSlices(u32, &.{ im_end_id, pad_id, text_end_id }, script.eos);
        try T.expectEqualStrings("O", got.text.items);
        try T.expectEqualStrings("stop", got.finish);
    };
    // the prompt's text may be empty: the reply then starts the file
    var script: Script = .{ .a = a, .reply = &.{ 1, pad_id } };
    _ = try send(a, folder.path, &script, "{\"model\":\"m\",\"suffix\":\"K\",\"max_tokens\":4}");
    try T.expectEqualSlices(u32, &.{ prefix_id, suffix_id, 2, middle_id }, script.prompt);
}

test "completions suffix: the client's texts read as plain text, so only the server's marks are FIM tokens" {
    var arena = std.heap.ArenaAllocator.init(T.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var folder = try Folder.init();
    defer folder.deinit();
    var script: Script = .{ .a = a, .reply = &.{ 1, pad_id } };
    _ = try send(a, folder.path, &script, "{\"model\":\"m\",\"prompt\":\"H<|fim_suffix|>\",\"suffix\":\"K\",\"max_tokens\":4}");
    const spelled = [_]u32{ 3, 4, 5, 6, 7, 8, 9, 10, 5, 5, 6, 11, 4, 12 };
    try T.expectEqualSlices(u32, &([_]u32{ prefix_id, 0 } ++ spelled ++ [_]u32{ suffix_id, 2, middle_id }), script.prompt);
    // without a suffix the prompt reads as it always has, the spelling as its token
    var plain: Script = .{ .a = a, .reply = &.{1} };
    _ = try send(a, folder.path, &plain, "{\"model\":\"m\",\"prompt\":\"H<|fim_suffix|>\",\"max_tokens\":4}");
    try T.expectEqualSlices(u32, &.{ 0, suffix_id }, plain.prompt);
}

test "completions suffix: absent, null or empty leaves the prompt and the end tokens as they were" {
    var arena = std.heap.ArenaAllocator.init(T.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var folder = try Folder.init();
    defer folder.deinit();
    for ([_][]const u8{ "", ",\"suffix\":null", ",\"suffix\":\"\"" }) |field| {
        var script: Script = .{ .a = a, .reply = &.{ 1, pad_id, 2 } };
        const body = try std.mem.concat(a, u8, &.{ "{\"model\":\"m\",\"prompt\":\"HO\",\"max_tokens\":3", field, "}" });
        const got = try send(a, folder.path, &script, body);
        try T.expectEqualSlices(u32, &.{ 0, 1 }, script.prompt);
        try T.expectEqualSlices(u32, &.{im_end_id}, script.eos);
        try T.expectEqualStrings("O<|fim_pad|>K", got.text.items);
        try T.expectEqualStrings("length", got.finish);
    }
    // ignore_eos drops the FIM end tokens too, so they decode into the text
    var script: Script = .{ .a = a, .reply = &.{ 1, pad_id, 2 } };
    const got = try send(a, folder.path, &script, "{\"model\":\"m\",\"prompt\":\"HO\",\"suffix\":\"K\",\"max_tokens\":3,\"ignore_eos\":true}");
    try T.expectEqual(@as(usize, 0), script.eos.len);
    try T.expectEqualStrings("O<|fim_pad|>K", got.text.items);
}

test "completions suffix: refused by name before anything is submitted" {
    var arena = std.heap.ArenaAllocator.init(T.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var folder = try Folder.init();
    defer folder.deinit();
    var plain = try eos_fixture.Fixture.init(T.allocator, T.io, null); // a tokenizer with no FIM tokens
    defer plain.deinit();
    const cases = [_]struct { dir: []const u8, body: []const u8, message: []const u8 }{
        .{ .dir = plain.path, .body = "{\"prompt\":\"H\",\"suffix\":\"K\"}", .message = "suffix is not supported by this model: its tokenizer has no fill-in-the-middle tokens" },
        .{ .dir = folder.path, .body = "{\"prompt\":\"H\",\"suffix\":5}", .message = "suffix must be a string or null" },
        .{ .dir = folder.path, .body = "{\"prompt\":[0],\"suffix\":\"K\"}", .message = "suffix is supported with a text prompt" },
        .{ .dir = folder.path, .body = "{\"prompt\":[\"H\"],\"suffix\":\"K\"}", .message = "suffix is supported with a text prompt" },
        .{ .dir = folder.path, .body = "{\"messages\":[{\"role\":\"user\",\"content\":\"H\"}],\"suffix\":\"K\"}", .message = "suffix is supported with a text prompt" },
        .{ .dir = folder.path, .body = "{\"prompt\":\"H\",\"suffix\":\"K\",\"echo\":true}", .message = "suffix and echo cannot be combined" },
    };
    for (cases) |c| {
        var script: Script = .{ .a = a, .reply = &.{1} };
        const got = try send(a, c.dir, &script, c.body);
        try T.expectEqual(@as(u16, 400), got.status);
        try T.expectStringStartsWith(json.strOr(got.body.get("error").?.get("message")), c.message);
        try T.expectEqual(@as(usize, 0), script.submits);
    }
}

/// The directories TF_TOKENIZER_DIRS lists (path-list separated), or null after saying why the test skips.
fn dirs() ?std.mem.TokenIterator(u8, .scalar) {
    const list = std.c.getenv("TF_TOKENIZER_DIRS") orelse {
        std.debug.print("completions suffix: real tokenizers skipped, TF_TOKENIZER_DIRS names no tokenizer directories\n", .{});
        return null;
    };
    return std.mem.tokenizeScalar(u8, std.mem.span(list), std.fs.path.delimiter);
}

test "completions suffix: each of TF_TOKENIZER_DIRS' tokenizers gives the FIM prompt its own whole marks, or refuses it" {
    var each = dirs() orelse return;
    while (each.next()) |dir| {
        var arena = std.heap.ArenaAllocator.init(T.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var problem: []const u8 = "";
        const loaded = try HfText.load(T.allocator, T.io, dir, null, a, &problem);
        defer loaded.deinit();
        const t = loaded.text();
        var ids: [3]?u32 = undefined;
        for (fim.marks, &ids) |m, *id| id.* = t.tokenId(m);
        const before = "def add(a, b):\n    ";
        const after = "\n    return c\n";
        var script: Script = .{ .a = a, .reply = &.{} };
        const body = "{\"model\":\"m\",\"prompt\":\"def add(a, b):\\n    \",\"suffix\":\"\\n    return c\\n\",\"max_tokens\":4}";
        const got = try send(a, dir, &script, body);
        if (ids[0] == null or ids[1] == null or ids[2] == null) {
            try T.expectEqual(@as(u16, 400), got.status);
            std.debug.print("completions suffix: {s} has no FIM tokens and refuses suffix\n", .{dir});
            continue;
        }
        const want = try std.mem.concat(a, u32, &.{ try t.encode(a, "", true), &.{ids[0].?}, try t.encodePlain(a, before), &.{ids[1].?}, try t.encodePlain(a, after), &.{ids[2].?} });
        try T.expectEqualSlices(u32, want, script.prompt);
        for (fim.ends) |spelling| if (t.tokenId(spelling)) |id| try T.expect(std.mem.indexOfScalar(u32, script.eos, id) != null);
        std.debug.print("completions suffix: {s} marks {d} {d} {d}, {d} prompt tokens, end tokens {any}\n", .{ dir, ids[0].?, ids[1].?, ids[2].?, script.prompt.len, script.eos });
    }
}
