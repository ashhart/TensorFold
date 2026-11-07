//! A DSML call's /v1/messages stream: its input_json_delta pieces join to the whole reply's tool_use input.
const std = @import("std");
const posix = @import("sys").posix; // the layer serve.zig installs its stop handlers with
const api = @import("engine_api");
const json = @import("json");
const model_text = @import("model_text.zig");
const serve = @import("serve.zig");
const d = @import("drain_test.zig");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// A DeepSeek-V4.1 call after prose, written as the model writes it.
const reply = "Writing it.\n\n<\u{ff5c}DSML\u{ff5c} calls>\n<\u{ff5c}DSML\u{ff5c} invoke name=\"write_file\">\n" ++
    "<\u{ff5c}DSML\u{ff5c} parameter name=\"path\" string=\"true\">notes.txt</\u{ff5c}DSML\u{ff5c} parameter>\n" ++
    "<\u{ff5c}DSML\u{ff5c} parameter name=\"content\" string=\"true\">hello \"world\"\nline two</\u{ff5c}DSML\u{ff5c} parameter>\n" ++
    "</\u{ff5c}DSML\u{ff5c} invoke>\n</\u{ff5c}DSML\u{ff5c} calls>";

/// The reply above one byte a token, a character a round, then the stop; bytes decode as themselves.
const ByteEngine = struct {
    const reply_ids = blk: {
        var out: [reply.len]u32 = undefined;
        for (reply, &out) |byte, *id| id.* = byte;
        break :blk out;
    };

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 4096 };
    }

    fn submit(_: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        var at: usize = 0;
        while (at < reply_ids.len) {
            const n = std.unicode.utf8ByteSequenceLength(reply[at]) catch 1;
            sink.event(sink.ctx, id, &.{ .tokens = reply_ids[at..][0..n] });
            at += n;
        }
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }

    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }

    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }

    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        const decoded = try a.alloc(u8, ids.len);
        for (ids, decoded) |id, *byte| byte.* = @intCast(id);
        return decoded;
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

    fn render(_: *anyopaque, a: Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        return .{ .text = try (a.dupe(u8, "prompt")) };
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

fn noop(_: *anyopaque) void {}

var listen_port: std.atomic.Value(u16) = .init(0);

fn onListen(port: u16) void {
    listen_port.store(port, .release);
}

/// serve.run on its own thread, its exit code kept for the join.
const Served = struct {
    code: u8 = 255,

    fn run(x: *Served, gpa: Allocator, io: std.Io, setup: serve.Setup) void {
        x.code = serve.run(gpa, io, .{ .host = "127.0.0.1", .port = 0 }, setup);
    }
};

/// The request's body: one tool offered, whole or streamed.
fn body(a: Allocator, stream: bool) ![]const u8 {
    const tail = ",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"tools\":[{\"name\":\"write_file\"," ++
        "\"input_schema\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}}}}]}";
    return std.mem.concat(a, u8, &.{ "{\"model\":\"x\",\"max_tokens\":400,\"stream\":", if (stream) "true" else "false", tail });
}

/// A Messages reply's input_json_delta pieces, in order.
fn inputPieces(a: Allocator, sse: []const u8) ![]const []const u8 {
    const event = "event: content_block_delta\ndata: ";
    var out: std.ArrayList([]const u8) = .empty;
    var at: usize = 0;
    while (std.mem.indexOfPos(u8, sse, at, event)) |e| {
        const start = e + event.len;
        const end = std.mem.indexOfScalarPos(u8, sse, start, '\n') orelse return error.NoDataLine;
        at = end;
        const delta = (try json.parseText(a, sse[start..end])).ok.get("delta") orelse return error.BadEvent;
        if (delta.typeIs("input_json_delta")) try out.append(a, delta.strField("partial_json") orelse return error.BadEvent);
    }
    return out.items;
}

test "a DSML call's /v1/messages input_json_delta pieces join to the whole reply's input" {
    const gpa = testing.allocator;
    const io = testing.io;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var env = std.process.Environ.Map.init(gpa);
    defer env.deinit();
    try env.put("TENSORFOLD_NO_LIVE", "1");
    try env.put("TF_DRAIN_S", "5");
    var eng: ByteEngine = .{};
    var x: Served = .{};
    listen_port.store(0, .release);
    const t = try std.Thread.spawn(.{}, Served.run, .{ &x, gpa, io, serve.Setup{
        .engine = eng.engine(),
        .text = eng.text(),
        .served = "test",
        .environ = &env,
        .on_listen = onListen,
        .stop = .{ .ctx = &eng, .close = noop },
    } });
    while (listen_port.load(.acquire) == 0) std.Io.sleep(io, .fromMilliseconds(2), .awake) catch {};
    const port = listen_port.load(.acquire);
    const whole = try d.request(a, io, port, "POST", "/v1/messages", try body(a, false));
    const stream = try d.request(a, io, port, "POST", "/v1/messages", try body(a, true));
    try posix.raise(.TERM);
    t.join();
    try testing.expectEqual(@as(u8, 0), x.code);
    try testing.expectEqual(@as(u16, 200), whole.status);
    try testing.expectEqual(@as(u16, 200), stream.status);

    var input: ?json.Value = null;
    for ((try json.parseText(a, whole.body)).ok.get("content").?.array) |block| if (block.typeIs("tool_use")) {
        input = block.get("input");
    };
    const expected = try json.stringify(a, input orelse return error.NoToolUse, .{ .ascii = false, .compact = true });
    try testing.expectEqualStrings("{\"path\":\"notes.txt\",\"content\":\"hello \\\"world\\\"\\nline two\"}", expected);
    // the path went out with the header, then the content in pieces as it was written
    const pieces = try inputPieces(a, stream.body);
    if (pieces.len < 10) std.debug.print("{d} input_json_delta pieces\n", .{pieces.len});
    try testing.expect(pieces.len >= 10);
    try testing.expectEqualStrings(expected, try std.mem.concat(a, u8, pieces));
}
