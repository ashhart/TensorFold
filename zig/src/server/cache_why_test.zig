//! #3686 on the server: TF_CACHE_WHY's reply field, the done line's fields and the /metrics series, by reason.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const server = @import("server.zig");
const log = @import("log.zig");
const cache_why = @import("cache_why.zig");
const model_text = @import("model_text.zig");
const Conn = @import("http_conn.zig").Conn;
const Why = api.prompt_cache.why.Why;
const Reason = api.prompt_cache.why.Reason;
const t = std.testing;
const Allocator = std.mem.Allocator;

/// Every clock stands still and entropy counts up, so two runs write the same bytes.
const Clock = struct {
    threadlocal var table: std.Io.VTable = undefined;
    threadlocal var next: u64 = 0;
    fn randomSecure(_: ?*anyopaque, bytes: []u8) std.Io.RandomSecureError!void {
        for (bytes) |*b| {
            next += 1;
            b.* = @truncate(next *% 0x9e37_79b9);
        }
    }
    fn now(_: ?*anyopaque, _: std.Io.Clock) std.Io.Timestamp {
        return .{ .nanoseconds = 1_800_000_000_000_000_000 };
    }
    fn io() std.Io {
        table = t.io.vtable.*;
        table.now = now;
        table.randomSecure = randomSecure;
        return .{ .userdata = t.io.userdata, .vtable = &table };
    }
};

/// Bytes as tokens; every prompt renders as "prompt".
const ByteText = struct {
    with_starts: bool = false,
    fn text(b: *@This()) model_text.Text {
        return .{ .ctx = b, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource, .token_bytes = tokenBytes } };
    }
    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const out = try a.alloc(u32, input.len);
        for (input, out) |byte, *id| id.* = byte;
        return out;
    }
    fn decode(_: *anyopaque, a: Allocator, tokens: []const u32) Allocator.Error![]u8 {
        const out = try a.alloc(u8, tokens.len);
        for (tokens, out) |id, *byte| byte.* = @intCast(id);
        return out;
    }
    fn tokenBytes(ctx: *anyopaque, a: Allocator, id: u32) Allocator.Error!?[]u8 {
        const self: *ByteText = @ptrCast(@alignCast(ctx));
        return if (self.with_starts) try decode(ctx, a, &.{id}) else null;
    }
    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }
    fn tokenString(_: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        return decode(undefined, a, &.{id});
    }
    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }
    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }
    fn render(ctx: *anyopaque, a: Allocator, _: json.Value, opts: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        const self: *ByteText = @ptrCast(@alignCast(ctx));
        if (self.with_starts) {
            if (opts.message_starts) |out| out.* = try a.dupe(usize, &.{ 0, 4, 8 });
            return .{ .text = try a.dupe(u8, "p000p111p222") };
        }
        return .{ .text = try a.dupe(u8, "prompt") };
    }
    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// Restores `why.resumed` tokens, writes "hi" and ends saying `why`.
const Script = struct {
    why: ?Why,
    fn engine(s: *@This()) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 4096, .prompt_cache = true };
    }
    fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        sink.event(sink.ctx, id, &.{ .prefilled = if (s.why) |w| w.resumed else 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{ 'h', 'i' } });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop, .stats = .{ .rounds = 2, .prompt_cache = s.why } } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// The whole reply a route wrote, held in memory.
const Wire = struct {
    a: Allocator,
    bytes: std.ArrayList(u8) = .empty,
    fn write(p: *anyopaque, bytes: []const u8) error{Closed}!void {
        const w: *@This() = @ptrCast(@alignCast(p));
        w.bytes.appendSlice(w.a, bytes) catch return error.Closed;
    }
};

const Line = struct {
    bytes: std.ArrayList(u8) = .empty,
    fn line(ctx: *anyopaque, message: []const u8) void {
        const l: *Line = @ptrCast(@alignCast(ctx));
        l.bytes.appendSlice(t.allocator, message) catch {};
    }
};

const ReplyResult = struct { body: []const u8, line: []const u8, counts: cache_why.Counts };

const body = "{\"model\":\"m\",\"max_tokens\":2,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}";

/// One chat completion on a server with TF_CACHE_WHY `on`: its body, its done line and the reason counts.
fn chat(a: Allocator, on: bool, why: ?Why) !ReplyResult {
    return reply(a, on, why, body, true, false);
}

fn reply(a: Allocator, on: bool, why: ?Why, request_body: []const u8, is_chat: bool, with_starts: bool) !ReplyResult {
    Clock.next = 0;
    var lines: Line = .{};
    defer lines.bytes.deinit(t.allocator);
    log.Test.sink = .{ .ctx = &lines, .call = Line.line };
    defer log.Test.sink = null;
    var text: ByteText = .{ .with_starts = with_starts };
    var script: Script = .{ .why = why };
    const srv = try server.Server.init(t.allocator, Clock.io(), script.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false, .cache_why = on }, null);
    defer srv.deinit();
    try t.expectEqual(@as(u32, 0), srv.info.prefill_step);
    if (!is_chat) srv.chunks.openers = &.{112};
    var wire: Wire = .{ .a = a };
    var headers = [_]@import("http_conn.zig").Header{.{ .name = "Content-Length", .value = try std.fmt.allocPrint(a, "{d}", .{request_body.len}) }};
    var c: Conn = .{ .fd = -1, .peer = "", .buf = try a.dupe(u8, request_body), .end = request_body.len, .gpa = a, .version = "HTTP/1.1", .method = "POST", .path = "/v1/chat/completions", .headers = &headers, .write_hooks = .{ .ctx = &wire, .write = Wire.write } };
    @import("openai.zig").post(srv, &c, a, is_chat);
    const raw = wire.bytes.items;
    const split = std.mem.indexOf(u8, raw, "\r\n\r\n") orelse return error.BadReply;
    const done = std.mem.indexOf(u8, lines.bytes.items, "done ") orelse return error.NoDoneLine;
    const end = std.mem.indexOfScalarPos(u8, lines.bytes.items, done, '\n') orelse lines.bytes.items.len;
    return .{ .body = raw[split + 4 ..], .line = try a.dupe(u8, lines.bytes.items[done..end]), .counts = srv.metrics.cache };
}

const evicted: Why = .{ .reason = .evicted, .shared = 40, .resumed = 0 };

test "cache why: off, a reply has no prompt_cache field; on, the field is its only difference" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const off = try chat(a, false, evicted);
    const on = try chat(a, true, evicted);
    try t.expect(std.mem.indexOf(u8, off.body, "prompt_cache") == null);
    const plain = (try json.parse(a, off.body)).ok;
    const told = (try json.parse(a, on.body)).ok;
    const field = told.get("tensorfold").?.get("prompt_cache").?;
    try t.expectEqualStrings("evicted", field.strField("reason").?);
    try t.expectEqual(@as(?i64, 40), field.get("shared").?.int64());
    try t.expectEqual(@as(?i64, 0), field.get("resumed").?.int64());
    try t.expectEqual(@as(?i64, 40), field.get("lost").?.int64());
    try t.expect(field.get("message") == null); // this template's message openers are unknown
    _ = told.get("tensorfold").?.object.orderedRemove("prompt_cache");
    try told.object.put(a, "id", plain.get("id").?); // each reply draws its own id
    try t.expectEqualStrings(try json.stringify(a, plain, .{}), try json.stringify(a, told, .{}));
}

test "cache why: the done line names the reason, and N and the loss past none; an engine that says nothing adds nothing" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try t.expect(std.mem.endsWith(u8, (try chat(a, false, evicted)).line, " cache=evicted at=40 lost=40"));
    try t.expect(std.mem.endsWith(u8, (try chat(a, false, .{ .reason = .none, .shared = 1, .resumed = 1 })).line, " cache=none"));
    const silent = try chat(a, false, null);
    try t.expect(std.mem.indexOf(u8, silent.line, "cache=") == null);
    for (silent.counts.lookups) |n| try t.expectEqual(@as(u64, 0), n);
}

test "cache why: /metrics has one lookups and one lost-tokens series a reason, and they count each finished request" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const counts = (try chat(a, false, evicted)).counts;
    var out: std.Io.Writer.Allocating = .init(a);
    try counts.render(&out.writer, "tensorfold:");
    var series: usize = 0;
    var lines = std.mem.splitScalar(u8, out.written(), '\n');
    while (lines.next()) |l| {
        if (l.len == 0 or l[0] == '#') continue;
        series += 1;
        const open = std.mem.indexOf(u8, l, "{reason=\"").? + "{reason=\"".len;
        const label = l[open..std.mem.indexOfScalarPos(u8, l, open, '"').?];
        try t.expect(std.meta.stringToEnum(Reason, label) != null); // only the fixed set: the labels are bounded
    }
    try t.expectEqual(2 * api.prompt_cache.why.reasons, series);
    try t.expect(std.mem.indexOf(u8, out.written(), "tensorfold:prompt_cache_lookups_total{reason=\"evicted\"} 1\n") != null);
    try t.expect(std.mem.indexOf(u8, out.written(), "tensorfold:prompt_cache_lost_tokens_total{reason=\"evicted\"} 40\n") != null);
    var sum: u64 = 0;
    for (counts.lookups) |n| sum += n;
    try t.expectEqual(@as(u64, 1), sum);
}

test "cache why: the message holding N uses rendered starts, including the last message" {
    const starts = [_]u32{ 3, 6 };
    try t.expectEqual(@as(?u32, 0), cache_why.messageAt(&starts, 2));
    try t.expectEqual(@as(?u32, 1), cache_why.messageAt(&starts, 4));
    try t.expectEqual(@as(?u32, 2), cache_why.messageAt(&starts, 8));
    try t.expectEqual(@as(?u32, null), cache_why.messageAt(&.{}, 4));
}

test "cache why: TF_CACHE_WHY takes 0 or 1 through the settings table" {
    const env = @import("sys").environment;
    try t.expect(try env.value(bool, "TF_CACHE_WHY", "1", false));
    try t.expect(!try env.value(bool, "TF_CACHE_WHY", null, false));
    try t.expectError(error.InvalidEnvironment, env.value(bool, "TF_CACHE_WHY", "yes please", false));
}

test "cache why: rendered message starts survive zero prefill step" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const got = try reply(a, true, .{ .reason = .evicted, .shared = 5, .resumed = 0 }, body, true, true);
    const field = (try json.parse(a, got.body)).ok.get("tensorfold").?.get("prompt_cache").?;
    try t.expect(field.get("message") != null);
    try t.expectEqual(@as(?i64, 1), field.get("message").?.int64());
}

test "cache why: raw completion opener words never imply a message" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const raw = "{\"model\":\"m\",\"max_tokens\":2,\"prompt\":\"p000p111p222\"}";
    const got = try reply(a, true, .{ .reason = .evicted, .shared = 5, .resumed = 0 }, raw, false, false);
    const field = (try json.parse(a, got.body)).ok.get("tensorfold").?.get("prompt_cache").?;
    try t.expect(field.get("message") == null);
}
