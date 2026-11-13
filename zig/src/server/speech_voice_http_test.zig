//! Real sockets: --manage-voices adds and deletes catalog voices for admin keys, --request-voices takes inline voices.
const std = @import("std");
const api = @import("engine_api");
const server = @import("server.zig");
const auth = @import("auth.zig");
const listener = @import("listener.zig");
const voice = @import("drama_voice.zig");
const wire = @import("speech_stream_test.zig");
const KeyFile = @import("auth_scope_test.zig").KeyFile;
const posix = @import("sys").posix;
const t = std.testing;
const Allocator = std.mem.Allocator;

const tone =
    \\{"format":"drama3-http-voice/1","source":{"kind":"synthetic","note":"Generated fixture tone."},
    \\"description":"Warm fixture.","references":[{"text":"A small cloud passes.","codes":[[7,1,2,3,4,5,6,7,8,9]]}]}
;

const Fake = struct {
    hold: std.atomic.Value(bool) = .init(false),
    held: std.atomic.Value(bool) = .init(false),
    encodes: std.atomic.Value(u32) = .init(0),
    frames: std.ArrayList([11]i32) = .empty,

    fn engine(e: *Fake) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .submit_speech = speech, .cancel = cancel, .status = status, .memory = memory, .encode_voice = encode } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .text_generation = false, .speech = .{ .sample_rate = 44100, .frame_length = 2048, .max_frames = 4, .max_prompt_tokens = 4096 } };
    }
    fn speech(ctx: *anyopaque, id: api.Id, request: *const api.SpeechRequest, sink: api.Sink) api.SpeechSubmitError!void {
        const e: *Fake = @ptrCast(@alignCast(ctx));
        e.frames.clearRetainingCapacity();
        e.frames.appendSlice(t.allocator, request.frames) catch return error.Busy;
        e.held.store(true, .release);
        while (e.hold.load(.acquire)) std.Thread.yield() catch {};
        @memset(request.pcm[0..256], 0.125);
        request.result.* = .{ .samples = 256, .reason = .stop };
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    /// Codes from the file's bytes: one row a byte past the 44-byte header, so a test can predict them.
    fn encode(ctx: *anyopaque, gpa: Allocator, wav: []const u8) api.EncodeVoiceError![][10]i32 {
        const e: *Fake = @ptrCast(@alignCast(ctx));
        _ = e.encodes.fetchAdd(1, .monotonic);
        if (wav.len <= 44 or !std.mem.eql(u8, wav[0..4], "RIFF")) return error.BadWav;
        const codes = try gpa.alloc([10]i32, wav.len - 44);
        for (codes, wav[44..]) |*row, byte| row.* = @splat(byte);
        return codes;
    }
    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        return error.Closed;
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

const Harness = struct {
    srv: *server.Server = undefined,
    text: wire.Bytes = .{},
    listener: listener.Listener = undefined,
    accept: std.Thread = undefined,
    stop: std.atomic.Value(bool) = .init(false),

    fn start(h: *Harness, fake: *Fake, config: server.Config, keys: ?*auth.Store) !void {
        h.srv = try server.Server.init(t.allocator, t.io, fake.engine(), h.text.text(), config, keys);
        errdefer h.srv.deinit();
        h.listener = try listener.Listener.open(.{ .ip4 = .loopback(0) });
        errdefer h.listener.close();
        h.accept = try std.Thread.spawn(.{}, server.Server.serve, .{ h.srv, h.listener, &h.stop });
    }
    fn deinit(h: *Harness) void {
        h.stop.store(true, .release);
        h.accept.join();
        var waited: usize = 0;
        while (h.srv.open_connections.load(.acquire) > 0) : (waited += 1) {
            if (waited == 5000) @panic("voice fixture did not quiesce");
            std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
        }
        h.listener.close();
        h.srv.deinit();
    }
    /// The whole reply to one request on its own connection.
    fn ask(h: *Harness, a: Allocator, method: []const u8, path: []const u8, key: []const u8, body: []const u8) ![]const u8 {
        const address: std.Io.net.IpAddress = .{ .ip4 = .loopback(h.listener.port()) };
        var storage: std.Io.Threaded.PosixAddress = undefined;
        const len = std.Io.Threaded.addressToPosix(&address, &storage);
        const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
        if (posix.errno(rc) != .SUCCESS) return error.Socket;
        const fd: posix.socket_t = @intCast(rc);
        defer _ = posix.system.close(fd);
        if (posix.errno(posix.system.connect(fd, &storage.any, len)) != .SUCCESS) return error.Connect;
        const auth_line = if (key.len > 0) try std.fmt.allocPrint(a, "Authorization: Bearer {s}\r\n", .{key}) else "";
        const raw = try std.fmt.allocPrint(a, "{s} {s} HTTP/1.1\r\nHost: localhost\r\n{s}Content-Length: {d}\r\nConnection: close\r\n\r\n{s}", .{ method, path, auth_line, body.len, body });
        var sent: usize = 0;
        while (sent < raw.len) {
            const n = posix.system.write(fd, raw[sent..].ptr, raw.len - sent);
            if (posix.errno(n) != .SUCCESS or n == 0) return error.Write;
            sent += @intCast(n);
        }
        var out: std.ArrayList(u8) = .empty;
        var buffer: [4096]u8 = undefined;
        while (true) {
            var fds = [_]posix.pollfd{.{ .fd = fd, .events = posix.POLL.IN, .revents = 0 }};
            if (try posix.poll(&fds, 5000) == 0) return error.Timeout;
            const n = posix.system.read(fd, &buffer, buffer.len);
            if (posix.errno(n) != .SUCCESS) return error.Read;
            if (n == 0) return out.items;
            try out.appendSlice(a, buffer[0..@intCast(n)]);
        }
    }
};

fn expectReply(reply: []const u8, status: []const u8, text: []const u8) !void {
    if (!std.mem.startsWith(u8, reply, status) or std.mem.indexOf(u8, reply, text) == null) {
        std.debug.print("expected {s} with {s}, got:\n{s}\n", .{ status, text, reply });
        return error.TestUnexpectedResult;
    }
}

fn wavBase64(a: Allocator, samples: []const u8) ![]const u8 {
    const wav = try std.mem.concat(a, u8, &.{ "RIFF\x00\x00\x00\x00WAVE", "fmt and data headers stand here.", samples });
    const out = try a.alloc(u8, std.base64.standard.Encoder.calcSize(wav.len));
    return std.base64.standard.Encoder.encode(out, wav);
}

test "voices HTTP: without --manage-voices and --request-voices every refusal stays as it was" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var fake: Fake = .{};
    defer fake.frames.deinit(t.allocator);
    var h: Harness = .{};
    try h.start(&fake, .{ .served_name = "m", .model_ids = &.{"m"}, .keep_warm_s = 0 }, null);
    defer h.deinit();
    try expectReply(try h.ask(a, "POST", "/v1/voices", "", "{\"name\":\"tone\"," ++ tone[1..]), "HTTP/1.1 404", "unknown path /v1/voices");
    try expectReply(try h.ask(a, "DELETE", "/v1/voices/tone", "", ""), "HTTP/1.1 404", "no stored response has id None");
    const as_null = try h.ask(a, "POST", "/v1/audio/speech", "", "{\"model\":\"m\",\"input\":\"hi\",\"voice\":null}");
    const as_object = try h.ask(a, "POST", "/v1/audio/speech", "", "{\"model\":\"m\",\"input\":\"hi\",\"voice\":" ++ tone ++ "}");
    try expectReply(as_object, "HTTP/1.1 400", "voice must be a string");
    try t.expectEqualStrings(as_null[std.mem.indexOf(u8, as_null, "\r\n\r\n").?..], as_object[std.mem.indexOf(u8, as_object, "\r\n\r\n").?..]);
    try t.expectEqual(@as(u32, 0), fake.encodes.load(.monotonic));
}

test "voices HTTP: admin keys add and delete voices; a request already running keeps its voice" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var keys_file = try KeyFile.init("app[inference]: infer-secret\nmonitor[read]: read-secret\nops[admin]: admin-secret\n");
    defer keys_file.deinit();
    var problem: []const u8 = "";
    var keys = try auth.Store.init(t.allocator, &.{}, "", keys_file.path, false, &problem);
    defer keys.deinit();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    var fake: Fake = .{};
    defer fake.frames.deinit(t.allocator);
    var h: Harness = .{};
    try h.start(&fake, .{ .served_name = "m", .model_ids = &.{"m"}, .keep_warm_s = 0, .voice = .{ .dir = dir, .manage = true } }, &keys);
    defer h.deinit();
    const add_tone = "{\"name\":\"tone\"," ++ tone[1..];
    for ([_][]const u8{ "infer-secret", "read-secret" }) |key| {
        try expectReply(try h.ask(a, "POST", "/v1/voices", key, add_tone), "HTTP/1.1 403", "");
        try expectReply(try h.ask(a, "DELETE", "/v1/voices/tone", key, ""), "HTTP/1.1 403", "");
    }
    const admin = "admin-secret";
    try expectReply(try h.ask(a, "POST", "/v1/voices", admin, "{\"name\":\"default\"," ++ tone[1..]), "HTTP/1.1 400", "not default");
    try expectReply(try h.ask(a, "POST", "/v1/voices", admin, "{\"name\":\"x\",\"format\":\"drama3-http-voice/1\",\"references\":[{\"text\":\"A.\",\"codes\":[[1,2,3,4,5,6,7,8,9,10]]}]}"), "HTTP/1.1 400", "source.kind");
    const unattested_wav = try std.fmt.allocPrint(a, "{{\"name\":\"clone\",\"text\":\"A small cloud passes.\",\"audio\":\"{s}\"}}", .{try wavBase64(a, "\x05\x06")});
    try expectReply(try h.ask(a, "POST", "/v1/voices", admin, unattested_wav), "HTTP/1.1 400", "source.kind");
    try t.expectEqual(@as(u32, 0), fake.encodes.load(.monotonic)); // refused before the codec ran
    try expectReply(try h.ask(a, "POST", "/v1/voices", admin, add_tone), "HTTP/1.1 201", "\"added\": true");
    try expectReply(try h.ask(a, "POST", "/v1/voices", admin, add_tone), "HTTP/1.1 200", "\"added\": false");
    const written = try voice.parse(a, try tmp.dir.readFileAlloc(t.io, "tone.json", a, .limited(voice.max_file_bytes)));
    try t.expectEqualStrings("A small cloud passes.", written[0].transcript);
    try t.expectEqualSlices([10]i32, (try voice.parse(a, tone))[0].codes, written[0].codes);
    const wav_body = try std.fmt.allocPrint(a, "{{\"name\":\"clone\",\"text\":\"A small cloud passes.\",\"source\":{{\"kind\":\"self\",\"note\":\"My own voice.\"}},\"audio\":\"{s}\"}}", .{try wavBase64(a, "\x05\x06")});
    try expectReply(try h.ask(a, "POST", "/v1/voices", admin, wav_body), "HTTP/1.1 201", "\"id\": \"clone\"");
    const clone = try voice.parse(a, try tmp.dir.readFileAlloc(t.io, "clone.json", a, .limited(voice.max_file_bytes)));
    try t.expectEqualSlices([10]i32, &.{ @splat(5), @splat(6) }, clone[0].codes);
    try expectReply(try h.ask(a, "GET", "/v1/voices", "read-secret", ""), "HTTP/1.1 200", "\"voices\": [\"default\", \"clone\", \"tone\"]");
    // a request reading "tone" keeps its frames while an admin deletes it
    fake.hold.store(true, .release);
    const Speaker = struct {
        h: *Harness,
        reply: []const u8 = "",
        fn run(s: *@This(), out: Allocator) void {
            s.reply = s.h.ask(out, "POST", "/v1/audio/speech", "infer-secret", "{\"model\":\"m\",\"input\":\"hello\",\"voice\":\"tone\"}") catch "failed";
        }
    };
    var speaker: Speaker = .{ .h = &h };
    var speaker_arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer speaker_arena.deinit();
    const thread = try std.Thread.spawn(.{}, Speaker.run, .{ &speaker, speaker_arena.allocator() });
    var joined = false;
    defer if (!joined) {
        fake.hold.store(false, .release);
        thread.join();
    };
    while (!fake.held.load(.acquire)) std.Thread.yield() catch {};
    try expectReply(try h.ask(a, "DELETE", "/v1/voices/tone", admin, ""), "HTTP/1.1 200", "\"deleted\": true");
    try t.expectError(error.FileNotFound, tmp.dir.access(t.io, "tone.json", .{}));
    fake.hold.store(false, .release);
    thread.join();
    joined = true;
    try expectReply(speaker.reply, "HTTP/1.1 200", "RIFF");
    const want = try voice.promptFor(a, h.text.text(), "hello", try voice.parse(a, tone));
    try t.expectEqualSlices([11]i32, want.frames, fake.frames.items);
    try expectReply(try h.ask(a, "POST", "/v1/audio/speech", "infer-secret", "{\"model\":\"m\",\"input\":\"hello\",\"voice\":\"tone\"}"), "HTTP/1.1 400", "unknown voice 'tone'; see /v1/voices");
    try expectReply(try h.ask(a, "DELETE", "/v1/voices/tone", admin, ""), "HTTP/1.1 404", "no such voice");
}

test "voices HTTP: --request-voices takes an attested inline voice as the catalog's own" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(t.io, .{ .sub_path = "tone.json", .data = tone });
    var fake: Fake = .{};
    defer fake.frames.deinit(t.allocator);
    var h: Harness = .{};
    try h.start(&fake, .{ .served_name = "m", .model_ids = &.{"m"}, .keep_warm_s = 0, .voice = .{ .dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path}), .inline_voices = true } }, null);
    defer h.deinit();
    try expectReply(try h.ask(a, "POST", "/v1/audio/speech", "", "{\"model\":\"m\",\"input\":\"A cloud.\",\"voice\":\"tone\"}"), "HTTP/1.1 200", "RIFF");
    const from_catalog = try a.dupe([11]i32, fake.frames.items);
    try expectReply(try h.ask(a, "POST", "/v1/audio/speech", "", "{\"model\":\"m\",\"input\":\"A cloud.\",\"voice\":" ++ tone ++ "}"), "HTTP/1.1 200", "RIFF");
    try t.expectEqualSlices([11]i32, from_catalog, fake.frames.items);
    try expectReply(try h.ask(a, "POST", "/v1/audio/speech", "", "{\"model\":\"m\",\"input\":\"hi\",\"voice\":{\"format\":\"drama3-http-voice/1\",\"references\":[{\"text\":\"A.\",\"codes\":[[1,2,3,4,5,6,7,8,9,10]]}]}}"), "HTTP/1.1 400", "source.kind");
    try expectReply(try h.ask(a, "POST", "/v1/audio/speech", "", "{\"model\":\"m\",\"input\":\"hi\",\"speaker\":\"tone\",\"voice\":" ++ tone ++ "}"), "HTTP/1.1 400", "takes no speaker");
    try expectReply(try h.ask(a, "POST", "/v1/audio/speech", "", "{\"model\":\"m\",\"input\":\"hi\",\"ref_audio\":\"x.wav\"}"), "HTTP/1.1 400", "ref_audio is not supported by this model");
    try expectReply(try h.ask(a, "POST", "/v1/voices", "", "{\"name\":\"x\"," ++ tone[1..]), "HTTP/1.1 404", "unknown path");
}
