//! POST /v1/voices and DELETE /v1/voices/NAME under --manage-voices: attested catalog voices added and deleted live.
const std = @import("std");
const json = @import("json");
const prompt = @import("drama_prompt");
const errors = @import("errors.zig");
const http_body = @import("http_body.zig");
const routes = @import("routes.zig");
const voice = @import("drama_voice.zig");
const tool = @import("drama_voice_tool.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Value = json.Value;
const Allocator = std.mem.Allocator;

const unattested = "a voice needs source.kind synthetic, self or consented and a source.note";
const bad_voice = "a voice needs 1 to 4 references, each a transcript a speech request accepts and its codes";

/// One frame stands in for the codes, so a request the catalog would refuse never reaches the encoder.
const placeholder = [1][10]i32{@splat(0)};

fn refuse(conn: *Conn, a: Allocator, code: u16, message: []const u8) void {
    const o = json.newObject(a) catch return;
    o.put(a, "message", .{ .string = message }) catch return;
    o.put(a, "type", .{ .string = if (code >= 500) "server_error" else "invalid_request_error" }) catch return;
    const wrapped = json.newObject(a) catch return;
    wrapped.put(a, "error", .{ .object = o }) catch return;
    routes.sendValue(conn, a, code, .{ .object = wrapped });
}

const Made = struct { file: []const u8, refs: []const prompt.Reference, description: []const u8 };

/// The attested voice file a request describes: a voice file's own fields, or a WAV with its transcript.
fn made(srv: *Server, a: Allocator, body: Value) !Made {
    const description = voice.describeDoc(body) catch return error.BadDescription;
    const source = body.get("source") orelse .null;
    const kind = source.strField("kind") orelse "";
    const note = source.strField("note") orelse "";
    const audio = body.get("audio") orelse .null;
    if (audio == .null) {
        const refs = try voice.parseDoc(a, body);
        return .{ .file = try tool.document(a, kind, note, description, refs), .refs = refs, .description = description };
    }
    if (body.get("references") != null) return error.AudioAndReferences;
    if (audio != .string) return error.BadAudio;
    const text = body.strField("text") orelse return error.BadVoice;
    _ = try tool.document(a, kind, note, description, &.{.{ .transcript = text, .codes = &placeholder }});
    const decoder = std.base64.standard.Decoder;
    const wav = try a.alloc(u8, decoder.calcSizeForSlice(audio.string) catch return error.BadAudio);
    decoder.decode(wav, audio.string) catch return error.BadAudio;
    const encode = srv.engine.vtable.encode_voice orelse return error.NoEncoder;
    const codes = try encode(srv.engine.ctx, a, wav);
    if (codes.len == 0 or codes.len > voice.max_frames) return error.BadWav;
    const refs = try a.dupe(prompt.Reference, &.{.{ .transcript = text, .codes = codes }});
    return .{ .file = try tool.document(a, kind, note, description, refs), .refs = refs, .description = description };
}

pub fn post(srv: *Server, conn: *Conn, a: Allocator) void {
    var cx: errors.Cx = .{ .a = a };
    const body = http_body.readJson(conn, &cx) catch |e| return if (e == error.OutOfMemory) refuse(conn, a, 500, "out of memory") else refuse(conn, a, 400, cx.message);
    const name = body.strField("name") orelse "";
    if (body != .object or !voice.validName(name)) return refuse(conn, a, 400, "name must be 1 to 64 letters, digits, '-' or '_', starting with a letter or digit, and not default");
    const voice_file = made(srv, a, body) catch |e| return switch (e) {
        error.MissingAttestation => refuse(conn, a, 400, unattested),
        error.BadVoice => refuse(conn, a, 400, bad_voice),
        error.BadDescription => refuse(conn, a, 400, "description must be UTF-8 text of at most 200 bytes"),
        error.AudioAndReferences => refuse(conn, a, 400, "send audio with its text, or a voice file's references, not both"),
        error.BadAudio => refuse(conn, a, 400, "audio must be a WAV file in standard base64"),
        error.BadWav => refuse(conn, a, 400, "audio must be a RIFF WAVE file (PCM 8, 16, 24 or 32-bit, or float32) of up to 30 seconds"),
        error.NoEncoder => refuse(conn, a, 400, "this engine does not encode reference audio: send a voice file's references"),
        error.Closed => refuse(conn, a, 503, "the speech engine is closed"),
        error.OutOfMemory => refuse(conn, a, 500, "out of memory"),
        else => refuse(conn, a, 500, "the reference audio could not be encoded"),
    };
    const added = srv.voices.put(name, voice_file.file, voice_file.refs, voice_file.description) catch |e| return switch (e) {
        error.TooManyVoices => refuse(conn, a, 409, "the catalog holds its most voices: delete one first"),
        error.OutOfMemory => refuse(conn, a, 500, "out of memory"),
        else => refuse(conn, a, 500, "the voice file could not be written to --voices"),
    };
    const reply = json.newObject(a) catch return;
    reply.put(a, "id", .{ .string = name }) catch return;
    reply.put(a, "description", .{ .string = voice_file.description }) catch return;
    reply.put(a, "added", .{ .bool = added }) catch return;
    routes.sendValue(conn, a, if (added) 201 else 200, .{ .object = reply });
}

pub fn delete(srv: *Server, conn: *Conn, a: Allocator, name: []const u8) void {
    srv.voices.remove(name, srv.config.voice.default) catch |e| return switch (e) {
        error.UnknownVoice => refuse(conn, a, 404, "no such voice in the catalog"),
        error.DefaultVoice => refuse(conn, a, 409, "the configured --default-voice cannot be deleted"),
        error.OutOfMemory => refuse(conn, a, 500, "out of memory"),
        else => refuse(conn, a, 500, "the voice file could not be removed from --voices"),
    };
    const reply = json.newObject(a) catch return;
    reply.put(a, "id", .{ .string = name }) catch return;
    reply.put(a, "deleted", .{ .bool = true }) catch return;
    routes.sendValue(conn, a, 200, .{ .object = reply });
}
