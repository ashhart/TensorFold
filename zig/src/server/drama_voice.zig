//! Startup-owned voice references; only catalog names, never request paths, select files.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const prompt = @import("drama_prompt");
const text_mod = @import("model_text.zig");
const checks = @import("drama_prompt.zig");
const errors = @import("errors.zig");
const Allocator = std.mem.Allocator;
const Cx = errors.Cx;

pub const Voice = struct { name: []const u8, references: []const prompt.Reference };
pub const max_frames = (30 * 44100 + 2047) / 2048;
pub const max_file_bytes = 256 * 1024;

pub fn validName(name: []const u8) bool {
    if (name.len == 0 or name.len > 64 or !std.ascii.isAlphanumeric(name[0]) or std.ascii.eqlIgnoreCase(name, "default")) return false;
    for (name) |c| if (!std.ascii.isAlphanumeric(c) and c != '-' and c != '_') return false;
    return true;
}

fn text(value: ?[]const u8, limit: usize) ![]const u8 {
    const s = value orelse return error.BadVoice;
    if (s.len > limit or std.mem.trim(u8, s, " \t\r\n").len == 0 or !std.unicode.utf8ValidateSlice(s)) return error.BadVoice;
    return s;
}

pub fn parse(a: Allocator, bytes: []const u8) ![]const prompt.Reference {
    if (bytes.len > max_file_bytes) return error.BadVoice;
    const doc = switch (try json.parse(a, bytes)) {
        .ok => |v| v,
        else => return error.BadVoice,
    };
    if (!std.mem.eql(u8, doc.strField("format") orelse "", "drama3-http-voice/1")) return error.BadVoice;
    const source = doc.get("source") orelse return error.MissingAttestation;
    const kind = source.strField("kind") orelse return error.MissingAttestation;
    if (!std.mem.eql(u8, kind, "synthetic") and !std.mem.eql(u8, kind, "self") and !std.mem.eql(u8, kind, "consented")) return error.MissingAttestation;
    _ = text(source.strField("note"), 4096) catch return error.MissingAttestation;
    const values = (doc.get("references") orelse return error.BadVoice).items() orelse return error.BadVoice;
    if (values.len == 0 or values.len > 4) return error.BadVoice;
    const refs = try a.alloc(prompt.Reference, values.len);
    for (values, refs) |value, *ref| {
        const transcript = try text(value.strField("text"), 4096);
        if (try checks.problem(a, transcript) != null) return error.BadVoice;
        const rows = (value.get("codes") orelse return error.BadVoice).items() orelse return error.BadVoice;
        if (rows.len == 0 or rows.len > max_frames) return error.BadVoice;
        const codes = try a.alloc([10]i32, rows.len);
        for (rows, codes) |row, *code| {
            const cells = row.items() orelse return error.BadVoice;
            if (cells.len != 10) return error.BadVoice;
            for (cells, 0..) |cell, i| {
                const n = cell.int64() orelse return error.BadVoice;
                if (n < 0 or n >= (if (i == 0) @as(i64, 4096) else 1024)) return error.BadVoice;
                code[i] = @intCast(n);
            }
        }
        ref.* = .{ .transcript = transcript, .codes = codes };
    }
    return refs;
}

pub fn load(a: Allocator, io: std.Io, path: []const u8) ![]const Voice {
    var dir = try std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true });
    defer dir.close(io);
    var iter = dir.iterate();
    var voices: std.ArrayList(Voice) = .empty;
    while (try iter.next(io)) |entry| {
        if (!std.mem.endsWith(u8, entry.name, ".json")) continue;
        const name = entry.name[0 .. entry.name.len - 5];
        if (entry.kind != .file or !validName(name)) return error.BadVoiceName;
        if (voices.items.len >= 256) return error.TooManyVoices;
        const bytes = try dir.readFileAlloc(io, entry.name, a, .limited(max_file_bytes + 1));
        try voices.append(a, .{ .name = try a.dupe(u8, name), .references = try parse(a, bytes) });
    }
    std.mem.sort(Voice, voices.items, {}, struct {
        fn less(_: void, left: Voice, right: Voice) bool {
            return std.mem.lessThan(u8, left.name, right.name);
        }
    }.less);
    return voices.items;
}

pub fn select(voices: []const Voice, default: []const u8, requested: []const u8) ![]const prompt.Reference {
    const name = if (requested.len == 0 or std.ascii.eqlIgnoreCase(requested, "default")) default else requested;
    if (std.mem.eql(u8, name, "default")) return &.{};
    for (voices) |voice| if (std.mem.eql(u8, voice.name, name)) return voice.references;
    return error.UnknownVoice;
}

const Tokenizer = struct {
    value: text_mod.Text,
    pub fn specialTokenId(t: Tokenizer, s: []const u8) ?u32 {
        return t.value.tokenId(s);
    }
    pub fn encode(t: Tokenizer, a: Allocator, s: []const u8) text_mod.Error![]u32 {
        return t.value.encode(a, s, false);
    }
};

pub fn build(a: Allocator, tokenizer: text_mod.Text, input: []const u8, refs: []const prompt.Reference) !prompt.Built {
    return prompt.build(a, Tokenizer{ .value = tokenizer }, input, refs);
}

/// A voice's prompt: its own ids and frame rows.
pub const Prompt = struct { prompt: []const u32, frames: []const [11]i32, reference_rows: u32 = 0 };

/// A voice's prompt ids and frame rows for an input.
pub fn promptFor(a: Allocator, tokenizer: text_mod.Text, input: []const u8, refs: []const prompt.Reference) !Prompt {
    const built = try build(a, tokenizer, input, refs);
    const frames = built.frames;
    const ids = try a.alloc(u32, frames.len);
    for (frames, ids) |row, *token| token.* = @intCast(row[0]);
    return .{ .prompt = ids, .frames = frames, .reference_rows = built.reference_rows };
}

/// A voice's prompt for a speech request: its ids and frame rows; null when the request named no voice.
pub fn requestPrompt(cx: *Cx, tokenizer: text_mod.Text, sp: api.SpeechInfo, input: []const u8, refs: []const prompt.Reference) errors.Refused!?Prompt {
    if (refs.len == 0) return null;
    if (try checks.problem(cx.a, input)) |message| return cx.refuse(message);
    const made = promptFor(cx.a, tokenizer, input, refs) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => cx.other("the tokenizer cannot encode this voice prompt"),
    };
    if (made.frames.len > sp.max_prompt_tokens)
        return cx.fail(.request, checks.error_prefix ++ " a prompt of {d} rows exceeds the maximum allowed length of {d} rows", .{ made.frames.len, sp.max_prompt_tokens });
    return made;
}

pub fn listing(a: Allocator, voices: []const Voice) !json.Value {
    const data = try a.alloc(json.Value, voices.len + 1);
    data[0] = .{ .string = "default" };
    for (voices, data[1..]) |voice, *slot| slot.* = .{ .string = voice.name };
    const object = try json.newObject(a);
    try object.put(a, "voices", .{ .array = data });
    return .{ .object = object };
}

test "voices: synthetic parsing, attestation, bounds and selection" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const fixture =
        \\{"format":"drama3-http-voice/1","source":{"kind":"synthetic","note":"Generated integer codes."},
        \\"references":[{"text":"A small cloud passes.","codes":[[4095,1,2,3,4,5,6,7,8,1023]]}]}
    ;
    const refs = try parse(a, fixture);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "tone.json", .data = fixture });
    const loaded = try load(a, std.testing.io, path);
    try std.testing.expectEqualStrings("tone", loaded[0].name);
    try std.testing.expectEqualSlices([10]i32, refs[0].codes, loaded[0].references[0].codes);
    const padded = try a.alloc(u8, max_file_bytes + 1);
    @memcpy(padded[0..fixture.len], fixture);
    @memset(padded[fixture.len..], ' ');
    for ([_]usize{ max_file_bytes - 1, max_file_bytes }) |size| {
        _ = try parse(a, padded[0..size]);
        try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "tone.json", .data = padded[0..size] });
        const boundary = try load(a, std.testing.io, path);
        try std.testing.expectEqualSlices([10]i32, refs[0].codes, boundary[0].references[0].codes);
    }
    try std.testing.expectError(error.BadVoice, parse(a, padded));
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "tone.json", .data = padded });
    try std.testing.expectError(error.StreamTooLong, load(a, std.testing.io, path));
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "tone.json", .data = "{\"format\":\"drama3-http-voice/1\"}" });
    try std.testing.expectError(error.MissingAttestation, load(a, std.testing.io, path));
    try std.testing.expectEqual(@as(i32, 4095), refs[0].codes[0][0]);
    const voices = [_]Voice{.{ .name = "tone", .references = refs }};
    try std.testing.expectEqual(refs.ptr, (try select(&voices, "tone", "default")).ptr);
    try std.testing.expectEqual(refs.ptr, (try select(&voices, "default", "tone")).ptr);
    try std.testing.expectEqual(@as(usize, 0), (try select(&.{}, "default", "")).len);
    try std.testing.expectError(error.UnknownVoice, select(&voices, "default", "../tone"));
    try std.testing.expectEqualStrings("{\"voices\": [\"default\", \"tone\"]}", try json.stringify(a, try listing(a, &voices), .{}));
    const doc = (try json.parse(a, fixture)).ok;
    _ = doc.object.swapRemove("source");
    try std.testing.expectError(error.MissingAttestation, parse(a, try json.stringify(a, doc, .{})));
    for ([_][]const u8{ "", " ", "null", "{\"kind\":\"synthetic\"}", "{\"kind\":\"other\",\"note\":\"n\"}" }) |bad| {
        const bytes = try std.fmt.allocPrint(a, "{{\"format\":\"drama3-http-voice/1\",\"source\":{s}}}", .{bad});
        try std.testing.expectError(if (bad.len == 0 or std.mem.eql(u8, bad, " ")) error.BadVoice else error.MissingAttestation, parse(a, bytes));
    }
    for ([_][]const u8{ "[]", "[[0]]", "[[-1,0,0,0,0,0,0,0,0,0]]", "[[4096,0,0,0,0,0,0,0,0,0]]", "[[0,1024,0,0,0,0,0,0,0,0]]", "[[0.0,0,0,0,0,0,0,0,0,0]]" }) |codes| {
        const bytes = try std.fmt.allocPrint(a, "{{\"format\":\"drama3-http-voice/1\",\"source\":{{\"kind\":\"synthetic\",\"note\":\"Generated.\"}},\"references\":[{{\"text\":\"A cloud.\",\"codes\":{s}}}]}}", .{codes});
        try std.testing.expectError(error.BadVoice, parse(a, bytes));
    }
    const oversized = try a.alloc(u8, max_file_bytes + 1);
    try std.testing.expectError(error.BadVoice, parse(a, oversized));
    for ([_][]const u8{ "default", "DEFAULT", "../tone", "a/b", "" }) |name| try std.testing.expect(!validName(name));
    const bounded = (try json.parse(a, fixture)).ok;
    const reference = bounded.get("references").?.array[0];
    const code = reference.get("codes").?.array[0];
    const many = try a.alloc(json.Value, max_frames + 1);
    @memset(many, code);
    try reference.object.put(a, "codes", .{ .array = many });
    try std.testing.expectError(error.BadVoice, parse(a, try json.stringify(a, bounded, .{})));
    try reference.object.put(a, "codes", .{ .array = many[0..max_frames] });
    _ = try parse(a, try json.stringify(a, bounded, .{}));
    try reference.object.put(a, "codes", .{ .array = many[0..1] });
    const long = try a.alloc(u8, 4097);
    @memset(long, 'a');
    try reference.object.put(a, "text", .{ .string = long });
    try std.testing.expectError(error.BadVoice, parse(a, try json.stringify(a, bounded, .{})));
    try reference.object.put(a, "text", .{ .string = long[0..4096] });
    _ = try parse(a, try json.stringify(a, bounded, .{}));
    try reference.object.put(a, "text", .{ .string = "A cloud." });
    const repeated = try a.alloc(json.Value, 5);
    @memset(repeated, reference);
    try bounded.object.put(a, "references", .{ .array = repeated });
    try std.testing.expectError(error.BadVoice, parse(a, try json.stringify(a, bounded, .{})));
    try bounded.object.put(a, "references", .{ .array = repeated[0..4] });
    try std.testing.expectEqual(@as(usize, 4), (try parse(a, try json.stringify(a, bounded, .{}))).len);
}

test "voices: real tokenizer rows equal the CLI path without server postprocessing" {
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var tok = try @import("tokenizer").parse(gpa,
        \\{"model":{"type":"WordLevel","vocab":{"[UNK]":0,"<|semantic:0|>":1000},"unk_token":"[UNK]"}}
    );
    defer tok.deinit();
    var hf: @import("hf_text.zig").HfText = .{
        .gpa = gpa,
        .arena = std.heap.ArenaAllocator.init(gpa),
        .tok = tok,
        .prefix = &.{17},
        .suffix = &.{18},
    };
    defer hf.arena.deinit();
    const codes = [_][10]i32{.{ 7, 1, 2, 3, 4, 5, 6, 7, 8, 9 }};
    const refs = [_]prompt.Reference{.{ .transcript = "A small cloud passes.", .codes = &codes }};
    const cli_rows = try prompt.build(a, &tok, "A cloud.", &refs);
    const http_rows = try build(a, hf.text(), "A cloud.", &refs);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(cli_rows.frames), std.mem.sliceAsBytes(http_rows.frames));
    try std.testing.expectEqual(cli_rows.reference_rows, http_rows.reference_rows);
}
