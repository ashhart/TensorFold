//! drama3-voice: an attested catalog file from reference WAVs and transcripts, coded by `drama3-generate --encode`.
const std = @import("std");
const builtin = @import("builtin");
const json = @import("json");
const prompt = @import("drama_prompt");
const voice = @import("drama_voice.zig");
const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;

const usage =
    \\usage: drama3-voice --codec CODEC --ref-audio REF.wav --ref-text "..." [--ref-audio REF.wav --ref-text "..."]...
    \\                    --source-kind synthetic|self|consented --source-note "..." --out DIR/NAME.json
    \\                    [--generate drama3-generate]
    \\
    \\The model licence forbids imitating an identifiable person's voice without their prior consent or another legal right.
    \\
;
const unattested = "drama3-voice: refused, a voice file needs --source-kind synthetic, self or consented and a --source-note\n";

/// One frame stands in for the codes, so a file the catalog would refuse never reaches the encoder.
const placeholder = [1][10]i32{@splat(0)};

const Flag = enum { @"--codec", @"--ref-audio", @"--ref-text", @"--source-kind", @"--source-note", @"--out", @"--generate" };

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(a);
    const args = try a.alloc([]const u8, argv.len - 1);
    for (args, argv[1..]) |*x, y| x.* = y;
    var buf: [4096]u8 = undefined;
    var err = std.Io.File.stderr().writerStreaming(init.io, &buf);
    defer err.interface.flush() catch {};
    return run(a, init.io, &err.interface, args);
}

pub fn run(a: Allocator, io: std.Io, err: *Writer, argv: []const []const u8) !u8 {
    var codec: ?[]const u8 = null;
    var audio: std.ArrayList([]const u8) = .empty;
    var texts: std.ArrayList([]const u8) = .empty;
    var kind: ?[]const u8 = null;
    var note: ?[]const u8 = null;
    var out: ?[]const u8 = null;
    var generate: []const u8 = "drama3-generate";
    if (argv.len % 2 != 0) return refuse(err, usage);
    var i: usize = 0;
    while (i < argv.len) : (i += 2) {
        const v = argv[i + 1];
        switch (std.meta.stringToEnum(Flag, argv[i]) orelse return refuse(err, usage)) {
            .@"--codec" => codec = v,
            .@"--ref-audio" => try audio.append(a, v),
            .@"--ref-text" => try texts.append(a, v),
            .@"--source-kind" => kind = v,
            .@"--source-note" => note = v,
            .@"--out" => out = v,
            .@"--generate" => generate = v,
        }
    }
    if (codec == null or out == null or audio.items.len != texts.items.len) return refuse(err, usage);
    const base = std.fs.path.basename(out.?);
    if (!std.mem.endsWith(u8, base, ".json") or !voice.validName(base[0 .. base.len - 5]))
        return refuse(err, "drama3-voice: refused, --out must be NAME.json with a catalog name other than default\n");
    if (kind == null or note == null) return refuse(err, unattested);
    const refs = try a.alloc(prompt.Reference, texts.items.len);
    for (refs, texts.items) |*ref, text| ref.* = .{ .transcript = text, .codes = &placeholder };
    _ = document(a, kind.?, note.?, refs) catch |e| return refuse(err, switch (e) {
        error.MissingAttestation => unattested,
        error.BadVoice => "drama3-voice: refused, give 1 to 4 references, each a transcript a speech request accepts\n",
        else => return e,
    });
    const cwd = std.Io.Dir.cwd();
    const parent = try cwd.openDir(io, std.fs.path.dirname(out.?) orelse ".", .{});
    defer parent.close(io);
    // Trust the encoder and same-UID processes; isolate stale paths, other users and concurrent runs.
    var work_buf: [256]u8 = undefined;
    const work = while (true) {
        var nonce: [8]u8 = undefined;
        io.random(&nonce);
        const name = try std.fmt.bufPrint(&work_buf, ".{s}.{x}", .{ base, std.mem.readInt(u64, &nonce, .native) });
        parent.createDir(io, name, .fromMode(0o700)) catch |e| switch (e) {
            error.PathAlreadyExists => continue,
            else => return e,
        };
        break name;
    };
    var unopened = true;
    errdefer if (unopened) {
        parent.deleteDir(io, work) catch |cleanup_error| {
            err.print("drama3-voice: cleanup failed for {s}/{s}: {t}\n", .{
                std.fs.path.dirname(out.?) orelse ".", work, cleanup_error,
            }) catch {};
        };
    };
    const work_path = try std.fs.path.join(a, &.{ std.fs.path.dirname(out.?) orelse ".", work });
    const private = try parent.openDir(io, work, .{ .iterate = true, .follow_symlinks = false });
    unopened = false;
    defer private.close(io);
    const result = encode(a, io, err, generate, codec.?, audio.items, refs, kind.?, note.?, parent, private, base, work_path);
    cleanup(io, parent, private, work) catch |e| {
        const published = if (result) |status| status == 0 else |_| false;
        try err.print("drama3-voice: {s} for {s}: {t}\n", .{
            if (published) "WARNING: cleanup failed" else "cleanup failed", work_path, e,
        });
    };
    return result;
}

fn cleanup(io: std.Io, parent: std.Io.Dir, private: std.Io.Dir, work: []const u8) !void {
    try private.setPermissions(io, .fromMode(0o700));
    var it = private.iterate();
    while (try it.next(io)) |entry| try private.deleteTree(io, entry.name);
    try parent.deleteDir(io, work);
}

fn encode(
    a: Allocator,
    io: std.Io,
    err: *Writer,
    generate: []const u8,
    codec: []const u8,
    audio: []const []const u8,
    refs: []prompt.Reference,
    kind: []const u8,
    note: []const u8,
    parent: std.Io.Dir,
    private: std.Io.Dir,
    base: []const u8,
    work_path: []const u8,
) !u8 {
    for (refs, audio, 0..) |*ref, wav, r| {
        const name = try std.fmt.allocPrint(a, "{d}.codes", .{r});
        const codes_path = try std.fs.path.join(a, &.{ work_path, name });
        var child = std.process.spawn(io, .{ .argv = &.{ generate, "--codec", codec, "--ref-audio", wav, "--encode", codes_path } }) catch |e| {
            try err.print("drama3-voice: cannot run {s}: {t}\n", .{ generate, e });
            return 1;
        };
        const term = try child.wait(io);
        if (!term.success()) {
            try err.print("drama3-voice: {s} {f} encoding {s}\n", .{ generate, term, wav });
            return 1;
        }
        const file = private.openFile(io, name, .{ .follow_symlinks = false }) catch |e| switch (e) {
            error.FileNotFound => {
                try err.print("drama3-voice: {s} wrote no codes for {s}\n", .{ generate, wav });
                return 1;
            },
            else => return e,
        };
        defer file.close(io);
        var reader = file.reader(io, &.{});
        const bytes = try reader.interface.allocRemainingAlignedSentinel(
            a,
            .limited(@sizeOf([10]i32) * voice.max_frames + 1),
            .of([10]i32),
            null,
        );
        if (bytes.len % @sizeOf([10]i32) != 0) return error.BadCodes;
        ref.codes = std.mem.bytesAsSlice([10]i32, bytes);
    }
    try private.writeFile(io, .{ .sub_path = base, .data = try document(a, kind, note, refs), .flags = .{ .exclusive = true } });
    try private.rename(base, parent, base, io);
    return 0;
}

fn refuse(err: *Writer, message: []const u8) !u8 {
    try err.writeAll(message);
    return 2;
}

/// The voice file's JSON, accepted by the catalog's own parser.
fn document(a: Allocator, kind: []const u8, note: []const u8, refs: []const prompt.Reference) ![]const u8 {
    var out: Writer.Allocating = .init(a);
    const w = &out.writer;
    try w.writeAll("{\"format\": \"drama3-http-voice/1\", \"source\": {\"kind\": ");
    try json.writeString(w, kind, .{ .ascii = false });
    try w.writeAll(", \"note\": ");
    try json.writeString(w, note, .{ .ascii = false });
    try w.writeAll("},\n\"references\": [");
    for (refs, 0..) |ref, r| {
        try w.writeAll(if (r == 0) "\n{\"text\": " else ",\n{\"text\": ");
        try json.writeString(w, ref.transcript, .{ .ascii = false });
        try w.writeAll(", \"codes\": [");
        for (ref.codes, 0..) |frame, f| {
            try w.writeAll(if (f == 0) "[" else ", [");
            for (frame, 0..) |code, c| try w.print("{s}{d}", .{ if (c == 0) "" else ", ", code });
            try w.writeByte(']');
        }
        try w.writeAll("]}");
    }
    try w.writeAll("\n]}\n");
    _ = try voice.parse(a, out.written());
    return out.written();
}

fn le(comptime v: u32) [4]u8 {
    return std.mem.toBytes(std.mem.nativeToLittle(u32, v));
}

/// A 0.1 s 440 Hz tone as 16-bit mono 44.1 kHz PCM.
fn tone(a: Allocator) ![]u8 {
    const n = 4410;
    const wav = try a.alloc(u8, 44 + 2 * n);
    wav[0..44].* = "RIFF".* ++ le(36 + 2 * n) ++ "WAVEfmt ".* ++ le(16) ++ le(1 | 1 << 16) ++ le(44100) ++ le(2 * 44100) ++ le(2 | 16 << 16) ++ "data".* ++ le(2 * n);
    for (0..n) |i| {
        const s = 8192 * @sin(2 * std.math.pi * 440 * @as(f32, @floatFromInt(i)) / 44100);
        std.mem.writeInt(i16, wav[44 + 2 * i ..][0..2], @intFromFloat(@round(s)), .little);
    }
    return wav;
}

const Case = struct {
    a: Allocator,
    tmp: std.testing.TmpDir,
    err: Writer.Allocating,

    fn path(c: *Case, name: []const u8) ![]const u8 {
        return std.fmt.allocPrint(c.a, ".zig-cache/tmp/{s}/{s}", .{ c.tmp.sub_path, name });
    }

    fn script(c: *Case, name: []const u8, text: []const u8) ![]const u8 {
        try c.tmp.dir.writeFile(std.testing.io, .{ .sub_path = name, .data = text, .flags = .{ .permissions = .fromMode(0o755) } });
        return c.path(name);
    }

    fn call(c: *Case, parts: []const []const []const u8) !u8 {
        c.err.clearRetainingCapacity();
        return run(c.a, std.testing.io, &c.err.writer, try std.mem.concat(c.a, []const u8, parts));
    }

    fn read(c: *Case, name: []const u8) ![]const u8 {
        return c.tmp.dir.readFileAlloc(std.testing.io, name, c.a, .unlimited);
    }

    /// Private work directories left behind by runs writing tone.json.
    fn leftovers(c: *Case) !usize {
        var dir = try c.tmp.dir.openDir(std.testing.io, ".", .{ .iterate = true });
        defer dir.close(std.testing.io);
        var it = dir.iterate();
        var n: usize = 0;
        while (try it.next(std.testing.io)) |entry| n += @intFromBool(std.mem.startsWith(u8, entry.name, ".tone.json."));
        return n;
    }
};

test "voice tool: a synthetic tone's encoder codes become an attested file the catalog loads" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var c: Case = .{ .a = arena.allocator(), .tmp = std.testing.tmpDir(.{}), .err = .init(arena.allocator()) };
    defer c.tmp.cleanup();
    var codes: [3][10]i32 = undefined;
    for (&codes, 0..) |*frame, f| for (frame, 0..) |*code, i| {
        code.* = @intCast((f * 10 + i) * 389 % @as(usize, if (i == 0) 4096 else 1024));
    };
    try c.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "grid.codes", .data = std.mem.sliceAsBytes(&codes) });
    try c.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "tone.wav", .data = try tone(c.a) });
    const wav = try c.path("tone.wav");
    const encoder = try c.script("encoder", try std.fmt.allocPrint(c.a,
        \\#!/bin/sh
        \\[ "$1 $2 $3 $5" = '--codec synthetic.pth --ref-audio --encode' ] && [ ! -e "$6" ] && cmp -s "$4" {s} && exec cp {s} "$6"
        \\exit 3
        \\
    , .{ wav, try c.path("grid.codes") }));
    const out = try c.path("tone.json");
    try c.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "unrelated.txt", .data = "unrelated" });
    try c.tmp.dir.symLink(std.testing.io, "unrelated.txt", "tone.json", .{});
    const texts = [_][]const u8{ "<|speaker:0|> A small cloud passes.", "\u{3000}<|speaker:15|> Rain taps on a tin roof." };
    try std.testing.expectEqual(@as(u8, 0), try c.call(&.{&.{
        "--codec",       "synthetic.pth", "--generate",    encoder,
        "--ref-audio",   wav,             "--ref-text",    texts[0],
        "--ref-audio",   wav,             "--ref-text",    texts[1],
        "--source-kind", "synthetic",     "--source-note", "A generated 440 Hz tone.",
        "--out",         out,
    }}));
    try std.testing.expectEqualStrings("", c.err.written());
    try std.testing.expectEqual(@as(usize, 0), try c.leftovers());
    try std.testing.expectEqualStrings("unrelated", try c.read("unrelated.txt"));
    try std.testing.expectEqual(.file, (try c.tmp.dir.statFile(std.testing.io, "tone.json", .{ .follow_symlinks = false })).kind);
    const loaded = try voice.load(c.a, std.testing.io, try c.path(""));
    try std.testing.expectEqualStrings("tone", loaded[0].name);
    for (loaded[0].references, texts) |ref, text| {
        try std.testing.expectEqualStrings(text, ref.transcript);
        try std.testing.expectEqualSlices([10]i32, &codes, ref.codes);
    }
    const doc = (try json.parse(c.a, try c.read("tone.json"))).ok;
    try std.testing.expectEqualStrings("synthetic", doc.get("source").?.strField("kind").?);
    try std.testing.expectEqualStrings("A generated 440 Hz tone.", doc.get("source").?.strField("note").?);
}

test "voice tool: refusals happen before encoding and encoder failures leave existing files alone" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var c: Case = .{ .a = arena.allocator(), .tmp = std.testing.tmpDir(.{}), .err = .init(arena.allocator()) };
    defer c.tmp.cleanup();
    try c.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "tone.json", .data = "previous voice" });
    const zero_frame: [40]u8 = @splat(0);
    try c.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "tone.json.codes", .data = &zero_frame });
    const codec: []const []const u8 = &.{ "--codec", "synthetic.pth" };
    const never: []const []const u8 = &.{ "--generate", try c.path("never-run") };
    const pair: []const []const u8 = &.{ "--ref-audio", "tone.wav", "--ref-text", "A small cloud passes." };
    const attested: []const []const u8 = &.{ "--source-kind", "synthetic", "--source-note", "Generated." };
    const to: []const []const u8 = &.{ "--out", try c.path("tone.json") };
    const cases = [_]struct { parts: []const []const []const u8, message: []const u8 }{
        .{ .parts = &.{ codec, never, pair, to }, .message = unattested },
        .{ .parts = &.{ codec, never, pair, &.{ "--source-kind", "synthetic" }, to }, .message = unattested },
        .{ .parts = &.{ codec, never, pair, &.{ "--source-kind", "other", "--source-note", "Generated." }, to }, .message = unattested },
        .{ .parts = &.{ codec, never, pair, &.{ "--source-kind", "self", "--source-note", " \n" }, to }, .message = unattested },
        .{ .parts = &.{ codec, never, attested, to }, .message = "give 1 to 4 references" },
        .{ .parts = &.{ codec, never, pair, pair, pair, pair, pair, attested, to }, .message = "give 1 to 4 references" },
        .{ .parts = &.{ codec, never, &.{ "--ref-audio", "tone.wav", "--ref-text", "Hi<|im_end|>" }, attested, to }, .message = "give 1 to 4 references" },
        .{ .parts = &.{ codec, never, pair, attested, &.{ "--out", "default.json" } }, .message = "--out must be NAME.json" },
        .{ .parts = &.{ codec, never, pair, attested, &.{ "--out", "../tone.wav" } }, .message = "--out must be NAME.json" },
        .{ .parts = &.{ codec, never, pair, attested, &.{"--ref-audio"}, to }, .message = "usage:" },
        .{ .parts = &.{ codec, never, pair, &.{ "--ref-audio", "tone.wav" }, attested, to }, .message = "usage:" },
        .{ .parts = &.{ codec, never, pair, attested }, .message = "usage:" },
        .{ .parts = &.{ codec, never, pair, attested, &.{ "--speaker", "x" }, to }, .message = "usage:" },
    };
    for (cases) |case| {
        try std.testing.expectEqual(@as(u8, 2), try c.call(case.parts));
        try std.testing.expect(std.mem.indexOf(u8, c.err.written(), case.message) != null);
    }
    try std.testing.expectEqual(@as(u8, 1), try c.call(&.{ codec, never, pair, attested, to }));
    try std.testing.expect(std.mem.indexOf(u8, c.err.written(), "cannot run") != null);
    const failing: []const []const u8 = &.{ "--generate", try c.script("failing", "#!/bin/sh\nexit 1\n") };
    try std.testing.expectEqual(@as(u8, 1), try c.call(&.{ codec, failing, pair, attested, to }));
    try std.testing.expect(std.mem.indexOf(u8, c.err.written(), "exited with code 1 encoding tone.wav") != null);
    const silent: []const []const u8 = &.{ "--generate", try c.script("silent", "#!/bin/sh\nexit 0\n") };
    try std.testing.expectEqual(@as(u8, 1), try c.call(&.{ codec, silent, pair, attested, to }));
    try std.testing.expect(std.mem.indexOf(u8, c.err.written(), "wrote no codes for tone.wav") != null);
    const short: []const []const u8 = &.{ "--generate", try c.script("short", "#!/bin/sh\nprintf abc > \"$6\"\n") };
    try std.testing.expectError(error.BadCodes, c.call(&.{ codec, short, pair, attested, to }));
    const wide: []const []const u8 = &.{ "--generate", try c.script("wide", "#!/bin/sh\nhead -c 40 /dev/zero | tr '\\0' '\\377' > \"$6\"\n") };
    try std.testing.expectError(error.BadVoice, c.call(&.{ codec, wide, pair, attested, to }));
    const linked: []const []const u8 = &.{ "--generate", try c.script("linked", try std.fmt.allocPrint(c.a, "#!/bin/sh\nln -s \"$PWD/{s}\" \"$6\"\n", .{try c.path("tone.json.codes")})) };
    try std.testing.expectError(error.SymLinkLoop, c.call(&.{ codec, linked, pair, attested, to }));
    try std.testing.expectEqualStrings("previous voice", try c.read("tone.json"));
    try std.testing.expectEqualSlices(u8, &zero_frame, try c.read("tone.json.codes"));
    try std.testing.expectEqual(@as(usize, 0), try c.leftovers());
}

test "voice tool: cleanup restores private directory permissions after encoder exit" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    for ([_]u8{ 0, 1 }) |exit_code| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        var c: Case = .{ .a = arena.allocator(), .tmp = std.testing.tmpDir(.{}), .err = .init(arena.allocator()) };
        defer c.tmp.cleanup();
        const encoder = try c.script("deny", try std.fmt.allocPrint(c.a,
            \\#!/bin/sh
            \\printf x > "$6"
            \\chmod 000 "$(dirname "$6")"
            \\exit {d}
            \\
        , .{exit_code}));
        const result = c.call(&.{&.{
            "--codec",       "synthetic.pth",         "--generate",    encoder,
            "--ref-audio",   "tone.wav",              "--ref-text",    "A small cloud passes.",
            "--source-kind", "synthetic",             "--source-note", "Generated.",
            "--out",         try c.path("tone.json"),
        }});
        if (result) |status| {
            try std.testing.expectEqual(@as(u8, 1), status);
        } else |e| {
            try std.testing.expectEqual(error.AccessDenied, e);
        }
        try std.testing.expectEqual(@as(usize, 0), try c.leftovers());
    }
}

test "voice tool: cleanup failure warns after publication and accompanies encoder failure" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    for ([_]u8{ 0, 1 }) |exit_code| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        var c: Case = .{ .a = arena.allocator(), .tmp = std.testing.tmpDir(.{}), .err = .init(arena.allocator()) };
        defer c.tmp.cleanup();
        try c.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "tone.json", .data = "previous" });
        const encoder = try c.script("leftovers", try std.fmt.allocPrint(c.a,
            \\#!/bin/sh
            \\set -eu
            \\dir=$(dirname "$6")
            \\mkdir "$dir/nested"
            \\printf x > "$dir/nested/file"
            \\chmod 500 "$dir/nested"
            \\printf '%s' "$dir" > {s}
            \\head -c 40 /dev/zero > "$6"
            \\exit {d}
            \\
        , .{ try c.path("work-path"), exit_code }));
        const result = c.call(&.{&.{
            "--codec",       "synthetic.pth",         "--generate",    encoder,
            "--ref-audio",   "tone.wav",              "--ref-text",    "A small cloud passes.",
            "--source-kind", "synthetic",             "--source-note", "Generated.",
            "--out",         try c.path("tone.json"),
        }});
        const work_path = try c.read("work-path");
        const nested = try std.Io.Dir.cwd().openDir(std.testing.io, try std.fs.path.join(c.a, &.{ work_path, "nested" }), .{ .iterate = true });
        defer nested.close(std.testing.io);
        try nested.setPermissions(std.testing.io, .fromMode(0o700));
        try std.testing.expectEqual(exit_code, try result);
        try std.testing.expectEqual(@as(usize, 1), try c.leftovers());
        try std.testing.expect(std.mem.indexOf(u8, c.err.written(), work_path) != null);
        try std.testing.expect(std.mem.indexOf(u8, c.err.written(), "cleanup failed") != null);
        if (exit_code == 0) {
            try std.testing.expect(std.mem.indexOf(u8, c.err.written(), "WARNING") != null);
            _ = try voice.parse(c.a, try c.read("tone.json"));
        } else {
            try std.testing.expect(std.mem.indexOf(u8, c.err.written(), "exited with code 1") != null);
            try std.testing.expectEqualStrings("previous", try c.read("tone.json"));
        }
    }
}

test "voice tool: allocation failures leave no private directory and encoding uses owner-only access" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    var fixture = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer fixture.deinit();
    var c: Case = .{ .a = fixture.allocator(), .tmp = std.testing.tmpDir(.{}), .err = .init(fixture.allocator()) };
    defer c.tmp.cleanup();
    const encoder = try c.script("mode", "#!/bin/sh\n[ \"$(stat -c %a \"$(dirname \"$6\")\")\" = 700 ] || exit 3\nhead -c 40 /dev/zero > \"$6\"\n");
    const args: []const []const u8 = &.{
        "--codec",       "synthetic.pth",         "--generate",    encoder,
        "--ref-audio",   "tone.wav",              "--ref-text",    "A small cloud passes.",
        "--source-kind", "synthetic",             "--source-note", "Generated.",
        "--out",         try c.path("tone.json"),
    };
    var fail_index: usize = 0;
    while (true) : (fail_index += 1) {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        var failing = std.testing.FailingAllocator.init(arena.allocator(), .{ .fail_index = fail_index });
        c.err.clearRetainingCapacity();
        const result = run(failing.allocator(), std.testing.io, &c.err.writer, args);
        try std.testing.expectEqual(@as(usize, 0), try c.leftovers());
        if (failing.has_induced_failure) {
            if (result) |_| return error.ExpectedAllocationFailure else |e| try std.testing.expect(e == error.OutOfMemory or e == error.WriteFailed);
        } else {
            try std.testing.expectEqual(@as(u8, 0), try result);
            break;
        }
    }
}
