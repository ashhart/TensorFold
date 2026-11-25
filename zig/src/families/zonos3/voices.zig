//! A release's default voices (gguf/voices/<language>/<clip>.codes.npy and .embedding.npy), ready without an encoder.
const std = @import("std");
const cfg = @import("config.zig");
const npy = @import("npy.zig");

pub const dir = "gguf/voices";
pub const max_voices = 256;

pub const Voice = struct {
    name: []const u8, // the clip's first word in lower case, e.g. "savannah"
    clip: []const u8, // the clip's file stem, e.g. "Savannah_en-American_F_1"
    language: []const u8, // the folder it sits in
    codes: []const [cfg.codebooks]i32, // the clip in the codec's codes: the reference prefix
    embedding: [cfg.speaker_dim]f32,
};

fn lower(a: std.mem.Allocator, s: []const u8) ![]u8 {
    const out = try a.dupe(u8, s);
    for (out) |*c| c.* = std.ascii.toLower(c.*);
    return out;
}

fn one(a: std.mem.Allocator, io: std.Io, folder: std.Io.Dir, language: []const u8, stem: []const u8, max_frames: usize) !Voice {
    var nb: [300]u8 = undefined;
    const code_bytes = try folder.readFileAlloc(io, try std.fmt.bufPrint(&nb, "{s}.codes.npy", .{stem}), a, .limited(1 << 24));
    const codes = try npy.parse(code_bytes);
    if (codes.rank != 2 or codes.shape[1] != cfg.codebooks or codes.shape[0] == 0) return error.BadVoice;
    const flat = try codes.ints(a);
    const frames = @min(codes.shape[0], max_frames);
    const rows = try a.alloc([cfg.codebooks]i32, frames);
    for (rows, 0..) |*row, i| {
        row.* = flat[i * cfg.codebooks ..][0..cfg.codebooks].*;
        for (row) |code| if (code < 0 or code >= cfg.codebook_size) return error.BadVoice;
    }
    const emb_bytes = try folder.readFileAlloc(io, try std.fmt.bufPrint(&nb, "{s}.embedding.npy", .{stem}), a, .limited(1 << 20));
    const emb = try npy.parse(emb_bytes);
    if (emb.kind != .f32 or emb.numel() != cfg.speaker_dim) return error.BadVoice;
    var v: Voice = .{ .name = "", .clip = try a.dupe(u8, stem), .language = try a.dupe(u8, language), .codes = rows, .embedding = undefined };
    const values = try emb.floats(a);
    v.embedding = values[0..cfg.speaker_dim].*;
    for (v.embedding) |x| if (!std.math.isFinite(x)) return error.BadVoice;
    const word = stem[0 .. std.mem.indexOfScalar(u8, stem, '_') orelse stem.len];
    v.name = try lower(a, word);
    return v;
}

/// Every voice under `root`'s voices folder, sorted by name; `a` is an arena. At most `max_frames` of each clip.
pub fn load(a: std.mem.Allocator, io: std.Io, root: []const u8, max_frames: usize) ![]const Voice {
    const path = try std.fs.path.join(a, &.{ root, dir });
    var top = std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true }) catch |e| switch (e) {
        error.FileNotFound => return &.{},
        else => return e,
    };
    defer top.close(io);
    var voices: std.ArrayList(Voice) = .empty;
    var languages = top.iterate();
    while (try languages.next(io)) |lang| {
        if (lang.kind != .directory) continue;
        var folder = try top.openDir(io, lang.name, .{ .iterate = true });
        defer folder.close(io);
        var files = folder.iterate();
        while (try files.next(io)) |entry| {
            const tail = ".codes.npy";
            if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, tail)) continue;
            if (voices.items.len >= max_voices) return error.TooManyVoices;
            try voices.append(a, try one(a, io, folder, lang.name, entry.name[0 .. entry.name.len - tail.len], max_frames));
        }
    }
    std.mem.sort(Voice, voices.items, {}, struct {
        fn less(_: void, l: Voice, r: Voice) bool {
            return std.mem.lessThan(u8, l.clip, r.clip);
        }
    }.less);
    // a first word two clips share names neither: those go by their whole stem
    for (voices.items, 0..) |*v, i| {
        var shared = false;
        for (voices.items, 0..) |other, j| shared = shared or (i != j and std.ascii.eqlIgnoreCase(other.clip[0..@min(other.clip.len, v.name.len)], v.name) and
            (other.clip.len == v.name.len or other.clip[v.name.len] == '_'));
        if (shared) v.name = try lower(a, v.clip);
    }
    return voices.items;
}

pub fn find(voices: []const Voice, name: []const u8) ?*const Voice {
    for (voices) |*v| if (std.ascii.eqlIgnoreCase(v.name, name) or std.mem.eql(u8, v.clip, name)) return v;
    return null;
}
