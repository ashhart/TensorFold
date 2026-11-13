//! The served voice catalog: --voices DIR at startup, then admin adds and deletes, read under one lock.
const std = @import("std");
const json = @import("json");
const prompt = @import("drama_prompt");
const voice = @import("drama_voice.zig");
const Allocator = std.mem.Allocator;
const Voice = voice.Voice;

/// Readers copy what they need under the lock, so a change frees the old catalog at once and no request sees it go.
pub const Store = struct {
    gpa: Allocator = undefined,
    io: std.Io = undefined,
    /// --voices DIR: where added voices are written and deleted ones removed; null keeps changes in memory.
    dir: ?[]const u8 = null,
    mutex: std.Io.Mutex = .init,
    arena: ?*std.heap.ArenaAllocator = null,
    voices: []const Voice = &.{},

    /// The catalog in `dir`, sorted by name, as startup loads it.
    pub fn load(s: *Store, dir: []const u8) !void {
        var scratch: std.heap.ArenaAllocator = .init(s.gpa);
        defer scratch.deinit();
        try s.swap(try voice.load(scratch.allocator(), s.io, dir));
    }

    pub fn deinit(s: *Store) void {
        if (s.arena) |arena| free(s.gpa, arena);
        s.arena = null;
        s.voices = &.{};
    }

    /// The references `requested` selects (as drama_voice.select), copied into `a`.
    pub fn select(s: *Store, a: Allocator, default: []const u8, requested: []const u8) ![]const prompt.Reference {
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        const refs = try voice.select(s.voices, default, requested);
        const copy = try a.alloc(prompt.Reference, refs.len);
        for (refs, copy) |ref, *out| out.* = .{ .transcript = try a.dupe(u8, ref.transcript), .codes = try a.dupe([10]i32, ref.codes) };
        return copy;
    }

    /// GET /v1/voices's reply, its names and descriptions copied into `a`.
    pub fn listing(s: *Store, a: Allocator) !json.Value {
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        const copy = try a.alloc(Voice, s.voices.len);
        for (s.voices, copy) |v, *out| out.* = .{ .name = try a.dupe(u8, v.name), .references = &.{}, .description = try a.dupe(u8, v.description) };
        return voice.listing(a, copy);
    }

    pub fn count(s: *Store) usize {
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        return s.voices.len;
    }

    /// Adds or replaces `name`, first writing `file` (its attested voice file) when there is a directory; true: added.
    pub fn put(s: *Store, name: []const u8, file: []const u8, refs: []const prompt.Reference, description: []const u8) !bool {
        if (!voice.validName(name)) return error.BadVoiceName;
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        var next: std.ArrayList(Voice) = .empty;
        defer next.deinit(s.gpa);
        for (s.voices) |v| if (!std.mem.eql(u8, v.name, name)) try next.append(s.gpa, v);
        const added = next.items.len == s.voices.len;
        if (added and next.items.len >= max_voices) return error.TooManyVoices;
        try next.append(s.gpa, .{ .name = name, .references = refs, .description = description });
        if (s.dir) |dir| try write(s.gpa, s.io, dir, name, file);
        try s.swap(next.items);
        return added;
    }

    /// Deletes `name`, and its file when there is a directory; the configured default voice stays.
    pub fn remove(s: *Store, name: []const u8, default: []const u8) !void {
        if (!voice.validName(name)) return error.UnknownVoice;
        if (std.mem.eql(u8, name, default)) return error.DefaultVoice;
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        var next: std.ArrayList(Voice) = .empty;
        defer next.deinit(s.gpa);
        for (s.voices) |v| if (!std.mem.eql(u8, v.name, name)) try next.append(s.gpa, v);
        if (next.items.len == s.voices.len) return error.UnknownVoice;
        if (s.dir) |dir| try unlink(s.io, dir, name);
        try s.swap(next.items);
    }

    /// The catalog becomes a sorted copy of `voices` and the old one is freed; the caller holds the lock or is alone.
    pub fn swap(s: *Store, voices: []const Voice) !void {
        const arena = try s.gpa.create(std.heap.ArenaAllocator);
        arena.* = .init(s.gpa);
        errdefer free(s.gpa, arena);
        const a = arena.allocator();
        const copy = try a.alloc(Voice, voices.len);
        for (voices, copy) |v, *out| {
            const refs = try a.alloc(prompt.Reference, v.references.len);
            for (v.references, refs) |ref, *r| r.* = .{ .transcript = try a.dupe(u8, ref.transcript), .codes = try a.dupe([10]i32, ref.codes) };
            out.* = .{ .name = try a.dupe(u8, v.name), .references = refs, .description = try a.dupe(u8, v.description) };
        }
        std.mem.sort(Voice, copy, {}, struct {
            fn less(_: void, left: Voice, right: Voice) bool {
                return std.mem.lessThan(u8, left.name, right.name);
            }
        }.less);
        if (s.arena) |old| free(s.gpa, old);
        s.arena = arena;
        s.voices = copy;
    }
};

pub const max_voices = 256;

fn free(gpa: Allocator, arena: *std.heap.ArenaAllocator) void {
    arena.deinit();
    gpa.destroy(arena);
}

/// NAME.json through a fresh hidden temporary in the same directory, renamed over any older file.
fn write(a: Allocator, io: std.Io, path: []const u8, name: []const u8, file: []const u8) !void {
    var dir = try std.Io.Dir.cwd().openDir(io, path, .{});
    defer dir.close(io);
    var nonce: [8]u8 = undefined;
    io.random(&nonce);
    const temp = try std.fmt.allocPrint(a, ".{s}.json.{x}", .{ name, std.mem.readInt(u64, &nonce, .native) });
    defer a.free(temp);
    const final = try std.fmt.allocPrint(a, "{s}.json", .{name});
    defer a.free(final);
    try dir.writeFile(io, .{ .sub_path = temp, .data = file, .flags = .{ .exclusive = true } });
    errdefer dir.deleteFile(io, temp) catch {};
    try dir.rename(temp, dir, final, io);
}

fn unlink(io: std.Io, path: []const u8, name: []const u8) !void {
    var dir = try std.Io.Dir.cwd().openDir(io, path, .{});
    defer dir.close(io);
    var buf: [80]u8 = undefined;
    dir.deleteFile(io, try std.fmt.bufPrint(&buf, "{s}.json", .{name})) catch |e| if (e != error.FileNotFound) return e;
}

test {
    _ = @import("drama_voice_store_test.zig");
}
