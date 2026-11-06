//! A checkpoint folder's text files: its JSON documents, the tokenizer they configure and its special tokens.
const std = @import("std");
const tokenizer = @import("tokenizer");
const Allocator = std.mem.Allocator;

pub fn readJson(io: std.Io, a: Allocator, dir: []const u8, name: []const u8) std.json.Value {
    const path = std.fs.path.join(a, &.{ dir, name }) catch return .null;
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20)) catch return .null;
    return std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch .null;
}

/// Optional EOS documents are absent only on FileNotFound; malformed present data refuses the text load.
pub fn readEosJson(io: std.Io, a: Allocator, dir: []const u8, name: []const u8, limit: usize) !std.json.Value {
    const path = try std.fs.path.join(a, &.{ dir, name });
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(limit)) catch |e| switch (e) {
        error.FileNotFound => return .null,
        else => return e,
    };
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{ .allocate = .alloc_always });
    if (doc != .object) return error.BadEos;
    return doc;
}

/// The checkpoint's tokenizer.json, else the tokenizer/ folder a pipeline keeps it in; `path` names what was read.
pub fn openTokenizer(io: std.Io, a: Allocator, gpa: Allocator, dir: []const u8, path: *[]const u8) !tokenizer.Tokenizer {
    path.* = dir;
    return tokenizer.loadTokenizer(io, gpa, dir) catch |e| {
        if (e != error.FileNotFound) return e;
        // the join lives in the caller's arena, so the fallback costs no allocation of its own
        path.* = try std.fs.path.join(a, &.{ dir, "tokenizer" });
        return tokenizer.loadTokenizer(io, gpa, path.*);
    };
}

/// Every ``*_token`` string (or ``{"content": ...}``) in tokenizer_config.json, as apply_chat_template passes them.
pub fn specialsOf(a: Allocator, config: std.json.Value) !std.json.Value {
    var out: std.json.ObjectMap = .empty;
    if (config == .object) {
        var it = config.object.iterator();
        while (it.next()) |e| {
            if (!std.mem.endsWith(u8, e.key_ptr.*, "_token")) continue;
            switch (e.value_ptr.*) {
                .string => try out.put(a, e.key_ptr.*, e.value_ptr.*),
                .object => |o| if (o.get("content")) |c| if (c == .string) try out.put(a, e.key_ptr.*, c),
                else => {},
            }
        }
    }
    return .{ .object = out };
}
