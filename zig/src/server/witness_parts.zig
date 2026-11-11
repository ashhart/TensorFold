//! The execution witness's server parts: the running binary, the text files as loaded, preparation and environment.
const std = @import("std");
const builtin = @import("builtin");
const json = @import("json");
const Key = @import("core").digest_cache.Key;
const file_identity = @import("sys").file_identity;
const model_text = @import("model_text.zig");
const Config = @import("server.zig").Config;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Allocator = std.mem.Allocator;
const Io = std.Io;
/// Set at shutdown: a hash in progress ends at its next block with error.Stopped.
pub const Stop = std.atomic.Value(bool);

/// The running executable, held open from startup: its open inode stays the running code if its path is replaced.
pub const Binary = struct {
    file: Io.File,
    key: Key,

    /// Linux's /proc/self/exe names the running inode; elsewhere the path the process was started from.
    pub fn openRunning(a: Allocator, io: Io) !Binary {
        if (builtin.os.tag == .linux) return openPath(io, "/proc/self/exe");
        const path = try std.process.executablePathAlloc(io, a);
        defer a.free(path);
        return openPath(io, path);
    }

    pub fn openPath(io: Io, path: []const u8) !Binary {
        const file = try Io.Dir.cwd().openFile(io, path, .{});
        errdefer file.close(io);
        return .{ .file = file, .key = try Key.of(file, io) };
    }

    pub fn close(b: Binary, io: Io) void {
        b.file.close(io);
    }

    /// The file's plain SHA-256, with the stamps taken at open unchanged before and after the read (else FilesChanged).
    pub fn digest(b: Binary, io: Io, stop: ?*const Stop) ![32]u8 {
        if (!b.key.equal(try Key.of(b.file, io))) return error.FilesChanged;
        const bytes = file_identity.hashUntil(io, b.file, 1 << 31, stop) catch |e|
            return if (e == error.IdentityFileChanged) error.FilesChanged else e;
        if (!b.key.equal(try Key.of(b.file, io))) return error.FilesChanged;
        return bytes;
    }
};

/// Every file a checkpoint's text load may read (tokenizer, template, special tokens, EOS); absent ones count too.
pub const text_files = [_][]const u8{ "config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "special_tokens_map.json", "chat_template.jinja", "chat_template.json", "tokenizer.model", "vocab.json", "merges.txt", "joint_head_config.json" };

/// The text files' stamps from before the load reads them; `digest` hashes them later only if no stamp moved.
pub const Text = struct {
    arena: std.heap.ArenaAllocator,
    labels: []const []const u8,
    paths: []const []const u8,
    keys: []?Key,

    /// Stamps each file before the text load; `override` is --chat-template's file, labelled as the flag.
    pub fn stamp(gpa: Allocator, io: Io, dir: []const u8, override: ?[]const u8) !*Text {
        const t = try gpa.create(Text);
        errdefer gpa.destroy(t);
        t.arena = .init(gpa);
        errdefer t.arena.deinit();
        const a = t.arena.allocator();
        const n = text_files.len + @intFromBool(override != null);
        const labels = try a.alloc([]const u8, n);
        const paths = try a.alloc([]const u8, n);
        for (text_files, 0..) |name, i| {
            labels[i] = name;
            paths[i] = try std.fs.path.join(a, &.{ dir, name });
        }
        if (override) |path| {
            labels[n - 1] = "--chat-template";
            paths[n - 1] = try a.dupe(u8, path);
        }
        t.labels = labels;
        t.paths = paths;
        t.keys = try a.alloc(?Key, n);
        for (paths, t.keys) |path, *k| k.* = try keyOf(io, path);
        return t;
    }

    pub fn deinit(t: *Text) void {
        const gpa = t.arena.child_allocator;
        t.arena.deinit();
        gpa.destroy(t);
    }

    /// Each file's label and bytes (or its absence), when every stamp still equals the one taken before the load.
    pub fn digest(t: *const Text, io: Io, stop: ?*const Stop) ![32]u8 {
        var h = Sha256.init(.{});
        h.update("text-v1");
        for (t.labels, t.paths, t.keys) |label, path, want| {
            field(&h, label);
            const k = want orelse {
                if (try keyOf(io, path) != null) return error.FilesChanged;
                field(&h, "absent");
                continue;
            };
            const file = Io.Dir.cwd().openFile(io, path, .{}) catch return error.FilesChanged;
            defer file.close(io);
            if (!k.equal(try Key.of(file, io))) return error.FilesChanged;
            const bytes = file_identity.hashUntil(io, file, 1 << 30, stop) catch |e|
                return if (e == error.IdentityFileChanged) error.FilesChanged else e;
            if (!k.equal(try Key.of(file, io))) return error.FilesChanged;
            field(&h, &bytes);
        }
        return h.finalResult();
    }

    fn keyOf(io: Io, path: []const u8) !?Key {
        const file = Io.Dir.cwd().openFile(io, path, .{}) catch |e| return if (e == error.FileNotFound) null else e;
        defer file.close(io);
        return try Key.of(file, io);
    }
};

/// What shapes a request before the engine: the server's defaults, and the template and EOS the text resolved.
pub fn preparation(a: Allocator, config: Config, text: model_text.Text) ![32]u8 {
    var h = Sha256.init(.{});
    h.update("preparation-v1");
    var buf: [64]u8 = undefined;
    const numbers = [_]i64{ config.default_max_tokens, config.thinking_budget, config.thinking_reserve, config.seed_salt };
    for (numbers) |n| field(&h, try std.fmt.bufPrint(&buf, "{d}", .{n}));
    const flags = [_]bool{ config.enable_thinking, config.loop_guard, config.loop_stop, config.use_drafts, config.prompt_seed };
    for (flags) |f| field(&h, if (f) "1" else "0");
    field(&h, config.reasoning_effort orelse "");
    for ([_]?json.Value{ config.default_sampling, config.instruct_sampling }) |v|
        field(&h, if (v) |x| try json.stringify(a, x, .{}) else "none");
    field(&h, text.templateSource());
    field(&h, std.mem.sliceAsBytes(text.eosIds()));
    return h.finalResult();
}

/// Every TF_* and TENSORFOLD_* variable but secrets, sorted by name, as the process started with them.
pub fn environment(a: Allocator, env: ?*const std.process.Environ.Map) ![32]u8 {
    var names: std.ArrayList([]const u8) = .empty;
    defer names.deinit(a);
    if (env) |m| for (m.keys()) |name| if (settingName(name) and !secret(name)) try names.append(a, name);
    std.mem.sort([]const u8, names.items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);
    var h = Sha256.init(.{});
    h.update("environment-v1");
    for (names.items) |name| {
        field(&h, name);
        field(&h, env.?.get(name).?);
    }
    return h.finalResult();
}

fn settingName(name: []const u8) bool {
    return std.mem.startsWith(u8, name, "TF_") or std.mem.startsWith(u8, name, "TENSORFOLD_");
}

/// Names whose values are credentials or point at them; they never change an execution.
pub fn secret(name: []const u8) bool {
    const exact = [_][]const u8{ "TENSORFOLD_API_KEY", "TENSORFOLD_TOOL_ID_KEY_FILE" };
    for (exact) |x| if (std.mem.eql(u8, name, x)) return true;
    for ([_][]const u8{ "_API_KEY", "_SECRET", "_PASSWORD" }) |tail| if (std.mem.endsWith(u8, name, tail)) return true;
    return false;
}

/// A length-prefixed field, so no two field lists hash alike.
pub fn field(h: *Sha256, bytes: []const u8) void {
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, bytes.len, .little);
    h.update(&len);
    h.update(bytes);
}
