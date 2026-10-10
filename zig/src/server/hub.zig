//! A model argument as a checkpoint directory: a path, or a Hugging Face repo id already in the local cache.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// ``owner/name`` with Hugging Face's allowed characters.
pub fn isRepoId(text: []const u8) bool {
    const slash = std.mem.indexOfScalar(u8, text, '/') orelse return false;
    if (std.mem.lastIndexOfScalar(u8, text, '/') != slash) return false;
    for ([_][]const u8{ text[0..slash], text[slash + 1 ..] }) |part| {
        if (part.len == 0 or part.len > 96) return false;
        for (part) |ch| if (!(std.ascii.isAlphanumeric(ch) or ch == '-' or ch == '_' or ch == '.')) return false;
    }
    return true;
}

fn isDir(io: std.Io, path: []const u8) bool {
    var dir = std.Io.Dir.cwd().openDir(io, path, .{}) catch return false;
    dir.close(io);
    return true;
}

/// The checkpoint directory for ``model``, or null with ``problem`` set (no downloads: the cache must hold it).
pub fn resolve(a: Allocator, io: std.Io, environ: ?*const std.process.Environ.Map, model: []const u8, problem: *[]const u8) !?[]const u8 {
    const home = if (environ) |e| e.get("HOME") orelse "" else "";
    const expanded = if (std.mem.startsWith(u8, model, "~/")) try std.fs.path.join(a, &.{ home, model[2..] }) else model;
    if (isDir(io, expanded)) return expanded;
    if (!isRepoId(model)) {
        problem.* = try std.fmt.allocPrint(a, "{s} is neither a directory nor a Hugging Face repo id (owner/name)", .{model});
        return null;
    }
    const get = struct {
        fn f(env: ?*const std.process.Environ.Map, name: []const u8) ?[]const u8 {
            return if (env) |e| e.get(name) else null;
        }
    }.f;
    const hub = get(environ, "HF_HUB_CACHE") orelse if (get(environ, "HF_HOME")) |h| try std.fs.path.join(a, &.{ h, "hub" }) else try std.fs.path.join(a, &.{ home, ".cache", "huggingface", "hub" });
    const slash = std.mem.indexOfScalar(u8, model, '/').?;
    const repo = try std.fmt.allocPrint(a, "models--{s}--{s}", .{ model[0..slash], model[slash + 1 ..] });
    const ref_path = try std.fs.path.join(a, &.{ hub, repo, "refs", "main" });
    const ref = std.Io.Dir.cwd().readFileAlloc(io, ref_path, a, .limited(256)) catch {
        problem.* = try std.fmt.allocPrint(a, "{s} is not in the Hugging Face cache; the native engine does not download: run tensorfold pull {s} first", .{ model, model });
        return null;
    };
    const rev = std.mem.trim(u8, ref, " \r\n");
    // The ref names the snapshot dir: only a 40-hex git sha is ever joined into a cache path.
    var sha_ok = rev.len == 40;
    for (rev) |ch| sha_ok = sha_ok and ((ch >= '0' and ch <= '9') or (ch >= 'a' and ch <= 'f') or (ch >= 'A' and ch <= 'F'));
    if (!sha_ok) {
        problem.* = try std.fmt.allocPrint(a, "{s}'s cached ref does not name a revision; run tensorfold pull {s}", .{ model, model });
        return null;
    }
    const snapshot = try std.fs.path.join(a, &.{ hub, repo, "snapshots", rev });
    if (!isDir(io, snapshot)) {
        problem.* = try std.fmt.allocPrint(a, "{s}'s cached snapshot is missing; run tensorfold pull {s}", .{ model, model });
        return null;
    }
    return snapshot;
}

/// The name a server answers to: --name, else the repo's or directory's last part.
pub fn servedName(name: []const u8, model: []const u8, dir: []const u8) []const u8 {
    if (name.len > 0) return name;
    const source = if (isRepoId(model)) std.mem.trimEnd(u8, model, "/") else std.mem.trimEnd(u8, dir, "/");
    return std.fs.path.basename(source);
}

test "repo ids" {
    try std.testing.expect(isRepoId("TensorFold/Qwen3.8-27B-MLX-4bit"));
    try std.testing.expect(!isRepoId("/models/x") and !isRepoId("a/b/c") and !isRepoId("plain"));
    try std.testing.expectEqualStrings("x", servedName("", "/models/x/", "/models/x/"));
}
