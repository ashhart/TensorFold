//! Tokenizer identity check shared by storage startup and later index startup.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;

pub fn tokenizerSha(gpa: Allocator, io: std.Io, model_dir: []const u8, limit: usize, problem: *[]const u8) ![64]u8 {
    if (model_dir.len == 0) {
        problem.* = "the snapshot names a tokenizer.json SHA-256, and the server has no model folder to read tokenizer.json from";
        return error.NoModelDir;
    }
    const path = try std.fs.path.join(gpa, &.{ model_dir, "tokenizer.json" });
    defer gpa.free(path);
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(limit)) catch |e| {
        problem.* = switch (e) {
            error.FileNotFound => "the snapshot names a tokenizer.json SHA-256, and the model folder has no tokenizer.json to check it against",
            error.StreamTooLong => "the snapshot names a tokenizer.json SHA-256, and the model's tokenizer.json is over the read limit (256 MiB), so it cannot be checked",
            error.OutOfMemory => "out of memory reading the model's tokenizer.json",
            else => "the snapshot names a tokenizer.json SHA-256, and the model's tokenizer.json cannot be read to check it",
        };
        return e;
    };
    defer gpa.free(bytes);
    var out: [Sha256.digest_length]u8 = undefined;
    Sha256.hash(bytes, &out, .{});
    return std.fmt.bytesToHex(out, .lower);
}
