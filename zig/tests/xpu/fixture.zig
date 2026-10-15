//! Reference tensors the op tests compare with: .bin files read at run time from TF_FIXTURES_DIR.

const std = @import("std");

var io: std.Io = undefined;
var dir: []const u8 = "";

/// The runner calls this once; `dir` is TF_FIXTURES_DIR or $TF_FIXTURES_DIR.
pub fn init(io_: std.Io, dir_: []const u8) void {
    io = io_;
    dir = dir_;
}

/// Page-aligned bytes of `dir`/`name`.bin, never freed (a test process is short).
pub fn load(name: []const u8) ![]const u8 {
    const path = try std.fmt.allocPrint(std.heap.page_allocator, "{s}/{s}.bin", .{ dir, name });
    defer std.heap.page_allocator.free(path);
    return std.Io.Dir.cwd().readFileAlloc(io, path, std.heap.page_allocator, .limited(1 << 30)) catch |e| {
        std.log.err("fixture {s}: {t}", .{ path, e });
        return e;
    };
}
