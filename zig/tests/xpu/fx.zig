//! Reference tensors read at run time: `name`.bin under TF_FIXTURES_DIR (default $TF_FIXTURES_DIR).

const std = @import("std");
const xpu = @import("xpu");

fn dir() []const u8 {
    if (std.c.getenv("TF_FIXTURES_DIR")) |v| return std.mem.span(v);
    const home = std.mem.span(std.c.getenv("HOME") orelse "/root");
    return std.fmt.allocPrint(std.heap.page_allocator, "{s}/tensorfold-fixtures", .{home}) catch "";
}

/// The bytes of `name`.bin, never freed (a test process is short).
pub fn load(name: []const u8) ![]const u8 {
    const file = try std.fmt.allocPrint(std.heap.page_allocator, "{s}.bin", .{name});
    return xpu.loader.readFile(std.heap.page_allocator, dir(), file) catch |e| {
        std.log.err("fixture {s}/{s}: {t}", .{ dir(), file, e });
        return e;
    };
}
