//! Pin/unpin talks to the running server so its in-memory graph and persisted file change in one transaction.
const std = @import("std");
const stream = @import("benchmark_stream.zig");
pub const Command = struct { url: []const u8, pinned: bool, key_file: ?[]const u8 };
pub const usage = "usage: tensorfold slide {pin|unpin} ID [--server URL] [--api-key-file FILE]\n";

pub fn parse(a: std.mem.Allocator, argv: []const [:0]const u8) !Command {
    if (argv.len < 2) return error.BadArguments;
    const pinned = std.mem.eql(u8, argv[0], "pin");
    if (!pinned and !std.mem.eql(u8, argv[0], "unpin")) return error.BadArguments;
    const id = std.fmt.parseInt(u32, argv[1], 10) catch return error.BadArguments;
    if (id == 0) return error.BadArguments;
    var server: []const u8 = "http://127.0.0.1:8080";
    var key_file: ?[]const u8 = null;
    var seen_server = false;
    var i: usize = 2;
    while (i < argv.len) : (i += 2) {
        if (i + 1 == argv.len or argv[i + 1].len == 0) return error.BadArguments;
        if (std.mem.eql(u8, argv[i], "--server") and !seen_server) {
            server = argv[i + 1];
            seen_server = true;
        } else if (std.mem.eql(u8, argv[i], "--api-key-file") and key_file == null) {
            key_file = argv[i + 1];
        } else return error.BadArguments;
    }
    _ = stream.splitUrl(server) orelse return error.BadArguments;
    var base = std.mem.trimEnd(u8, server, "/");
    if (std.mem.endsWith(u8, base, "/v1")) base = base[0 .. base.len - 3];
    return .{ .url = try std.fmt.allocPrint(a, "{s}/v1/slide/facts/{d}/pin", .{ base, id }), .pinned = pinned, .key_file = key_file };
}

pub fn main(init: std.process.Init, argv: []const [:0]const u8) !u8 {
    const a = init.arena.allocator();
    const c = parse(a, argv) catch {
        std.debug.print("{s}", .{usage});
        return 2;
    };
    const key: ?[]const u8 = if (c.key_file) |file| blk: {
        const bytes = std.Io.Dir.cwd().readFileAlloc(init.io, file, a, .limited(64 << 10)) catch {
            std.debug.print("tensorfold: cannot read the API key file\n", .{});
            return 1;
        };
        var lines = std.mem.tokenizeAny(u8, bytes, "\r\n");
        const line = std.mem.trim(u8, lines.next() orelse "", " \t");
        if (line.len == 0 or line.len > 256) {
            std.debug.print("tensorfold: key file needs a nonempty first key of at most 256 bytes\n", .{});
            return 2;
        }
        break :blk line;
    } else null;
    var client: std.http.Client = .{ .allocator = a, .io = init.io };
    defer client.deinit();
    const got = stream.fetch(a, init.io, &client, .{ .method = .POST, .url = c.url, .body = if (c.pinned) "{\"pinned\":true}" else "{\"pinned\":false}", .token = key, .limit = 64 << 10, .timeout_s = 30 });
    if (got.failed != .none or got.status != 200) {
        std.debug.print("tensorfold: pin change failed (HTTP {d}, {s}): {s}\n", .{ got.status, @tagName(got.failed), got.body });
        return 1;
    }
    try std.Io.File.stdout().writeStreamingAll(init.io, got.body);
    try std.Io.File.stdout().writeStreamingAll(init.io, "\n");
    return 0;
}

test "fact usage: slide CLI targets the pin route and rejects malformed ids, URLs and options" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const p = try parse(a, &.{ "pin", "7" });
    try std.testing.expectEqualStrings("http://127.0.0.1:8080/v1/slide/facts/7/pin", p.url);
    try std.testing.expect(p.pinned and p.key_file == null);
    const u = try parse(a, &.{ "unpin", "9", "--server", "https://server.example.test/api/v1/", "--api-key-file", "keyfile" });
    try std.testing.expectEqualStrings("https://server.example.test/api/v1/slide/facts/9/pin", u.url);
    try std.testing.expect(!u.pinned);
    try std.testing.expectEqualStrings("keyfile", u.key_file.?);
    try std.testing.expectError(error.BadArguments, parse(a, &.{ "pin", "0" }));
    try std.testing.expectError(error.BadArguments, parse(a, &.{ "pin", "7", "--server", "https://user:secret@server.example.test" }));
    try std.testing.expectError(error.BadArguments, parse(a, &.{ "pin", "7", "--server", "http://server.example.test?q=x" }));
    try std.testing.expectError(error.BadArguments, parse(a, &.{ "pin", "7", "--api-key-file" }));
}
