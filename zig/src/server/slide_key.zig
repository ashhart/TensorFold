//! The --slide key gate: a server that rewrites its weights starts only with a key its own key store accepts.
const std = @import("std");
const auth = @import("auth.zig");
const cli = @import("cli.zig");
const Allocator = std.mem.Allocator;

/// Why `--slide` cannot start (no key, or one the server's store refuses; `env_keys`: TENSORFOLD_API_KEY), or null.
pub fn slideKeyless(a: Allocator, args: cli.Args, env_keys: []const u8) !?[]const u8 {
    if (!args.slide) return null;
    var problem: []const u8 = "";
    // the store serve.zig builds after the load, built here before it, so the gate and the store agree
    var given = auth.Store.init(a, args.api_key, env_keys, null, false, &problem) catch |e| {
        if (e != error.KeyFile) return e;
        const text = "--slide needs a key this server accepts, and --api-key or TENSORFOLD_API_KEY gives one it " ++
            "refuses: {s}";
        return try std.fmt.allocPrint(a, text, .{problem});
    };
    defer given.deinit();
    if (args.api_key_file) |file| {
        var read = auth.Store.init(a, &.{}, "", file, false, &problem) catch |e| {
            if (e != error.KeyFile) return e;
            const text = "--slide needs a key this server can read, and --api-key-file {s} gives none: {s}";
            return try std.fmt.allocPrint(a, text, .{ file, problem });
        };
        read.deinit();
        return null;
    }
    if (given.enabled()) return null;
    const text = "--slide lets any client that reaches this server rewrite the weights in {s} through " ++
        "/v1/slide/learn: add --api-key KEY or --api-key-file FILE so only your clients can teach it";
    return try std.fmt.allocPrint(a, text, .{args.model});
}

test "--slide starts only with a key: --api-key, --api-key-file or TENSORFOLD_API_KEY" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: cli.Usage = .{};
    const bare = try cli.parse(a, &.{ "/models/copy", "--slide" }, &u);
    const why = (try slideKeyless(a, bare, "")) orelse return error.TestExpectedRefusal;
    try std.testing.expect(has(why, "/models/copy") and has(why, "--api-key KEY or --api-key-file FILE"));
    try std.testing.expect(try slideKeyless(a, bare, " , ") != null); // separators alone are no key
    try std.testing.expect(try slideKeyless(a, bare, "sk-env") == null);
    const keyed = try cli.parse(a, &.{ "m", "--slide", "--api-key", "sk-1" }, &u);
    try std.testing.expect(try slideKeyless(a, keyed, "") == null);
    try std.testing.expect(try slideKeyless(a, try cli.parse(a, &.{"m"}, &u), "") == null); // serving alone needs none
}

test "--slide refuses before any load a key file or a key the server's store would refuse, naming the file" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var u: cli.Usage = .{};
    const file = try std.fmt.allocPrintSentinel(a, ".zig-cache/tmp/{s}/keys", .{tmp.sub_path}, 0);
    const keyed = try cli.parse(a, &.{ "m", "--slide", "--api-key-file", file }, &u);
    const missing = (try slideKeyless(a, keyed, "")) orelse return error.TestExpectedRefusal;
    try std.testing.expect(has(missing, file));
    try tmp.dir.writeFile(io, .{ .sub_path = "keys", .data = "ops: sk-file\n" });
    try std.testing.expectEqual(@as(c_int, 0), std.c.fchmodat(std.posix.AT.FDCWD, file, 0o644, 0));
    const open = (try slideKeyless(a, keyed, "")) orelse return error.TestExpectedRefusal;
    try std.testing.expect(has(open, file) and has(open, "chmod 600"));
    try std.testing.expectEqual(@as(c_int, 0), std.c.fchmodat(std.posix.AT.FDCWD, file, 0o600, 0));
    try std.testing.expect(try slideKeyless(a, keyed, "") == null);
    try tmp.dir.writeFile(io, .{ .sub_path = "keys", .data = "" });
    const empty = (try slideKeyless(a, keyed, "")) orelse return error.TestExpectedRefusal;
    try std.testing.expect(has(empty, file) and has(empty, "no keys"));
    const bare = try cli.parse(a, &.{ "m", "--slide" }, &u);
    for ([_][]const u8{ "\u{a0}", "sk-a, sk b", "\x1c" }) |keys| { // not trimmed as space, so keys the store refuses
        const why = (try slideKeyless(a, bare, keys)) orelse return error.TestExpectedRefusal;
        try std.testing.expect(has(why, "TENSORFOLD_API_KEY") and has(why, "without whitespace"));
    }
    const blank = try cli.parse(a, &.{ "m", "--slide", "--api-key", " " }, &u);
    try std.testing.expect(try slideKeyless(a, blank, "") != null);
}

fn has(text: []const u8, part: []const u8) bool {
    return std.mem.indexOf(u8, text, part) != null;
}
