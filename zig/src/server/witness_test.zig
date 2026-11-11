//! Execution witness parts: what each one binds, the startup refusal and the flags outside the first path.
const std = @import("std");
const cli = @import("cli.zig");
const witness = @import("witness.zig");
const parts = @import("witness_parts.zig");
const fixture = @import("witness_fixture.zig");
const ByteText = @import("embedding_protocol_test.zig").ByteText;
const t = std.testing;

fn parse(a: std.mem.Allocator, argv: []const []const u8) !cli.Args {
    var u: cli.Usage = .{};
    return cli.parse(a, argv, &u) catch |e| {
        std.debug.print("{s}\n", .{u.message});
        return e;
    };
}

test "execution witness: the flag is off by default and needs a key or the owner-only Unix socket" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const off = try parse(a, &.{"m"});
    try t.expect(!off.execution_witness);
    try t.expectEqual(@as(?[]const u8, null), witness.refusal(off, false));
    const on = try parse(a, &.{ "m", "--execution-witness" });
    try t.expect(on.execution_witness);
    try t.expect(std.mem.indexOf(u8, witness.refusal(on, false).?, "--unix-socket") != null);
    try t.expectEqual(@as(?[]const u8, null), witness.refusal(on, true));
    try t.expect(!witness.keysNamed(a, on, "") and !witness.keysNamed(a, on, " , "));
    try t.expect(witness.keysNamed(a, on, "sk-env") and witness.keysNamed(a, try parse(a, &.{ "m", "--api-key", "sk-flag" }), ""));
    try t.expect(witness.keysNamed(a, try parse(a, &.{ "m", "--api-key-file", "/tmp/keys" }), ""));
    if (@import("builtin").os.tag != .windows) {
        const socket = try parse(a, &.{ "m", "--execution-witness", "--unix-socket", "/tmp/tf.sock" });
        try t.expectEqual(@as(?[]const u8, null), witness.refusal(socket, false));
    }
}

test "execution witness: each flag outside the first path names itself, and the plain serve has none" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try t.expectEqual(@as(?[]const u8, null), witness.excluded(try parse(a, &.{ "m", "--execution-witness" })));
    const cases = [_]struct { flag: []const u8, argv: []const []const u8 }{
        .{ .flag = "--slide", .argv = &.{"--slide"} },
        .{ .flag = "--learn-conversations", .argv = &.{"--learn-conversations"} },
        .{ .flag = "--learn-conversations", .argv = &.{ "--prompt-cache-disk", "/tmp/kept" } },
        .{ .flag = "--learn", .argv = &.{"--learn"} },
        .{ .flag = "--learn", .argv = &.{ "--learn-dir", "/tmp/kept" } },
        .{ .flag = "--compact-at", .argv = &.{ "--compact-at", "auto" } },
        .{ .flag = "--compact-memory", .argv = &.{ "--compact-memory", "/tmp/notes" } },
        .{ .flag = "--cortheon", .argv = &.{"--cortheon"} },
        .{ .flag = "--recall", .argv = &.{"--recall"} },
        .{ .flag = "--drafter", .argv = &.{ "--drafter", "/tmp/drafter" } },
        .{ .flag = "--record", .argv = &.{ "--record", "/tmp/rows" } },
        .{ .flag = "--vision", .argv = &.{"--vision"} },
    };
    for (cases) |c| {
        const argv = try std.mem.concat(a, []const u8, &.{ &.{ "m", "--execution-witness" }, c.argv });
        try t.expectEqualStrings(c.flag, witness.excluded(try parse(a, argv)).?);
    }
    // the build-gated flags: set as their parsers set them
    for ([_]struct { []const u8, cli.Args }{ .{ "--spill-gib", .{ .spill_gib = 1 } }, .{ "--mtp-drafts", .{ .mtp_drafts = 2 } }, .{ "--tp", .{ .tp = 2 } }, .{ "--speed-up", .{ .speed_up = "a" } } }) |c|
        try t.expectEqualStrings(c[0], witness.excluded(c[1]).?);
    try t.expectEqual(@as(?[]const u8, null), witness.excluded(.{ .mtp_drafts = 0, .spill_gib = 0 }));
}

test "execution witness: a changed template or tokenizer file changes the text part, a change after loading is unknown" {
    var c = try fixture.Checkpoint.init();
    defer c.deinit();
    const io = t.io;
    const first = try parts.Text.stamp(t.allocator, io, c.path, null);
    defer first.deinit();
    const base = try first.digest(io, null);
    try t.expectEqualSlices(u8, &base, &try first.digest(io, null));
    try c.rewrite("chat_template.jinja", "{{ messages }}!");
    try t.expectError(error.FilesChanged, first.digest(io, null)); // the bytes loaded are not the bytes now on disk
    const template = try parts.Text.stamp(t.allocator, io, c.path, null);
    defer template.deinit();
    const changed = try template.digest(io, null);
    try t.expect(!std.mem.eql(u8, &base, &changed));
    try c.rewrite("tokenizer.json", "{\"model\":{\"vocab\":{\"b\":0}}}");
    const tokenizer = try parts.Text.stamp(t.allocator, io, c.path, null);
    defer tokenizer.deinit();
    try t.expect(!std.mem.eql(u8, &changed, &try tokenizer.digest(io, null)));
    // a file that appears after the load (an absent sidecar) is a change too
    try c.write("special_tokens_map.json", "{}");
    try t.expectError(error.FilesChanged, tokenizer.digest(io, null));
    // --chat-template's file counts under its flag
    try c.write("override.jinja", "{{ other }}");
    const override_path = try std.fs.path.join(t.allocator, &.{ c.path, "override.jinja" });
    defer t.allocator.free(override_path);
    const overridden = try parts.Text.stamp(t.allocator, io, c.path, override_path);
    defer overridden.deinit();
    const with_override = try overridden.digest(io, null);
    try c.rewrite("override.jinja", "{{ another }}");
    const again = try parts.Text.stamp(t.allocator, io, c.path, override_path);
    defer again.deinit();
    try t.expect(!std.mem.eql(u8, &with_override, &try again.digest(io, null)));
}

test "execution witness: the binary part is the open file's SHA-256, and a rewrite after opening is unknown" {
    var c = try fixture.Checkpoint.init();
    defer c.deinit();
    const io = t.io;
    try c.write("bin", "first build");
    try c.write("other", "second build");
    const bin = try std.fs.path.join(t.allocator, &.{ c.path, "bin" });
    defer t.allocator.free(bin);
    const other = try std.fs.path.join(t.allocator, &.{ c.path, "other" });
    defer t.allocator.free(other);
    const running = try parts.Binary.openPath(io, bin);
    defer running.close(io);
    var want: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash("first build", &want, .{});
    try t.expectEqualSlices(u8, &want, &try running.digest(io, null));
    var stopped: parts.Stop = .init(true);
    try t.expectError(error.Stopped, running.digest(io, &stopped)); // a shutdown ends the hash at its next block
    try t.expectError(error.Stopped, textDigest(io, c.path, &stopped));
    const second = try parts.Binary.openPath(io, other);
    defer second.close(io);
    try t.expect(!std.mem.eql(u8, &want, &try second.digest(io, null)));
    // an in-place rewrite of the open file is not the code that started
    try c.rewrite("other", "patched build");
    try t.expectError(error.FilesChanged, second.digest(io, null));
    // a path replaced by rename never reads as the new file: the open inode's bytes, or unknown if unlinking stamps it
    try c.write("next", "next build");
    try c.tmp.dir.rename("next", c.tmp.dir, "bin", io);
    if (running.digest(io, null)) |got| try t.expectEqualSlices(u8, &want, &got) else |e| try t.expectEqual(error.FilesChanged, e);
    const self = try parts.Binary.openRunning(t.allocator, io);
    defer self.close(io);
    _ = try self.digest(io, null);
}

test "execution witness: preparation follows the serving defaults, environment the settings but not secrets" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text = ByteText.model();
    const base: @import("server.zig").Config = .{ .served_name = "m", .model_ids = &.{"m"} };
    const first = try parts.preparation(a, base, text);
    try t.expectEqualSlices(u8, &first, &try parts.preparation(a, base, text));
    var longer = base;
    longer.default_max_tokens = 8192;
    try t.expect(!std.mem.eql(u8, &first, &try parts.preparation(a, longer, text)));
    var salted = base;
    salted.seed_salt = 1;
    try t.expect(!std.mem.eql(u8, &first, &try parts.preparation(a, salted, text)));
    var renamed = base;
    renamed.served_name = "another-name";
    try t.expectEqualSlices(u8, &first, &try parts.preparation(a, renamed, text));
    var env: std.process.Environ.Map = .init(a);
    try env.put("PATH", "/bin");
    try env.put("TF_QWEN_KV_PAGES", "1");
    try env.put("TENSORFOLD_API_KEY", "secret-one");
    const settings = try parts.environment(a, &env);
    try env.put("PATH", "/usr/bin");
    try env.put("TENSORFOLD_API_KEY", "secret-two");
    try env.put("BRAVE_SEARCH_API_KEY", "other");
    try t.expectEqualSlices(u8, &settings, &try parts.environment(a, &env));
    try env.put("TF_QWEN_KV_PAGES", "0");
    try t.expect(!std.mem.eql(u8, &settings, &try parts.environment(a, &env)));
    try env.put("TF_QWEN_KV_PAGES", "1");
    try env.put("TENSORFOLD_PREFILL_TOKENS", "512");
    try t.expect(!std.mem.eql(u8, &settings, &try parts.environment(a, &env)));
}

test "execution witness: content is restart-stable, the witness binds instance and epoch, and each part moves both" {
    const engine: @import("engine_api").execution_witness.Known = .{ .path = "test/fake", .digest = @splat(1) };
    const server: [witness.components.len - 1][32]u8 = .{ @splat(2), @splat(3), @splat(4), @splat(5) };
    const a = witness.compose(@splat(9), 1, engine, server);
    const restarted = witness.compose(@splat(8), 1, engine, server);
    try t.expectEqualSlices(u8, &a.content, &restarted.content);
    try t.expect(!std.mem.eql(u8, &a.witness, &restarted.witness));
    try t.expect(!std.mem.eql(u8, &a.witness, &witness.compose(@splat(9), 2, engine, server).witness));
    for (0..witness.components.len) |i| {
        var e = engine;
        var s = server;
        if (i == 0) e.digest[0] ^= 1 else s[i - 1][0] ^= 1;
        const moved = witness.compose(@splat(9), 1, e, s);
        try t.expect(!std.mem.eql(u8, &a.content, &moved.content));
        try t.expect(!std.mem.eql(u8, &a.witness, &moved.witness));
    }
    var other = engine;
    other.path = "test/other";
    try t.expect(!std.mem.eql(u8, &a.content, &witness.compose(@splat(9), 1, other, server).content));
}

fn textDigest(io: std.Io, dir: []const u8, stop: *const parts.Stop) ![32]u8 {
    const text = try parts.Text.stamp(t.allocator, io, dir, null);
    defer text.deinit();
    return text.digest(io, stop);
}
