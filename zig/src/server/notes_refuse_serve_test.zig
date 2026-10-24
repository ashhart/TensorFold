//! serve.run's --notes refusals: a writer or engine erase cannot reach stops startup before the store opens.
const std = @import("std");
const posix = @import("sys").posix;
const serve = @import("serve.zig");
const check = @import("notes/tokenizer_check.zig");
const drain = @import("drain_test.zig");
const Echo = @import("notes_storage_http_test.zig").Echo;
const Bytes = @import("notes_storage_bytes_test.zig").Bytes;
const testing = std.testing;

const Started = struct {
    code: u8 = 255,
    fn run(x: *Started, args: @import("cli.zig").Args, setup: serve.Setup) void {
        x.code = serve.run(testing.allocator, testing.io, args, setup);
    }
};

fn onListen(port: u16) void {
    drain.listen_port.store(port, .release);
}

test "--notes is refused with a writer or an engine erase cannot clear, before its journal opens" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    const path = try std.fs.path.join(a, &.{ dir, "snap.json" });
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data = "{}" });
    var problem: []const u8 = "";
    const sha = try check.tokenizerSha(a, io, dir, 256, &problem);
    const tokenizer = try std.fmt.allocPrint(a, "\"tokenizer\":{{\"name\":\"bytes\",\"sha256\":\"{s}\"}}", .{sha});
    const snapshot = try std.mem.replaceOwned(u8, a, @embedFile("notes/testdata/l6_bytes_snapshot.json"), "\"tokenizer\":{\"name\":\"bytes\"}", tokenizer);
    try tmp.dir.writeFile(io, .{ .sub_path = "snap.json", .data = snapshot });
    var env = std.process.Environ.Map.init(testing.allocator);
    defer env.deinit();
    try env.put("TENSORFOLD_NO_LIVE", "1");
    var bytes: Bytes = .{};
    const args: @import("cli.zig").Args = .{ .host = "127.0.0.1", .port = 0, .notes = path };

    // an engine that can clear its states starts and takes a write, so the journal exists
    var clear: Echo = .{ .cache = true, .supports_forget = true };
    try accepted(a, args, .{ .engine = clear.engine(), .text = bytes.text(), .served = "echo", .model_dir = dir, .environ = &env, .on_listen = onListen }, true);
    // each refusal comes before the store opens: a start that opened it would append its run to the journal
    const size = (try tmp.dir.statFile(io, "snap.json.journal", .{})).size;
    // a request log keeps chat bodies erase does not search: refused before the store opens
    {
        try env.put("TENSORFOLD_REQUEST_LOG", try std.fs.path.join(a, &.{ dir, "requests.jsonl" }));
        defer _ = env.swapRemove("TENSORFOLD_REQUEST_LOG");
        var echo: Echo = .{};
        try testing.expectEqual(@as(u8, 2), serve.run(testing.allocator, io, args, .{ .engine = echo.engine(), .text = bytes.text(), .served = "echo", .model_dir = dir, .environ = &env }));
        try testing.expectEqual(size, (try tmp.dir.statFile(io, "snap.json.journal", .{})).size);
    }
    // an engine with a prompt cache and no prompt-state erase: refused
    {
        var echo: Echo = .{ .cache = true };
        try testing.expectEqual(@as(u8, 2), serve.run(testing.allocator, io, args, .{ .engine = echo.engine(), .text = bytes.text(), .served = "echo", .model_dir = dir, .environ = &env }));
        try testing.expectEqual(size, (try tmp.dir.statFile(io, "snap.json.journal", .{})).size);
    }
    // a peer that does not acknowledge dropped states: refused
    {
        var echo: Echo = .{ .cache = true, .supports_forget = true, .forgotten = .{ .held = 1 } };
        try testing.expectEqual(@as(u8, 2), serve.run(testing.allocator, io, args, .{ .engine = echo.engine(), .text = bytes.text(), .served = "echo", .model_dir = dir, .environ = &env }));
        try testing.expectEqual(size, (try tmp.dir.statFile(io, "snap.json.journal", .{})).size);
    }
    // the control: an accepted start opens the store and appends its run, so the size check can fail
    try accepted(a, args, .{ .engine = clear.engine(), .text = bytes.text(), .served = "echo", .model_dir = dir, .environ = &env, .on_listen = onListen }, false);
    try testing.expect((try tmp.dir.statFile(io, "snap.json.journal", .{})).size > size);
}

/// serve.run on its own thread until it listens; it lists notes (and writes one), then SIGTERM stops it cleanly.
fn accepted(a: std.mem.Allocator, args: @import("cli.zig").Args, setup: serve.Setup, write: bool) !void {
    const io = testing.io;
    drain.listen_port.store(0, .release);
    var x: Started = .{};
    const t = try std.Thread.spawn(.{}, Started.run, .{ &x, args, setup });
    while (drain.listen_port.load(.acquire) == 0 and x.code == 255) std.Io.sleep(io, .fromMilliseconds(2), .awake) catch {};
    try testing.expectEqual(@as(u8, 255), x.code); // listening, not refused
    const port = drain.listen_port.load(.acquire);
    try testing.expectEqual(@as(u16, 200), (try drain.request(a, io, port, "GET", "/v1/notes", "")).status);
    if (write) try testing.expectEqual(@as(u16, 201), (try drain.request(a, io, port, "POST", "/v1/notes", "{\"text\":\"My favourite colour is green.\"}")).status);
    try posix.raise(.TERM);
    t.join();
    try testing.expectEqual(@as(u8, 0), x.code);
}
