//! `tensorfold log` over records the reader must not mistake for the writer's own: added fields and bare values.
const std = @import("std");
const log = @import("journal_log.zig");
const t = std.testing;
const Allocator = std.mem.Allocator;

/// The null stamp the request log's records get: its writer keeps no time.
const no_stamp = "-------------------";

fn runLog(a: Allocator, o: log.Options) !struct { u8, []const u8, []const u8 } {
    var out: std.Io.Writer.Allocating = .init(a);
    var err: std.Io.Writer.Allocating = .init(a);
    const code = try log.runWith(a, t.io, &out.writer, &err.writer, null, o);
    return .{ code, out.written(), err.written() };
}

/// One request and its reply, each carrying a field the reader has never seen on that shape.
fn addedFieldPair(a: Allocator, tmp: t.TmpDir) ![]const u8 {
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const path = try std.fmt.allocPrint(a, "{s}/requests.jsonl", .{dir});
    const data = try std.mem.concat(a, u8, &.{
        "{\"type\":\"request\",\"id\":\"chatcmpl-11\",\"body\":{\"model\":\"m11\"},\"route\":\"/v1/chat/completions\"}\n",
        "{\"type\":\"reply\",\"id\":\"chatcmpl-11\",\"status\":200,\"stream\":false,\"finish_reason\":\"stop\",\"text\":\"Nine.\",\"usage\":{\"prompt_tokens\":3,\"cached_tokens\":1,\"completion_tokens\":2},\"timings\":{\"queue_ms\":0.2,\"ttft_ms\":null,\"decode_tps\":30},\"truncated\":false,\"cache_hits\":7}\n",
    });
    try std.Io.Dir.cwd().writeFile(t.io, .{ .sub_path = path, .data = data });
    return path;
}

test "log: a field the reader does not know on either shape still reads as that shape" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try addedFieldPair(a, tmp);
    const code, const out, const err = try runLog(a, .{ .path = path });
    try t.expectEqual(@as(u8, 0), code);
    try t.expectEqualStrings("", err);
    // the pair still pairs: a field the reader does not know must not turn either record into a bare body
    try t.expectEqualStrings(no_stamp ++ "  chatcmpl-11  -  m11  200 finished  prompt 3  out 2  ttft -  30.0 tok/s\n", out);
    _, const shown, _ = try runLog(a, .{ .command = .show, .id = "chatcmpl-11", .path = path });
    try t.expect(std.mem.indexOf(u8, shown, "  text: Nine.") != null);
}

test "log: the record readers answer null for a value that is not an object" {
    const scalar: std.json.Value = .{ .integer = 1 };
    try t.expect(log.str(scalar, "model") == null);
    try t.expect(log.int(scalar, "status") == null);
    try t.expect(log.num(scalar, "ttft_ms") == null);
}
