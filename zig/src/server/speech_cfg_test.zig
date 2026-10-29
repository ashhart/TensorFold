//! Exercise real speech routing, request frames and response headers with a deterministic engine sink.
const std = @import("std");
const harness = @import("speech_stream_test.zig");
const cfg = @import("drama_cfg.zig");
const voice = @import("drama_voice.zig");
const family = @import("drama_prompt");
const gpa = std.testing.allocator;

test "whole and streamed reference CFG carries only the plain prompt; default stays unguided" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var text: harness.Bytes = .{};
    const guide = try cfg.guide(a, text.text(), .{ .gamma = 1.5 }, 1.5, "hi", true);
    const codes = [_][10]i32{.{ 7, 1, 2, 3, 4, 5, 6, 7, 8, 9 }};
    const refs = [_]family.Reference{.{ .transcript = "A small cloud passes.", .codes = &codes }};
    const voices = [_]voice.Voice{.{ .name = "tone", .references = &refs }};
    const Case = struct { options: ?cfg.Options, header: []const u8, referenced: bool, gamma: f32, guided: bool, says: ?[]const u8 };
    const cases = [_]Case{
        .{ .options = null, .header = "X-Cfg-Gamma: nonsense\r\n", .referenced = true, .gamma = 1, .guided = false, .says = null },
        .{ .options = .{ .gamma = 1.5 }, .header = "X-Cfg-Gamma: nonsense\r\n", .referenced = true, .gamma = 1.5, .guided = true, .says = "X-Cfg-Gamma: 1.5\r\n" },
        .{ .options = .{ .gamma = 1.5 }, .header = "", .referenced = false, .gamma = 1, .guided = false, .says = "X-Cfg-Gamma: 1.5\r\n" },
        .{ .options = .{ .gamma = 1.5, .request_gamma = true }, .header = "X-Cfg-Gamma: 1\r\n", .referenced = true, .gamma = 1, .guided = false, .says = "X-Cfg-Gamma: 1\r\n" },
        .{ .options = .{ .gamma = 1.5, .request_gamma = true }, .header = "X-Cfg-Gamma: 3\r\n", .referenced = true, .gamma = 3, .guided = true, .says = "X-Cfg-Gamma: 3\r\n" },
        .{ .options = .{ .force_guided = true }, .header = "", .referenced = true, .gamma = 1, .guided = true, .says = "X-Cfg-Gamma: 1\r\n" },
    };
    for ([_]bool{ false, true }) |stream| for (cases) |case| {
        const input = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"input\":\"hi\",\"voice\":\"{s}\",\"response_format\":\"pcm\",\"stream\":{s}}}", .{ if (case.referenced) "tone" else "default", if (stream) "true" else "false" });
        var script: harness.Script = .{ .steps = &.{harness.frame}, .inline_ = true, .voices = &voices, .expected_unconditioned = if (case.guided) guide.rows else &.{} };
        var served: harness.Served = undefined;
        try served.open(&script, .{ .served_name = "m", .model_ids = &.{"m"}, .cfg = case.options }, case.header, "HTTP/1.1", input, 2000);
        defer served.deinit();
        const head = try served.readHead();
        const success = std.mem.indexOf(u8, head, " 200 ") != null;
        const header_ok = if (case.says) |says| std.mem.indexOf(u8, head, says) != null else std.mem.indexOf(u8, head, "X-Cfg-Gamma") == null;
        _ = try served.readRest();
        served.join(&script);
        try std.testing.expect(success and header_ok);
        try std.testing.expectEqual(@as(usize, 1), script.takes);
        try std.testing.expectEqual(case.gamma, script.cfg_gamma);
        try std.testing.expect(script.unconditioned_equal);
    };
}

test "invalid opted-in CFG header is refused before engine submission" {
    for ([_][]const u8{ "nan", "inf", "0.5", "6", "wrong" }) |bad| {
        var header_buf: [64]u8 = undefined;
        const header = try std.fmt.bufPrint(&header_buf, "X-Cfg-Gamma: {s}\r\n", .{bad});
        var script: harness.Script = .{ .steps = &.{}, .inline_ = true };
        var served: harness.Served = undefined;
        try served.open(&script, .{ .served_name = "m", .model_ids = &.{"m"}, .cfg = .{ .request_gamma = true } }, header, "HTTP/1.1", "{\"model\":\"m\",\"input\":\"hi\"}", 2000);
        defer served.deinit();
        _ = try served.readHead();
        const body = try served.readRest();
        served.join(&script);
        try std.testing.expectEqual(@as(usize, 0), script.takes);
        try std.testing.expect(std.mem.indexOf(u8, body, "X-Cfg-Gamma must be a number") != null);
    }
}
