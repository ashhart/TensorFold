const std = @import("std");
const log = @import("log.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Cx = @import("errors.zig").Cx;
const Metadata = @import("speech_metadata.zig").Metadata;

const Capture = struct {
    bytes: [256]u8 = undefined,
    len: usize = 0,

    fn line(ctx: *anyopaque, text: []const u8) void {
        const self: *Capture = @ptrCast(@alignCast(ctx));
        @memcpy(self.bytes[self.len..][0..text.len], text);
        self.len += text.len;
    }
};

pub fn check(comptime redraw: anytype) !void {
    var srv: Server = undefined;
    srv.config = .{ .served_name = "m", .model_ids = &.{"m"}, .redraw = .{ .register = .{ .st = 6, .high = 80, .high_hz = 250 } } };
    var conn: Conn = undefined;
    var cx: Cx = .{ .a = std.testing.allocator };
    var spare: []f32 = &.{};
    var capture: Capture = .{};
    log.Test.sink = .{ .ctx = &capture, .call = Capture.line };
    defer log.Test.sink = null;
    for ([_]f32{ std.math.nan(f32), 0 }, [_][]const u8{ "NonFinite", "no voiced phrase comparison" }) |sample, reason| {
        capture.len = 0;
        var pcm = [_]f32{sample};
        var metadata: Metadata = .{ .enabled = false, .register_on = true };
        const take = redraw(&srv, &conn, &cx, "register-test", .{ .sample_rate = 44100, .frame_length = 2048, .max_frames = 1024, .max_prompt_tokens = 4096 }, .{ .input = "hi", .prompt = &.{}, .sampling = .{ .seed = 0 }, .format = .pcm, .stream = false }, 0, undefined, 0, .{ .pcm = &pcm, .reason = .stop }, &spare, &metadata).?;
        try std.testing.expect(take.jump == null);
        try std.testing.expect(take.pcm.ptr == &pcm);
        try std.testing.expect(!metadata.register_tripped and !metadata.register_redrawn);
        try std.testing.expectEqual(@as(usize, 0), spare.len);
        var expected: [256]u8 = undefined;
        try std.testing.expectEqualStrings(try std.fmt.bufPrint(&expected, "[tensorfold] speech register-test register unmeasured: {s}\n", .{reason}), capture.bytes[0..capture.len]);
    }
}
