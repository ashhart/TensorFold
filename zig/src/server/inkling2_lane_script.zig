//! Scripted tokens through the actual lane commit/EOS logic, with the route's actual request ends.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");

pub fn submit(a: std.mem.Allocator, id: api.Id, request: *const api.Request, sink: api.Sink, reply: []const u32, stride: usize) !void {
    var stream = try lanes.Stream.init(a, .{
        .id = "inkling-required-stop",
        .prompt = request.prompt,
        .max_new = request.max_tokens,
        .eos = request.eos,
        .drafts = false,
    });
    defer stream.deinit(a);
    var start: usize = 0;
    while (start < reply.len and !stream.finished) {
        const end = @min(start + stride, reply.len);
        const before = stream.emitted().len;
        const landed = try stream.commit(a, reply[start..end], &.{});
        if (landed > 0) sink.event(sink.ctx, id, &.{ .tokens = stream.emitted()[before..] });
        start += landed;
    }
    sink.event(sink.ctx, id, &.{ .finished = .{ .reason = if (stream.reason == .stop) .stop else .length } });
}
