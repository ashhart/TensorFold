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
        .think_budget = request.think_budget,
        .think_close = request.think_close,
        .think_end = if (request.think_end) |end| end else -1,
    });
    defer stream.deinit(a);
    var start: usize = 0;
    while (!stream.finished) {
        var forced: [1]u32 = undefined;
        var tokens: []const u32 = &.{};
        var scripted = false;
        if (stream.popForce()) |token| {
            forced[0] = token;
            tokens = &forced;
        } else if (stream.cutsNext()) {
            forced[0] = try stream.startClose(a);
            tokens = &forced;
            if (start < reply.len) start += 1;
        } else {
            if (start >= reply.len) break;
            const end = @min(start + stride, reply.len);
            tokens = reply[start..end];
            if (stream.thinkCut(tokens)) |cut| tokens = tokens[0..cut];
            scripted = true;
        }
        const before = stream.emitted().len;
        const landed = try stream.commit(a, tokens, &.{});
        if (landed > 0) sink.event(sink.ctx, id, &.{ .tokens = stream.emitted()[before..] });
        if (scripted) start += landed;
    }
    sink.event(sink.ctx, id, &.{ .finished = .{ .reason = if (stream.reason == .stop) .stop else .length } });
}
