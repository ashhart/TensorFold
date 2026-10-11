//! A server init that fails after its idle keepalive started stops that ticker and frees all it took.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const model_text = @import("model_text.zig");
const server = @import("server.zig");
const Allocator = std.mem.Allocator;

/// Ticks of the fake queue; a file-level count, so a ticker a failed init left running touches no freed memory.
var ticks: std.atomic.Value(u32) = .init(0);

/// An engine with a keepalive target, as a Metal host has, and nothing to serve.
const Queue = struct {
    fn engine(q: *Queue) api.Engine {
        return .{ .ctx = q, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .keepalive = keepalive } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{};
    }

    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        return error.Closed;
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }

    fn keepalive(ctx: *anyopaque) ?api.keepalive.Target {
        return .{ .ctx = ctx, .tick = tick };
    }

    fn tick(_: *anyopaque) void {
        _ = ticks.fetchAdd(1, .monotonic);
    }
};

/// A vocabulary with `</think>` whose encoder fails: init's think-close step, after the keepalive starts, fails.
const BrokenText = struct {
    fn text(t: *BrokenText) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }

    fn encode(_: *anyopaque, _: Allocator, _: []const u8, _: bool) model_text.Error![]u32 {
        return error.Template;
    }

    fn decode(_: *anyopaque, a: Allocator, _: []const u32) Allocator.Error![]u8 {
        return a.dupe(u8, "");
    }

    fn tokenId(_: *anyopaque, piece: []const u8) ?u32 {
        return if (std.mem.eql(u8, piece, "</think>")) 7 else null;
    }

    fn tokenString(_: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }

    fn render(_: *anyopaque, a: Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        return a.dupe(u8, "prompt");
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

test "an init that fails after the keepalive started stops its thread and frees everything it allocated" {
    // counted over c_allocator: a ticker left running keeps memory nothing frees, so the counts differ
    var counted = std.testing.FailingAllocator.init(std.heap.c_allocator, .{});
    var queue: Queue = .{};
    var text: BrokenText = .{};
    const config: server.Config = .{ .served_name = "m", .model_ids = &.{"m"}, .keep_warm_s = 900 };
    try std.testing.expectError(error.Template, server.Server.init(counted.allocator(), std.testing.io, queue.engine(), text.text(), config, null));
    try std.testing.expect(counted.allocations > 0); // the server, its arena and the keepalive were taken
    // Keepalive.stop joins the ticker's thread before it frees the ticker: every byte back means the thread is gone
    try std.testing.expectEqual(counted.allocated_bytes, counted.freed_bytes);
    try std.testing.expectEqual(counted.allocations, counted.deallocations);
}
