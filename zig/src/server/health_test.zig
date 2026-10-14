//! The open /health body: its keys in order, the release, backend and GPU after them, and valid JSON.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const model_text = @import("model_text.zig");
const server = @import("server.zig");
const status_routes = @import("status_routes.zig");
const Allocator = std.mem.Allocator;

/// The keys an open /health sent before it named the release, backend and GPU.
const earlier = [_][]const u8{ "status", "model", "model_ids", "max_batch_size", "warming", "memory", "live" };

/// Four lanes, some memory, nothing running.
const Idle = struct {
    fn engine(e: *Idle) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{ .lanes = 4 };
    }

    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        return error.Closed;
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return .{ .active = 3, .cache = 2, .peak = 5 };
    }
};

/// A template-free vocabulary of bytes.
const Bytes = struct {
    fn text(t: *Bytes) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }

    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }

    fn decode(_: *anyopaque, a: Allocator, _: []const u32) Allocator.Error![]u8 {
        return a.dupe(u8, "");
    }

    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
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

/// The open /health body a server with ``identity`` sends, parsed back as a client reads it.
fn served(a: Allocator, identity: server.Identity) !std.json.ObjectMap {
    var engine: Idle = .{};
    var text: Bytes = .{};
    const srv = try server.Server.init(std.testing.allocator, std.testing.io, engine.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .identity = identity }, null);
    defer srv.deinit();
    const body = try json.stringify(a, try status_routes.openHealth(srv, a, false), .{});
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, body, .{});
    return doc.object;
}

test "the open health names the release, backend and GPU after its earlier keys, unchanged" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const o = try served(arena.allocator(), .{ .version = "9.8.7", .backend = "metal", .device = "Example \"GPU\" 1" });
    try std.testing.expectEqual(earlier.len + 3, o.count());
    for (earlier ++ [_][]const u8{ "version", "backend", "device" }, o.keys()) |want, got| try std.testing.expectEqualStrings(want, got);
    try std.testing.expectEqualStrings("9.8.7", o.get("version").?.string);
    try std.testing.expectEqualStrings("metal", o.get("backend").?.string);
    try std.testing.expectEqualStrings("Example \"GPU\" 1", o.get("device").?.string);
    try std.testing.expectEqual(@as(i64, 4), o.get("max_batch_size").?.integer);
    try std.testing.expectEqual(@as(i64, 5), o.get("memory").?.object.get("peak").?.integer);
}

test "the open health leaves out what is unknown, and an engine-less build sends only the earlier keys" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const none = try served(arena.allocator(), .{});
    try std.testing.expectEqual(earlier.len, none.count());
    for (earlier, none.keys()) |want, got| try std.testing.expectEqualStrings(want, got);
    const no_gpu = try served(arena.allocator(), .{ .version = "9.8.7", .backend = "cuda" });
    try std.testing.expectEqual(earlier.len + 2, no_gpu.count());
    try std.testing.expect(no_gpu.get("device") == null);
    try std.testing.expectEqualStrings("cuda", no_gpu.get("backend").?.string);
}
