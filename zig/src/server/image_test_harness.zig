//! Image-route tests using the production connection hooks and fake one-picture engines, without sockets.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const root = @import("root.zig");
const Conn = @import("http_conn.zig").Conn;
const Bytes = @import("speech_stream_test.zig").Bytes;

pub const caps: api.ImageInfo = .{ .max_side = 1536, .max_pixels = 1572864, .max_steps = 200, .max_prompt_tokens = 4096 };
pub const Stub = struct {
    calls: usize = 0,
    finished: usize = 0,
    active: usize = 0,
    maximum_active: usize = 0,
    cancels: usize = 0,
    seeds: [4]u64 = @splat(0),
    reject_at: ?usize = null,
    reject: api.ImageSubmitError = error.InvalidRequest,
    fail_at: ?usize = null,
    cancel_at: ?usize = null,
    hold_at: ?usize = null,
    leave_at: ?usize = null,
    pending: ?struct { id: api.Id, request: *const api.ImageRequest, sink: api.Sink } = null,

    pub fn engine(s: *Stub) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .submit_image = image } };
    }
    fn cast(ctx: *anyopaque) *Stub {
        return @ptrCast(@alignCast(ctx));
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .text_generation = false, .image = caps };
    }
    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        return error.Closed;
    }
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
    fn cancel(ctx: *anyopaque, id: api.Id) void {
        const s = cast(ctx);
        s.cancels += 1;
        if (s.pending) |p| {
            std.debug.assert(id == p.id);
            s.pending = null;
            s.active -= 1;
            s.finished += 1;
            p.request.result.* = .{ .reason = .cancelled };
            p.sink.event(p.sink.ctx, id, &.{ .finished = .{ .reason = .cancelled } });
        }
    }
    fn image(ctx: *anyopaque, id: api.Id, request: *const api.ImageRequest, sink: api.Sink) api.ImageSubmitError!void {
        const s = cast(ctx);
        s.calls += 1;
        if (s.reject_at == s.calls) return s.reject;
        if (s.calls > s.seeds.len or s.active != 0) return error.Busy;
        s.seeds[s.calls - 1] = request.seed;
        s.active += 1;
        s.maximum_active = @max(s.maximum_active, s.active);
        if (s.hold_at == s.calls) {
            s.pending = .{ .id = id, .request = request, .sink = sink };
            return;
        }
        var hash: std.crypto.hash.sha2.Sha256 = .init(.{});
        hash.update(std.mem.sliceAsBytes(request.prompt));
        hash.update(std.mem.sliceAsBytes(request.negative));
        inline for (.{ "width", "height", "steps", "guidance", "seed" }) |field| hash.update(std.mem.asBytes(&@field(request, field)));
        const bytes = hash.finalResult();
        request.png.appendSlice(request.alloc, "\x89PNG\r\n\x1a\n") catch return error.Busy;
        request.png.appendSlice(request.alloc, &bytes) catch return error.Busy;
        const reason: api.Reason = if (s.fail_at == s.calls) .failed else if (s.cancel_at == s.calls) .cancelled else .stop;
        request.result.* = .{ .bytes = request.png.items.len, .reason = reason };
        s.active -= 1;
        s.finished += 1;
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = reason, .message = if (reason == .failed) "fake image pass failed" else "" } });
    }
    pub fn gone(ctx: *anyopaque) bool {
        const s = cast(ctx);
        return if (s.leave_at) |after| s.calls >= after else false;
    }
};

pub const Wire = struct {
    a: std.mem.Allocator,
    bytes: std.ArrayList(u8) = .empty,
    pub fn write(ctx: *anyopaque, bytes: []const u8) error{Closed}!void {
        const w: *Wire = @ptrCast(@alignCast(ctx));
        w.bytes.appendSlice(w.a, bytes) catch return error.Closed;
    }
};

pub const Reply = struct { status: u16, value: json.Value, bytes: []const u8 };

pub fn ask(a: std.mem.Allocator, stub: *Stub, body: []const u8) !Reply {
    var text: Bytes = .{};
    const srv = try root.Server.init(std.testing.allocator, std.testing.io, stub.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
    defer srv.deinit();
    var wire: Wire = .{ .a = a };
    var headers = [_]@import("http_conn.zig").Header{.{ .name = "Content-Length", .value = try std.fmt.allocPrint(a, "{d}", .{body.len}) }};
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = try a.dupe(u8, body), .end = body.len, .gpa = a, .method = "POST", .path = "/v1/images/generations", .version = "HTTP/1.1", .headers = &headers, .write_hooks = .{ .ctx = &wire, .write = Wire.write }, .gone_hook = .{ .ctx = stub, .call = Stub.gone } };
    defer conn.deinit();
    @import("routes.zig").dispatch(srv, &conn, a);
    const split = std.mem.indexOf(u8, wire.bytes.items, "\r\n\r\n") orelse return error.NoReply;
    return .{ .status = try std.fmt.parseInt(u16, wire.bytes.items[9..12], 10), .value = (try json.parse(a, wire.bytes.items[split + 4 ..])).ok, .bytes = wire.bytes.items };
}
