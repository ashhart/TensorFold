//! A fake denoiser that gates each step on its actual HTTP progress write, with borrowed preview scratch.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const root = @import("root.zig");
const http = @import("http_conn.zig");
const Bytes = @import("speech_stream_test.zig").Bytes;
const t = std.testing;

pub const Fake = struct {
    calls: usize = 0,
    progress: bool = true,
    previews: bool = true,
    gate: bool = true,
    inline_: bool = false,
    ignore_cancel: bool = false,
    reject: ?api.ImageSubmitError = null,
    reject_at: ?usize = null,
    fail_step: ?u32 = null,
    early_stop: ?u32 = null,
    invalid_step: bool = false,
    oversized_preview: bool = false,
    missing_preview: bool = false,
    leave_after: ?u32 = null,
    delivered: std.atomic.Value(u32) = .init(0),
    steps_done: std.atomic.Value(u32) = .init(0),
    preview_reads: std.atomic.Value(u32) = .init(0),
    finished: std.atomic.Value(bool) = .init(false),
    cancelled: std.atomic.Value(bool) = .init(false),
    live: std.atomic.Value(bool) = .init(false),
    gate_timeout: std.atomic.Value(bool) = .init(false),
    cancels: usize = 0,
    thread: ?std.Thread = null,
    observed: bool = false,
    seed: u64 = 0,

    pub fn engine(f: *Fake) api.Engine {
        return .{ .ctx = f, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .submit_image = image } };
    }
    fn cast(ctx: *anyopaque) *Fake {
        return @ptrCast(@alignCast(ctx));
    }
    fn info(ctx: *anyopaque) api.Info {
        const f = cast(ctx);
        const cap: api.ImageInfo = .{ .max_side = 1536, .max_pixels = 1572864, .max_steps = 200, .max_prompt_tokens = 4096, .progress = f.progress, .previews = f.previews };
        return .{ .text_generation = false, .image = cap };
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
    fn cancel(ctx: *anyopaque, _: api.Id) void {
        const f = cast(ctx);
        f.cancels += 1;
        if (!f.ignore_cancel) f.cancelled.store(true, .release);
    }
    fn image(ctx: *anyopaque, id: api.Id, request: *const api.ImageRequest, sink: api.Sink) api.ImageSubmitError!void {
        const f = cast(ctx);
        f.calls += 1;
        if (f.reject) |e| if (f.reject_at == null or f.reject_at == f.calls) return e;
        f.join();
        f.seed = request.seed;
        f.delivered.store(0, .release);
        f.steps_done.store(0, .release);
        f.preview_reads.store(0, .release);
        f.finished.store(false, .release);
        f.cancelled.store(false, .release);
        f.observed = request.observer != null;
        if (f.inline_) return work(f, id, request, sink);
        f.thread = std.Thread.spawn(.{}, work, .{ f, id, request, sink }) catch return error.Busy;
    }
    fn work(f: *Fake, id: api.Id, request: *const api.ImageRequest, sink: api.Sink) void {
        var reason: api.Reason = .stop;
        for (1..request.steps + 1) |n| {
            const step: u32 = @intCast(n);
            if (f.cancelled.load(.acquire)) {
                reason = .cancelled;
                break;
            }
            f.steps_done.store(step, .release);
            var preview: [12]u8 = undefined;
            @memcpy(preview[0..8], "\x89PNG\r\n\x1a\n");
            std.mem.writeInt(u32, preview[8..12], step, .little);
            const due = request.wantsPreview(step);
            if (due) _ = f.preview_reads.fetchAdd(1, .monotonic);
            const keep = request.reportStep(if (f.invalid_step) 0 else step, if (due and !f.missing_preview) .{ .png = &preview, .width = if (f.oversized_preview) 129 else 64, .height = 64 } else null);
            @memset(&preview, 0xa5);
            if (!keep) {
                reason = .cancelled;
                break;
            }
            if (f.observed and f.gate) {
                var tries: usize = 0;
                while (f.delivered.load(.acquire) < step and !f.cancelled.load(.acquire)) : (tries += 1) {
                    if (tries == 2000) {
                        f.gate_timeout.store(true, .release);
                        break;
                    }
                    std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
                }
            }
            if (f.fail_step == step) {
                reason = .failed;
                break;
            }
            if (f.early_stop == step) break;
        }
        if (f.cancelled.load(.acquire)) reason = .cancelled;
        if (reason == .stop) {
            var hash: std.crypto.hash.sha2.Sha256 = .init(.{});
            hash.update(std.mem.sliceAsBytes(request.prompt));
            hash.update(std.mem.sliceAsBytes(request.negative));
            inline for (.{ "seed", "width", "height", "steps", "guidance" }) |field| hash.update(std.mem.asBytes(&@field(request, field)));
            const bytes = hash.finalResult();
            request.png.appendSlice(request.alloc, "\x89PNG\r\n\x1a\n") catch {};
            request.png.appendSlice(request.alloc, &bytes) catch {};
        }
        request.result.* = .{ .bytes = request.png.items.len, .reason = reason };
        f.finished.store(true, .release);
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = reason, .message = if (reason == .failed) "fake denoising failed" else "" } });
    }
    pub fn join(f: *Fake) void {
        if (f.thread) |thread| thread.join();
        f.thread = null;
    }
    fn gone(ctx: *anyopaque) bool {
        const f = cast(ctx);
        return if (f.leave_after) |step| f.delivered.load(.acquire) >= step else false;
    }
};

const Wire = struct {
    a: std.mem.Allocator,
    fake: *Fake,
    fail_step: ?u32,
    fail_index: ?usize,
    fail_head: bool,
    bytes: std.ArrayList(u8) = .empty,
    fn write(ctx: *anyopaque, bytes: []const u8) error{Closed}!void {
        const w: *Wire = @ptrCast(@alignCast(ctx));
        if (w.fail_head and std.mem.startsWith(u8, bytes, "HTTP/")) return error.Closed;
        w.bytes.appendSlice(w.a, bytes) catch return error.Closed;
        if (std.mem.startsWith(u8, bytes, "event: image_generation.progress\n")) {
            const at = std.mem.indexOf(u8, bytes, "data: ").? + 6;
            const value = (json.parse(w.a, bytes[at..]) catch return error.Closed).ok;
            const step: u32 = @intCast(value.get("step").?.int64().?);
            const index: usize = @intCast(value.get("index").?.int64().?);
            if (!w.fake.finished.load(.acquire)) w.fake.live.store(true, .release);
            w.fake.delivered.store(step, .release);
            if (w.fail_step == step and (w.fail_index == null or w.fail_index == index)) return error.Closed;
        }
    }
};

pub const Reply = struct { status: u16, body: []const u8, wire: []const u8 };
pub fn ask(a: std.mem.Allocator, fake: *Fake, body: []const u8, fail_step: ?u32, fail_head: bool) !Reply {
    return askAt(a, fake, body, fail_step, fail_head, null);
}
pub fn askAt(a: std.mem.Allocator, fake: *Fake, body: []const u8, fail_step: ?u32, fail_head: bool, fail_index: ?usize) !Reply {
    var text: Bytes = .{};
    const srv = try root.Server.init(t.allocator, t.io, fake.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
    defer srv.deinit();
    defer fake.join();
    var wire: Wire = .{ .a = a, .fake = fake, .fail_step = fail_step, .fail_index = fail_index, .fail_head = fail_head };
    var headers = [_]http.Header{.{ .name = "Content-Length", .value = try std.fmt.allocPrint(a, "{d}", .{body.len}) }};
    var conn: http.Conn = .{ .fd = -1, .peer = "", .buf = try a.dupe(u8, body), .end = body.len, .gpa = a, .method = "POST", .path = "/v1/images/generations", .version = "HTTP/1.1", .headers = &headers, .write_hooks = .{ .ctx = &wire, .write = Wire.write }, .gone_hook = .{ .ctx = fake, .call = Fake.gone } };
    defer conn.deinit();
    @import("routes.zig").dispatch(srv, &conn, a);
    const split = std.mem.indexOf(u8, wire.bytes.items, "\r\n\r\n");
    return .{ .status = if (split != null) try std.fmt.parseInt(u16, wire.bytes.items[9..12], 10) else 0, .body = if (split) |at| wire.bytes.items[at + 4 ..] else "", .wire = wire.bytes.items };
}
