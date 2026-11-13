//! One image job whose step observer feeds SSE; request storage stays owned until its finished event.
const api = @import("engine_api");
const errors = @import("errors.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Queue = @import("image_progress_queue.zig").Queue;
pub const Writer = @import("image_progress_wire.zig").Writer;

/// The refusal for an engine that would not take a picture, the same for whole replies and streams.
pub fn submitRefused(cx: *errors.Cx, e: api.ImageSubmitError) errors.Refused {
    return switch (e) {
        error.Busy => cx.fail(.capacity, "the engine is busy; retry shortly", .{}),
        error.Closed => cx.fail(.capacity, "the scheduler is closed", .{}),
        error.InvalidRequest => cx.fail(.request, "the engine refused this image request's prompt, size or steps", .{}),
        error.Unsupported => cx.fail(.capacity, "this model does not serve images", .{}),
    };
}

/// Streams one picture's steps and PNG; false: the client left or a write failed, and the engine has finished.
pub fn generate(srv: *Server, c: *Conn, cx: *errors.Cx, request: api.ImageRequest, interval: u32, index: usize, writer: *Writer) errors.Refused!bool {
    if (c.peerGone()) return false;
    var queue: Queue = .{ .io = srv.io, .gpa = srv.gpa, .steps = request.steps, .preview_interval = interval };
    defer queue.deinit();
    var owned = request;
    owned.observer = queue.observer();
    const id = srv.next_id.fetchAdd(1, .monotonic);
    if (srv.keepalive) |k| k.begin();
    defer if (srv.keepalive) |k| k.end();
    srv.engine.submitImage(id, &owned, queue.sink()) catch |e| return submitRefused(cx, e);
    writer.start() catch {
        queue.cancelAndWait(srv.engine, id);
        return false;
    };
    while (true) {
        const next = queue.next();
        defer if (next.frame) |frame| queue.freeFrame(frame);
        if (next.problem) |why| {
            queue.cancelAndWait(srv.engine, id);
            return cx.fail(.server, "{s}", .{why});
        }
        if (c.peerGone()) {
            queue.cancelAndWait(srv.engine, id);
            return false;
        }
        if (next.frame) |frame| {
            writer.progress(index, request.seed, frame.step, frame.total) catch {
                queue.cancelAndWait(srv.engine, id);
                return false;
            };
            if (frame.preview) |preview| writer.preview(index, request.seed, frame.step, frame.total, interval, preview) catch {
                queue.cancelAndWait(srv.engine, id);
                return false;
            };
            continue;
        }
        if (!next.finished) continue;
        request.result.reason = next.reason;
        if (next.reason == .failed)
            return cx.fail(.server, "{s}", .{if (next.message.len > 0) next.message else "the image generation failed"});
        if (next.reason == .cancelled) return cx.fail(.capacity, "the image generation was cancelled", .{});
        writer.completed(index, request.seed, request.png.items[0..@min(request.result.bytes, request.png.items.len)]) catch return false;
        return true;
    }
}
