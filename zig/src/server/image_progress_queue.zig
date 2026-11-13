//! Bounded image-only progress handed from an engine thread to the request's writer.
const std = @import("std");
const api = @import("engine_api");

pub const Frame = struct { step: u32, total: u32, preview: ?api.ImagePreview = null };
pub const Read = struct { frame: ?Frame = null, finished: bool, reason: api.Reason, message: []const u8, problem: ?[]const u8 };

pub const Queue = struct {
    io: std.Io,
    gpa: std.mem.Allocator,
    steps: u32,
    preview_interval: u32,
    mutex: std.Io.Mutex = .init,
    changed: std.Io.Condition = .init,
    cancelled: std.atomic.Value(bool) = .init(false),
    frames: [200]Frame = undefined,
    published: usize = 0,
    consumed: usize = 0,
    finished: bool = false,
    reason: api.Reason = .stop,
    message: []const u8 = "",
    problem: ?[]const u8 = null,

    pub fn observer(q: *Queue) api.ImageObserver {
        return .{ .ctx = q, .publish = publish, .preview_interval = q.preview_interval };
    }
    pub fn sink(q: *Queue) api.Sink {
        return .{ .ctx = q, .event = event };
    }
    fn cast(ctx: *anyopaque) *Queue {
        return @ptrCast(@alignCast(ctx));
    }
    fn fail(q: *Queue, message: []const u8) bool {
        q.problem = message;
        q.cancelled.store(true, .release);
        q.changed.signal(q.io);
        return false;
    }
    fn publish(ctx: *anyopaque, step: u32, total: u32, preview: ?api.ImagePreview) bool {
        const q = cast(ctx);
        q.mutex.lockUncancelable(q.io);
        defer q.mutex.unlock(q.io);
        if (q.cancelled.load(.acquire) or q.finished) return false;
        if (total != q.steps or step != q.published + 1 or step > total or q.published == q.frames.len)
            return q.fail("the image engine reported invalid step progress");
        const due = q.preview_interval > 0 and step % q.preview_interval == 0;
        if (due != (preview != null)) return q.fail("the image engine did not honor preview_interval");
        var frame: Frame = .{ .step = step, .total = total };
        if (preview) |p| {
            if (p.width == 0 or p.height == 0 or p.width > api.ImagePreview.max_side or p.height > api.ImagePreview.max_side or
                p.png.len == 0 or p.png.len > api.ImagePreview.max_bytes)
                return q.fail("the image preview exceeds the 128-side or 81920-byte limit");
            frame.preview = .{ .png = q.gpa.dupe(u8, p.png) catch return q.fail("out of memory copying an image preview"), .width = p.width, .height = p.height };
        }
        q.frames[q.published] = frame;
        q.published += 1;
        q.changed.signal(q.io);
        return true;
    }
    fn event(ctx: *anyopaque, _: api.Id, value: *const api.Event) void {
        const q = cast(ctx);
        if (value.* != .finished) return;
        q.mutex.lockUncancelable(q.io);
        defer q.mutex.unlock(q.io);
        if (q.finished) return;
        q.reason = value.finished.reason;
        q.message = q.gpa.dupe(u8, value.finished.message) catch "";
        if (q.reason != .failed and q.reason != .cancelled and q.published != q.steps)
            q.problem = "the image engine finished before reporting every step";
        q.finished = true;
        q.changed.signal(q.io);
    }
    pub fn next(q: *Queue) Read {
        q.mutex.lockUncancelable(q.io);
        defer q.mutex.unlock(q.io);
        if (q.consumed == q.published and !q.finished and q.problem == null)
            q.changed.waitTimeout(q.io, &q.mutex, .{ .duration = .{ .raw = .fromMilliseconds(50), .clock = .awake } }) catch {};
        var out: Read = .{ .finished = q.finished, .reason = q.reason, .message = q.message, .problem = q.problem };
        if (q.consumed < q.published) {
            out.frame = q.frames[q.consumed];
            q.consumed += 1;
        }
        return out;
    }
    pub fn cancelAndWait(q: *Queue, engine: api.Engine, id: api.Id) void {
        q.cancelled.store(true, .release);
        q.mutex.lockUncancelable(q.io);
        const done = q.finished;
        q.mutex.unlock(q.io);
        if (!done) engine.cancel(id);
        q.mutex.lockUncancelable(q.io);
        defer q.mutex.unlock(q.io);
        while (!q.finished) q.changed.waitUncancelable(q.io, &q.mutex);
    }
    pub fn freeFrame(q: *Queue, frame: Frame) void {
        if (frame.preview) |p| q.gpa.free(p.png);
    }
    pub fn deinit(q: *Queue) void {
        for (q.frames[q.consumed..q.published]) |frame| q.freeFrame(frame);
        q.gpa.free(q.message);
    }
};
