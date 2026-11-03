//! The shared learner on the exclusive lane thread; changed weights also invalidate the native in-place resume.
const std = @import("std");
const api = @import("engine_api");
const q = @import("tensorfold").qwen27;
const Shared = @import("slide_adapter.zig").Adapter(q.slide);
const Host = @import("qwen27_served.zig").Host;

pub const Bridge = struct {
    host: *Host,
    dir: []u8,
    adapter: Shared,

    pub fn init(h: *Host, io: std.Io, dir: []const u8) !*Bridge {
        const b = try h.gpa.create(Bridge);
        errdefer h.gpa.destroy(b);
        const copy = try h.gpa.dupe(u8, dir);
        b.* = .{ .host = h, .dir = copy, .adapter = Shared.init(h.gpa, io, q.slide.Model{ .m = h.model, .dir = copy, .follow = h.runner.residency }) };
        return b;
    }
    pub fn deinit(b: *Bridge) void {
        const a = b.host.gpa;
        b.adapter.deinit();
        a.free(b.dir);
        a.destroy(b);
    }
    pub fn conversations(b: *const Bridge) bool {
        return b.adapter.learner.readsConversations();
    }
    pub fn hook(b: *Bridge) api.Learner {
        return .{ .ctx = b, .begin = begin, .step = step, .abort = abort };
    }
    fn begin(ctx: *anyopaque, request: *const api.LearnRequest, sink: api.LearnSink) !void {
        const b: *Bridge = @ptrCast(@alignCast(ctx));
        const h = b.adapter.hook();
        try h.begin(h.ctx, request, sink);
    }
    fn step(ctx: *anyopaque) api.Learner.Step {
        const b: *Bridge = @ptrCast(@alignCast(ctx));
        const h = b.adapter.hook();
        const result = h.step(h.ctx);
        if (result.changed) {
            // The lane host clears its saved RAM/disk cache. Slot 0 has another route around that store.
            q.learning_epoch.changed(b.host.back);
        }
        return result;
    }
    fn abort(ctx: *anyopaque) void {
        const b: *Bridge = @ptrCast(@alignCast(ctx));
        const h = b.adapter.hook();
        h.abort(h.ctx);
    }
};

/// Check links before mapping weights or creating a Metal device.
pub fn refused(a: std.mem.Allocator, io: std.Io, dir: []const u8, problem: *[]const u8) !bool {
    const path = try std.fs.path.join(a, &.{ dir, "config.json" });
    defer a.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20));
    defer a.free(text);
    const config = try q.config.parse(a, text, null);
    if (!q.train_admission.supported(config)) {
        problem.* = "--slide on this Metal engine requires the dense Qwen3.8-27B geometry";
        return true;
    }
    if (try @import("metal_slide.zig").slideWidths(a, io, dir, q.slide.family)) |why| {
        problem.* = why;
        return true;
    }
    if (try @import("slide_adapter.zig").refusal(@import("tensorfold").shard_edit, a, io, dir, "`cp -cRL` (an APFS clone that follows links)")) |why| {
        problem.* = why;
        return true;
    }
    return false;
}
