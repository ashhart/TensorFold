//! Nemotron's Sliding Weights learner behind the lane host's learn hook: lessons in, learn events out.
const std = @import("std");
const api = @import("engine_api");
const tf = @import("tensorfold");
const slide = tf.nemotron.slide;
const Allocator = std.mem.Allocator;

pub const Adapter = struct {
    learner: slide.Learner,
    arena: std.heap.ArenaAllocator,
    sink: ?api.LearnSink = null,

    pub fn init(gpa: Allocator, io: std.Io, b: *tf.nemotron.backend.Metal, dir: []const u8) !Adapter {
        return .{ .learner = try slide.Learner.init(gpa, io, b, dir), .arena = .init(gpa) };
    }

    pub fn deinit(a: *Adapter) void {
        a.learner.deinit();
        a.arena.deinit();
    }

    pub fn hook(a: *Adapter) api.Learner {
        return .{ .ctx = a, .begin = begin, .step = step, .abort = abort };
    }

    fn begin(ctx: *anyopaque, request: *const api.LearnRequest, sink: api.LearnSink) anyerror!void {
        const a: *Adapter = @ptrCast(@alignCast(ctx));
        _ = a.arena.reset(.retain_capacity);
        const al = a.arena.allocator();
        try a.learner.begin(.{ .train = try examples(al, request.train), .held = try examples(al, request.held), .near = try examples(al, request.near), .keep = try examples(al, request.keep), .save = request.save, .undo = request.undo, .steps = request.steps, .more = request.more });
        a.sink = sink;
    }

    fn step(ctx: *anyopaque) api.Learner.Step {
        const a: *Adapter = @ptrCast(@alignCast(ctx));
        const s = a.learner.step();
        const failed = if (s.report) |r| r == .failed else false;
        if (s.report) |r| a.emit(switch (r) {
            .learned => |x| .{ .learned = .{ .recalled = x.recalled, .steps = x.steps, .loss = x.loss } },
            .saved => |n| .{ .saved = .{ .tensors = n } },
            .failed => |message| .{ .done = .{ .message = message } },
        });
        if (s.done and !failed) a.emit(.{ .done = .{} });
        return .{ .done = s.done, .changed = s.changed };
    }

    fn abort(ctx: *anyopaque) void {
        const a: *Adapter = @ptrCast(@alignCast(ctx));
        a.learner.abort();
        a.emit(.{ .done = .{ .message = "the engine closed" } });
    }

    fn emit(a: *Adapter, event: api.LearnEvent) void {
        const sink = a.sink orelse return;
        sink.event(sink.ctx, &event);
    }
};

fn examples(al: Allocator, xs: []const api.Example) ![]slide.Example {
    const out = try al.alloc(slide.Example, xs.len);
    for (xs, out) |x, *o| o.* = .{ .ids = x.ids, .start = x.start };
    return out;
}
