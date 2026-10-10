//! GLM-5.3-Flash's Sliding Weights learner (layers.44 shared-expert down_proj) behind the lane host's learn hook.
const std = @import("std");
const api = @import("engine_api");
const tf = @import("tensorfold");
const lw = tf.glm.lw_train;
const lwl = tf.glm.lw_learner;
const Allocator = std.mem.Allocator;

pub const Adapter = struct {
    learner: lw.Learner,
    arena: std.heap.ArenaAllocator,
    sink: ?api.LearnSink = null,

    pub fn init(gpa: Allocator, eng: *tf.glm.engine.Engine) Adapter {
        return .{ .learner = lw.Learner.init(gpa, eng), .arena = .init(gpa) };
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
        try a.learner.begin(.{ .train = try examples(al, request.train), .held = try examples(al, request.held), .near = try examples(al, request.near), .keep = try examples(al, request.keep), .undo = request.undo, .steps = request.steps, .more = request.more, .commit = request.commit, .save = request.save });
        a.sink = sink;
    }

    fn step(ctx: *anyopaque) api.Learner.Step {
        const a: *Adapter = @ptrCast(@alignCast(ctx));
        const s = a.learner.step();
        const failed = if (s.report) |r| r == .failed else false;
        if (s.report) |r| a.emit(switch (r) {
            .learned => |x| .{ .learned = .{ .recalled = x.recalled, .steps = x.steps, .loss = x.loss } },
            .failed => |message| .{ .done = .{ .message = message } },
        });
        if (s.done and !failed) a.emit(.{ .done = .{} });
        if (s.done) if (a.learner.trainer) |t| std.log.info("slide: {d} GPU captures ({d:.1} s), CPU tail {d:.1} s so far", .{ t.captures, t.seconds_gpu, t.seconds_cpu });
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

fn examples(al: Allocator, xs: []const api.Example) ![]lwl.Example {
    const out = try al.alloc(lwl.Example, xs.len);
    for (xs, out) |x, *o| o.* = .{ .ids = x.ids, .start = x.start };
    return out;
}
