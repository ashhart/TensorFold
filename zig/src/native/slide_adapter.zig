//! A family's Living Weights learner behind the lane host's learn hook: lessons in, learn events out.
const std = @import("std");
const api = @import("engine_api");
const Allocator = std.mem.Allocator;

/// Neurons a learn event carries at most; a fact takes about ten.
const max_neurons = 64;

/// Why --slide cannot learn into `dir` (`Checks`: linked and plainShared files; `copy`: the copy to make), or null.
pub fn refusal(comptime Checks: type, a: Allocator, io: std.Io, dir: []const u8, copy: []const u8) !?[]const u8 {
    if (@hasDecl(Checks, "plainShared")) if (try Checks.plainShared(a, io, dir)) |name| return try std.fmt.allocPrint(a, "--slide adds learned neurons beside every shared expert, and {s} in {s} is unquantized, as TensorFold 1.0.3 to 1.0.6 saved their lessons: serve this folder without --slide (what it learned stays), or teach a fresh copy", .{ name, dir });
    if (try Checks.linked(a, io, dir)) |name| return try std.fmt.allocPrint(a, "--slide rewrites the model's own files, and {s} in {s} is a link to data another file shares (a Hugging Face cache's blob, or a hard link), so learning would change that file too: serve a copy made with {s}", .{ name, dir, copy });
    return null;
}

/// The hook over `slide`, a backend's learner (its Learner, Example and Lesson): Metal's or CUDA's.
pub fn Adapter(comptime slide: type) type {
    return struct {
        const Self = @This();

        learner: slide.Learner,
        arena: std.heap.ArenaAllocator,
        sink: ?api.LearnSink = null,
        sent: ?api.LearnProgress = null, // the last progress sent, so each move goes out once
        neurons: [max_neurons]api.Neuron = undefined, // the open lesson's neurons for the event being sent

        pub fn init(gpa: Allocator, io: std.Io, model: anytype) Self {
            return .{ .learner = slide.Learner.init(gpa, io, model), .arena = .init(gpa) };
        }

        pub fn deinit(a: *Self) void {
            a.learner.deinit();
            a.arena.deinit();
        }

        pub fn hook(a: *Self) api.Learner {
            return .{ .ctx = a, .begin = begin, .step = step, .abort = abort };
        }

        fn begin(ctx: *anyopaque, request: *const api.LearnRequest, sink: api.LearnSink) anyerror!void {
            const a: *Self = @ptrCast(@alignCast(ctx));
            _ = a.arena.reset(.retain_capacity);
            const al = a.arena.allocator();
            try a.learner.begin(.{ .train = try examples(al, request.train), .think = try examples(al, request.think), .held = try examples(al, request.held), .near = try examples(al, request.near), .keep = try examples(al, request.keep), .asked = try examples(al, request.asked), .unasked = try examples(al, request.unasked), .kept = try examples(al, request.kept), .undo = request.undo, .steps = request.steps, .more = request.more, .reroute = request.reroute, .drop = request.drop, .save = request.save, .retire = try retired(al, request.retire), .unretire = request.unretire, .forget = try retired(al, request.forget), .unforget = request.unforget });
            a.sink = sink;
            a.sent = null;
            a.progress();
        }

        fn step(ctx: *anyopaque) api.Learner.Step {
            const a: *Self = @ptrCast(@alignCast(ctx));
            const s = a.learner.step();
            a.progress();
            const failed = if (s.report) |r| r == .failed else false;
            if (s.report) |r| a.emit(switch (r) {
                .learned => |x| .{ .learned = .{ .recalled = x.recalled, .steps = x.steps, .loss = x.loss, .neurons = a.placed() } },
                .failed => |message| .{ .done = .{ .message = message } },
            });
            if (s.done and !failed) a.emit(.{ .done = .{} });
            return .{ .done = s.done, .changed = s.changed, .stale = s.stale };
        }

        fn abort(ctx: *anyopaque) void {
            const a: *Self = @ptrCast(@alignCast(ctx));
            a.learner.abort();
            a.emit(.{ .done = .{ .message = "the engine closed" } });
        }

        fn emit(a: *Self, event: api.LearnEvent) void {
            const sink = a.sink orelse return;
            sink.event(sink.ctx, &event);
        }

        /// The open pocket lesson's progress, sent when its phase or unit moved.
        fn progress(a: *Self) void {
            if (comptime slide.Learner.Progress == void) return;
            const p = a.learner.progress() orelse return;
            const phase: api.LearnPhase = switch (p.phase) {
                .read => .reading,
                .fit => .fitting,
                .values => .values,
                .check => .checking,
                .done => return,
            };
            var now: api.LearnProgress = .{ .phase = phase, .at = @intCast(p.at), .of = @intCast(p.of), .loss = p.loss };
            if (a.sent) |last| if (last.phase == now.phase and last.at == now.at) return;
            a.sent = now;
            if (phase == .values or phase == .checking) now.neurons = a.placed();
            a.emit(.{ .progress = now });
        }

        /// The open lesson's neurons as placed, in this adapter's buffer until the next call.
        fn placed(a: *Self) []const api.Neuron {
            var buf: [max_neurons]slide.Learner.Placed = undefined;
            const got = a.learner.placed(&buf);
            for (got, a.neurons[0..got.len]) |x, *o| o.* = .{ .layer = @intCast(x.layer), .neuron = @intCast(x.neuron), .key = switch (x.key) {
                inline else => |k| @field(api.NeuronKey, @tagName(k)),
            } };
            return a.neurons[0..got.len];
        }

        fn examples(al: Allocator, xs: []const api.Example) ![]slide.Example {
            const out = try al.alloc(slide.Example, xs.len);
            for (xs, out) |x, *o| o.* = .{ .ids = x.ids, .start = x.start, .decide = x.decide, .subject = x.subject, .value = x.value, .context = x.context };
            return out;
        }

        /// Each neuron to zero as the learner names it: (layer, neuron).
        fn retired(al: Allocator, ns: []const api.Neuron) ![]const [2]u16 {
            const out = try al.alloc([2]u16, ns.len);
            for (ns, out) |n, *o| o.* = .{ n.layer, n.neuron };
            return out;
        }
    };
}
