//! Clef's joint schema head as the CUDA Qwen3.5 lanes' readout: cuda_joint_schema behind cuda_readout.Readout.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const lanes = @import("lanes");
const js = @import("cuda_joint_schema");
const E = @import("cuda_engine.zig");
const Readout = @import("cuda_readout.zig").Readout;

/// The bounds the head plans: the design's 64 questions and 1,024 options, rows up to the window, as on HIP.
pub const max_questions = 64;
pub const max_options = 1024;
pub const max_rows = 16384;

/// The bounds a `window`-token engine plans for its readouts; any prompt piece up to the engine's rows.
pub fn bounds(window: usize) js.plan.Bounds {
    return .{ .rows = @min(max_rows, window), .questions = max_questions, .options = max_options, .piece = E.max_rows };
}

/// Device bytes each stream's readout state takes beside its cache.
pub fn streamBytes(cfg: core.joint_schema.Config, window: usize) usize {
    return js.plan.Lane.of(cfg, bounds(window)).total;
}

pub const Joint = struct {
    head: *js.Head,
    stream: cuda.Stream,

    /// `dir`'s head checked by the host loader and uploaded, over `e`'s packed BF16 lm_head; refused by name.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, e: *E.Engine, dir: []const u8, window: usize, problem: *core.joint_schema.Problem) !*Joint {
        const table = switch (e.head) {
            .ql => |w| switch (w) {
                .bf16 => |q| q.codes,
                else => return error.QwenHeadNeedsBf16,
            },
            else => return error.QwenHeadNeedsBf16,
        };
        var head = try core.joint_schema.load.open(gpa, io, dir, problem);
        defer head.close(io);
        if (head.weights.cfg.hidden_size != e.g.hidden) {
            problem.set("joint_head_config.json: hidden_size {d} is not the backbone's {d}", .{ head.weights.cfg.hidden_size, e.g.hidden });
            return error.BadConfig;
        }
        var w = try js.weights.Weights.upload(gpa, e.d, &head.file, &head.weights);
        errdefer w.deinit();
        const j = try gpa.create(Joint);
        errdefer gpa.destroy(j);
        j.* = .{ .head = try js.Head.init(gpa, e.d, w, bounds(window), table, e.g.vocab), .stream = e.stream };
        return j;
    }

    pub fn readout(j: *Joint) Readout {
        const b = j.head.bounds;
        return .{ .ctx = j, .info = .{ .max_rows = @intCast(b.rows), .max_questions = @intCast(b.questions), .max_options = @intCast(b.options) }, .vtable = &vtable };
    }

    const vtable: Readout.VTable = .{ .open = openState, .consume = consume, .finish = finish, .close = close, .deinit = deinit };

    fn of(ctx: *anyopaque) *Joint {
        return @ptrCast(@alignCast(ctx));
    }

    fn openState(ctx: *anyopaque, spec: *const lanes.readout.Spec, prompt: []const u32) anyerror!*Readout.State {
        const j = of(ctx);
        const st = try js.state.open(j.head, .{ .question_spans = spec.question_spans, .question_types = spec.question_types, .option_spans = spec.option_spans, .option_question = spec.option_question, .prompt = prompt }, j.stream);
        return @ptrCast(st);
    }

    fn consume(ctx: *anyopaque, st: *Readout.State, rows: u64, first_row: usize, count: usize, stream: cuda.Stream) anyerror!void {
        try js.consume(of(ctx).head, @ptrCast(@alignCast(st)), rows, first_row, count, stream);
    }

    fn finish(ctx: *anyopaque, st: *Readout.State, logits: []u16) anyerror!void {
        try js.finish(of(ctx).head, @ptrCast(@alignCast(st)), logits, of(ctx).stream);
    }

    fn close(ctx: *anyopaque, st: *Readout.State) void {
        js.state.close(of(ctx).head, @ptrCast(@alignCast(st)));
    }

    fn deinit(ctx: *anyopaque) void {
        const j = of(ctx);
        const gpa = j.head.gpa;
        j.head.deinit();
        gpa.destroy(j);
    }
};
