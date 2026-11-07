//! Clef's joint schema head as the HIP lane backend's readout: hip/joint_schema behind hip_lanes.Readout.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const lanes = @import("lanes");
const Readout = @import("hip_lanes.zig").Readout;

const js = hip.joint_schema;

/// The bounds the head plans: the design's 64 questions and 1,024 options, rows up to the window.
pub const max_questions = 64;
pub const max_options = 1024;
pub const max_rows = 16384;

/// The head's files beside the checkpoint: the device head opens when the config is there.
pub fn present(io: std.Io, a: std.mem.Allocator, dir: []const u8) bool {
    const path = std.fs.path.join(a, &.{ dir, core.joint_schema.load.config_file }) catch return false;
    std.Io.Dir.cwd().access(io, path, .{}) catch return false;
    return true;
}

/// The checked head's weights on the device, before the engine sizes its streams (they count with the model's).
pub const Loaded = struct {
    weights: js.weights.Weights,
    cfg: core.joint_schema.Config,

    /// `dir`'s head checked by the host loader, its file mapped only while the weights go up.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, r: *const hip.Runtime, dir: []const u8, hidden: usize, problem: *core.joint_schema.Problem) !Loaded {
        var head = try core.joint_schema.load.open(gpa, io, dir, problem);
        defer head.close(io);
        if (head.weights.cfg.hidden_size != hidden) {
            problem.set("joint_head_config.json: hidden_size {d} is not the backbone's {d}", .{ head.weights.cfg.hidden_size, hidden });
            return error.BadConfig;
        }
        return .{ .weights = try js.weights.Weights.upload(gpa, r, &head.file, &head.weights), .cfg = head.weights.cfg };
    }

    pub fn deinit(l: *Loaded) void {
        l.weights.deinit();
    }
};

/// The bounds a `window`-token engine with `piece`-row prompt chunks plans for its readouts.
pub fn bounds(window: usize, piece: usize) js.plan.Bounds {
    return .{ .rows = @min(max_rows, window), .questions = max_questions, .options = max_options, .piece = piece };
}

pub const Joint = struct {
    head: *js.Head,
    stream: hip.Stream,

    pub fn readout(j: *Joint) Readout {
        const b = j.head.bounds;
        return .{ .ctx = j, .info = .{ .max_rows = @intCast(b.rows), .max_questions = @intCast(b.questions), .max_options = @intCast(b.options) }, .vtable = &vtable };
    }

    pub fn deinit(j: *Joint) void {
        j.head.deinit();
    }

    const vtable: Readout.VTable = .{ .open = open, .consume = consume, .finish = finish, .close = close };

    fn of(ctx: *anyopaque) *Joint {
        return @ptrCast(@alignCast(ctx));
    }

    fn open(ctx: *anyopaque, spec: *const lanes.readout.Spec, prompt: []const u32) anyerror!*Readout.State {
        const st = try js.state.open(of(ctx).head, .{ .question_spans = spec.question_spans, .question_types = spec.question_types, .option_spans = spec.option_spans, .option_question = spec.option_question, .prompt = prompt });
        return @ptrCast(st);
    }

    fn consume(ctx: *anyopaque, st: *Readout.State, rows: hip.ops.Tensor, first_row: usize, count: usize, stream: hip.Stream) anyerror!void {
        if (rows.kind != .bf16) return error.ReadoutNeedsBf16Rows;
        try js.consume(of(ctx).head, @ptrCast(@alignCast(st)), rows.ptr, first_row, count, stream);
    }

    fn finish(ctx: *anyopaque, st: *Readout.State, logits: []u16) anyerror!void {
        try js.finish(of(ctx).head, @ptrCast(@alignCast(st)), logits, of(ctx).stream);
    }

    fn close(ctx: *anyopaque, st: *Readout.State) void {
        js.state.close(of(ctx).head, @ptrCast(@alignCast(st)));
    }
};
