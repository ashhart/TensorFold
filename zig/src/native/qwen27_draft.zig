//! Slot 0's drafter on the lane backend (DFlash2 or DSpine), which borrows the target model and its runner.
const std = @import("std");
const mtl = @import("metal");
const q = @import("tensorfold").qwen27;
const Any = q.drafter.Any;
pub const Draft = struct {
    gpa: std.mem.Allocator,
    runner: *q.decode_round.Runner,
    model: Any,
    owns_model: bool,
    generation: q.dflash.generation.GenerationOf(Any),
    previous_capture: bool,
    previous_taps: [q.taps.max_planes]usize,
    previous_count: usize,

    /// The drafter at `dir` (DFlash2 at `bits`, or DSpine, which reads the target's head again from `target_dir`).
    pub fn open(gpa: std.mem.Allocator, io: std.Io, target: *q.model.Model, runner: *q.decode_round.Runner, dir: []const u8, bits: u8, target_dir: []const u8) !*Draft {
        if (runner.model != target) return error.TargetBinding;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const model = try Any.load(gpa, io, target, dir, bits, target_dir);
        errdefer model.deinit();
        return attach(gpa, runner, model, true);
    }
    pub fn attach(gpa: std.mem.Allocator, runner: *q.decode_round.Runner, model: Any, owns_model: bool) !*Draft {
        if (!model.binds(runner.model)) return error.TargetBinding;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const d = try gpa.create(Draft);
        errdefer gpa.destroy(d);
        const previous_capture = runner.capture_taps;
        const previous_taps = runner.taps.ids;
        const previous_count = runner.taps.count;
        // Generation keeps a pointer to the drafter: it is d.model, at its final address
        d.* = .{ .gpa = gpa, .runner = runner, .model = model, .owns_model = owns_model, .generation = undefined, .previous_capture = previous_capture, .previous_taps = previous_taps, .previous_count = previous_count };
        d.generation = q.dflash.generation.GenerationOf(Any).init(gpa, runner, &d.model) catch |err| {
            runner.capture_taps = previous_capture;
            runner.taps.ids = previous_taps;
            runner.taps.count = previous_count;
            return err;
        };
        model.profile(runner.trace); // the target's TF_Q27_PROFILE trace times the drafter's passes too
        return d;
    }
    /// What the host reports: "DFlash2" or "DSpine".
    pub fn label(d: *const Draft) []const u8 {
        return d.model.label();
    }
    pub fn reset(d: *Draft) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        try d.generation.settle();
        try d.runner.reset(0);
        try d.model.reset();
    }
    pub fn promptChunk(d: *Draft, ids: []const u32, last: bool) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        try d.generation.settle();
        try d.generation.promptChunk(ids, last);
    }
    pub fn decodeChunk(d: *Draft, ids: []const u32, last: bool) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        try d.generation.settle();
        try d.generation.decodeChunk(ids, last);
    }
    pub fn advance(d: *Draft, token: u32) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        try d.generation.settle();
        try d.generation.advance(token);
    }
    pub fn deinit(d: *Draft) void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        d.generation.deinit();
        d.runner.capture_taps = d.previous_capture;
        d.runner.taps.ids = d.previous_taps;
        d.runner.taps.count = d.previous_count;
        d.model.profile(null); // a borrowed drafter outlives the host and its trace
        if (d.owns_model) d.model.deinit();
        d.gpa.destroy(d);
    }
};
