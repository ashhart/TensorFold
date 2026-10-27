//! The DFlash2 drafter for slot 0 of the lane backend, which borrows the target model and its runner.
const std = @import("std");
const mtl = @import("metal");
const q = @import("tensorfold").qwen27;
const df = q.dflash;
pub const Draft = struct {
    gpa: std.mem.Allocator,
    runner: *q.decode_round.Runner,
    model: *df.runtime_model.Model,
    owns_model: bool,
    generation: df.generation.Generation,
    previous_capture: bool,
    previous_taps: [5]usize,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, target: *q.model.Model, runner: *q.decode_round.Runner, dir: []const u8, mode: df.operators.Mode) !*Draft {
        if (runner.model != target) return error.TargetBinding;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const model = try df.runtime_model.Model.load(gpa, io, target, dir, mode);
        errdefer model.deinit();
        return attach(gpa, runner, model, true);
    }
    pub fn attach(gpa: std.mem.Allocator, runner: *q.decode_round.Runner, model: *df.runtime_model.Model, owns_model: bool) !*Draft {
        if (model.backend.target != runner.model) return error.TargetBinding;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const d = try gpa.create(Draft);
        errdefer gpa.destroy(d);
        const previous_capture = runner.capture_taps;
        const previous_taps = runner.taps.ids;
        const generation = df.generation.Generation.init(gpa, runner, model) catch |err| {
            runner.capture_taps = previous_capture;
            runner.taps.ids = previous_taps;
            return err;
        };
        d.* = .{ .gpa = gpa, .runner = runner, .model = model, .owns_model = owns_model, .generation = generation, .previous_capture = previous_capture, .previous_taps = previous_taps };
        model.backend.profile(runner.trace); // the target's TF_Q27_PROFILE trace times the drafter's passes too
        return d;
    }
    pub fn reset(d: *Draft) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        try d.runner.reset(0);
        try d.model.reset();
    }
    pub fn promptChunk(d: *Draft, ids: []const u32, last: bool) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        try d.generation.promptChunk(ids, last);
    }
    pub fn decodeChunk(d: *Draft, ids: []const u32, last: bool) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        try d.generation.decodeChunk(ids, last);
    }
    pub fn advance(d: *Draft, token: u32) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        try d.generation.advance(token);
    }
    pub fn deinit(d: *Draft) void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        d.generation.deinit();
        d.runner.capture_taps = d.previous_capture;
        d.runner.taps.ids = d.previous_taps;
        d.model.backend.profile(null); // a borrowed drafter outlives the host and its trace
        if (d.owns_model) d.model.deinit();
        d.gpa.destroy(d);
    }
};
