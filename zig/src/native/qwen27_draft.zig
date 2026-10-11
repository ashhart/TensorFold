//! A serialized host owns this draft helper while the target model and stable runner remain borrowed.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
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
    tokens: []u32 = &.{},
    first_batch: bool = true,

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
        return d;
    }
    fn retire(d: *Draft) void {
        if (d.tokens.len != 0) d.gpa.free(d.tokens);
        d.tokens = &.{};
    }
    pub fn reset(d: *Draft) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        try d.runner.reset(0);
        try d.model.reset();
        d.first_batch = true;
    }
    pub fn promptChunk(d: *Draft, ids: []const u32, last: bool) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        try d.generation.promptChunk(ids, last);
        if (last) d.first_batch = true;
    }
    pub fn decodeChunk(d: *Draft, ids: []const u32, last: bool) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        try d.generation.decodeChunk(ids, last);
        if (last) d.first_batch = true;
    }
    pub fn advance(d: *Draft, token: u32) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        try d.generation.advance(token);
        d.first_batch = true;
    }
    pub fn batch(d: *Draft, budget: u32, eos: []const u32) !api.serial_host.Batch {
        d.retire();
        if (budget == 0) return .{ .tokens = &.{}, .stats = .{} };
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const result = try d.generation.runRound(budget, 15, eos, !d.first_batch);
        errdefer d.gpa.free(result.tokens);
        if (result.tokens.len == 0 or result.tokens.len > @min(budget, 16) or result.verified < result.rounds) return error.BadDraftBatch;
        d.tokens = result.tokens;
        d.first_batch = false;
        return .{ .tokens = d.tokens, .stats = .{ .rounds = result.rounds, .drafted = result.verified - result.rounds, .accepted = result.matched, .min_rows = result.min_rows } };
    }
    pub fn deinit(d: *Draft) void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        d.retire();
        d.generation.deinit();
        d.runner.capture_taps = d.previous_capture;
        d.runner.taps.ids = d.previous_taps;
        if (d.owns_model) d.model.deinit();
        d.gpa.destroy(d);
    }
};
