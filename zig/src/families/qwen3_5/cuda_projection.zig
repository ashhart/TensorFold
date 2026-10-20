//! Qwen3.5's projections: the row tiles, the prompt GEMM and the grouped form.
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const int8_prompt = @import("cuda_prompt_int8.zig");

const Engine = engine.Engine;
const Proj = engine.Proj;

/// out (rows, n) = x (rows, k) through `l`: row tiles with batch-independent bits or a prompt GEMM.
pub fn proj(e: *Engine, l: Proj, x: u64, out: u64, rows: usize, prompt: bool) !void {
    return projSums(e, l, x, e.xs, out, rows, prompt, e.tables.ptr + Engine.params_at);
}

/// `proj` with group sums `xs` and device counts `params` (0: all real); prompts past 16 rows take a prompt GEMM.
pub fn projSums(e: *Engine, l: Proj, x: u64, xs: u64, out: u64, rows: usize, prompt: bool, params: u64) !void {
    return project(e, l, x, xs, out, rows, prompt, params, true);
}

pub fn projAgain(e: *Engine, l: Proj, x: u64, out: u64, rows: usize, prompt: bool) !void {
    return project(e, l, x, e.xs, out, rows, prompt, e.tables.ptr + Engine.params_at, false);
}

/// `projAgain` of each of `ls` into `outs` over one x; on the split form's GPUs one segmented launch a row tile.
pub fn projMany(e: *Engine, ls: []const Proj, x: u64, outs: []const u64, rows: usize, prompt: bool) !void {
    return projectMany(e, ls, x, e.xs, outs, rows, prompt, e.tables.ptr + Engine.params_at, false);
}

/// `projMany` with shared input sums, preserving the caller's prompt or native projection role.
pub fn projManySums(e: *Engine, ls: []const Proj, x: u64, xs: u64, outs: []const u64, rows: usize, prompt: bool, params: u64) !void {
    return projectMany(e, ls, x, xs, outs, rows, prompt, params, true);
}

pub const projectMany = @import("cuda_projection_many.zig").projectMany;

pub fn project(e: *Engine, l: Proj, x: u64, xs: u64, out: u64, rows: usize, prompt: bool, params: u64, sums: bool) !void {
    if (l == .ql) return @import("cuda_modelopt.zig").project(e, l.ql, x, out, rows, prompt);
    const n_out = l.outputs();
    const k = l.inputs();
    if (prompt and e.int8.on) switch (l) {
        // every prompt row of an int8 request takes int8; an exceptional chunk returns to the default precision
        .a4 => |q| if (try int8_prompt.projectRows(e, q, x, xs, out, rows, sums, params, k)) return,
        .an => {},
        .ql => unreachable,
    };
    if (prompt and rows > 16) return switch (l) {
        .a4 => |q| e.affine.prompt(e.stream, x, q, out, rows),
        .an => |q| e.wide.prompt(e.stream, x, q, out, rows),
        .ql => unreachable,
    };
    if (sums) try e.run(e.fns.sums, .{ rows, (k / 64 + 63) / 64, 1 }, .{ 64, 1, 1 }, .{ x, xs, @as(c_int, @intCast(k)), params });
    const tile: usize = switch (l) {
        .a4 => e.affine.verifyRows(),
        .an, .ql => 16,
    };
    var at: usize = 0;
    while (at < rows) : (at += tile) {
        const n = @min(tile, rows - at);
        const xa = x + at * k * 2;
        const sa = xs + at * (k / 64) * 4;
        const oa = out + at * n_out * 2;
        switch (l) {
            .a4 => |q| try e.affine.verify(e.stream, xa, sa, q, oa, n),
            .an => |q| try e.wide.decode(e.stream, xa, sa, q, oa, n),
            .ql => unreachable,
        }
    }
}
