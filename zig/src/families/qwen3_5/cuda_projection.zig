//! Qwen3.5's projections: the row tiles, the prompt GEMM and the grouped form.
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const int8_prompt = @import("cuda_prompt_int8.zig");
const projection_many = @import("cuda_projection_many.zig");

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
    return projectMany(e, ls, x, e.xs, outs, rows, prompt, e.tables.ptr + Engine.params_at, false, false);
}

/// The landed forward's shared-input projections; on GB10 their prompt CTAs may share one launch.
pub fn projManyPrompt(e: *Engine, ls: []const Proj, x: u64, outs: []const u64, rows: usize, prompt: bool) !void {
    return projectMany(e, ls, x, e.xs, outs, rows, prompt, e.tables.ptr + Engine.params_at, false, true);
}

/// `projMany` with shared input sums, preserving the caller's prompt or native projection role.
pub fn projManySums(e: *Engine, ls: []const Proj, x: u64, xs: u64, outs: []const u64, rows: usize, prompt: bool, params: u64) !void {
    return projectMany(e, ls, x, xs, outs, rows, prompt, params, true, false);
}

pub const projectMany = projection_many.projectMany;

pub fn project(e: *Engine, l: Proj, x: u64, xs: u64, out: u64, rows: usize, prompt: bool, params: u64, sums: bool) !void {
    if (l == .ql) return @import("cuda_modelopt.zig").project(e, l.ql, x, out, rows, prompt);
    const n_out = l.outputs();
    const k = l.inputs();
    // EXL3 and BF16 matrices use their own row kernels at every width.
    switch (l) {
        .x3 => |q| return if (prompt and rows > 16) e.kit.?.dense.prompt(e.stream, q, x, .bf16, rows, out, .bf16) else e.kit.?.dense.decode(e.stream, q, x, .bf16, rows, out, .bf16),
        .bf => |q| {
            // The small BF16 row kernel accepts at most 128 rows, including during a full prompt pass.
            var at: usize = 0;
            while (at < rows) : (at += 128) {
                try e.kit.?.b16.denseRows(e.stream, x + at * q.k * 2, q.w, out + at * q.n * 2, @min(rows - at, 128), q.n, q.k);
            }
            return;
        },
        else => {},
    }
    if (prompt and e.int8.on) switch (l) {
        // every prompt row of an int8 request takes int8; an exceptional chunk returns to the default precision
        .a4 => |q| if (try int8_prompt.projectRows(e, q, x, xs, out, rows, sums, params, k)) return else int8_prompt.fellBack(e),
        .an, .x3, .bf => {},
        .ql => unreachable,
    };
    if (prompt and rows > 16) return switch (l) {
        .a4 => |q| if (qwen38LargePromptTile(e, q.n, q.k, rows)) e.affine.promptHalf(e.stream, x, q, out, rows) else e.affine.prompt(e.stream, x, q, out, rows),
        .an => |q| e.wide.prompt(e.stream, x, q, out, rows),
        .ql => unreachable,
        .x3, .bf => unreachable,
    };
    if (sums) try e.run(e.fns.sums, .{ rows, (k / 64 + 63) / 64, 1 }, .{ 64, 1, 1 }, .{ x, xs, @as(c_int, @intCast(k)), params });
    const tile: usize = switch (l) {
        .a4 => e.affine.verifyRows(),
        .an, .ql => 16,
        .x3, .bf => unreachable,
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
            .x3, .bf => unreachable,
        }
    }
}

fn qwen38LargePromptTile(e: *const Engine, n: usize, k: usize, rows: usize) bool {
    return e.affine.pdl and e.g.hidden == 5120 and e.g.intermediate == 17408 and e.g.layers == 64 and e.g.experts == 0 and projection_many.qwen38LargePromptShape(n, k, rows);
}
