//! Dispatch dense BF16 and packed affine projections for the CUDA Qwen3.5 engine.
const Engine = @import("cuda_engine.zig").Engine;
const Proj = @import("cuda_layers.zig").Proj;
pub fn run(e: *Engine, l: Proj, x: u64, xs: u64, out: u64, rows: usize, prompt: bool, params: u64, sums: bool) !void {
    if (l == .bf16) {
        const p = l.bf16;
        const plan = try @import("cuda_bf16.zig").projectionPlan(e, rows, p.n, p.k);
        const ws = e.bf16_workspace.?;
        return plan.run(x, p.w, out, ws.ptr, ws.len, e.stream.handle);
    }
    const n_out = l.outputs();
    const k = l.inputs();
    if (prompt and rows > 16) return switch (l) {
        .a4 => |q| e.affine.prompt(e.stream, x, q, out, rows),
        .an => |q| e.wide.prompt(e.stream, x, q, out, rows),
        .bf16 => unreachable,
    };
    if (sums) try e.run(e.fns.sums, .{ rows, (k / 64 + 63) / 64, 1 }, .{ 64, 1, 1 }, .{ x, xs, @as(c_int, @intCast(k)), params });
    const tile: usize = switch (l) {
        .a4 => e.affine.verifyRows(),
        .an => 16,
        .bf16 => unreachable,
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
            .bf16 => unreachable,
        }
    }
}
