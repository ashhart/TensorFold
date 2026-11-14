//! Compact eight-slot prompt reference for direct-output byte checks, isolated from production serving.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const common = cuda.exl3_experts;
const Experts = glm.exl3.Experts;
const Kernels = glm.exl3.Kernels;
const Scratch = glm.exl3.Scratch;
const experts = glm.experts;
const top_k = glm.exl3.top_k;
const chunk_rows = glm.exl3.chunk_rows;
fn layer(w: Experts, count: usize, dims: usize, width: usize, limit: f32) common.Layer {
    return .{ .gate = w.gt, .up = w.ut, .down = w.dt, .gate_k2 = w.k2, .up_k2 = w.k2, .down_k2 = w.k2, .suh_g = w.suh_g, .suh_u = w.suh_u, .svh_g = w.svh_g, .svh_u = w.svh_u, .suh_d = w.suh_d, .svh_d = w.svh_d, .experts = count, .dims = dims, .width = width, .codebook = .mcg, .limit = limit, .act_mode = 0, .uniform_k2 = 8, .shared_input = w.shx };
}
fn shape(dims: usize, width: usize) common.Layer {
    const w: Experts = .{ .gt = 0, .ut = 0, .dt = 0, .k2 = 0, .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0, .shx = false };
    return layer(w, cuda.grouped.max_experts, dims, width, 0);
}
fn picksBytes(rows: usize) usize {
    const cap: usize = @min(rows, chunk_rows);
    return std.mem.alignForward(usize, cap * top_k * 4, 256);
}
fn copyRows(stream: cuda.Stream, dst: u64, dpitch: usize, src: u64, spitch: usize, width: usize, height: usize, what: []const u8) !void {
    const m: cuda.abi.Memcpy2D = .{ .src = src, .src_pitch = spitch, .dst = dst, .dst_pitch = dpitch, .width = width, .height = height };
    try stream.d.check(stream.d.api.cuMemcpy2DAsync_v2(&m, stream.handle), what);
}
fn compact(k: *const Kernels, ek: *const experts.Kernels, stream: cuda.Stream, w: Experts, sc: Scratch, plan_buf: cuda.grouped.Plan, x: u64, x_stride: usize, picks: u64, slots: usize, rows: usize, count: usize, dims: usize, width: usize, limit: f32, prompt: bool, ey: u64) !void {
    const l = layer(w, count, dims, width, limit);
    const cap: usize = @min(rows, chunk_rows);
    const arena = sc.picks + picksBytes(rows);
    const fused = prompt and common.prompt.eligible(l);
    var at: usize = 0;
    while (at < rows) : (at += chunk_rows) {
        const n: usize = @min(chunk_rows, rows - at);
        try copyRows(stream, sc.picks, top_k * 4, picks + at * slots * 4, slots * 4, top_k * 4, n, "EXL3 compact route rows");
        const output = if (fused) blk: {
            const scratch = try common.prompt.scratch(arena, l, cap, top_k);
            const y = arena + (try common.prompt.bytes(l, cap, top_k));
            try k.prompt.run(k, ek.router, stream, l, x + at * x_stride * 2, .bf16, x_stride, sc.picks, n, scratch, plan_buf, y, null);
            break :blk y;
        } else blk: {
            const scratch = try common.scratch(arena, l, cap, top_k);
            try k.runSlots(stream, l, x + at * x_stride * 2, x_stride, .bf16, sc.picks, n, scratch);
            break :blk scratch.y;
        };
        try copyRows(stream, ey + at * slots * dims * 4, slots * dims * 4, output, top_k * dims * 4, top_k * dims * 4, n, "EXL3 routed output slots");
    }
}

pub fn scratchBytes(rows: usize, dims: usize, width: usize, shx: bool) usize {
    var l = shape(dims, width);
    l.shared_input = shx;
    const n: usize = @min(rows, chunk_rows);
    return picksBytes(rows) + (common.prompt.bytes(l, n, top_k) catch unreachable) + n * top_k * dims * 4;
}
pub fn run(k: *const Kernels, ek: *const experts.Kernels, stream: cuda.Stream, w: Experts, sc: Scratch, plan_buf: cuda.grouped.Plan, x: u64, x_stride: usize, picks: u64, slots: usize, rows: usize, count: usize, dims: usize, width: usize, limit: f32, ey: u64) !void {
    return compact(k, ek, stream, w, sc, plan_buf, x, x_stride, picks, slots, rows, count, dims, width, limit, true, ey);
}
