//! GLM-5.3-Flash's MoE on CUDA, a rank's share of the Python family's moe_block: top-8 plus shared, grouped NVFP4.

const std = @import("std");
const cuda = @import("cuda");
const cuda_pdl = @import("cuda_pdl.zig");
const glue = @import("cuda_glue.zig");
const experts = @import("cuda_experts.zig");
const exl3 = @import("cuda_exl3.zig");
const prof_mod = @import("cuda_prof.zig");
const projection = @import("cuda_proj.zig");

/// One MoE layer on a rank: router (bf16 [count][dims]) and fp32 bias, routed experts, shared [gate | up] and down.
pub const Layer = struct {
    router: u64,
    bias: u64,
    routed: cuda.experts.Layer, // NVFP4 experts, or only the shape (width, dims, limit) when `exl3` or `tfq3` holds them
    exl3: ?exl3.Experts = null,
    tfq3: ?cuda.tfq3.MoeLayer = null,
    count: usize,
    shared_gu: projection.Proj, // as the checkpoint stores them: bf16, block FP8 or the MTP's MXFP8
    shared_down: projection.Proj,
    shared: usize,
    top_k: usize,
    scale: f32,
    norm: bool,
};

/// The block's scratch for up to `rows` rows (forward.Buffers' MoE part).
pub const Scratch = struct {
    mlog: u64, // fp32 [rows][count]
    router_part: u64, // fp32 [router_ks][rows][count]
    pick: u64, // i32 [rows][top_k + 1]
    wts: u64, // fp32 [rows][top_k + 1]
    plan: cuda.grouped.Plan,
    eact: u64, // bf16 [rows (top_k + 1)][width]
    ey: u64, // [rows][top_k + 1][dims]: fp32, bf16 in a prompt chunk
    sgu: u64, // bf16 [rows][2 shared]
    sact: u64, // bf16 [rows][shared]
    sxs: u64, // fp32 [rows][shared / 64]
    sy: u64, // fp32 [rows][dims]
    mm_part: u64, // fp32 split-K partials of the shared expert's matmuls
    x3: exl3.Scratch = .{ .picks = 0, .xg = 0, .xu = 0, .z = 0 }, // EXL3 experts (eact holds their down inputs)
};

/// The kernels the block runs: the captured Triton set, the grouped NVFP4 experts, the plan, and glm_ops.
pub const Kernels = struct {
    set: glue.Set,
    experts: *const experts.Kernels,
    exl3: *const exl3.Kernels,
    tfq3: ?*const cuda.tfq3.Kernels = null,
    put_slot: cuda.Function,
    prof: ?*prof_mod.Prof = null,
    direct: bool = false, // TF_GLM_MOE_DIRECT=1 (opt-in): see `direct`
};

/// TF_GLM_MOE_DIRECT=1: a decode window's EXL3 experts on flat picks, their slot rows summed in place (no copies).
fn direct(k: Kernels, l: Layer, rows: usize, prompt: bool) bool {
    return k.direct and !prompt and l.exl3 != null and l.top_k == exl3.top_k and rows <= exl3.chunk_rows;
}

/// forward.moe_block: out [rows][dims] fp32 partial; `prompt`: 64-pair items, the prompt kernels, bf16 slot rows.
pub fn block(k: Kernels, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, prompt: bool, out: u64) !void {
    try body(k, stream, l, s, x, x_stride, rows, prompt);
    try fill(k, stream, l, s, x, x_stride, 0, rows, prompt, out);
}

/// Router and top-k selection, with `pick` optionally a segment of a paired plan's picks.
pub fn prepare(k: Kernels, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, pick: u64, flat_picks: ?u64) !void {
    const d = l.routed.dims;
    const stamp = prof_mod.span(k.prof, stream, .router);
    defer stamp.end();
    try glue.router(k.set, stream, x, x_stride, l.router, s.mlog, s.router_part, rows, d, l.count);
    try glue.select(k.set, stream, s.mlog, l.bias, pick, s.wts, rows, l.top_k, l.count, l.scale, l.norm, flat_picks);
}

/// The block up to its fill: the router, the top-k pick, the plan and the routed experts into the slot rows.
pub fn body(k: Kernels, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, prompt: bool) !void {
    const d = l.routed.dims;
    const slots = l.top_k + 1;
    const shared_id: i32 = @intCast(l.count);
    const flat = direct(k, l, rows, prompt);
    try prepare(k, stream, l, s, x, x_stride, rows, s.pick, if (flat) s.x3.picks else null);
    const stamp = prof_mod.span(k.prof, stream, .experts);
    defer stamp.end();
    if (k.prof) |p| p.mark(stream, .router);
    defer if (k.prof) |p| p.mark(stream, .routed);
    if (flat) return exl3.runDirect(k.exl3, stream, l.exl3.?, s.x3, x, x_stride, rows, l.count, d, l.routed.width, l.routed.limit);
    if (l.exl3) |w| return exl3.run(k.exl3, k.experts, stream, w, s.x3, s.plan, x, x_stride, s.pick, slots, rows, l.count, d, l.routed.width, l.routed.limit, prompt, s.ey, s.eact);
    if (l.tfq3) |tl| {
        const tk = k.tfq3 orelse return error.FormatNotLoaded;
        return tk.moe(stream, tl, x, x_stride, s.pick, @intCast(slots), s.ey, @intCast(slots), rows, @intCast(l.top_k), l.routed.limit, 0, true);
    }
    const plan: experts.Plan = .{ .route = s.plan, .slots = slots, .experts = l.count + 1, .prompt = prompt, .fact = !prompt and k.experts.factored(rows) and l.routed.format == .nvfp4 };
    try experts.route(k.experts, stream, s.pick, rows * slots, plan);
    try experts.run(k.experts, stream, .gate_up, l.routed, plan, x, x_stride, rows, s.eact, shared_id);
    try experts.run(k.experts, stream, if (prompt) .down else .down_f32, l.routed, plan, s.eact, l.routed.width, rows, s.ey, shared_id);
}

/// Shared expert and its slot, before the slot sum; paired passes call this on the original exchange pieces.
pub fn shared(k: Kernels, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, lo: usize, hi: usize, prompt: bool) !void {
    const d = l.routed.dims;
    const slots = l.top_k + 1;
    const n = hi - lo;
    const bf16_slots = prompt and l.exl3 == null and l.tfq3 == null;
    const eb: usize = if (bf16_slots) 2 else 4;
    try projection.matmul(k.set, stream, l.shared_gu, x + lo * x_stride * 2, x_stride, s.sgu + lo * 2 * l.shared * 2, s.mm_part, n, 2 * l.shared, d, false);
    try glue.swiglu(k.set, stream, s.sgu + lo * 2 * l.shared * 2, s.sact + lo * l.shared * 2, s.sxs + lo * (l.shared / 64) * 4, l.routed.limit, n, l.shared);
    try projection.matmul(k.set, stream, l.shared_down, s.sact + lo * l.shared * 2, l.shared, s.sy + lo * d * 4, s.mm_part, n, d, l.shared, true);
    if (direct(k, l, hi, prompt)) { // body ran these rows direct: a decode window, filled whole; fill sums it (combineTail)
        if (lo != 0) return error.BadWindow;
        return;
    }
    var a: cuda.Args = .{};
    a.add(s.sy + lo * d * 4);
    a.add(s.ey + lo * slots * d * eb);
    for ([_]usize{ n, d, slots, l.top_k, @intFromBool(bf16_slots) }) |v| a.add(@as(c_int, @intCast(v)));
    try cuda_pdl.launch(k.put_slot, .{ .grid = .{ .x = @intCast((n * d + 255) / 256) }, .block = .{ .x = 256 } }, stream, &a);
}

/// The already-filled slot rows, summed in slot order into out fp32 (any range, same bits).
pub fn combine(k: Kernels, stream: cuda.Stream, l: Layer, s: Scratch, lo: usize, hi: usize, prompt: bool, out: u64) !void {
    const bf16_slots = prompt and l.exl3 == null and l.tfq3 == null;
    const eb: usize = if (bf16_slots) 2 else 4;
    try glue.combine(k.set, stream, s.ey + lo * (l.top_k + 1) * l.routed.dims * eb, !bf16_slots, s.wts + lo * (l.top_k + 1) * 4, out, hi - lo, l.routed.dims, l.top_k + 1);
}

/// moe_block's fill of rows lo..hi: shared expert into the last slot, slot sum into out fp32 (any range, same bits).
pub fn fill(k: Kernels, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, lo: usize, hi: usize, prompt: bool, out: u64) !void {
    const stamp = prof_mod.span(k.prof, stream, .shared_expert);
    defer stamp.end();
    try shared(k, stream, l, s, x, x_stride, lo, hi, prompt);
    if (direct(k, l, hi, prompt)) { // TF_GLM_MOE_DIRECT: the routed sum straight from the flat picks, then the shared slot
        const d = l.routed.dims;
        const y = try exl3.directOut(l.exl3.?, s.x3, hi, l.count, d, l.routed.width, l.routed.limit);
        return glue.combineTail(k.set, stream, y, s.sy, s.wts, out, hi - lo, d, l.top_k + 1);
    }
    try combine(k, stream, l, s, lo, hi, prompt, out);
}
