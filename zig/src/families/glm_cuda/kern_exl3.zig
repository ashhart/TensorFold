//! GLM-5.3-Flash on CUDA: the EXL3 routed experts' launches (glm/exl3.cu), re-exported by kern.zig.

const std = @import("std");
const cuda = @import("cuda");
const kern = @import("kern.zig");
const K = kern.K;
const Plan = kern.Plan;
const D = kern.D;
const TOPK = kern.TOPK;
const EXPERTS = kern.EXPERTS;
const LIMIT = kern.LIMIT;
const maxItems = kern.maxItems;

/// One layer's EXL3 experts on this rank: trellis words and fp16 scales (weights.expertsExl3).
pub const Exl3 = struct { gt: u64, ut: u64, dt: u64, suh_g: u64, suh_u: u64, svh_g: u64, svh_u: u64, suh_d: u64, svh_d: u64 };

/// Scratch for P routed pairs: the rotated inputs (fp16) and the splits' partials.
pub const Exl3Scratch = struct { xg: u64, xu: u64, xd: u64, z: u64 };

pub const EXL3_SK: i64 = 4; // gate/up K slices in decode windows (down: 1); prompt chunks take 1

/// The EXL3 experts' prompt kernel (TF_GLM_EXL3_PROMPT, read once):
///   fused (the default): the transposed design (exl3.cu, "prompt chunks, the transposed design"): items of 64 pairs,
///     the weights decoded into A fragments, gate and up in one block with the SwiGLU epilogue, the down projection
///     with its epilogue; no partials in memory, one accumulation chain an output (a chunking-independent result).
///   staged: items of 128 pairs, the four warps over columns, rows and words through shared memory, K split
///     TF_GLM_EXL3_PROMPT_SK ways; wide: its 8-warp block (256 columns a block).
///   old: the grouped decode kernel on items of 64 pairs, warps over K (the engine's original prompt path).
pub const PromptKernel = enum { fused, staged, wide, old };
var prompt_kernel: ?PromptKernel = null;
pub fn promptKernel() PromptKernel {
    if (prompt_kernel == null) {
        prompt_kernel = .fused;
        if (std.c.getenv("TF_GLM_EXL3_PROMPT")) |v| prompt_kernel = std.meta.stringToEnum(PromptKernel, std.mem.span(v)) orelse .fused;
    }
    return prompt_kernel.?;
}
pub fn promptOld() bool {
    return promptKernel() == .old;
}
pub fn promptWide() bool {
    return promptKernel() == .wide;
}
pub fn promptFused() bool {
    return promptKernel() == .fused;
}
/// A prompt plan's tile: the pairs an item holds.
pub fn promptTile() i64 {
    return if (promptOld() or promptFused()) 64 else 128;
}
/// The staged prompt kernel's K splits (TF_GLM_EXL3_PROMPT_SK=1|2|4, 2): a short prompt's few pairs an expert get SK
/// times the blocks at a fraction of the K each. One value for every chunk, so a row's bits do not depend on the
/// chunking (a resumed prompt equals a fresh one); the chunk scratch is sized for it. Read once.
var prompt_sk: ?i64 = null;
pub fn promptSplit() i64 {
    if (prompt_sk == null) {
        const v: i64 = if (std.c.getenv("TF_GLM_EXL3_PROMPT_SK")) |v| (std.fmt.parseInt(i64, std.mem.span(v), 10) catch 2) else 2;
        prompt_sk = if (v == 1 or v == 2 or v == 4) v else 2;
    }
    return prompt_sk.?;
}

/// The routed pairs' outputs (pair = row * 8 + slot, picks [R, 8]) -> y [R, ys, D] fp32 at slots 0..7, through
/// the plan's items (16 pairs each in a window, 64 or 128 in a prompt chunk).
/// ``shx``: every expert of the layer shares one input rotation for gate and up (weights' exl3_shx): a prompt chunk
/// rotates each token once (sc.xg holds a row a token) and the staged kernel reads a pair's row as pair >> 3.
pub fn exl3Routed(k: *K, x: u64, x_stride: i64, picks: u64, plan: Plan, w: Exl3, sc: Exl3Scratch, y: u64, ys: i64, R: i64, width: i64, prompt: bool, shx: bool) !void {
    const P = R * TOPK;
    const items = maxItems(P, EXPERTS, plan.tile);
    const fused = prompt and promptFused();
    const staged = prompt and !promptOld() and !fused;
    const wide = staged and promptWide();
    const sk: i64 = if (staged) promptSplit() else if (prompt) 1 else EXL3_SK;
    const rows_x = (staged or fused) and shx;
    if (rows_x) {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@as(i32, @intCast(x_stride)));
        a.add(w.suh_g);
        a.add(sc.xg);
        a.add(@as(i32, @intCast(D)));
        try k.go(try k.ext(.exl3, "glm_exl3_rot_rows"), .{ .x = @intCast(R), .y = @intCast(@divExact(D, 128)) }, 32, 0, &a);
    } else {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@as(i32, @intCast(x_stride)));
        a.add(picks);
        a.add(w.suh_g);
        a.add(w.suh_u);
        a.add(sc.xg);
        a.add(sc.xu);
        a.add(@as(i32, @intCast(D)));
        try k.go(try k.ext(.exl3, "glm_exl3_rot_in"), .{ .x = @intCast(P), .y = @intCast(@divExact(D, 128)), .z = 2 }, 32, 0, &a);
    }
    if (fused) { // gate and up with their epilogue -> sc.xd; the down projection with its epilogue -> y
        {
            var a: cuda.Args = .{};
            a.add(sc.xg);
            a.add(if (rows_x) sc.xg else sc.xu);
            a.add(w.gt);
            a.add(w.ut);
            a.add(plan.items);
            a.add(plan.counts);
            a.add(plan.members);
            a.add(w.svh_g);
            a.add(w.svh_u);
            a.add(w.suh_d);
            a.add(sc.xd);
            a.add(@as(i32, @intCast(D)));
            a.add(@as(i32, @intCast(width)));
            a.add(@as(i32, if (rows_x) 3 else 0));
            a.add(LIMIT);
            try k.go(try k.ext(.exl3, "glm_exl3_prompt_gu"), .{ .x = @intCast(@divExact(width, 128)), .y = @intCast(items), .z = 1 }, 256, 0, &a);
        }
        {
            var a: cuda.Args = .{};
            a.add(sc.xd);
            a.add(w.dt);
            a.add(plan.items);
            a.add(plan.counts);
            a.add(plan.members);
            a.add(w.svh_d);
            a.add(y);
            a.add(@as(i32, @intCast(width)));
            a.add(@as(i32, @intCast(D)));
            a.add(@as(i32, @intCast(ys)));
            try k.go(try k.ext(.exl3, "glm_exl3_prompt_dn"), .{ .x = @intCast(@divExact(D, 256)), .y = @intCast(items), .z = 1 }, 256, 0, &a);
        }
        return;
    }
    const nt: i64 = if (wide) 16 else if (staged) 8 else if (prompt) 2 else 8;
    const threads: u32 = if (wide) 256 else 128;
    const gname: []const u8 = if (wide) "glm_exl3_prompt128w" else if (staged) "glm_exl3_prompt128" else if (prompt) "glm_exl3_grouped64" else "glm_exl3_grouped";
    // the staged kernel's grid: x = (column block, matrix, split), one item's blocks adjacent (its rows read from L2)
    {
        var a: cuda.Args = .{};
        a.add(sc.xg);
        a.add(if (rows_x) sc.xg else sc.xu);
        a.add(w.gt);
        a.add(w.ut);
        a.add(plan.items);
        a.add(plan.counts);
        a.add(plan.members);
        a.add(sc.z);
        a.add(@as(i32, @intCast(D)));
        a.add(@as(i32, @intCast(width)));
        a.add(@as(i32, @intCast(P)));
        if (staged) a.add(@as(i32, if (rows_x) 3 else 0));
        a.add(@as(i32, @intCast(sk)));
        if (staged) a.add(@as(i32, 2));
        const ncb: i64 = @divExact(width, 16 * nt);
        const grid: cuda.Dim3 = if (staged) .{ .x = @intCast(ncb * 2 * sk), .y = @intCast(items), .z = 1 } else .{ .x = @intCast(items), .y = @intCast(ncb), .z = @intCast(2 * sk) };
        try k.go(try k.ext(.exl3, gname), grid, threads, 0, &a);
    }
    {
        var a: cuda.Args = .{};
        a.add(sc.z);
        a.add(picks);
        a.add(w.svh_g);
        a.add(w.svh_u);
        a.add(w.suh_d);
        a.add(sc.xd);
        a.add(@as(i32, @intCast(P)));
        a.add(@as(i32, @intCast(width)));
        a.add(@as(i32, @intCast(sk)));
        a.add(LIMIT);
        try k.go(try k.ext(.exl3, "glm_exl3_gateup"), .{ .x = @intCast(P), .y = @intCast(@divExact(width, 128)) }, 32, 0, &a);
    }
    {
        var a: cuda.Args = .{};
        a.add(sc.xd);
        a.add(sc.xd);
        a.add(w.dt);
        a.add(w.dt);
        a.add(plan.items);
        a.add(plan.counts);
        a.add(plan.members);
        a.add(sc.z);
        a.add(@as(i32, @intCast(width)));
        a.add(@as(i32, @intCast(D)));
        a.add(@as(i32, @intCast(P)));
        if (staged) a.add(@as(i32, 0));
        a.add(@as(i32, 1));
        if (staged) a.add(@as(i32, 1));
        const ncb: i64 = @divExact(D, 16 * nt);
        const grid: cuda.Dim3 = if (staged) .{ .x = @intCast(ncb), .y = @intCast(items), .z = 1 } else .{ .x = @intCast(items), .y = @intCast(ncb), .z = 1 };
        try k.go(try k.ext(.exl3, gname), grid, threads, 0, &a);
    }
    {
        var a: cuda.Args = .{};
        a.add(sc.z);
        a.add(picks);
        a.add(w.svh_d);
        a.add(y);
        a.add(@as(i32, @intCast(P)));
        a.add(@as(i32, @intCast(D)));
        a.add(@as(i32, 1));
        a.add(@as(i32, @intCast(ys)));
        try k.go(try k.ext(.exl3, "glm_exl3_down"), .{ .x = @intCast(P), .y = @intCast(@divExact(D, 128)) }, 32, 0, &a);
    }
}
