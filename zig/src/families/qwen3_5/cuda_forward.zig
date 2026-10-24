//! The landed CUDA forward with optional learner taps and pocket contributions.
const E = @import("cuda_engine.zig");
const cuda = @import("cuda");
const sig = @import("cuda_signatures.zig");
const hooks = @import("cuda_learn_hooks.zig");
const geometry = @import("cuda_geometry.zig");
const head_dim = geometry.head_dim;
/// Prompt experts keep the shipped arithmetic boundary, independently of verification capacity.
pub const prompt_moe_cutoff: usize = 16;
/// qwen35_attn_tc.cuh's bf16 tile buffers: two of a 32-key tile's keys and values, 512 bytes a row each.
pub const attn_tiles = 4 * 32 * 512;
const ids_at = E.Engine.ids_at;
const segs_at = E.Engine.segs_at;
const rowseg_at = E.Engine.rowseg_at;
const rowpar_at = E.Engine.rowpar_at;
const rowanc_at = E.Engine.rowanc_at;
const params_at = E.Engine.params_at;
const mpos_at = E.Engine.mpos_at;
const src_at = E.Engine.src_at;

pub fn promptMoe(prompt: bool, rows: usize) bool {
    return prompt and rows > prompt_moe_cutoff;
}

pub fn body(e: *E.Engine, shape: anytype) !void {
    const g = e.g;
    const total: usize = shape.total;
    const nseg: usize = shape.nseg;
    const record = shape.record;
    const prompt = shape.prompt;
    const heads: usize = if (record) total else 1;
    const t_ids = e.tables.ptr + ids_at;
    const t_segs = e.tables.ptr + segs_at;
    const t_rows = e.tables.ptr + rowseg_at;
    const t_par = e.tables.ptr + rowpar_at;
    const t_anc = e.tables.ptr + rowanc_at;
    const t_params = e.tables.ptr + params_at;
    const geo = e.geo;

    try e.embed(t_ids, e.h, total, t_params);
    // the producers of the out, o and down inputs write their group sums; a prompt GEMM past 16 rows needs none
    const sums: u64 = if (prompt and total > 16) 0 else e.xs;
    // an image chunk's rows take the tower's features, and rotate at their staged t, h, w positions
    const mpos: u64 = if (e.features != 0) e.tables.ptr + mpos_at else 0;
    if (e.features != 0) try e.run(e.fns.splice, .{ total, 1, 1 }, .{ 256, 1, 1 }, .{ e.h, e.features, e.tables.ptr + src_at, @as(c_int, @intCast(g.hidden)), t_params });
    var di: usize = 0;
    var ai: c_int = 0;
    for (e.layers, 0..) |l, i| {
        try e.norm(i != 0, l.input_norm, total);
        if (e.learning) |k| try k.input(i, e.h, total);
        switch (l.mixer) {
            .delta => |m| {
                const layer: c_int = @intCast(di);
                try e.projMany(&.{ m.qkv, m.z, m.a, m.b }, e.x, &.{ e.qkv, e.z, e.a, e.b }, total, prompt);
                if (e.learning) |k| try k.delta(i, e.qkv, e.z, e.b, e.a, total);
                try e.run(e.fns.gdn_pre, .{ 1, 2 * g.k_heads + g.v_heads, total }, .{ 32, 1, 1 }, .{ geo, e.qkv, t_segs, t_rows, t_par, layer, m.conv, e.a, e.b, m.a_log, m.dt_bias, e.q, e.k, e.v, e.g_, e.beta, t_params });
                // one recorded window of 2+ rows: gdn_pre logged its tails, conv_commit would only return
                if (!(record and nseg == 1 and total > 1)) try e.run(e.fns.conv_commit, .{ (g.convWidth() + 255) / 256, nseg, 1 }, .{ 256, 1, 1 }, .{ geo, e.qkv, t_segs, t_par, layer, t_params });
                if (prompt and nseg == 1) try @import("cuda_prompt.zig").delta(e, di, total) else try e.deltaScan(record, shape.slots, nseg, layer);
                try e.run(e.fns.gdn_post, .{ 1, g.v_heads, total }, .{ 32, 1, 1 }, .{ geo, e.y, e.z, m.norm, g.eps, e.mix, t_params, sums });
                try e.projAgain(m.out, e.mix, e.r, total, prompt);
                di += 1;
            },
            .attention => |m| {
                const aw = g.attnWidth();
                try e.projMany(&.{ m.q, m.k, m.v }, e.x, &.{ e.qraw, e.kraw, e.vraw }, total, prompt);
                if (e.learning) |k| try k.attention(i, e.qraw, e.kraw, e.vraw, total);
                try e.run(e.fns.head_norm, .{ total * g.q_heads, 1, 1 }, .{ 64, 1, 1 }, .{ e.qraw, m.q_norm, e.qn, @as(c_int, @intCast(g.q_heads)), @as(c_int, @intCast(2 * aw)), @as(c_int, 2 * head_dim), g.eps, t_params });
                try e.run(e.fns.head_norm, .{ total * g.kv_heads, 1, 1 }, .{ 64, 1, 1 }, .{ e.kraw, m.k_norm, e.kn, @as(c_int, @intCast(g.kv_heads)), @as(c_int, @intCast(g.kv_heads * head_dim)), @as(c_int, head_dim), g.eps, t_params });
                try launch(e, if (e.kv8) e.fp8.rope else e.fns.rope, sig.rope, .{ total, g.q_heads + g.kv_heads, 1 }, .{ head_dim, 1, 1 }, .{ geo, e.qn, e.kn, e.vraw, t_segs, t_rows, t_anc, @as(c_int, ai), t_params, mpos });
                if (shape.paged) {
                    try @import("cuda_attention.zig").partial(e, shape, ai);
                    try launch(e, e.fns.attn_merge, sig.attn_merge, .{ g.q_heads, total, 1 }, .{ 32, 1, 1 }, .{ geo, e.pm, e.pl, e.po, t_segs, t_rows, t_anc, t_params, e.y });
                } else if (prompt and nseg == 1 and (g.q_heads / g.kv_heads) % 2 == 0) {
                    // a prompt chunk's rows over its own cache: tensor-core tiles in place of the decode partials
                    try @import("cuda_prompt.zig").run(e, @intCast(ai), total);
                } else {
                    const args = .{ geo, e.qn, t_segs, t_rows, t_anc, @as(c_int, ai), t_params, @as(f32, 1.0 / 16.0), e.pm, e.pl, e.po };
                    const groups = (total + 64 / group - 1) / (64 / group);
                    const rows_args = .{ geo, e.qn, t_segs, t_rows, t_anc, @as(c_int, ai), t_params, @as(f32, 1.0 / 16.0), @as(c_int, @intCast(total)), e.pm, e.pl, e.po };
                    if (e.kv8) {
                        try launch(e, e.fp8.attn_partial, sig.attn_partial8, .{ total, max_chunks, g.kv_heads }, .{ 128, 1, 1 }, args);
                        try launch(e, e.fp8.attn_rows, sig.attn_rows8, .{ groups, max_chunks, g.kv_heads }, .{ 256, 1, 1 }, rows_args);
                    } else {
                        try shared(e, e.fns.attn_partial, sig.attn_partial, .{ total, max_chunks, g.kv_heads }, 128, args);
                        try shared(e, e.fns.attn_rows, sig.attn_rows, .{ groups, max_chunks, g.kv_heads }, 256, rows_args);
                    }
                    try launch(e, e.fns.attn_merge, sig.attn_merge, .{ g.q_heads, total, 1 }, .{ 32, 1, 1 }, .{ geo, e.pm, e.pl, e.po, t_segs, t_rows, t_anc, t_params, e.y });
                }
                try launch(e, e.fns.attn_gate, sig.attn_gate, .{ (total * aw + 255) / 256, 1, 1 }, .{ 256, 1, 1 }, .{ geo, e.y, e.qraw, e.mix, @as(c_int, @intCast(total)), t_params, sums });
                if (e.learning) |k| try k.gated(i, e.mix, total);
                try e.projAgain(m.o, e.mix, e.r, total, prompt);
                ai += 1;
            },
        }
        try e.norm(true, l.post_norm, total);
        if (e.learning) |k| try k.post(i, e.h, e.x, total);
        switch (l.mlp) {
            .dense => |m| {
                try e.projAgain(m.gate, e.x, e.gate, total, prompt);
                try e.projAgain(m.up, e.x, e.up, total, prompt);
                if (e.learning) |k| try k.mlp(i, e.gate, e.up, total);
                const n: c_longlong = @intCast(total * g.intermediate);
                try e.run(e.fns.mlp_act, .{ (total * g.intermediate + 1023) / 1024, 1, 1 }, .{ 256, 1, 1 }, .{ e.gate, e.up, e.act, n, t_params, @as(c_int, @intCast(g.intermediate)), sums });
                try e.projAgain(m.down, e.act, e.r, total, prompt);
                if (e.mlp) |k| try k.f(k.ctx, i, e.x, e.r, total, e.taught) else try e.pockets.add(e.stream, i, e.x, e.r, total, e.taught);
            },
            .moe => |m| try e.routedRows(m.router, m.experts, e.x, e.r, total, promptMoe(prompt, total), t_params),
        }
        if (shape.taps) for (e.taps[0..e.n_taps], 0..) |t, ti| if (t == i) {
            try e.run(e.fns.tap, .{ total, (g.hidden + 2047) / 2048, 1 }, .{ 256, 1, 1 }, .{ e.h, e.r, t_segs, t_rows, @as(c_int, @intCast(ti)), @as(c_int, @intCast(g.hidden)), t_params });
        };
    }
    try finish(e, total, heads, shape.hidden);
}

/// A launch whose arguments are checked against the kernel's .cu signature: a wrong count or kind does not build.
fn launch(e: *E.Engine, f: cuda.Function, comptime s: sig.Sig, grid: [3]usize, block: [3]usize, args: anytype) !void {
    sig.check(s, @TypeOf(args));
    return e.run(f, grid, block, args);
}

/// A bf16 decode attention launch with its two tile buffers of dynamic shared memory.
fn shared(e: *E.Engine, f: cuda.Function, comptime s: sig.Sig, grid: [3]usize, threads: u32, args: anytype) !void {
    sig.check(s, @TypeOf(args));
    var a: cuda.Args = .{};
    inline for (args) |v| a.add(v);
    const cfg: cuda.Config = .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = threads }, .shared = attn_tiles };
    try cuda.launch.launch(f, cfg, e.stream, &a);
}

/// Layer inputs are already kept; hidden reads need neither the final norm nor the vocabulary head.
pub fn finish(e: anytype, total: usize, heads: usize, hidden: bool) !void {
    if (hidden) return;
    const g = e.g;
    try e.norm(true, e.final_norm, total);
    if (e.learning) |k| try k.final(e.h, total);
    // the head over the kept rows
    try e.proj(e.head, e.x + (total - heads) * g.hidden * 2, e.logits, heads, false);
    if (e.learning) |k| try hooks.logits(k, e.logits, e.stream);
}

test "prompt MoE precision: target piece arithmetic keeps the shipped 16-row boundary" {
    const std = @import("std");
    try std.testing.expectEqual(@as(usize, 64), E.most_window);
    var mismatches: usize = 0;
    for ([_]usize{ 16, 17, 32, 64, 65 }) |rows| {
        const expected = rows > 16;
        if (promptMoe(true, rows) != expected) {
            std.debug.print("prompt piece {d}: expected prompt experts {any}\n", .{ rows, expected });
            mismatches += 1;
        }
        if (promptMoe(false, rows)) {
            std.debug.print("verify window {d}: expected decode experts\n", .{rows});
            mismatches += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 0), mismatches);
}

test "prompt MoE precision: every prompt piece through 128 and admitted verify width keeps its recipe" {
    const std = @import("std");
    for (1..129) |rows| try std.testing.expectEqual(rows > 16, promptMoe(true, rows));
    for (1..E.most_window + 1) |rows| try std.testing.expect(!promptMoe(false, rows));
}
