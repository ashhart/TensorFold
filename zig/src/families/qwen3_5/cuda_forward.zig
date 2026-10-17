//! The landed CUDA forward with optional learner taps and pocket contributions.
const E = @import("cuda_engine.zig");
const hooks = @import("cuda_learn_hooks.zig");
const geometry = @import("cuda_geometry.zig");
const head_dim = geometry.head_dim;
const max_chunks = 64;
const most_window = E.most_window;
const ids_at = E.Engine.ids_at;
const segs_at = E.Engine.segs_at;
const rowseg_at = E.Engine.rowseg_at;
const rowpar_at = E.Engine.rowpar_at;
const rowanc_at = E.Engine.rowanc_at;
const params_at = E.Engine.params_at;
const mpos_at = E.Engine.mpos_at;
const src_at = E.Engine.src_at;

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
    const group = g.q_heads / g.kv_heads;

    try e.embed(t_ids, e.h, total, t_params);
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
                try e.proj(m.qkv, e.x, e.qkv, total, prompt);
                try e.proj(m.z, e.x, e.z, total, prompt);
                try e.proj(m.a, e.x, e.a, total, prompt);
                try e.proj(m.b, e.x, e.b, total, prompt);
                if (e.learning) |k| try k.delta(i, e.qkv, e.z, e.b, e.a, total);
                try e.run(e.fns.gdn_pre, .{ 1, 2 * g.k_heads + g.v_heads, total }, .{ 32, 1, 1 }, .{ geo, e.qkv, t_segs, t_rows, t_par, layer, m.conv, e.a, e.b, m.a_log, m.dt_bias, e.q, e.k, e.v, e.g_, e.beta, t_params });
                try e.run(e.fns.conv_commit, .{ (g.convWidth() + 255) / 256, nseg, 1 }, .{ 256, 1, 1 }, .{ geo, e.qkv, t_segs, t_par, layer, t_params });
                try e.run(e.fns.gdn_chain, .{ 1, geometry.dk / 4, g.v_heads * nseg }, .{ 32, 4, 1 }, .{ geo, e.q, e.k, e.v, e.g_, e.beta, t_segs, t_par, layer, e.y, t_params });
                try e.run(e.fns.gdn_post, .{ 1, g.v_heads, total }, .{ 32, 1, 1 }, .{ geo, e.y, e.z, m.norm, g.eps, e.mix, t_params });
                try e.proj(m.out, e.mix, e.r, total, prompt);
                di += 1;
            },
            .attention => |m| {
                const aw = g.attnWidth();
                try e.proj(m.q, e.x, e.qraw, total, prompt);
                try e.proj(m.k, e.x, e.kraw, total, prompt);
                try e.proj(m.v, e.x, e.vraw, total, prompt);
                if (e.learning) |k| try k.attention(i, e.qraw, e.kraw, e.vraw, total);
                try e.run(e.fns.head_norm, .{ total * g.q_heads, 1, 1 }, .{ 64, 1, 1 }, .{ e.qraw, m.q_norm, e.qn, @as(c_int, @intCast(g.q_heads)), @as(c_int, @intCast(2 * aw)), @as(c_int, 2 * head_dim), g.eps, t_params });
                try e.run(e.fns.head_norm, .{ total * g.kv_heads, 1, 1 }, .{ 64, 1, 1 }, .{ e.kraw, m.k_norm, e.kn, @as(c_int, @intCast(g.kv_heads)), @as(c_int, @intCast(g.kv_heads * head_dim)), @as(c_int, head_dim), g.eps, t_params });
                try e.run(if (e.kv8) e.fp8.rope else e.fns.rope, .{ total, 1, 1 }, .{ head_dim, 1, 1 }, .{ geo, e.qn, e.kn, e.vraw, t_segs, t_rows, t_anc, ai, t_params, mpos });
                try e.run(if (e.kv8) e.fp8.attn_partial else e.fns.attn_partial, .{ max_chunks, g.kv_heads, total }, .{ 32 * group, 1, 1 }, .{ geo, e.qn, t_segs, t_rows, t_anc, ai, t_params, @as(f32, 1.0 / 16.0), e.pm, e.pl, e.po });
                try e.run(e.fns.attn_merge, .{ g.q_heads, total, 1 }, .{ 32, 1, 1 }, .{ geo, e.pm, e.pl, e.po, t_segs, t_rows, t_anc, t_params, e.y });
                try e.run(e.fns.attn_gate, .{ (total * aw + 255) / 256, 1, 1 }, .{ 256, 1, 1 }, .{ geo, e.y, e.qraw, e.mix, @as(c_int, @intCast(total)), t_params });
                if (e.learning) |k| try k.gated(i, e.mix, total);
                try e.proj(m.o, e.mix, e.r, total, prompt);
                ai += 1;
            },
        }
        try e.norm(true, l.post_norm, total);
        if (e.learning) |k| try k.post(i, e.h, e.x, total);
        switch (l.mlp) {
            .dense => |m| {
                try e.proj(m.gate, e.x, e.gate, total, prompt);
                try e.proj(m.up, e.x, e.up, total, prompt);
                if (e.learning) |k| try k.mlp(i, e.gate, e.up, total);
                const n: c_longlong = @intCast(total * g.intermediate);
                try e.run(e.fns.mlp_act, .{ (total * g.intermediate + 255) / 256, 1, 1 }, .{ 256, 1, 1 }, .{ e.gate, e.up, e.act, n, t_params, @as(c_int, @intCast(g.intermediate)) });
                try e.proj(m.down, e.act, e.r, total, prompt);
                if (e.mlp) |k| try k.f(k.ctx, i, e.x, e.r, total, e.taught) else try e.pockets.add(e.stream, i, e.x, e.r, total, e.taught);
            },
            .moe => |m| try e.routedRows(m.router, m.experts, e.x, e.r, total, prompt and total > most_window),
        }
        if (shape.taps) for (e.taps[0..e.n_taps], 0..) |t, ti| if (t == i) {
            try e.run(e.fns.tap, .{ total, 1, 1 }, .{ 256, 1, 1 }, .{ e.h, e.r, t_segs, t_rows, @as(c_int, @intCast(ti)), @as(c_int, @intCast(g.hidden)), t_params });
        };
    }
    try e.norm(true, e.final_norm, total);
    if (e.learning) |k| try k.final(e.h, total);
    // the head over the kept rows
    try e.proj(e.head, e.x + (total - heads) * g.hidden * 2, e.logits, heads, false);
    if (e.learning) |k| try hooks.logits(k, e.logits, e.stream);
}
