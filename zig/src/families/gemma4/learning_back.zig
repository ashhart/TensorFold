//! The full dense reverse chain keeps the pre-tail pocket seam and combines the global layer's tied K/V uses.
const std = @import("std");
const core = @import("core");
const stage = @import("learning_stage.zig");
fn name(t: anytype, l: usize, suffix: []const u8, b: *[224]u8) ![]const u8 {
    return stage.named(t, l, suffix, b);
}
pub fn run(t: anytype) !void {
    const s = t.shape;
    const h = s.hidden;
    try t.ops().zero(t.buf(.push));
    var mask_all: u64 = 0;
    for (t.seq.nudge.layers) |mask| mask_all |= mask;
    if (mask_all == 0) return;
    const low: usize = @ctz(mask_all);
    var p: core.living_rows.Params = .{};
    p.u[0] = @intCast(t.rows);
    p.u[1] = @intCast(s.vocab);
    p.f[0] = 30;
    try t.ops().gemma(.softcap, p, &.{ .{ 0, try t.taped(null, .head_raw) }, .{ 1, t.vec(.logits, s.vocab) }, .{ 8, t.vec(.capped, s.vocab) } });
    try stage.matrixBack(t, "language_model.model.embed_tokens", s.vocab, h, t.vec(.capped, s.vocab), t.vec(.norm, h), false);
    try stage.norm(t, "language_model.model.norm.weight", try t.taped(s.layers - 1, .residual), t.vec(.norm, h), t.vec(.grad, h), h, 1, true);
    var l = s.layers;
    while (l > low) {
        l -= 1;
        var b: [224]u8 = undefined;
        const scalar = try t.reader.get(try name(t, l, "layer_scalar", &b));
        if (scalar.dtype != .bf16 or scalar.bytes.len != 2) return error.GemmaTrainingScalar;
        try t.write(try t.buf(.stage).plus(0, 2), scalar.bytes);
        p = .{};
        p.u[0] = @intCast(t.rows);
        p.u[1] = @intCast(h);
        try t.ops().gemma(.scale, p, &.{ .{ 0, t.vec(.grad, h) }, .{ 1, try t.buf(.stage).plus(0, 2) }, .{ 8, t.vec(.grad, h) } });
        try stage.norm(t, try name(t, l, "post_feedforward_layernorm.weight", &b), try t.taped(l, .mlp_output), t.vec(.grad, h), t.vec(.branch, h), h, 1, true);
        for (t.seq.nudge.layers, t.seq.nudge.rows, 0..) |mask, row, key| {
            if (mask & (@as(u64, 1) << @intCast(l)) != 0 and row < t.rows)
                try t.ops().move(try t.vec(.branch, h).plus(row * h * 4, h * 4), try t.buf(.push).plus(t.vector(key, l) * 4, h * 4), false, 1);
        }
        if (l == low) break;
        try stage.matrixBack(t, try name(t, l, "mlp.down_proj", &b), h, s.intermediate, t.vec(.branch, h), t.vec(.act, s.intermediate), false);
        p.u[1] = @intCast(s.intermediate);
        try t.ops().gemma(.geglu, p, &.{ .{ 0, try t.taped(l, .gate) }, .{ 1, try t.taped(l, .up) }, .{ 2, t.vec(.act, s.intermediate) }, .{ 8, t.vec(.gate, s.intermediate) }, .{ 9, t.vec(.up, s.intermediate) } });
        try stage.matrixBack(t, try name(t, l, "mlp.gate_proj", &b), s.intermediate, h, t.vec(.gate, s.intermediate), t.vec(.mixed, h), false);
        try stage.matrixBack(t, try name(t, l, "mlp.up_proj", &b), s.intermediate, h, t.vec(.up, s.intermediate), t.vec(.mixed, h), true);
        try pocket(t, l);
        try stage.norm(t, try name(t, l, "pre_feedforward_layernorm.weight", &b), try t.taped(l, .mid), t.vec(.mixed, h), t.vec(.norm, h), h, 1, true);
        try t.ops().move(t.vec(.norm, h), t.vec(.grad, h), true, 1);
        try stage.norm(t, try name(t, l, "post_attention_layernorm.weight", &b), try t.taped(l, .attn_output), t.vec(.grad, h), t.vec(.branch, h), h, 1, true);
        try attention(t, l);
        const input = if (l == 0) try t.taped(null, .input) else try t.taped(l - 1, .residual);
        try stage.norm(t, try name(t, l, "input_layernorm.weight", &b), input, t.vec(.mixed, h), t.vec(.norm, h), h, 1, true);
        try t.ops().move(t.vec(.norm, h), t.vec(.grad, h), true, 1);
    }
}
fn pocket(t: anytype, l: usize) !void {
    const stored = t.r.pockets.layers[l] orelse return;
    if (stored.used == 0) return;
    const Ref = @TypeOf(t.buf(.dummy));
    const weights: Ref = .{ .buffer = stored.buffer, .len = stored.values.len * 4 };
    var p: core.living_rows.Params = .{};
    p.u[0] = @intCast(t.rows);
    p.u[1] = @intCast(t.shape.hidden);
    p.u[2] = @intCast(stored.used);
    p.u[3] = t.from[l];
    p.u[4] = 64;
    p.u[12] = 1;
    p.f[0] = t.seq.scale;
    try t.ops().row(.pocket_coeff, p, &.{ .{ 0, try t.taped(l, .mlp_input) }, .{ 1, weights }, .{ 2, t.vec(.branch, t.shape.hidden) }, .{ 8, t.buf(.coeff) } });
    try t.ops().row(.pocket_back, p, &.{ .{ 0, t.buf(.coeff) }, .{ 1, weights }, .{ 8, t.vec(.mixed, t.shape.hidden) } });
}
pub fn attention(t: anytype, l: usize) !void {
    const s = t.shape;
    const dim = s.dim(l);
    const kv = s.kv(l);
    const qw = s.heads * dim;
    const kw = kv * dim;
    var b: [224]u8 = undefined;
    try stage.matrixBack(t, try name(t, l, "self_attn.o_proj", &b), s.hidden, qw, t.vec(.branch, s.hidden), t.vec(.context, qw), false);
    var p: core.living_qsa.Params = .{};
    p.u[0] = @intCast(t.rows);
    p.u[1] = @intCast(s.heads);
    p.u[2] = @intCast(kv);
    p.u[3] = @intCast(dim);
    p.u[4] = @intCast(t.pack);
    p.u[5] = @TypeOf(t.e.*).probability_mode;
    p.u[6] = @intCast(dim / 128);
    p.f[0] = 1;
    const q = try t.taped(l, .query);
    const k = try t.taped(l, .key);
    const v = try t.taped(l, .value);
    try t.ops().qsa(.scores, p, &.{ .{ 0, q }, .{ 1, k }, .{ 2, v }, .{ 3, t.vec(.context, qw) }, .{ 4, t.buf(.ids) }, .{ 5, t.buf(.counts) }, .{ 6, t.buf(.sparse) }, .{ 8, t.buf(.scores) }, .{ 9, t.buf(.dp) } });
    try t.ops().qsa(.softmax, p, &.{ .{ 0, t.buf(.scores) }, .{ 1, t.buf(.dp) }, .{ 2, t.buf(.ids) }, .{ 3, t.buf(.counts) }, .{ 4, t.buf(.sparse) }, .{ 8, t.buf(.probability) }, .{ 9, t.buf(.ds) } });
    try t.ops().qsa(.query, p, &.{ .{ 0, t.buf(.ds) }, .{ 1, k }, .{ 2, t.buf(.ids) }, .{ 3, t.buf(.counts) }, .{ 4, t.buf(.sparse) }, .{ 8, t.vec(.q, qw) } });
    try t.ops().qsa(.key_value, p, &.{ .{ 0, t.buf(.ds) }, .{ 1, t.buf(.probability) }, .{ 2, q }, .{ 3, t.vec(.context, qw) }, .{ 4, t.buf(.ids) }, .{ 5, t.buf(.counts) }, .{ 6, t.buf(.sparse) }, .{ 8, t.vec(.k, kw) }, .{ 9, t.vec(.v, kw) } });
    var inv: [256]f32 = undefined;
    t.e.frequencies(l, inv[0 .. dim / 2]);
    try t.write(t.buf(.frequency), std.mem.sliceAsBytes(inv[0 .. dim / 2]));
    p = .{};
    p.u[0] = @intCast(t.rows);
    p.u[1] = @intCast(qw);
    p.u[2] = @intCast(dim);
    try t.ops().gemma(.rope, p, &.{ .{ 0, t.vec(.q, qw) }, .{ 1, t.buf(.frequency) }, .{ 8, t.vec(.raw, qw) } });
    try stage.norm(t, try name(t, l, "self_attn.q_norm.weight", &b), try t.taped(l, .q_raw), t.vec(.raw, qw), t.vec(.q, qw), dim, s.heads, true);
    try stage.matrixBack(t, try name(t, l, "self_attn.q_proj", &b), qw, s.hidden, t.vec(.q, qw), t.vec(.mixed, s.hidden), false);
    p.u[1] = @intCast(kw);
    try t.ops().gemma(.rope, p, &.{ .{ 0, t.vec(.k, kw) }, .{ 1, t.buf(.frequency) }, .{ 8, t.vec(.raw, kw) } });
    try stage.norm(t, try name(t, l, "self_attn.k_norm.weight", &b), try t.taped(l, .k_raw), t.vec(.raw, kw), t.vec(.k, kw), dim, kv, true);
    try stage.norm(t, "", try t.taped(l, .v_raw), t.vec(.v, kw), t.vec(.raw, kw), dim, kv, false);
    if (l % 6 == 5) {
        try t.ops().move(t.vec(.raw, kw), t.vec(.k, kw), true, 1);
    } else try stage.matrixBack(t, try name(t, l, "self_attn.v_proj", &b), kw, s.hidden, t.vec(.raw, kw), t.vec(.mixed, s.hidden), true);
    try stage.matrixBack(t, try name(t, l, "self_attn.k_proj", &b), kw, s.hidden, t.vec(.k, kw), t.vec(.mixed, s.hidden), true);
}
