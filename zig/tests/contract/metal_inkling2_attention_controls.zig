//! Count-checked, discriminating source edits for every Inkling 2 attention shader entry point.
const std = @import("std");
const contract = @import("contract");

pub const Which = enum { prep_sliding, prep_global, rel_table, decode_sliding, decode_global, prompt_m5, prompt_pre_m5 };
pub fn function(w: Which) []const u8 {
    return switch (w) {
        .prep_sliding, .prep_global => "tf_ink2_prep",
        .rel_table => "tf_ink2_rel_table",
        .decode_sliding, .decode_global => "tf_ink2_attn_scalar",
        .prompt_m5, .prompt_pre_m5 => "tf_ink2_attn_prompt",
    };
}
pub fn isPrompt(w: Which) bool {
    return w == .prompt_m5 or w == .prompt_pre_m5;
}
pub fn isAttention(w: Which) bool {
    return w == .decode_sliding or w == .decode_global or isPrompt(w);
}

pub const Mutation = enum {
    none,
    prep_swap_kv,
    prep_tau_off,
    rel_late,
    decode_late_bias,
    merge_no_sink,
    prompt_late_bias,
    prompt_pre_late_bias,
    pub fn name(m: Mutation) []const u8 {
        return switch (m) {
            .none => "",
            .prep_swap_kv => "control: metal inkling2 attention prep sliding with gated K/V inputs exchanged tf_ink2_prep",
            .prep_tau_off => "control: metal inkling2 attention global prep without tau tf_ink2_prep",
            .rel_late => "control: metal inkling2 attention relative projection one column late tf_ink2_rel_table",
            .decode_late_bias => "control: metal inkling2 attention sliding first key score boosted tf_ink2_attn_scalar",
            .merge_no_sink => "control: metal inkling2 attention global merge without sink tf_ink2_merge",
            .prompt_late_bias => "control: metal inkling2 prompt m5 first key bias boosted tf_ink2_attn_prompt",
            .prompt_pre_late_bias => "control: metal inkling2 prompt pre-m5 first key bias boosted tf_ink2_attn_prompt",
        };
    }
    pub fn gate(m: Mutation) ?contract.gates.Gate {
        return if (m == .none) null else .shapes;
    }
    pub fn which(m: Mutation) Which {
        return switch (m) {
            .none => unreachable,
            .prep_swap_kv => .prep_sliding,
            .prep_tau_off => .prep_global,
            .rel_late => .rel_table,
            .decode_late_bias => .decode_sliding,
            .merge_no_sink => .decode_global,
            .prompt_late_bias => .prompt_m5,
            .prompt_pre_late_bias => .prompt_pre_m5,
        };
    }
    const Edit = struct { from: []const u8, to: []const u8 };
    fn edit(m: Mutation) Edit {
        return switch (m) {
            .none => unreachable,
            .prep_swap_kv => .{
                .from = "const device bfloat* src = (key ? k_raw : v_raw) + (size_t(row) * a.kv_heads + h) * D;",
                .to = "const device bfloat* src = (key ? v_raw : k_raw) + (size_t(row) * a.kv_heads + h) * D;",
            },
            .prep_tau_off => .{
                .from = "const float tau = a.global ? 1.0f + a.alpha * precise::log(max(float(pos + 1) / a.floor_pos, 1.0f)) : 1.0f;",
                .to = "const float tau = 1.0f;",
            },
            .rel_late => .{
                .from = "acc += r[j] * float(proj[j * EXTENT + d]);",
                .to = "acc += r[j] * float(proj[j * EXTENT + (d + 1) % EXTENT]);",
            },
            .decode_late_bias => .{
                .from = "const float score = (dot_score * a.scale + bias) * LOG2E;",
                .to = "const float score = (dot_score * a.scale + bias + (kp == from ? 128.0f : 0.0f)) * LOG2E;",
            },
            .merge_no_sink => .{
                .from = "const float sk = float(sink[rh]) * LOG2E, mf = max(mm, sk), a0 = precise::exp2(mm - mf);",
                .to = "const float sk = -INFINITY, mf = max(mm, sk), a0 = precise::exp2(mm - mf);",
            },
            .prompt_late_bias, .prompt_pre_late_bias => .{
                .from = "const float bias = live && distance < EXTENT ? float(tab[prow * EXTENT + distance]) : 0.0f;",
                .to = "const float bias = (live && distance < EXTENT ? float(tab[prow * EXTENT + distance]) : 0.0f) + (live && kp == b_lo ? 128.0f : 0.0f);",
            },
        };
    }
};

pub fn apply(a: std.mem.Allocator, text: []const u8, m: Mutation) ![]u8 {
    if (m == .none) return a.dupe(u8, text);
    const e = m.edit();
    if (std.mem.count(u8, text, e.from) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, text, e.from, e.to);
}

test "Inkling 2 attention controls each hit one live statement and cannot apply twice" {
    const a = std.testing.allocator;
    const sources = @import("inkling2_metal");
    for (std.enums.values(Mutation)) |m| {
        if (m == .none) continue;
        const w = m.which();
        const plain = if (isPrompt(w)) (if (w == .prompt_pre_m5) sources.prompt_attention_pre_m5 else sources.prompt_attention) else sources.attention;
        const edited = try apply(a, plain, m);
        defer a.free(edited);
        try std.testing.expect(std.mem.indexOf(u8, edited, m.edit().to) != null);
        try std.testing.expectError(error.ControlEditMissing, apply(a, edited, m));
        try std.testing.expectEqual(contract.gates.Gate.shapes, m.gate().?);
    }
    const unchanged = try apply(a, sources.attention, .none);
    defer a.free(unchanged);
    try std.testing.expectEqualStrings(sources.attention, unchanged);
}

fn fp64Clip(src: []const f64, dst: []f64, threshold: f64) void {
    var ss: f64 = 0;
    for (src) |x| ss += x * x;
    const f = @min(threshold / (@sqrt(ss) + 1e-6), 1);
    for (dst, src) |*y, x| y.* = x * f;
}

fn fp64Rms(src: []const f64, dst: []f64) void {
    var ss: f64 = 0;
    for (src) |x| ss += x * x;
    const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(src.len)) + 1e-6);
    for (dst, src) |*y, x| y.* = x * inv;
}

test "Inkling 2 gated FP64 prep exchanges only K/V" {
    const q = [4]f64{ 4, 0, 0, 0 };
    const rel = [4]f64{ 4, 0, 0, 0 };
    const k = [4]f64{ 4, 0, 0, 0 };
    const v = [4]f64{ 0, 3, 0, 0 };
    var q_base: [4]f64 = undefined;
    var q_swap: [4]f64 = undefined;
    var rel_base: [4]f64 = undefined;
    var rel_swap: [4]f64 = undefined;
    var k_base: [4]f64 = undefined;
    var k_swap: [4]f64 = undefined;
    var v_base: [4]f64 = undefined;
    var v_swap: [4]f64 = undefined;
    fp64Clip(&q, &q_base, 5);
    fp64Clip(&q, &q_swap, 5);
    fp64Clip(&rel, &rel_base, 5);
    fp64Clip(&rel, &rel_swap, 5);
    fp64Clip(&k, &k_base, 5);
    fp64Clip(&v, &k_swap, 5);
    fp64Rms(&v, &v_base);
    fp64Rms(&k, &v_swap);
    try std.testing.expectEqualSlices(f64, &q_base, &q_swap);
    try std.testing.expectEqualSlices(f64, &rel_base, &rel_swap);
    try std.testing.expectEqualSlices(f64, &.{ 4, 0, 0, 0 }, &k_base);
    try std.testing.expectEqualSlices(f64, &.{ 0, 3, 0, 0 }, &k_swap);
    const v_from_v = 3 * (1 / @sqrt(@as(f64, 9) / 4 + 1e-6));
    const v_from_k = 4 * (1 / @sqrt(@as(f64, 16) / 4 + 1e-6));
    try std.testing.expectEqualSlices(f64, &.{ 0, v_from_v, 0, 0 }, &v_base);
    try std.testing.expectEqualSlices(f64, &.{ v_from_k, 0, 0, 0 }, &v_swap);
}
