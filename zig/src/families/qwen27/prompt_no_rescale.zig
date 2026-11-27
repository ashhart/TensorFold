//! Opt-in Qwen prompt kernels which omit only exact identity softmax rescales.
const std = @import("std");

const Replacement = struct { from: []const u8, to: []const u8, count: usize };

fn edited(a: std.mem.Allocator, source: []const u8, comptime replacements: anytype) ![]u8 {
    var size = source.len;
    inline for (replacements) |r| {
        const count = std.mem.count(u8, source, r.from);
        if (count != r.count) return error.AttentionSource;
        if (r.to.len >= r.from.len) size += (r.to.len - r.from.len) * count else size -= (r.from.len - r.to.len) * count;
    }
    const out = try a.alloc(u8, size);
    var at: usize = 0;
    var written: usize = 0;
    while (at < source.len) {
        var first = source.len;
        var which = replacements.len;
        inline for (replacements, 0..) |r, i| {
            if (std.mem.indexOfPos(u8, source, at, r.from)) |found| if (found < first) {
                first = found;
                which = i;
            };
        }
        if (which == replacements.len) break;
        const r = replacements[which];
        @memcpy(out[written..][0 .. first - at], source[at..first]);
        written += first - at;
        @memcpy(out[written..][0..r.to.len], r.to);
        written += r.to.len;
        at = first + r.from.len;
    }
    @memcpy(out[written..][0 .. source.len - at], source[at..]);
    written += source.len - at;
    std.debug.assert(written == out.len);
    return out;
}

const max_and_factor =
    "  const float nm0 = max(r.m0, x0), nm1 = max(r.m1, x1);\n" ++
    "  const float f0 = (x0 == -INFINITY) ? 1.0f : fast::exp2(r.m0 - nm0);\n" ++
    "  const float f1 = (x1 == -INFINITY) ? 1.0f : fast::exp2(r.m1 - nm1);\n";
const guarded_factor =
    "  const float nm0 = max(r.m0, x0), nm1 = max(r.m1, x1);\n" ++
    "  const bool keep0 = !NR || r.m0 != nm0 || !isfinite(r.m0) || !isfinite(x0);\n" ++
    "  const bool keep1 = !NR || r.m1 != nm1 || !isfinite(r.m1) || !isfinite(x1);\n" ++
    "  const float f0 = keep0 && x0 != -INFINITY ? fast::exp2(r.m0 - nm0) : 1.0f;\n" ++
    "  const float f1 = keep1 && x1 != -INFINITY ? fast::exp2(r.m1 - nm1) : 1.0f;\n";
const old_denominator =
    "  if (x0 != -INFINITY) { r.l0 = r.l0 * f0 + y0; r.m0 = nm0; }\n" ++
    "  if (x1 != -INFINITY) { r.l1 = r.l1 * f1 + y1; r.m1 = nm1; }";
const new_denominator =
    "  if (x0 != -INFINITY) { r.l0 = keep0 ? r.l0 * f0 + y0 : r.l0 + y0; r.m0 = nm0; }\n" ++
    "  if (x1 != -INFINITY) { r.l1 = keep1 ? r.l1 * f1 + y1 : r.l1 + y1; r.m1 = nm1; }";

/// Tensor-unit prompt source; its usual names select the `NR=true` instantiations.
pub fn tensor(a: std.mem.Allocator, source: []const u8) ![]u8 {
    return edited(a, source, [_]Replacement{
        .{ .from = "template <bool EDGE, bool MASKED>\ninline void q27_prompt_step", .to = "template <bool EDGE, bool MASKED, bool NR>\ninline void q27_prompt_step", .count = 1 },
        .{ .from = max_and_factor, .to = guarded_factor, .count = 1 },
        .{ .from = old_denominator, .to = new_denominator, .count = 1 },
        .{ .from = "  for (int j = 0; j < 8; j++) for (int i = 0; i < 4; i++) { r.O[j][i] *= f0; r.O[j][4 + i] *= f1; }", .to = "  for (int j = 0; j < 8; j++) for (int i = 0; i < 4; i++) { if (keep0) r.O[j][i] *= f0; if (keep1) r.O[j][4 + i] *= f1; }", .count = 1 },
        .{ .from = "template <int SG, bool MASKED>\nkernel void q27_attn_prompt(", .to = "template <int SG, bool MASKED, bool NR>\nkernel void q27_attn_prompt(", .count = 1 },
        .{ .from = "q27_prompt_step<false, false>", .to = "q27_prompt_step<false, false, NR>", .count = 2 },
        .{ .from = "q27_prompt_step<true, false>", .to = "q27_prompt_step<true, false, NR>", .count = 1 },
        .{ .from = "q27_prompt_step<true, true>", .to = "q27_prompt_step<true, true, NR>", .count = 1 },
        .{ .from = "#define Q27_PROMPT(SG, M, NAME) template [[host_name(NAME #SG)]] kernel void q27_attn_prompt<SG, M>(device const bfloat*, device const Q27AttnReadPtrs*, constant Q27PromptArgs&, device float*, device float*, device float*, uint3, ushort, ushort);\n" ++
            "Q27_PROMPT(8, false, \"q27_attn_prompt_\")\n" ++
            "Q27_PROMPT(8, true, \"q27_attn_prompt_masked_\")", .to = "#define Q27_PROMPT(SG, M, NR, NAME) template [[host_name(NAME #SG)]] kernel void q27_attn_prompt<SG, M, NR>(device const bfloat*, device const Q27AttnReadPtrs*, constant Q27PromptArgs&, device float*, device float*, device float*, uint3, ushort, ushort);\n" ++
            "Q27_PROMPT(8, false, true, \"q27_attn_prompt_\")\n" ++
            "Q27_PROMPT(8, true, true, \"q27_attn_prompt_masked_\")", .count = 1 },
    });
}

/// Simdgroup-matrix prompt source; its usual names select the `NR=true` instantiations.
pub fn simdgroup(a: std.mem.Allocator, source: []const u8) ![]u8 {
    return edited(a, source, [_]Replacement{
        .{ .from = "template <bool EDGE, bool MASKED>\ninline void q27_sg_step", .to = "template <bool EDGE, bool MASKED, bool NR>\ninline void q27_sg_step", .count = 1 },
        .{ .from = max_and_factor, .to = guarded_factor, .count = 1 },
        .{ .from = old_denominator, .to = new_denominator, .count = 1 },
        .{ .from = "    for (int i = 0; i < 2; i++) { r.O[0][e].thread_elements()[i] *= f0; r.O[1][e].thread_elements()[i] *= f1; }", .to = "    for (int i = 0; i < 2; i++) { if (keep0) r.O[0][e].thread_elements()[i] *= f0; if (keep1) r.O[1][e].thread_elements()[i] *= f1; }", .count = 1 },
        .{ .from = "template <int SG, bool MASKED>\nkernel void q27_attn_prompt_sg(", .to = "template <int SG, bool MASKED, bool NR>\nkernel void q27_attn_prompt_sg(", .count = 1 },
        .{ .from = "q27_sg_step<false, false>", .to = "q27_sg_step<false, false, NR>", .count = 2 },
        .{ .from = "q27_sg_step<true, false>", .to = "q27_sg_step<true, false, NR>", .count = 1 },
        .{ .from = "q27_sg_step<true, true>", .to = "q27_sg_step<true, true, NR>", .count = 1 },
        .{ .from = "#define Q27_PROMPT_SG(SG, M, NAME) template [[host_name(NAME #SG)]] kernel void q27_attn_prompt_sg<SG, M>(device const bfloat*, device const Q27AttnReadPtrs*, constant Q27PromptArgs&, device float*, device float*, device float*, uint3, ushort, ushort);\n" ++
            "Q27_PROMPT_SG(8, false, \"q27_attn_prompt_sg_\")\n" ++
            "Q27_PROMPT_SG(8, true, \"q27_attn_prompt_sg_masked_\")", .to = "#define Q27_PROMPT_SG(SG, M, NR, NAME) template [[host_name(NAME #SG)]] kernel void q27_attn_prompt_sg<SG, M, NR>(device const bfloat*, device const Q27AttnReadPtrs*, constant Q27PromptArgs&, device float*, device float*, device float*, uint3, ushort, ushort);\n" ++
            "Q27_PROMPT_SG(8, false, true, \"q27_attn_prompt_sg_\")\n" ++
            "Q27_PROMPT_SG(8, true, true, \"q27_attn_prompt_sg_masked_\")", .count = 1 },
    });
}

/// A planted source fault: maximum changes no longer force the original rescale.
pub fn fault(a: std.mem.Allocator, source: []const u8, units: bool) ![]u8 {
    return if (units) edited(a, source, [_]Replacement{
        .{ .from = "r.m0 != nm0", .to = "false", .count = 1 },
        .{ .from = "r.m1 != nm1", .to = "false", .count = 1 },
    }) else edited(a, source, [_]Replacement{
        .{ .from = "r.m0 != nm0", .to = "false", .count = 2 },
        .{ .from = "r.m1 != nm1", .to = "false", .count = 2 },
    });
}

/// Host model of the shader guard; true retains the original denominator and accumulator arithmetic.
pub fn keep(old_max: f32, new_max: f32, tile_max: f32) bool {
    return old_max != new_max or !std.math.isFinite(old_max) or !std.math.isFinite(tile_max);
}

test "Qwen prompt rescale guard is exhaustive over ordinary and non-finite maxima" {
    const p = std.math.inf(f32);
    const n = std.math.nan(f32);
    const values = [_]f32{ -p, -1, -0.0, 0.0, 1, p, n };
    for (values) |old| for (values) |new| for (values) |tile| {
        const expected = !(old == new and std.math.isFinite(old) and std.math.isFinite(tile));
        try std.testing.expectEqual(expected, keep(old, new, tile));
    };
    try std.testing.expect(keep(1, 2, 2));
    try std.testing.expect(!keep(-0.0, 0.0, -0.0));
    for (values) |x| if (!std.math.isNan(x)) try std.testing.expectEqual(@as(u32, @bitCast(x)), @as(u32, @bitCast(x * 1.0)));
}

test "Qwen prompt source generation adds guarded tensor and simdgroup variants" {
    const a = std.testing.allocator;
    const m5 = try tensor(a, @embedFile("../../../kernels/metal/qwen27/attention_prompt.metal"));
    defer a.free(m5);
    const sg = try simdgroup(a, @embedFile("../../../kernels/metal/qwen27/attention_prompt_sg.metal"));
    defer a.free(sg);
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, m5, "bool NR"));
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, sg, "bool NR"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, m5, "Q27_PROMPT(8, false, true, \"q27_attn_prompt_\")"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, m5, "Q27_PROMPT(8, true, true, \"q27_attn_prompt_masked_\")"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, sg, "Q27_PROMPT_SG(8, false, true, \"q27_attn_prompt_sg_\")"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, sg, "Q27_PROMPT_SG(8, true, true, \"q27_attn_prompt_sg_masked_\")"));
    try std.testing.expectEqual(@as(usize, 5), std.mem.count(u8, m5, ", NR>"));
    try std.testing.expectEqual(@as(usize, 5), std.mem.count(u8, sg, ", NR>"));
}

test "Qwen paged prompt sources accept the same guarded variants and planted fault" {
    const a = std.testing.allocator;
    const paged = @import("../../../kernels/metal/qwen27/paged.zig");
    const m5 = try tensor(a, paged.attention_prompt);
    defer a.free(m5);
    const sg = try simdgroup(a, paged.attention_prompt_sg);
    defer a.free(sg);
    const body = try std.mem.concat(a, u8, &.{ m5, sg });
    defer a.free(body);
    const bad = try fault(a, body, false);
    defer a.free(bad);
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, bad, "r.m0 != nm0"));
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, bad, "r.m1 != nm1"));
    try std.testing.expectEqual(@as(usize, 4), std.mem.count(u8, bad, "!NR || false ||"));
    try std.testing.expectError(error.AttentionSource, fault(a, m5, false));
}
