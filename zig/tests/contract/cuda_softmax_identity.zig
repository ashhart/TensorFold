//! Host bit-reference for CUDA's finite stable-max old-state rescale guard.
const std = @import("std");

const Style = enum { plain, plain_exp2, legacy_equal_exp2, fused, two_products, two_products_exp2 };
const State = struct { m: f32, l: f32, o: f32 };
const Case = struct {
    old: State,
    new_m: f32,
    part_m: f32,
    new_l: f32,
    new_o: f32,
};
const Site = struct { name: []const u8, style: Style };

const sites = [_]Site{
    .{ .name = "block and sparse decode", .style = .fused },
    .{ .name = "idol decode and prompt", .style = .two_products },
    .{ .name = "prefill and row-prefill", .style = .two_products },
    .{ .name = "tree tiles with masks", .style = .two_products },
    .{ .name = "window decode and verify", .style = .two_products },
    .{ .name = "window rows and split merge", .style = .two_products },
    .{ .name = "common MLA tiles", .style = .fused },
    .{ .name = "FlashNext tile/chunk/chart", .style = .two_products },
    .{ .name = "Nemotron chunk and prefetch", .style = .fused },
    .{ .name = "paged segmented attention", .style = .fused },
    .{ .name = "Qwen scalar/KV8/draft/MTP", .style = .plain },
    .{ .name = "Qwen tensor decode/verify", .style = .fused },
    .{ .name = "Qwen chunk merges", .style = .fused },
    .{ .name = "Inkling decode/prompt folds", .style = .two_products_exp2 },
    .{ .name = "H3 exp2 verify", .style = .plain_exp2 },
    .{ .name = "MiniMax exp2 tile", .style = .legacy_equal_exp2 },
    .{ .name = "Drama attention/codec/merge", .style = .plain },
    .{ .name = "GLM DFlash folds/merge", .style = .plain },
    .{ .name = "Muse decode/DFlash/merge", .style = .plain },
    .{ .name = "Moss and Qwen vision", .style = .plain },
    .{ .name = "HY4 sink and MLA merge", .style = .plain },
};

const controlled_sources = [_][]const u8{
    @embedFile("../../kernels/cuda/attn_fold.cuh"),
    @embedFile("../../kernels/cuda/common/mla.cuh"),
    @embedFile("../../kernels/cuda/drama3_attn.cuh"),
    @embedFile("../../kernels/cuda/drama3_codec.cu"),
    @embedFile("../../kernels/cuda/flashnext_attn.cu"),
    @embedFile("../../kernels/cuda/glm_dflash.cu"),
    @embedFile("../../kernels/cuda/glm_glue_attn.cuh"),
    @embedFile("../../kernels/cuda/h3_attention.cuh"),
    @embedFile("../../kernels/cuda/hy4_flash/attention_sink.cu"),
    @embedFile("../../kernels/cuda/idol_attention.cuh"),
    @embedFile("../../kernels/cuda/idol_decoder_attention.cuh"),
    @embedFile("../../kernels/cuda/inkling2/attention.cu"),
    @embedFile("../../kernels/cuda/inkling2/prompt_attention.cu"),
    @embedFile("../../kernels/cuda/minimax_m3_attn.cu"),
    @embedFile("../../kernels/cuda/moss_image_encoder.cu"),
    @embedFile("../../kernels/cuda/muse_spark/attention.cu"),
    @embedFile("../../kernels/cuda/muse_spark/dflash.cu"),
    @embedFile("../../kernels/cuda/nemotron_attention.cu"),
    @embedFile("../../kernels/cuda/paged_attn_segs.cuh"),
    @embedFile("../../kernels/cuda/prefill_attention.cu"),
    @embedFile("../../kernels/cuda/prefill_attention_rows.cuh"),
    @embedFile("../../kernels/cuda/qwen35.cu"),
    @embedFile("../../kernels/cuda/qwen35_attn.cuh"),
    @embedFile("../../kernels/cuda/qwen35_attn_tc.cuh"),
    @embedFile("../../kernels/cuda/qwen35_draft.cu"),
    @embedFile("../../kernels/cuda/qwen35_draft_frame.cuh"),
    @embedFile("../../kernels/cuda/qwen35_kv8.cu"),
    @embedFile("../../kernels/cuda/qwen36_mtp.cu"),
    @embedFile("../../kernels/cuda/qwen_vision.cu"),
    @embedFile("../../kernels/cuda/tree_attention.cu"),
    @embedFile("../../kernels/cuda/window_attention.cu"),
    @embedFile("../../kernels/cuda/window_rows.cuh"),
    @embedFile("../../kernels/cuda/window_split.cuh"),
};

test "CUDA identity source inventory has default-on compile-time controls" {
    const header = @embedFile("../../kernels/cuda/common/softmax_identity.cuh");
    try std.testing.expect(std.mem.indexOf(u8, header, "#define TF_CUDA_SOFTMAX_IDENTITY_GUARD 1") != null);
    const flag = "TF_CUDA_SOFTMAX_IDENTITY_GUARD";
    inline for (controlled_sources) |text| {
        try std.testing.expect(std.mem.indexOf(u8, text, "softmax_identity.cuh") != null);
        try std.testing.expect(std.mem.indexOf(u8, text, flag ++ " && ") != null);
    }
}

fn bits(x: f32) u32 {
    return @bitCast(x);
}

fn fromBits(x: u32) f32 {
    return @bitCast(x);
}

fn sameBits(a: State, b: State) bool {
    return bits(a.m) == bits(b.m) and bits(a.l) == bits(b.l) and bits(a.o) == bits(b.o);
}

fn expFor(comptime style: Style, x: f32) f32 {
    return switch (style) {
        .plain_exp2, .legacy_equal_exp2, .two_products_exp2 => @exp2(x),
        else => @exp(x),
    };
}

fn reference(c: Case, comptime style: Style) State {
    @setFloatMode(.strict);
    const a = if (style == .legacy_equal_exp2 and c.old.m == c.new_m) 1.0 else expFor(style, c.old.m - c.new_m);
    const b = expFor(style, c.part_m - c.new_m);
    return switch (style) {
        .plain, .plain_exp2, .legacy_equal_exp2 => .{ .m = c.new_m, .l = c.old.l * a + c.new_l, .o = c.old.o * a + c.new_o },
        .fused => .{ .m = c.new_m, .l = @mulAdd(f32, c.old.l, a, c.new_l), .o = @mulAdd(f32, c.old.o, a, c.new_o) },
        .two_products, .two_products_exp2 => .{ .m = c.new_m, .l = c.old.l * a + c.new_l * b, .o = c.old.o * a + c.new_o * b },
    };
}

fn optimized(c: Case, comptime style: Style) State {
    @setFloatMode(.strict);
    const same = std.math.isFinite(c.new_m) and c.new_m == c.old.m;
    const a = if (same or (style == .legacy_equal_exp2 and c.old.m == c.new_m)) 1.0 else expFor(style, c.old.m - c.new_m);
    const b = expFor(style, c.part_m - c.new_m);
    const fast: State = switch (style) {
        .plain, .plain_exp2, .legacy_equal_exp2, .fused => .{ .m = c.new_m, .l = c.old.l + c.new_l, .o = c.old.o + c.new_o },
        .two_products, .two_products_exp2 => .{ .m = c.new_m, .l = c.old.l + c.new_l * b, .o = c.old.o + c.new_o * b },
    };
    const legacy: State = switch (style) {
        .plain, .plain_exp2, .legacy_equal_exp2 => .{ .m = c.new_m, .l = c.old.l * a + c.new_l, .o = c.old.o * a + c.new_o },
        .fused => .{ .m = c.new_m, .l = @mulAdd(f32, c.old.l, a, c.new_l), .o = @mulAdd(f32, c.old.o, a, c.new_o) },
        .two_products, .two_products_exp2 => .{ .m = c.new_m, .l = c.old.l * a + c.new_l * b, .o = c.old.o * a + c.new_o * b },
    };
    return .{ .m = c.new_m, .l = if (same and std.math.isFinite(fast.l)) fast.l else legacy.l, .o = if (same and std.math.isFinite(fast.o)) fast.o else legacy.o };
}

// Deliberately broken control: equal infinities take the shortcut and lose the original NaNs.
pub fn plantedControl(c: Case) State {
    const same = c.new_m == c.old.m;
    const a = if (same) 1.0 else @exp(c.old.m - c.new_m);
    return if (same)
        .{ .m = c.new_m, .l = c.old.l + c.new_l, .o = c.old.o + c.new_o }
    else
        .{ .m = c.new_m, .l = @mulAdd(f32, c.old.l, a, c.new_l), .o = @mulAdd(f32, c.old.o, a, c.new_o) };
}

const nan = fromBits(0x7fc12345);
const pos_inf = std.math.inf(f32);
const neg_inf = -std.math.inf(f32);
const subnormal = fromBits(1);
const largest_subnormal = fromBits(0x007fffff);
const smallest_normal = fromBits(0x00800000);

const cases = [_]Case{
    .{ .old = .{ .m = 3.5, .l = subnormal, .o = -0.0 }, .new_m = 3.5, .part_m = 3.5, .new_l = largest_subnormal, .new_o = 0.0 },
    .{ .old = .{ .m = -0.0, .l = -0.0, .o = subnormal }, .new_m = 0.0, .part_m = -0.0, .new_l = 0.0, .new_o = smallest_normal },
    .{ .old = .{ .m = -17.25, .l = 0x1.921fb6p1, .o = -9.75 }, .new_m = -17.25, .part_m = -20.0, .new_l = 0x1p-120, .new_o = -0.0 },
    .{ .old = .{ .m = 1.0, .l = 2.5, .o = -7.0 }, .new_m = 2.0, .part_m = 2.0, .new_l = 1.0, .new_o = 11.0 },
    .{ .old = .{ .m = 2.0, .l = 2.5, .o = -7.0 }, .new_m = nan, .part_m = neg_inf, .new_l = 0.0, .new_o = 0.0 },
    .{ .old = .{ .m = nan, .l = 2.5, .o = -7.0 }, .new_m = nan, .part_m = nan, .new_l = 0.0, .new_o = 0.0 },
    .{ .old = .{ .m = pos_inf, .l = 2.5, .o = -7.0 }, .new_m = pos_inf, .part_m = pos_inf, .new_l = 0.0, .new_o = 0.0 },
    .{ .old = .{ .m = neg_inf, .l = 0.0, .o = 0.0 }, .new_m = neg_inf, .part_m = neg_inf, .new_l = 0.0, .new_o = 0.0 },
    .{ .old = .{ .m = 2.0, .l = 2.5, .o = -7.0 }, .new_m = pos_inf, .part_m = neg_inf, .new_l = 0.0, .new_o = 0.0 },
    .{ .old = .{ .m = 2.0, .l = 2.5, .o = -7.0 }, .new_m = neg_inf, .part_m = neg_inf, .new_l = 0.0, .new_o = 0.0 },
    .{ .old = .{ .m = 2.0, .l = nan, .o = -0.0 }, .new_m = 2.0, .part_m = 2.0, .new_l = 1.0, .new_o = nan },
    .{ .old = .{ .m = 2.0, .l = 2.5, .o = pos_inf }, .new_m = 2.0, .part_m = 2.0, .new_l = nan, .new_o = neg_inf },
};

fn checkStyle(comptime style: Style) !void {
    for (cases) |c| {
        const want = reference(c, style);
        const got = optimized(c, style);
        try std.testing.expect(sameBits(want, got));
    }
}

test "CUDA identity guard is bit-equal for all arithmetic families" {
    try checkStyle(.plain);
    try checkStyle(.plain_exp2);
    try checkStyle(.fused);
    try checkStyle(.legacy_equal_exp2);
    try checkStyle(.two_products);
    try checkStyle(.two_products_exp2);
}

test "CUDA served decode verify prompt tree split and merge variants" {
    for (sites) |site| switch (site.style) {
        .plain => for (cases) |c| try std.testing.expect(sameBits(reference(c, .plain), optimized(c, .plain))),
        .plain_exp2 => for (cases) |c| try std.testing.expect(sameBits(reference(c, .plain_exp2), optimized(c, .plain_exp2))),
        .legacy_equal_exp2 => for (cases) |c| try std.testing.expect(sameBits(reference(c, .legacy_equal_exp2), optimized(c, .legacy_equal_exp2))),
        .fused => for (cases) |c| try std.testing.expect(sameBits(reference(c, .fused), optimized(c, .fused))),
        .two_products => for (cases) |c| try std.testing.expect(sameBits(reference(c, .two_products), optimized(c, .two_products))),
        .two_products_exp2 => for (cases) |c| try std.testing.expect(sameBits(reference(c, .two_products_exp2), optimized(c, .two_products_exp2))),
    };
}

test "planted nonfinite-max control fails against the pre-guard arithmetic" {
    const c = cases[6];
    try std.testing.expect(!sameBits(reference(c, .fused), plantedControl(c)));
}

test "CUDA identity keeps incoming-product FMA and rejects separated rounding" {
    const weight: f32 = fromBits(0x3f800001);
    const value: f32 = fromBits(0x3f7ffffe);
    const old: f32 = fromBits(0x33800001);
    const legacy = @mulAdd(f32, weight, value, old * 1.0);
    const skipped = @mulAdd(f32, weight, value, old);
    const wrong = weight * value + old;
    try std.testing.expectEqual(bits(legacy), bits(skipped));
    try std.testing.expect(bits(legacy) != bits(wrong));
    const source = @embedFile("../../kernels/cuda/qwen35_attn.cuh");
    try std.testing.expect(std.mem.indexOf(u8, source, "? fmaf(b, bf(vb[i]), o[i]) : fmaf(b, bf(vb[i]), o[i] * a)") != null);
}

test "CUDA identity preserves MiniMax's legacy equal-nonfinite factor" {
    const body = @embedFile("../../kernels/cuda/minimax_m3_attn.cu");
    try std.testing.expect(std.mem.indexOf(u8, body, "const float f0 = m0 == mx0 ? 1.f : exp2f(m0 - mx0)") != null);
}
fn unsafe(text: []const u8) bool {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (std.mem.indexOf(u8, line, "same") != null and std.mem.indexOf(u8, line, "rowsum(") != null) return true;
    }
    return false;
}
test "CUDA identity full-mask row sums precede every row-dependent identity arm" {
    const body = @embedFile("../../kernels/cuda/prefill_attention_rows.cuh");
    try std.testing.expect(!unsafe(body));
    const sum = std.mem.indexOf(u8, body, "const float sum0 = rowsum(c0), sum1 = rowsum(c1);") orelse return error.MissingUnconditionalSum;
    const guard = std.mem.indexOf(u8, body, "l0 = (same0") orelse return error.MissingIdentityGuard;
    try std.testing.expect(sum < guard);
    try std.testing.expect(unsafe("l0 = (same0 && isfinite(rowsum(c0))) ? rowsum(c0) : 0;"));
}
test "CUDA identity pure rescale retains nonfinite accumulator operations" {
    const body = @embedFile("../../kernels/cuda/attn_fold.cuh");
    try std.testing.expect(std.mem.indexOf(u8, body, "!same0 || !isfinite(o[i][0])") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "!same1 || !isfinite(o[i][3])") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "l0 = (same0 && isfinite(__fadd_rn(l0, c0)))") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "l1 = (same1 && isfinite(__fadd_rn(l1, c1)))") != null);
}

test "CUDA identity guards never contain warp or block collectives" {
    for (controlled_sources) |source| {
        var lines = std.mem.splitScalar(u8, source, '\n');
        while (lines.next()) |line| {
            if (std.mem.indexOf(u8, line, "same") == null) continue;
            if (std.mem.indexOf(u8, line, "?") == null and std.mem.indexOf(u8, line, "if (") == null) continue;
            for ([_][]const u8{ "rowsum(", "warp_sum(", "warp_max(", "__shfl", "__syncwarp", "__syncthreads" }) |call| {
                try std.testing.expect(std.mem.indexOf(u8, line, call) == null);
            }
        }
    }
}
