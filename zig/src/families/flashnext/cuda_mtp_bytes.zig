//! The Flash Next MTP head's device bytes beyond the main model's: its weights, scratch and shared expert by form.
const std = @import("std");
const wt = @import("cuda_prompt_wt.zig");
const round_rows = @import("cuda_mtp.zig").round_rows;

const dims = wt.dims;

/// The split best's scratch: each warp's best and id bits, its exponent sum, then a counter a row (round_rows rows).
pub const split_bytes: usize = round_rows * 32 * 8 + round_rows * 32 * 4 + round_rows * 4;

/// A shared projection's stored form: the affine pair, a whole BF16 tensor, or MXFP8 codes with their scales.
pub const Format = enum { affine, bf16, mxfp8 };

/// The head's shared expert on the device: its intermediate width and each of the three projections' stored form.
pub const Shared = struct {
    width: usize,
    gate: Format = .affine,
    up: Format = .affine,
    down: Format = .affine,
};

/// Device bytes a shared projection of `n` rows by `k` takes in `form`: 5/8 a weight affine, two BF16, 33/32 MXFP8.
fn sharedBytes(n: usize, k: usize, form: Format) usize {
    const count = n * k;
    return switch (form) {
        .affine => count / 2 * 5 / 4,
        .bf16 => 2 * count,
        .mxfp8 => count + count / 32,
    };
}

/// Device bytes the head's weights take beyond the main model's: `fp8` routed experts, `shared` the shared expert.
pub fn headBytes(fp8: bool, shared: Shared) usize {
    const weights = 3 * wt.moe_w * dims;
    const nvfp4 = weights / 2 * 5 / 4;
    const routed = if (fp8) weights + 3 * (wt.moe_w / 128) * (dims / 128) * 4 else nvfp4;
    const gate_up = sharedBytes(shared.width, dims, shared.gate) + sharedBytes(shared.width, dims, shared.up);
    const experts = wt.experts_n * routed + gate_up + sharedBytes(dims, shared.width, shared.down);
    const attn = (wt.proj_n + dims) * dims / 2 * 5 / 4;
    const draft = 79_591 * dims / 2 * 5 / 4;
    return experts + attn + draft + 2 * dims * dims / 2 * 5 / 4 + @import("cuda_draw_batch.zig").bytes + @import("cuda_draft_draw.zig").scoreBytes(79_591) + split_bytes;
}

test "the head's bytes: NVFP4 experts as before, FP8 routed experts a byte a weight and their f32 block scales" {
    const before = (wt.experts_n + 1) * 3 * wt.moe_w * dims / 2 * 5 / 4 + (wt.proj_n + dims) * dims / 2 * 5 / 4 + 79_591 * dims / 2 * 5 / 4 + 2 * dims * dims / 2 * 5 / 4;
    try std.testing.expectEqual(@as(usize, 24_832), split_bytes);
    try std.testing.expectEqual(before + @import("cuda_draw_batch.zig").bytes + @import("cuda_draft_draw.zig").scoreBytes(79_591) + split_bytes, headBytes(false, .{ .width = wt.moe_w }));
    // an expert's three projections: 640 x 2560 bytes and 5 x 20 scales each, against 5/8 of a byte a weight
    try std.testing.expectEqual(wt.experts_n * (3 * (640 * 2560 + 5 * 20 * 4) - 3 * 640 * 2560 * 5 / 8), headBytes(true, .{ .width = wt.moe_w }) - headBytes(false, .{ .width = wt.moe_w }));
}

test "a 1280-wide shared expert adds its three projections: the 3,072,000 bytes the 640 width undercounted" {
    // gate, up and down at 5/8 of a byte a weight over the 640 rows a 1280-wide expert adds
    const added = 3 * (1280 - wt.moe_w) * dims / 2 * 5 / 4;
    try std.testing.expectEqual(@as(usize, 3_072_000), added);
    try std.testing.expectEqual(added, headBytes(false, .{ .width = 1280 }) - headBytes(false, .{ .width = wt.moe_w }));
    try std.testing.expectEqual(added, headBytes(true, .{ .width = 1280 }) - headBytes(true, .{ .width = wt.moe_w }));
    // the routed experts' format moves the head by the same bytes at either shared width
    try std.testing.expectEqual(headBytes(true, .{ .width = wt.moe_w }) - headBytes(false, .{ .width = wt.moe_w }), headBytes(true, .{ .width = 1280 }) - headBytes(false, .{ .width = 1280 }));
}

test "the shared expert's bytes follow the form it is stored in: affine, BF16 and MXFP8 at both widths" {
    const bf16: Shared = .{ .width = 1280, .gate = .bf16, .up = .bf16, .down = .bf16 };
    const mxfp8: Shared = .{ .width = 1280, .gate = .mxfp8, .up = .mxfp8, .down = .mxfp8 };
    // the whole expert against no shared rows: 9,830,400 weights at 5/8, two BF16 bytes, a code and a scale a 32
    try std.testing.expectEqual(@as(usize, 6_144_000), headBytes(false, .{ .width = 1280 }) - headBytes(false, .{ .width = 0 }));
    try std.testing.expectEqual(@as(usize, 19_660_800), headBytes(false, bf16) - headBytes(false, .{ .width = 0 }));
    try std.testing.expectEqual(@as(usize, 10_137_600), headBytes(false, mxfp8) - headBytes(false, .{ .width = 0 }));
    // the correction's own rows, 640 wider: 9,830,400 bytes BF16 and 5,068,800 MXFP8 against 3,072,000 affine
    const affine_added = headBytes(false, .{ .width = 1280 }) - headBytes(false, .{ .width = wt.moe_w });
    const bf16_added = headBytes(false, bf16) - headBytes(false, .{ .width = wt.moe_w, .gate = .bf16, .up = .bf16, .down = .bf16 });
    const mxfp8_added = headBytes(false, mxfp8) - headBytes(false, .{ .width = wt.moe_w, .gate = .mxfp8, .up = .mxfp8, .down = .mxfp8 });
    try std.testing.expectEqual(@as(usize, 9_830_400), bf16_added);
    try std.testing.expectEqual(@as(usize, 5_068_800), mxfp8_added);
    try std.testing.expectEqual(@as(usize, 6_758_400), bf16_added - affine_added);
    try std.testing.expectEqual(@as(usize, 1_996_800), mxfp8_added - affine_added);
    // one projection at another form is priced on its own: the same head's BF16 down projection costs the difference
    const bf16_down: Shared = .{ .width = 1280, .gate = .mxfp8, .up = .mxfp8, .down = .bf16 };
    try std.testing.expectEqual(sharedBytes(dims, 1280, .bf16) - sharedBytes(dims, 1280, .mxfp8), headBytes(false, bf16_down) - headBytes(false, mxfp8));
}
