//! Pure Inkling 2 Mini runtime launch geometry and independent FP64 references.
const std = @import("std");

pub const hidden = 2304;
pub const block_slots = 4;
pub const tap_channels = 4;
pub const max_rows = 256;
pub const threads: u32 = 256;
pub const visit_slots = 48;
pub const conv_states = 4;
pub const conv_widths = [conv_states]usize{ 1024, 1024, 2304, 2304 };
pub const conv_row_values = 1024 + 1024 + 2304 + 2304;
pub const window_rows = 16;
pub const max_commit_segments = 8;
pub const commit_tiles_per_slot = 26;
pub const RuntimeArgs = extern struct { count: u32, boundary: u32 };
pub const CommitArgs = extern struct { segments: u32, taps_stride: u32, accepted: [max_commit_segments]u32 };

pub const Kind = enum {
    copy_bf16,
    copy_f32,
    widen,
    attention_partial,
    mlp_partial,
    capture_tap,
    commit_histories,

    pub fn symbol(k: Kind) []const u8 {
        return switch (k) {
            .copy_bf16 => "tf_ink2_copy_bf16",
            .copy_f32 => "tf_ink2_copy_f32",
            .widen => "tf_ink2_widen",
            .attention_partial => "tf_ink2_attention_partial",
            .mlp_partial => "tf_ink2_mlp_partial",
            .capture_tap => "tf_ink2_capture_tap",
            .commit_histories => "tf_ink2_commit_histories",
        };
    }
    pub fn label(k: Kind) []const u8 {
        return switch (k) {
            .copy_bf16 => "metal inkling2 runtime copy_bf16 tf_ink2_copy_bf16",
            .copy_f32 => "metal inkling2 runtime copy_f32 tf_ink2_copy_f32",
            .widen => "metal inkling2 runtime widen tf_ink2_widen",
            .attention_partial => "metal inkling2 runtime attention_partial tf_ink2_attention_partial",
            .mlp_partial => "metal inkling2 runtime mlp_partial tf_ink2_mlp_partial",
            .capture_tap => "metal inkling2 runtime capture_tap tf_ink2_capture_tap",
            .commit_histories => "metal inkling2 runtime commit_histories tf_ink2_commit_histories",
        };
    }
};

/// `k` is one for pointwise helpers, a boundary for attention, or a DSpark tap channel.
pub const Call = struct { rows: usize, n: usize, k: usize };
pub const Geometry = struct { count: usize, threads: u32 = threads, tg_memory: u32 = 0 };

pub fn valid(kind: Kind, c: Call) bool {
    if (c.rows == 0 or c.rows > max_rows or c.n == 0 or c.rows > std.math.maxInt(u32) / c.n) return false;
    return switch (kind) {
        .copy_bf16, .copy_f32 => c.k == 1,
        .capture_tap => c.n == hidden and c.k < tap_channels,
        .commit_histories => c.rows <= max_commit_segments and c.n == conv_row_values and c.k <= window_rows,
        .widen, .attention_partial, .mlp_partial => c.n == hidden and switch (kind) {
            .attention_partial => c.k <= block_slots,
            else => c.k == 1,
        },
    };
}

pub fn geometry(c: Call) Geometry {
    return .{ .count = c.rows * c.n };
}

/// Values in the full destination for one capture launch; its dispatch still reads rows*n.
pub fn outputCount(kind: Kind, c: Call) usize {
    if (kind == .commit_histories) return c.rows * visit_slots * 3 * conv_row_values;
    const channels: usize = if (kind == .capture_tap) tap_channels else 1;
    return c.rows * c.n * channels;
}

pub fn args(kind: Kind, c: Call) RuntimeArgs {
    std.debug.assert(valid(kind, c) and kind != .commit_histories);
    return .{ .count = @intCast(c.rows * c.n), .boundary = if (kind == .attention_partial or kind == .capture_tap) @intCast(c.k) else 0 };
}

/// Completed blocks are [row][four BF16 blocks][H]; the separate partial is not stored here.
pub fn blockOffset(row: usize, boundary: usize) usize {
    std.debug.assert(row < max_rows and boundary >= 1 and boundary <= block_slots);
    return (row * block_slots + boundary - 1) * hidden;
}

/// DSpark features are [row][four taps][H]; a capture launch writes one tap channel.
pub fn tapOffset(row: usize, channel: usize) usize {
    std.debug.assert(row < max_rows and channel < tap_channels);
    return (row * tap_channels + channel) * hidden;
}

/// Round through binary32 and nearest-even BF16, returning an exactly representable FP64 value.
pub fn roundBf16(v: anytype) f64 {
    const x: f32 = @floatCast(v);
    const bits: u32 = @bitCast(x);
    const rounded: u16 = if (std.math.isNan(x))
        @truncate((bits >> 16) | 0x40)
    else
        @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
    const value: f32 = @bitCast(@as(u32, rounded) << 16);
    return value;
}

/// Boundary restarts the FP32 partial; otherwise the two exact BF16 inputs undergo one FP32 add.
pub fn attentionPartial(p: f32, x: f32, boundary: usize) f64 {
    std.debug.assert(boundary <= block_slots);
    return if (boundary != 0) x else @as(f32, p + x);
}

/// FP32 residual addition followed by the required BF16 endpoint.
pub fn mlpPartial(t: f32, x: f32) f64 {
    return roundBf16(@as(f32, t + x));
}

/// Committed three-tap history element for step `t` (`0..3`) after accepting `accepted` staged rows.
pub fn commitHistory(hist: [3]f32, taps: []const f32, accepted: usize, t: usize) f64 {
    std.debug.assert(t < 3 and accepted <= window_rows and (accepted + t < 3 or accepted + t - 3 < taps.len));
    const idx = accepted + t;
    return roundBf16(if (idx < 3) hist[idx] else taps[idx - 3]);
}
