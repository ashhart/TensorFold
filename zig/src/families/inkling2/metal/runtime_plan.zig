//! Pure Inkling 2 Mini runtime launch geometry and independent FP64 references.
const std = @import("std");

pub const hidden = 2304;
pub const block_slots = 4;
pub const tap_channels = 4;
pub const max_rows = 64;
pub const max_stage_rows = 512;
pub const visit_slots = 48;
pub const window_rows = 16;
pub const conv_states = 4;
pub const conv_widths = [conv_states]usize{ 1024, 1024, 2304, 2304 };
pub const conv_row_values = 1024 + 1024 + 2304 + 2304;
pub const threads: u32 = 256;
pub const RuntimeArgs = extern struct { count: u32, boundary: u32 };
pub const StageConvArgs = extern struct { rows: u32, width: u32, commit: u32, pairs: u32 };
pub const CommitHistoriesArgs = extern struct { offset: u32, slots: u32, tap_rows: u32 };

pub const Kind = enum {
    copy_bf16,
    copy_f32,
    widen,
    attention_partial,
    mlp_partial,
    capture_tap,
    stage_conv,
    commit_histories,

    pub fn symbol(k: Kind) []const u8 {
        return switch (k) {
            .copy_bf16 => "tf_ink2_copy_bf16",
            .copy_f32 => "tf_ink2_copy_f32",
            .widen => "tf_ink2_widen",
            .attention_partial => "tf_ink2_attention_partial",
            .mlp_partial => "tf_ink2_mlp_partial",
            .capture_tap => "tf_ink2_capture_tap",
            .stage_conv => "tf_ink2_stage_conv",
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
            .stage_conv => "metal inkling2 runtime stage_conv tf_ink2_stage_conv",
            .commit_histories => "metal inkling2 runtime commit_histories tf_ink2_commit_histories",
        };
    }
};

/// `k` is 1 for moves, boundary/tap channel, stage_conv mode (commit|(pairs-1)<<1), or commit_histories offset.
pub const Call = struct { rows: usize, n: usize, k: usize };
pub const Geometry = struct { count: usize, threads: u32 = threads, tg_memory: u32 = 0 };

pub fn stagePairs(k: usize) usize {
    return (k >> 1) + 1;
}

pub fn stageCommit(k: usize) bool {
    return (k & 1) != 0;
}

pub fn valid(kind: Kind, c: Call) bool {
    if (c.rows == 0 or c.n == 0) return false;
    return switch (kind) {
        .stage_conv => c.rows <= max_stage_rows and c.k <= 3 and c.rows + 3 <= std.math.maxInt(u32) / c.n and stagePairs(c.k) <= std.math.maxInt(u32) / c.n,
        .commit_histories => c.rows <= visit_slots and c.n >= 3 and c.n <= 3 + window_rows and c.k + 3 <= c.n,
        else => blk: {
            if (c.rows > max_rows or c.rows > std.math.maxInt(u32) / c.n) break :blk false;
            break :blk switch (kind) {
                .copy_bf16, .copy_f32 => c.k == 1,
                .capture_tap => c.n == hidden and c.k < tap_channels,
                .widen, .attention_partial, .mlp_partial => c.n == hidden and switch (kind) {
                    .attention_partial => c.k <= block_slots,
                    else => c.k == 1,
                },
                else => unreachable,
            };
        },
    };
}

pub fn geometry(c: Call) Geometry {
    return .{ .count = c.rows * c.n };
}

pub fn dispatchCount(kind: Kind, c: Call) usize {
    return switch (kind) {
        .stage_conv => stagePairs(c.k) * c.n,
        .commit_histories => c.rows * 3 * conv_row_values,
        else => c.rows * c.n,
    };
}

/// Values in the destination outputs for one launch.
pub fn outputCount(kind: Kind, c: Call) usize {
    return switch (kind) {
        .capture_tap => c.rows * c.n * tap_channels,
        .stage_conv => stagePairs(c.k) * (6 + c.rows) * c.n,
        .commit_histories => c.rows * 3 * conv_row_values,
        else => c.rows * c.n,
    };
}

pub fn args(kind: Kind, c: Call) RuntimeArgs {
    std.debug.assert(valid(kind, c));
    return .{ .count = @intCast(c.rows * c.n), .boundary = if (kind == .attention_partial or kind == .capture_tap) @intCast(c.k) else 0 };
}

pub fn stageConvArgs(c: Call) StageConvArgs {
    std.debug.assert(valid(.stage_conv, c));
    return .{ .rows = @intCast(c.rows), .width = @intCast(c.n), .commit = @intFromBool(stageCommit(c.k)), .pairs = @intCast(stagePairs(c.k)) };
}

pub fn commitHistoriesArgs(c: Call) CommitHistoriesArgs {
    std.debug.assert(valid(.commit_histories, c));
    return .{ .offset = @intCast(c.k), .slots = @intCast(c.rows), .tap_rows = @intCast(c.n) };
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

pub fn stageConvRef(hist_before: []const u16, raw: []const u16, width: usize, rows: usize, commit: bool, hist_out: []u16, staged_out: []u16) void {
    std.debug.assert(hist_before.len == 3 * width and raw.len == rows * width);
    std.debug.assert(hist_out.len == 3 * width and staged_out.len == (3 + rows) * width);
    @memcpy(staged_out[0 .. 3 * width], hist_before);
    @memcpy(staged_out[3 * width ..], raw);
    if (!commit) {
        @memcpy(hist_out, hist_before);
        return;
    }
    if (rows == 1) {
        @memcpy(hist_out[0 .. 2 * width], hist_before[width .. 3 * width]);
        @memcpy(hist_out[2 * width .. 3 * width], raw);
    } else if (rows == 2) {
        @memcpy(hist_out[0..width], hist_before[2 * width .. 3 * width]);
        @memcpy(hist_out[width .. 3 * width], raw);
    } else {
        @memcpy(hist_out, raw[(rows - 3) * width .. rows * width]);
    }
}

pub fn commitHistoriesRef(taps: []const u16, conv_out: []u16, slots: usize, tap_rows: usize, offset: usize) void {
    std.debug.assert(taps.len == slots * tap_rows * conv_row_values and conv_out.len == slots * 3 * conv_row_values);
    std.debug.assert(offset + 3 <= tap_rows);
    for (0..slots) |slot| {
        var tap_base: usize = slot * tap_rows * conv_row_values;
        var conv_base: usize = slot * 3 * conv_row_values;
        for (conv_widths) |w| {
            const src = tap_base + offset * w;
            @memcpy(conv_out[conv_base .. conv_base + 3 * w], taps[src .. src + 3 * w]);
            tap_base += tap_rows * w;
            conv_base += 3 * w;
        }
    }
}
