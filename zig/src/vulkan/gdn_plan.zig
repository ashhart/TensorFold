// The GDN geometry plans: split, conv prefill and decode, the prefill recurrence, decode and the gated norm.
const std = @import("std");
const dispatch = @import("dispatch.zig");
const gdn = @import("gdn.zig");

const Model = gdn.Model;
const Error = dispatch.Error;

/// The gate split's derived geometry, or the named reason it cannot run.
pub const SplitPlan = struct { tokens: u32, conv_channels: u32, section_elements: u32 };

pub fn splitPlan(conv_out_elements: u64, soa_elements: u64, g_elements: u64, beta_elements: u64, a_elements: u64, b_elements: u64, a_log_elements: u64, dt_elements: u64, tokens: u64) Error!SplitPlan {
    if (tokens == 0) return error.Invalid;
    const section = std.math.mul(u64, tokens, Model.linear_dim) catch return error.TooLarge;
    const rows = std.math.mul(u64, tokens, Model.heads) catch return error.TooLarge;
    const soa = std.math.mul(u64, section, 3) catch return error.TooLarge;
    const conv = std.math.mul(u64, tokens, Model.conv_channels) catch return error.TooLarge;
    if (conv_out_elements != conv or soa_elements != soa) return error.Invalid;
    if (g_elements != rows or beta_elements != rows or a_elements != rows or b_elements != rows) return error.Invalid;
    if (a_log_elements != Model.heads or dt_elements != Model.heads) return error.Invalid;
    return .{
        .tokens = try cast32(tokens),
        .conv_channels = try cast32(Model.conv_channels),
        .section_elements = try cast32(section),
    };
}

pub const ConvPrefillPlan = struct { tokens: u32, count: u32, l_in: u32, tokens_elements: u64, state_elements: u64 };

pub fn convPrefillPlan(conv_input_elements: u64, conv_state_in_elements: u64, qkv_elements: u64, weight_elements: u64, conv_state_out_elements: u64, tokens: u64) Error!ConvPrefillPlan {
    if (tokens == 0) return error.Invalid;
    const rows = std.math.mul(u64, tokens, Model.conv_channels) catch return error.TooLarge;
    const input = std.math.add(u64, rows, std.math.mul(u64, Model.conv_state_rows, Model.conv_channels) catch return error.TooLarge) catch return error.TooLarge;
    const state = std.math.mul(u64, Model.conv_state_rows, Model.conv_channels) catch return error.TooLarge;
    if (conv_input_elements != input) return error.Invalid;
    if (conv_state_in_elements != state or conv_state_out_elements != state) return error.Invalid;
    if (qkv_elements != rows) return error.Invalid;
    if (weight_elements != Model.conv_channels * Model.taps) return error.Invalid;
    return .{
        .tokens = try cast32(tokens),
        .count = try cast32(rows),
        .l_in = try cast32(tokens + Model.conv_state_rows),
        .tokens_elements = rows,
        .state_elements = state,
    };
}

pub const ConvDecodePlan = struct { count: u32, conv_channels: u32 };

pub fn convDecodePlan(conv_state_in_elements: u64, x_elements: u64, weight_elements: u64, activated_elements: u64, conv_state_out_elements: u64, soa_elements: u64, g_elements: u64, beta_elements: u64, a_elements: u64, b_elements: u64, a_log_elements: u64, dt_elements: u64) Error!ConvDecodePlan {
    const state = std.math.mul(u64, Model.conv_state_rows, Model.conv_channels) catch return error.TooLarge;
    const channels: u64 = Model.conv_channels;
    if (conv_state_in_elements != state or conv_state_out_elements != state) return error.Invalid;
    if (x_elements != channels or activated_elements != channels) return error.Invalid;
    if (weight_elements != channels * Model.taps) return error.Invalid;
    if (soa_elements != 3 * Model.linear_dim) return error.Invalid;
    if (g_elements != Model.heads or beta_elements != Model.heads) return error.Invalid;
    if (a_elements != Model.heads or b_elements != Model.heads) return error.Invalid;
    if (a_log_elements != Model.heads or dt_elements != Model.heads) return error.Invalid;
    return .{ .count = try cast32(channels), .conv_channels = try cast32(channels) };
}

pub const PrefillPlan = struct { tokens: u32, chunk: u32, chunks: u32, section_elements: u32 };

pub fn prefillPlan(soa_elements: u64, g_elements: u64, beta_elements: u64, h0_elements: u64, y_elements: u64, hf_elements: u64, snap_elements: u64, tokens: u64, chunk: u64) Error!PrefillPlan {
    if (tokens == 0 or chunk == 0) return error.Invalid;
    const section = std.math.mul(u64, tokens, Model.linear_dim) catch return error.TooLarge;
    const soa = std.math.mul(u64, section, 3) catch return error.TooLarge;
    const rows = std.math.mul(u64, tokens, Model.heads) catch return error.TooLarge;
    if (soa_elements != soa or g_elements != rows or beta_elements != rows) return error.Invalid;
    if (h0_elements != Model.state_elements or hf_elements != Model.state_elements) return error.Invalid;
    if (y_elements != section) return error.Invalid;
    const chunks = (tokens + chunk - 1) / chunk;
    const snap = std.math.mul(u64, chunks - 1, Model.state_elements) catch return error.TooLarge;
    if (snap_elements != snap) return error.Invalid;
    return .{
        .tokens = try cast32(tokens),
        .chunk = try cast32(chunk),
        .chunks = try cast32(chunks),
        .section_elements = try cast32(section),
    };
}

pub const DecodePlan = struct {};

pub fn decodePlan(soa_elements: u64, a_elements: u64, b_elements: u64, a_log_u16_elements: u64, dt_elements: u64, state_in_elements: u64, state_out_elements: u64, y_elements: u64) Error!DecodePlan {
    if (soa_elements != 3 * Model.linear_dim) return error.Invalid;
    if (a_elements != Model.heads or b_elements != Model.heads or dt_elements != Model.heads) return error.Invalid;
    // f32 A_log viewed as u16 pairs: heads * 2
    if (a_log_u16_elements != 2 * Model.heads) return error.Invalid;
    if (state_in_elements != Model.state_elements or state_out_elements != Model.state_elements) return error.Invalid;
    if (y_elements != Model.linear_dim) return error.Invalid;
    return .{};
}

pub const GatedNormPlan = struct { rows: u32 };

// `rows` counts tokens; dispatch rows*16 width-128 head rows with per-head weight [128], not [2048].
pub fn gatedNormPlan(out_elements: u64, y_elements: u64, z_elements: u64, weight_elements: u64, rows: u64, eps: f32) Error!GatedNormPlan {
    if (!std.math.isFinite(eps) or eps <= 0) return error.Invalid;
    if (rows == 0) return error.Invalid;
    const total = std.math.mul(u64, rows, Model.linear_dim) catch return error.TooLarge;
    if (out_elements != total or y_elements != total or z_elements != total) return error.Invalid;
    if (weight_elements != Model.head_dim) return error.Invalid;
    const head_rows = std.math.mul(u64, rows, Model.heads) catch return error.TooLarge;
    return .{ .rows = try cast32(head_rows) };
}

fn cast32(v: u64) Error!u32 {
    return std.math.cast(u32, v) orelse error.TooLarge;
}

const testing = std.testing;

test "splitPlan derives the geometry and refuses wrong shapes and overflow" {
    const t: u64 = 7;
    const section = t * Model.linear_dim;
    try testing.expectEqual(@as(u32, 7), (try splitPlan(t * Model.conv_channels, 3 * section, t * Model.heads, t * Model.heads, t * Model.heads, t * Model.heads, Model.heads, Model.heads, t)).tokens);
    try testing.expectError(error.Invalid, splitPlan(t * Model.conv_channels + 1, 3 * section, t * Model.heads, t * Model.heads, t * Model.heads, t * Model.heads, Model.heads, Model.heads, t));
    try testing.expectError(error.Invalid, splitPlan(t * Model.conv_channels, 3 * section, t * Model.heads, t * Model.heads, t * Model.heads, t * Model.heads, Model.heads + 1, Model.heads, t));
    try testing.expectError(error.Invalid, splitPlan(t * Model.conv_channels, 3 * section, t * Model.heads, t * Model.heads, t * Model.heads, t * Model.heads, Model.heads, Model.heads, 0));
    // tokens=2^44 yields section=2^55; u64 geometry fits, but u32 casts return TooLarge.
    const big: u64 = 1 << 44;
    const big_section = big * Model.linear_dim;
    try testing.expectError(error.TooLarge, splitPlan(big * Model.conv_channels, 3 * big_section, big * Model.heads, big * Model.heads, big * Model.heads, big * Model.heads, Model.heads, Model.heads, big));
}

test "convPrefillPlan checks the conv input, state and weight geometry" {
    const t: u64 = 5;
    const ch: u64 = Model.conv_channels;
    const state = Model.conv_state_rows * ch;
    try testing.expectEqual(@as(u32, 8), (try convPrefillPlan((t + 3) * ch, state, t * ch, ch * 4, state, t)).l_in);
    try testing.expectError(error.Invalid, convPrefillPlan((t + 2) * ch, state, t * ch, ch * 4, state, t));
    try testing.expectError(error.Invalid, convPrefillPlan((t + 3) * ch, state, t * ch, ch * 5, state, t));
    try testing.expectError(error.Invalid, convPrefillPlan((t + 3) * ch, state + ch, t * ch, ch * 4, state, t));
}

test "convDecodePlan refuses any torn decode geometry" {
    const ch: u64 = Model.conv_channels;
    const state = Model.conv_state_rows * ch;
    const heads: u64 = Model.heads;
    const ok = try convDecodePlan(state, ch, ch * 4, ch, state, 3 * Model.linear_dim, heads, heads, heads, heads, heads, heads);
    try testing.expect(ok.count == ch);
    try testing.expectError(error.Invalid, convDecodePlan(state, ch + 2, ch * 4, ch, state, 3 * Model.linear_dim, heads, heads, heads, heads, heads, heads));
    try testing.expectError(error.Invalid, convDecodePlan(state, ch, ch * 4, ch, state, 3 * Model.linear_dim, heads, heads, heads, heads, heads, heads + 1));
}

test "prefillPlan derives chunks and the snapshot size, refusing snap mismatch" {
    const t: u64 = 130;
    const chunk: u64 = 64;
    const section = t * Model.linear_dim;
    const state: u64 = Model.state_elements;
    // Three chunks need two interior snapshots; chunk c restores slot c-1, so size is (chunks-1)*state.
    const plan = try prefillPlan(3 * section, t * Model.heads, t * Model.heads, state, section, state, 2 * state, t, chunk);
    try testing.expectEqual(@as(u32, 3), plan.chunks);
    try testing.expectEqual(@as(u32, 64), plan.chunk);
    try testing.expectError(error.Invalid, prefillPlan(3 * section, t * Model.heads, t * Model.heads, state, section, state, 3 * state, t, chunk)); // too many slots
    try testing.expectError(error.Invalid, prefillPlan(3 * section, t * Model.heads, t * Model.heads, state, section, state, 1 * state, t, chunk)); // too few
    try testing.expectError(error.Invalid, prefillPlan(3 * section, t * Model.heads, t * Model.heads, state, section, state, 0, t, chunk)); // chunks>1 needs snap
    try testing.expectError(error.Invalid, prefillPlan(3 * section, t * Model.heads, t * Model.heads, state, section, state, 0, t, 0));
    const single = try prefillPlan(3 * section, t * Model.heads, t * Model.heads, state, section, state, 0, t, t);
    try testing.expectEqual(@as(u32, 1), single.chunks);
}

test "the common prefill graph admits decode and full prompt chunks" {
    const ch: u64 = Model.conv_channels;
    const conv_state = Model.conv_state_rows * ch;
    const state: u64 = Model.state_elements;
    for ([_]u64{ 1, 31, 32 }) |tokens| {
        const conv = try convPrefillPlan((tokens + Model.conv_state_rows) * ch, conv_state, tokens * ch, ch * Model.taps, conv_state, tokens);
        try testing.expectEqual(@as(u32, @intCast(tokens)), conv.tokens);
        try testing.expectEqual(@as(u32, @intCast(tokens + Model.conv_state_rows)), conv.l_in);
        const section = tokens * Model.linear_dim;
        const recurrence = try prefillPlan(3 * section, tokens * Model.heads, tokens * Model.heads, state, section, state, 0, tokens, tokens);
        try testing.expectEqual(@as(u32, 1), recurrence.chunks);
        try testing.expectEqual(@as(u32, @intCast(tokens)), recurrence.chunk);
    }
}

test "decodePlan checks the u16 A_log window and the fixed decode shapes" {
    const heads: u64 = Model.heads;
    _ = try decodePlan(3 * Model.linear_dim, heads, heads, 2 * heads, heads, Model.state_elements, Model.state_elements, Model.linear_dim);
    try testing.expectError(error.Invalid, decodePlan(3 * Model.linear_dim, heads, heads, heads, heads, Model.state_elements, Model.state_elements, Model.linear_dim));
    try testing.expectError(error.Invalid, decodePlan(3 * Model.linear_dim, heads, heads, 2 * heads, heads, Model.state_elements + 4, Model.state_elements, Model.linear_dim));
}

test "gatedNormPlan is the per-head 128 norm and refuses torn geometry" {
    const rows: u64 = 3;
    const total = rows * Model.linear_dim;
    // rows = tokens; the plan dispatches tokens*16 head-rows, weight [128]
    try testing.expectEqual(@as(u32, 3 * Model.heads), (try gatedNormPlan(total, total, total, Model.head_dim, rows, 1e-6)).rows);
    // the old 2048 global-row weight is REFUSED
    try testing.expectError(error.Invalid, gatedNormPlan(total, total, total, Model.linear_dim, rows, 1e-6));
    try testing.expectError(error.Invalid, gatedNormPlan(total + 2, total, total, Model.head_dim, rows, 1e-6));
    try testing.expectError(error.Invalid, gatedNormPlan(total, total, total, Model.head_dim, rows, 0));
    try testing.expectError(error.Invalid, gatedNormPlan(total, total, total, Model.head_dim, 0, 1e-6));
}
