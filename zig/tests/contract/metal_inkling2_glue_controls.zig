//! One host-checkable source-planted arithmetic fault for every Inkling 2 Mini glue kernel.
const std = @import("std");
const p = @import("inkling2_glue_plan");

const Edit = struct { from: []const u8, to: []const u8 };
fn edit(kind: p.Kind) Edit {
    return switch (kind) {
        .embed_norm => .{ .from = "float(weight[d]) * (float(x[d]) * inverse)", .to = "float(weight[0]) * (float(x[d]) * inverse)" },
        .ar => .{ .from = "weights[s] = dot * INKLING_AR_SCALE;", .to = "weights[s] = dot;" },
        .cached_ar => .{ .from = "if (t + 1 < a.sources) means[t] = block_inv_rms[t];", .to = "if (t + 1 < a.sources) means[t] = block_inv_rms[(t + 1) & 3];" },
        .block_inv_rms => .{ .from = "if (t == 0) inv_rms[size_t(row) * 4 + a.slot] = inv;", .to = "if (t == 0) inv_rms[size_t(row) * 4 + a.slot] = inv * 2.0f;" },
        .attn_partial_ar => .{ .from = "means[a.sources - 1] = part_inv;", .to = "means[a.sources - 1] = part_inv * 2.0f;" },
        .mlp_partial_ar => .{ .from = "const bfloat next_p = bfloat(pin_row[d] + float(conv_row[d]));", .to = "const bfloat next_p = bfloat(pin_row[d] - float(conv_row[d]));" },
        .conv_static => .{ .from = "weight[c * 4 + tap]) * float(taps", .to = "weight[c * 4 + 3 - tap]) * float(taps" },
        .conv_dynamic => .{ .from = "static_w[lag * a.channels + c]) + projected[4 * (c / 128) + lag];\n      sum", .to = "static_w[(3 - lag) * a.channels + c]) + projected[4 * (c / 128) + lag];\n      sum" },
        .conv_static_inplace => .{ .from = "weight[c * 4 + tap]) * float(x)", .to = "weight[c * 4 + 3 - tap]) * float(x)" },
        .conv_dynamic_inplace => .{ .from = "static_w[lag * a.channels + c]) + projected[4 * (c / 128) + lag];\n      const uint idx", .to = "static_w[(3 - lag) * a.channels + c]) + projected[4 * (c / 128) + lag];\n      const uint idx" },
        .router => .{ .from = "key[t] = scores[t] + bias[t];", .to = "key[t] = scores[t];" },
        .swiglu => .{ .from = "ink_cap(u, 25.0f)", .to = "ink_cap(g, 25.0f)" },
        .logits => .{ .from = "ink_cap(float(bfloat(dot)), 30.0f)", .to = "ink_cap(dot, 30.0f)" },
        .expert_pick => .{ .from = "ids[r * 8 + j]", .to = "ids[r * 8 + 7 - j]" },
        .expert_combine => .{ .from = "acc += weights[r * 8 + j] * float(picked[(size_t(r) * 8 + j) * a.channels + d]);", .to = "acc -= weights[r * 8 + j] * float(picked[(size_t(r) * 8 + j) * a.channels + d]);" },
    };
}

pub fn source(a: std.mem.Allocator, production: []const u8, kind: p.Kind) ![]u8 {
    const e = edit(kind);
    if (std.mem.count(u8, production, e.from) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, production, e.from, e.to);
}

pub fn name(kind: p.Kind) []const u8 {
    return switch (kind) {
        .embed_norm => "control: metal inkling2 glue embed_norm tf_inkling2_embed_norm",
        .ar => "control: metal inkling2 glue ar tf_inkling2_ar",
        .cached_ar => "control: metal inkling2 glue cached_ar tf_inkling2_cached_ar",
        .block_inv_rms => "control: metal inkling2 glue block_inv_rms tf_inkling2_block_inv_rms",
        .attn_partial_ar => "control: metal inkling2 glue attn_partial_ar tf_inkling2_attn_partial_ar",
        .mlp_partial_ar => "control: metal inkling2 glue mlp_partial_ar tf_inkling2_mlp_partial_ar",
        .conv_static => "control: metal inkling2 glue conv_static tf_inkling2_conv_static",
        .conv_dynamic => "control: metal inkling2 glue conv_dynamic tf_inkling2_conv_dynamic",
        .router => "control: metal inkling2 glue router tf_inkling2_router",
        .swiglu => "control: metal inkling2 glue swiglu tf_inkling2_swiglu",
        .logits => "control: metal inkling2 glue logits tf_inkling2_logits",
        .expert_pick => "control: metal inkling2 glue expert_pick tf_inkling2_expert_pick",
        .expert_combine => "control: metal inkling2 glue expert_combine tf_inkling2_expert_combine",
        .conv_static_inplace => "control: metal inkling2 glue conv_static_inplace tf_inkling2_conv_static_inplace",
        .conv_dynamic_inplace => "control: metal inkling2 glue conv_dynamic_inplace tf_inkling2_conv_dynamic_inplace",
    };
}

/// The router's NaN-key fallback planted out: the rank picks run on every row, or the flag reads expert 0's key alone.
pub const NanControl = enum { no_fallback, key0_flag };

fn nanEdit(c: NanControl) Edit {
    return switch (c) {
        .no_fallback => .{ .from = "  if (any_nan != 0) {", .to = "  if (false) {" },
        .key0_flag => .{ .from = "if (t == 0) any_nan = nan ? 1u : 0u;", .to = "if (t == 0) any_nan = isnan(key[0]) ? 1u : 0u;" },
    };
}

pub fn nanSource(a: std.mem.Allocator, production: []const u8, c: NanControl) ![]u8 {
    const e = nanEdit(c);
    if (std.mem.count(u8, production, e.from) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, production, e.from, e.to);
}

pub fn nanName(c: NanControl) []const u8 {
    return switch (c) {
        .no_fallback => "control: metal inkling2 glue router tf_inkling2_router, NaN keys without the serial fallback",
        .key0_flag => "control: metal inkling2 glue router tf_inkling2_router, the NaN flag from expert 0's key only",
    };
}

test "every router NaN-key control applies once" {
    const production = @import("inkling2_metal").glue;
    const a = std.testing.allocator;
    for (std.enums.values(NanControl)) |c| {
        const changed = try nanSource(a, production, c);
        defer a.free(changed);
        try std.testing.expectEqual(production.len - nanEdit(c).from.len + nanEdit(c).to.len, changed.len);
        try std.testing.expectError(error.ControlEditMissing, nanSource(a, changed, c));
    }
}

test "every glue control applies once and changes only its arithmetic anchor" {
    const production = @import("inkling2_metal").glue;
    const a = std.testing.allocator;
    for (std.enums.values(p.Kind)) |kind| {
        const e = edit(kind);
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, production, e.from));
        const changed = try source(a, production, kind);
        defer a.free(changed);
        const at = std.mem.indexOf(u8, production, e.from).?;
        try std.testing.expectEqualStrings(production[0..at], changed[0..at]);
        try std.testing.expectEqualStrings(e.to, changed[at..][0..e.to.len]);
        try std.testing.expectEqualStrings(production[at + e.from.len ..], changed[at + e.to.len ..]);
        try std.testing.expectError(error.ControlEditMissing, source(a, changed, kind));
        try std.testing.expectError(error.ControlEditMissing, source(a, "", kind));
    }
}

test "a repeated glue control anchor is ambiguous and refuses to prove a fault" {
    const a = std.testing.allocator;
    const production = @import("inkling2_metal").glue;
    for (std.enums.values(p.Kind)) |kind| {
        const twice = try std.mem.concat(a, u8, &.{ production, production });
        defer a.free(twice);
        try std.testing.expectError(error.ControlEditMissing, source(a, twice, kind));
    }
}

test "glue controls clean up and fail cleanly on OOM" {
    const production = @import("inkling2_metal").glue;
    for (std.enums.values(p.Kind)) |kind| {
        var fail = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
        try std.testing.expectError(error.OutOfMemory, source(fail.allocator(), production, kind));
    }
    for (std.enums.values(NanControl)) |c| {
        var fail = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
        try std.testing.expectError(error.OutOfMemory, nanSource(fail.allocator(), production, c));
    }
}
