//! One host-checkable source-planted fault for every Inkling 2 Small Metal op.
const std = @import("std");
const p = @import("inkling2_small_ops_plan");

const Edit = struct { from: []const u8, to: []const u8 };

fn edit(kind: p.Kind) Edit {
    return switch (kind) {
        .embed_norm => .{ .from = "float(weight[d]) * (float(x[d]) * inverse)", .to = "float(weight[0]) * (float(x[d]) * inverse)" },
        .ar => .{ .from = "weights[s] = dot * INK2_SMALL_AR_SCALE;", .to = "weights[s] = dot;" },
        .conv_dynamic => .{ .from = "static_w[lag * a.channels + c]", .to = "static_w[(3u - lag) * a.channels + c]" },
        .router => .{ .from = "key[e] = scores[e] + bias[e];", .to = "key[e] = scores[e];" },
        .logits => .{ .from = "ink2s_cap(float(bfloat(dot)), 30.0f)", .to = "ink2s_cap(dot, 30.0f)" },
        .attention_partial => .{
            .from = "blocks[size_t(row) * INK2_SMALL_BLOCKS * INK2_SMALL_H + (a.boundary - 1u) * INK2_SMALL_H + d] = p[i];",
            .to = "blocks[size_t(row) * INK2_SMALL_H + (a.boundary - 1u) * INK2_SMALL_H + d] = p[i];",
        },
        .capture_tap => .{
            .from = "y[size_t(row) * INK2_SMALL_TAPS * INK2_SMALL_H + a.boundary * INK2_SMALL_H + d] = x[i];",
            .to = "y[size_t(row) * INK2_SMALL_TAPS * INK2_SMALL_H + ((a.boundary + 1u) % INK2_SMALL_TAPS) * INK2_SMALL_H + d] = x[i];",
        },
    };
}

pub fn source(a: std.mem.Allocator, production: []const u8, kind: p.Kind) ![]u8 {
    const e = edit(kind);
    if (std.mem.count(u8, production, e.from) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, production, e.from, e.to);
}

pub fn name(kind: p.Kind) []const u8 {
    return switch (kind) {
        .embed_norm => "control: metal inkling2 small embed_norm tf_ink2_small_embed_norm",
        .ar => "control: metal inkling2 small ar tf_ink2_small_ar",
        .conv_dynamic => "control: metal inkling2 small conv_dynamic tf_ink2_small_conv_dynamic",
        .router => "control: metal inkling2 small router tf_ink2_small_router",
        .logits => "control: metal inkling2 small logits tf_ink2_small_logits",
        .attention_partial => "control: metal inkling2 small attention_partial tf_ink2_small_attention_partial",
        .capture_tap => "control: metal inkling2 small capture_tap tf_ink2_small_capture_tap",
    };
}

/// The Small router's NaN-key fallback planted out: skip fallback, or inspect expert 0 alone.
pub const NanControl = enum { no_fallback, key0_flag };

fn nanEdit(c: NanControl) Edit {
    return switch (c) {
        .no_fallback => .{ .from = "  if (any_nan != 0u) {", .to = "  if (false) {" },
        .key0_flag => .{ .from = "if (t == 0u) any_nan = saw_nan ? 1u : 0u;", .to = "if (t == 0u) any_nan = isnan(key[0]) ? 1u : 0u;" },
    };
}

pub fn nanSource(a: std.mem.Allocator, production: []const u8, c: NanControl) ![]u8 {
    const e = nanEdit(c);
    if (std.mem.count(u8, production, e.from) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, production, e.from, e.to);
}

pub fn nanName(c: NanControl) []const u8 {
    return switch (c) {
        .no_fallback => "control: metal inkling2 small router tf_ink2_small_router, NaN keys without the serial fallback",
        .key0_flag => "control: metal inkling2 small router tf_ink2_small_router, the NaN flag from expert 0's key only",
    };
}

test "every small_ops control applies once and changes only its discriminating anchor" {
    const production = @import("inkling2_metal").small_ops;
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

test "every small router NaN-key control applies once" {
    const production = @import("inkling2_metal").small_ops;
    const a = std.testing.allocator;
    for (std.enums.values(NanControl)) |c| {
        const changed = try nanSource(a, production, c);
        defer a.free(changed);
        try std.testing.expectEqual(production.len - nanEdit(c).from.len + nanEdit(c).to.len, changed.len);
        try std.testing.expectError(error.ControlEditMissing, nanSource(a, changed, c));
    }
}

test "a repeated small_ops control anchor is ambiguous and refuses to prove a fault" {
    const production = @import("inkling2_metal").small_ops;
    const a = std.testing.allocator;
    for (std.enums.values(p.Kind)) |kind| {
        const twice = try std.mem.concat(a, u8, &.{ production, production });
        defer a.free(twice);
        try std.testing.expectError(error.ControlEditMissing, source(a, twice, kind));
    }
}
