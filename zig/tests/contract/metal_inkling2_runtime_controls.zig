//! One host-checkable arithmetic or layout mutation for each Inkling 2 Mini runtime helper.
const std = @import("std");
const p = @import("inkling2_runtime_plan");

const Edit = struct { from: []const u8, to: []const u8 };
fn edit(kind: p.Kind) Edit {
    return switch (kind) {
        .copy_bf16 => .{
            .from = "kernel void tf_ink2_copy_bf16(const device bfloat* x [[buffer(0)]], device bfloat* y [[buffer(1)]], constant InkRuntimeArgs& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {\n  if (i < a.count) y[i] = x[i];",
            .to = "kernel void tf_ink2_copy_bf16(const device bfloat* x [[buffer(0)]], device bfloat* y [[buffer(1)]], constant InkRuntimeArgs& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {\n  if (i < a.count) y[i] = bfloat(0);",
        },
        .copy_f32 => .{
            .from = "kernel void tf_ink2_copy_f32(const device float* x [[buffer(0)]], device float* y [[buffer(1)]], constant InkRuntimeArgs& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {\n  if (i < a.count) y[i] = x[i];",
            .to = "kernel void tf_ink2_copy_f32(const device float* x [[buffer(0)]], device float* y [[buffer(1)]], constant InkRuntimeArgs& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {\n  if (i < a.count) y[i] = 0.0f;",
        },
        .widen => .{ .from = "if (i < a.count) y[i] = float(x[i]);", .to = "if (i < a.count) y[i] = 2.0f * float(x[i]);" },
        .attention_partial => .{ .from = "blocks[row * 4 * 2304 + (a.boundary - 1) * 2304 + d] = p[i];", .to = "blocks[row * 2304 + (a.boundary - 1) * 2304 + d] = p[i];" },
        .mlp_partial => .{ .from = "bfloat(t[i] + float(ink2_conv(", .to = "bfloat(t[i] - float(ink2_conv(" },
        .capture_tap => .{ .from = "y[row * 4 * 2304 + a.boundary * 2304 + d] = x[i];", .to = "y[row * 4 * 2304 + ((a.boundary + 1) % 4) * 2304 + d] = x[i];" },
    };
}

pub fn source(a: std.mem.Allocator, production: []const u8, kind: p.Kind) ![]u8 {
    const e = edit(kind);
    if (std.mem.count(u8, production, e.from) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, production, e.from, e.to);
}

pub fn name(kind: p.Kind) []const u8 {
    return switch (kind) {
        .copy_bf16 => "control: metal inkling2 runtime copy_bf16 tf_ink2_copy_bf16",
        .copy_f32 => "control: metal inkling2 runtime copy_f32 tf_ink2_copy_f32",
        .widen => "control: metal inkling2 runtime widen tf_ink2_widen",
        .attention_partial => "control: metal inkling2 runtime attention_partial tf_ink2_attention_partial",
        .mlp_partial => "control: metal inkling2 runtime mlp_partial tf_ink2_mlp_partial",
        .capture_tap => "control: metal inkling2 runtime capture_tap tf_ink2_capture_tap",
    };
}

test "every runtime control applies once and changes only its discriminating anchor" {
    const production = @import("inkling2_metal").runtime;
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

test "a repeated runtime control anchor is ambiguous and refuses to prove a fault" {
    const production = @import("inkling2_metal").runtime;
    const a = std.testing.allocator;
    for (std.enums.values(p.Kind)) |kind| {
        const twice = try std.mem.concat(a, u8, &.{ production, production });
        defer a.free(twice);
        try std.testing.expectError(error.ControlEditMissing, source(a, twice, kind));
    }
}
