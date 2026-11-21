//! One host-checkable source-planted arithmetic fault for every Inkling 2 Mini glue kernel.
const std = @import("std");
const p = @import("inkling2_glue_plan");

const Edit = struct { from: []const u8, to: []const u8 };
fn edit(kind: p.Kind) Edit {
    return switch (kind) {
        .embed_norm => .{ .from = "float(weight[d]) * (float(x[d]) * inverse)", .to = "float(weight[0]) * (float(x[d]) * inverse)" },
        .ar => .{ .from = "weights[s] = dot * INKLING_AR_SCALE;", .to = "weights[s] = dot;" },
        .conv_static => .{ .from = "weight[c * 4 + tap]", .to = "weight[c * 4 + 3 - tap]" },
        .conv_dynamic => .{ .from = "static_w[lag * a.channels + c]", .to = "static_w[(3 - lag) * a.channels + c]" },
        .router => .{ .from = "keys[t] = scores[t] + bias[t];", .to = "keys[t] = scores[t];" },
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
        .conv_static => "control: metal inkling2 glue conv_static tf_inkling2_conv_static",
        .conv_dynamic => "control: metal inkling2 glue conv_dynamic tf_inkling2_conv_dynamic",
        .router => "control: metal inkling2 glue router tf_inkling2_router",
        .swiglu => "control: metal inkling2 glue swiglu tf_inkling2_swiglu",
        .logits => "control: metal inkling2 glue logits tf_inkling2_logits",
        .expert_pick => "control: metal inkling2 glue expert_pick tf_inkling2_expert_pick",
        .expert_combine => "control: metal inkling2 glue expert_combine tf_inkling2_expert_combine",
    };
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
