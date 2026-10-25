//! The source compiler check covers every runtime library and all generated operator controls.
const std = @import("std");
const cfg = @import("config.zig");
pub const Entry = struct { name: []const u8, text: []const u8 };
fn add(a: std.mem.Allocator, list: *std.ArrayList(Entry), name: []const u8, text: []const u8) !void {
    try @import("metal_control.zig").audit(text);
    for (list.items) |item| if (std.mem.eql(u8, item.text, text)) return;
    try list.append(a, .{ .name = name, .text = text });
}
pub fn generate(a: std.mem.Allocator) ![]const Entry {
    var list: std.ArrayList(Entry) = .empty;
    const empty = struct {};
    const projection = @import("../../core/weight_only_metal_impl.zig").Adapter(empty, empty);
    const activation = @import("../../core/hc_activation_metal_impl.zig").Adapter(empty, empty);
    const pointwise = @import("../../core/hc_pointwise_metal_impl.zig").Adapter(empty, empty);
    const weight = @embedFile("../../../kernels/metal/core/weight_only.h");
    const act = @embedFile("../../../kernels/metal/core/hc_activation.h");
    for ([_]bool{ false, true }) |controls| {
        try add(a, &list, try std.fmt.allocPrint(a, "projection-{d}", .{@intFromBool(controls)}), try projection.source(a, weight, @embedFile("../../../kernels/metal/core/weight_only.metal"), controls));
        try add(a, &list, try std.fmt.allocPrint(a, "activation-{d}", .{@intFromBool(controls)}), try activation.source(a, weight, act, @embedFile("../../../kernels/metal/core/hc_activation.metal"), controls));
        try add(a, &list, try std.fmt.allocPrint(a, "pointwise-{d}", .{@intFromBool(controls)}), try pointwise.source(a, weight, act, @embedFile("../../../kernels/metal/core/hc_pointwise.metal"), controls));
    }
    for (0..3) |i| try add(a, &list, try std.fmt.allocPrint(a, "draw-{d}", .{i}), try @import("metal_draw.zig").source(a, i, @import("../flashnext/replay_glue_source.zig").text));
    for (0..2) |geometry| {
        var problem: cfg.Problem = .{};
        var c = try cfg.parse(a, @embedFile("fixtures/native-dense.json"), &problem);
        defer c.deinit();
        if (geometry == 1) {
            c.hidden = 5120;
            c.streams = 8;
            c.heads = 24;
            c.kv_heads = 4;
            c.index_heads = 4;
            c.index_budget = 512;
            c.index_ratio = 4;
            c.key_heads = 16;
            c.value_heads = 48;
        }
        const hc = try @import("metal_hc.zig").source(a, .{ .hidden = c.hidden, .streams = c.streams });
        const ple = try @import("metal_ple.zig").source(a, &c);
        for (std.enums.values(@import("hc_control_source.zig").Control)) |control| try add(a, &list, try std.fmt.allocPrint(a, "hc-{d}-{t}", .{ geometry, control }), try @import("hc_control_source.zig").source(a, hc, control));
        for (std.enums.values(@import("ple_control_source.zig").Control)) |control| try add(a, &list, try std.fmt.allocPrint(a, "ple-{d}-{t}", .{ geometry, control }), try @import("ple_control_source.zig").metal(a, ple, control));
        for ([_]@import("cuda_gdn.zig").Gate{ .silu, .sigmoid }) |gate| {
            const gdn = try @import("metal_gdn.zig").source(a, c.key_heads, c.value_heads, gate);
            for (std.enums.values(@import("gdn_control_source.zig").Control)) |control| try add(a, &list, try std.fmt.allocPrint(a, "gdn-{d}-{t}-{t}", .{ geometry, gate, control }), try @import("gdn_control_source.zig").metal(a, gdn, control));
        }
        const qsa = try @import("metal_attention_source.zig").source(a, .{ .heads = c.heads, .kv = c.kv_heads, .rotary = c.rotary, .index_heads = c.index_heads, .index_dim = c.index_dim, .ratio = c.index_ratio, .top = c.index_budget });
        const control = @import("qsa_control_source.zig");
        for (std.enums.values(control.Control)) |fault| {
            try add(a, &list, try std.fmt.allocPrint(a, "qsa-front-{d}-{t}", .{ geometry, fault }), try control.frontend(a, qsa, fault));
            try add(a, &list, try std.fmt.allocPrint(a, "qsa-scores-{d}-{t}", .{ geometry, fault }), try control.scores(a, qsa, fault));
            if (fault != .sums) try add(a, &list, try std.fmt.allocPrint(a, "qsa-selection-{d}-{t}", .{ geometry, fault }), try control.selection(a, qsa, fault));
            try add(a, &list, try std.fmt.allocPrint(a, "qsa-attention-{d}-{t}", .{ geometry, fault }), try control.attention(a, qsa, fault));
        }
    }
    return list.toOwnedSlice(a);
}
