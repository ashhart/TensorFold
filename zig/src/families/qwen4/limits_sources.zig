//! Resource probes compile the real specializations at admitted upper shapes, without substitute shader bodies.
const std = @import("std");
pub fn source(a: std.mem.Allocator, label: []const u8) ![]u8 {
    var control: @import("qsa_control_source.zig").Control = .production;
    for ([_][]const u8{ "bad_nan", "bad_repeat", "bad_tail", "bad_sums" }, 1..) |word, i| {
        if (std.mem.indexOf(u8, label, word) != null) control = @fromBackingInt(@intCast(i));
    }
    const text = try production(a, label);
    defer a.free(text);
    if (std.mem.startsWith(u8, label, "qwen4-native hc ")) return @import("hc_control_source.zig").source(a, text, @fromBackingInt(@intCast(@backingInt(control))));
    if (std.mem.startsWith(u8, label, "qwen4-native gdn-")) return @import("gdn_control_source.zig").metal(a, text, @fromBackingInt(@intCast(@backingInt(control))));
    if (std.mem.startsWith(u8, label, "qwen4-native ple ")) return @import("ple_control_source.zig").metal(a, text, @fromBackingInt(@intCast(@backingInt(control))));
    const qsa = @import("qsa_control_source.zig");
    if (std.mem.indexOf(u8, label, "qwen4_idx_scores") != null) return qsa.scores(a, text, control);
    if (std.mem.indexOf(u8, label, "qwen4_idx_select") != null) return qsa.selection(a, text, control);
    if (std.mem.indexOf(u8, label, "qwen4_attn_parts") != null or std.mem.indexOf(u8, label, "qwen4_attn_merge") != null) return qsa.attention(a, text, control);
    return qsa.frontend(a, text, control);
}
fn production(a: std.mem.Allocator, label: []const u8) ![]u8 {
    if (std.mem.startsWith(u8, label, "qwen4-native hc ")) return @import("metal_hc.zig").source(a, .{ .hidden = 5120, .streams = 8 });
    if (std.mem.startsWith(u8, label, "qwen4-native gdn-")) return @import("../flashnext/hybrid_sources.zig").Gdn.metal(a, @embedFile("../../../kernels/metal/decode/fn_gdn.metal"), @embedFile("../../../kernels/metal/flashnext/q4_gdn_step_af87f500308a9afb.metal"), 16, 48, std.mem.indexOf(u8, label, "gdn-sigmoid ") != null);
    if (std.mem.startsWith(u8, label, "qwen4-native qsa ")) return @import("metal_attention_source.zig").source(a, .{ .heads = 24, .kv = 4, .rotary = 64, .index_heads = 4, .index_dim = 128, .ratio = 4, .top = 512 });
    if (std.mem.startsWith(u8, label, "qwen4-native ple ")) {
        var p: @import("config.zig").Problem = .{};
        var c = try @import("config.zig").parse(a, @embedFile("fixtures/native-dense.json"), &p);
        defer c.deinit();
        c.hidden = 5120;
        c.streams = 8;
        return @import("metal_ple.zig").source(a, &c);
    }
    return error.UnknownQwen4Launch;
}
