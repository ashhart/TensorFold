//! Flash Next's recorded libraries and its maintained runtime source transformations.
const std = @import("std");
const ks = @import("kernel_sources");
const Set = @import("set.zig").Set;
const rotary = @import("../src/families/flashnext/rotary_cache.zig");
const tree = @import("../src/families/flashnext/tree.zig");
const rows = @import("../src/families/flashnext/rows_w.zig");
const hcrows = @import("../src/families/flashnext/rows_hc.zig");
const lane = @import("../src/families/flashnext/lane_w.zig");
const compile = @import("../src/families/flashnext/replay_compile.zig");
const fsr = @import("../src/families/flashnext/fsr_contract_source.zig");

pub fn add(s: *Set) !void {
    for (ks.flashnext_gen.sources) |file| {
        if (std.mem.indexOf(u8, file.name, "lane_qmm_bytes_grouped") != null) {
            const scalar = std.mem.endsWith(u8, file.name, "-lanes.metal");
            if (scalar == s.units) continue;
        }
        for ([_]bool{ false, true }) |cached|
            try s.raw("flashnext-recorded", file.name, try rotary.source(s.a, file.text, cached));
        if (std.mem.startsWith(u8, file.name, "q4_attn_parts")) {
            try s.raw("flashnext-tree", file.name, try tree.patch(s.a, file.text, &tree.attn_edits));
            try s.raw("flashnext-attn2", file.name, try tree.patch(s.a, file.text, &@import("../src/families/flashnext/attn2.zig").edits));
        }
        if (std.mem.startsWith(u8, file.name, "q4_ple_conv"))
            try s.raw("flashnext-tree", file.name, try tree.patch(s.a, file.text, &tree.conv_edits));
        if (std.mem.startsWith(u8, file.name, "lane_qmm_bytes_grouped") and std.mem.indexOf(u8, file.text, "constexpr int SK = ") != null) {
            for ([_]?[2]usize{ null, .{ 1, 4 } }) |groups|
                try s.raw("flashnext-tp", file.name, try @import("../src/families/flashnext/split.zig").laneSource(s.a, file.text, &.{.{ 1, 2 }}, 2, groups));
        }
        if (std.mem.startsWith(u8, file.name, "q4_gdn")) {
            const peer = "const int hv = int(threadgroup_position_in_grid.x);";
            if (std.mem.count(u8, file.text, peer) == 1) try s.raw("flashnext-tp", "gdn-peer", try @import("../src/families/flashnext/gdn_source.zig").peer(s.a, file.text));
            const from = "SO[((size_t(r) * NV + hv)";
            if (std.mem.count(u8, file.text, from) == 1) try s.raw("flashnext-prompt", "gdn-last-state", try @import("../src/families/flashnext/gdn_source.zig").last(s.a, file.text));
        }
    }
    try s.raw("flashnext", "glue", @import("../src/families/flashnext/replay_glue_source.zig").text);
    try s.raw("flashnext", "verdict", @import("../src/families/flashnext/verdict.zig").source);
    try s.raw("flashnext", "tree", tree.source);
    try s.raw("flashnext", "gate", @import("../src/families/flashnext/gate.zig").source);
    try s.raw("flashnext", "ple-scale", @import("../src/families/flashnext/ple_scale.zig").source);
    try s.raw("flashnext", "grammar", @import("../src/families/flashnext/grammar.zig").source);
    try s.raw("flashnext", "tp", @import("../src/families/flashnext/tp_source.zig").source);
    try s.raw("flashnext", "gx", ks.flashnext_gx);
    try s.raw("flashnext", "rotary-producer", try rotary.producer(s.a));
    inline for (.{ "flashnext_select", "flashnext_pool_pos", "flashnext_mrope" }) |name| for ([_]bool{ false, true }) |cached|
        try s.raw("flashnext", name, try rotary.source(s.a, @field(ks, name), cached));
    try s.fragments("flashnext", "prompt-attention", ks.flashnext_attn);
    try s.fragments("flashnext", "qmm6", ks.flashnext_qmm6);
    const expert = try fsr.recorded("qa_expert_gateup_fd30a642315017a8.metal");
    for ([_]bool{ false, true }) |packing| for ([_]bool{ false, true }) |ssd| {
        const xnew = try compile.xnewSource(s.a, expert, packing, ssd);
        try s.raw("flashnext", "xnew", xnew.text);
        try s.raw("flashnext", "prompt", try std.mem.concat(s.a, u8, &.{ xnew.header, ks.flashnext_prompt }));
        for ([_][]const u8{ "#define FZ_GDN_REPLAY 0\n", "#define FZ_GDN_REPLAY 1\n", "#define FZ_GDN_REPLAY 0\n#define FZ_GDN_TREE 1\n" }) |mode|
            try s.raw("flashnext", "gdn", try std.mem.concat(s.a, u8, &.{ xnew.header, mode, ks.flashnext_gdn }));
    };
    const hc = try fsr.recorded("qa_hc_down_row_85069d94acd4e9c1.metal");
    const cut = std.mem.indexOf(u8, hc, "[[max_total_threads_per_threadgroup") orelse return error.NoKernel;
    try s.raw("flashnext", "hc-up", try std.mem.concat(s.a, u8, &.{ hc[0..cut], @import("../src/families/flashnext/replay_hc_source.zig").text }));
    for ([_][]const u8{ fsr.gate, fsr.down, "fz_xgu", "tf_gather_qmm6_nax_32", "tf_gather_qmmw_nax_32" }) |symbol| {
        const text = try fsr.source(s.a, symbol);
        if (text.fragments) try s.fragments("flashnext-fsr", symbol, text.text) else try s.raw("flashnext-fsr", symbol, text.text);
    }
    try widths(s);
    try recordedWidths(s);
}

fn widths(s: *Set) !void {
    for ([_]u8{ 2, 3, 4, 5, 6, 8 }) |bits| for ([_]u16{ 32, 64, 128 }) |group| {
        for ([_]bool{ false, true }) |ssd| {
            const body = if (ssd) try @import("../src/families/flashnext/fsr_prompt_source.zig").patch(s.a, ks.flashnext_qmmw, true) else ks.flashnext_qmmw;
            try s.fragments("prompt-widths", "qmmw", try @import("../src/core/qmmw_source.zig").source(s.a, body, bits, group));
        }
        for ([_]u8{ 1, 2, 4 }) |rps| for ([_]u8{ 0, 1, 2 }) |nib| {
            if (bits != 4 and nib != 0) continue;
            try s.raw("flashnext-rows", "dense", try rows.source(s.a, .{ .n = 2560, .k = 2560, .bits = bits, .group = group, .ld = 2560, .rps = rps, .nib = nib }));
        };
        for ([_]bool{ false, true }) |mapped| for ([_]bool{ false, true }) |tiled| {
            try s.raw("flashnext-lane", "width", try lane.source(s.a, try lane.template(s.units), .{ .n = if (tiled) 2560 else 2576, .k = 2560, .bits = bits, .group = group, .tmr = 16, .ld = 2600, .mapped = mapped, .tiled = tiled }));
        };
    };
    for ([_][2]usize{ .{ 6, 32 }, .{ 4, 64 }, .{ 8, 32 } }) |format| for ([_]u8{ 0, 1, 2 }) |nib| {
        if (format[0] != 4 and nib != 0) continue;
        try s.raw("flashnext-hc", "rows", try hcrows.source(s.a, .{ .nd = 324, .low = 320, .down = format, .up = format, .nib = nib }));
    };
}

fn recordedWidths(s: *Set) !void {
    const variants = @import("../src/families/flashnext/variants.zig");
    for (ks.flashnext_gen.sources) |file| {
        const expert = std.mem.startsWith(u8, file.name, "qa_expert_");
        const hc = std.mem.startsWith(u8, file.name, "qa_hc_");
        if (!expert and !hc) continue;
        for ([_]u8{ 2, 3, 4, 5, 6, 8 }) |bits| for ([_]usize{ 32, 64, 128 }) |group| {
            var constants: std.ArrayList(variants.Const) = .empty;
            for ([_][]const u8{ "WB", "SWB", "BITS" }) |name| {
                if (std.mem.indexOf(u8, file.text, try std.fmt.allocPrint(s.a, "constexpr int {s} = ", .{name})) != null)
                    try constants.append(s.a, .{ .name = name, .value = bits });
            }
            for ([_][]const u8{ "WG", "SWG", "GS" }) |name| {
                if (std.mem.indexOf(u8, file.text, try std.fmt.allocPrint(s.a, "constexpr int {s} = ", .{name})) != null)
                    try constants.append(s.a, .{ .name = name, .value = group });
            }
            if (constants.items.len == 0) continue;
            for ([_]bool{ false, true }) |half_lanes| {
                const text = try variants.source(s.a, file.text, constants.items, half_lanes, false, std.mem.indexOf(u8, file.name, "gateup") != null);
                try s.raw("flashnext-widths", file.name, text);
                if (expert and bits == 4 and group == 32) {
                    const slotted = try variants.source(s.a, file.text, constants.items, half_lanes, true, std.mem.indexOf(u8, file.name, "gateup") != null);
                    try s.raw("flashnext-fsr-widths", file.name, slotted);
                }
            }
        };
    }
}
