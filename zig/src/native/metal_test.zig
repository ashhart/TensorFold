//! The Metal entry points' refusals before any load, its family capabilities and Nemotron's plain sharing.
const std = @import("std");
const metal = @import("metal.zig");
const tf = @import("tensorfold");
const lanes = tf.lanes;
const nemotron = tf.nemotron;
const open = metal.open;
const openFlashNext = metal.openFlashNext;
const families = metal.families;
const prompt_int8_refusal = metal.prompt_int8_refusal;

test "prompt precision: Metal refuses int8 before any model or family-specific load" {
    for ([_][]const u8{ "qwen3_5", "fish_omni", "nemotron_h", "gemma4", "unknown" }) |model| {
        var why: []const u8 = "";
        const opened = try open(std.testing.allocator, std.testing.allocator, std.testing.io, .{ .dir = "missing-prompt-int8-model", .model_type = model, .prompt_int8 = true }, &why);
        try std.testing.expect(opened == null);
        try std.testing.expectEqualStrings(prompt_int8_refusal, why);
    }
}

test "slide admission: Metal entry points refuse before checkpoint or device access" {
    const check = @import("slide_open_test.zig").check;
    const kimi = @import("kimi_metal.zig");
    for (families) |f| {
        if (std.mem.eql(u8, f.model_type, "nemotron_h") or std.mem.eql(u8, f.model_type, "qwen3_5") or std.mem.eql(u8, f.model_type, "qwen3_5_moe")) continue;
        if (std.mem.eql(u8, f.model_type, "kimi_k3") or std.mem.eql(u8, f.model_type, "kimi_k31")) {
            const want = "Kimi: learned states and on-disk prompt states are not implemented";
            try @import("slide_open_test.zig").message(open, f.model_type, want);
            try @import("slide_open_test.zig").message(kimi.open, f.model_type, want);
        } else try check(open, "Macs", &.{f.model_type});
    }
    try check(@import("gemma4.zig").open, "Macs", &.{"gemma4"});
    try check(@import("lfm2_moe.zig").open, "Macs", &.{"lfm2_moe"});
    try check(@import("step5.zig").open, "Macs", &.{"step5_vl"});
    try check(@import("spark25.zig").open, "Macs", &.{"spark2_5"});
    try check(@import("minimax_m3.zig").open, "Macs", &.{"minimax_m3_vl"});
    try check(@import("uocr.zig").open, "Macs", &.{"unlimited-ocr"});
    try check(openFlashNext, "Macs", &.{"qwen4_exp"});
}

test "Metal capabilities: fish_omni retains FP8 and includes its native NVFP4 expert format" {
    var fp8 = false;
    var nvfp4 = false;
    for (families) |family| if (std.mem.eql(u8, family.model_type, "fish_omni")) {
        for (family.formats) |format| {
            fp8 = fp8 or std.mem.eql(u8, format, "fp8-block128");
            nvfp4 = nvfp4 or std.mem.eql(u8, format, "nvfp4");
        }
    };
    try std.testing.expect(fp8 and nvfp4);
}

test "plain sharing: Nemotron without a head shares streams without widening its window" {
    var b: nemotron.backend.Metal = undefined;
    b.o = .{ .capacity = 4096, .streams = 4, .batch_rows = 32, .drafts = false };
    b.head = null;
    b.costs = .{};
    const facts = b.facts();
    try std.testing.expect(facts.plain_share and facts.hidden_rows);
    try std.testing.expectEqual(@as(u32, 1), facts.exact_width);
    try std.testing.expect(!facts.mtp and !facts.speculate and !facts.allocate_shared_drafts);
    var config = try lanes.Config.init(std.testing.allocator, facts, nemotron.backend.max_window, nemotron.backend.max_window - 1);
    defer config.deinit(std.testing.allocator);
    try std.testing.expect(config.family_streams and config.pipelined);
    try std.testing.expectEqual(@as(u32, 4), config.batch_streams);
    try std.testing.expectEqual(@as(u32, 1), config.family_width);
    try std.testing.expectEqual(@as(u32, 0), config.max_copy);
    try std.testing.expectEqual(@as(u32, 0), config.most_drafts);
    try std.testing.expect(!config.family_mtp);
}
