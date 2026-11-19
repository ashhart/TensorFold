//! The CUDA families: the registry the host opens from, and the capability entries each model_type lists.
const std = @import("std");
const api = @import("engine_api");
const nemotron = @import("nemotron");
const flashnext = @import("flashnext");
const lfm2 = @import("lfm2.zig");
const glm = @import("glm_cuda");
const minimax = @import("minimax_m3_cuda");
const qwen35 = @import("qwen35");
const kolibri1 = @import("kolibri1");
const gemma4 = @import("gemma4_cuda");
const spark25 = @import("spark25");
const dory = @import("dory");
const qwen3 = @import("qwen3_cuda");

/// CUDA families provide metadata, open their lane backend and explain their refusals.
pub const registry = .{ nemotron.native, nemotron.omni, glm.native, minimax.native, qwen35.native, qwen35.native_moe, flashnext.native, kolibri1.native, gemma4.native, gemma4.native31, spark25.native, dory.native, @import("dots_mocr_cuda").native, @import("lfm2_moe_cuda").native, @import("mimo_v2").native };
pub const families: []const api.Family = blk: {
    var out: [registry.len + 5]api.Family = undefined;
    for (registry, 0..) |F, i| out[i] = .{ .model_type = F.model_type, .formats = F.formats, .decision_formats = if (@hasDecl(F, "decision_formats")) F.decision_formats else &.{} };
    out[registry.len] = lfm2.family;
    out[registry.len + 1] = .{ .model_type = "fish_omni", .formats = &.{"fp8-block128"} };
    out[registry.len + 2] = .{ .model_type = "qwen4", .formats = @import("qwen4").family.native_host.formats };
    out[registry.len + 3] = .{ .model_type = "qwen4_text", .formats = @import("qwen4").family.native_host.formats };
    out[registry.len + 4] = .{ .model_type = qwen3.native.model_type, .formats = qwen3.native.formats };
    const final = out;
    break :blk &final;
};

test "every registered family and d1's decision engine are listed for capabilities" {
    try std.testing.expectEqual(@as(usize, registry.len + 5), families.len);
    try std.testing.expectEqualStrings("nemotron_h", families[0].model_type);
    try std.testing.expectEqualStrings("lfm2_moe", families[registry.len - 2].model_type);
    try std.testing.expectEqualStrings("mimo_v2", families[registry.len - 1].model_type);
    try std.testing.expectEqual(@as(usize, 1), families[registry.len - 1].formats.len);
    try std.testing.expectEqualStrings("exl3", families[registry.len - 1].formats[0]);
    try std.testing.expectEqualStrings("lfm2_vl", families[registry.len].model_type);
    try std.testing.expectEqualStrings("qwen4", families[registry.len + 2].model_type);
    try std.testing.expectEqualStrings("qwen4_text", families[registry.len + 3].model_type);
    try std.testing.expectEqualStrings("qwen3", families[registry.len + 4].model_type);
}

test "dots.mocr serves its publisher's BF16 weights on CUDA through the dense engine" {
    for (families) |f| if (std.mem.eql(u8, f.model_type, "dots_ocr")) {
        try std.testing.expectEqual(@as(usize, 1), f.formats.len);
        return std.testing.expectEqualStrings("bf16", f.formats[0]);
    };
    return error.NoDotsFamily;
}

test "GLM capabilities: CUDA offers the EXL3 format its loader accepts" {
    const config_json = "{\"quantization_config\":{\"quant_method\":\"exl3\",\"bits\":4,\"codebook\":\"mcg\"}}";
    var why: @import("core").modelopt.Why = .{};
    try std.testing.expect(glm.exl3.isExl3(std.testing.allocator, config_json));
    try glm.exl3.admit(std.testing.allocator, config_json, &why);
    for (families) |family| if (std.mem.eql(u8, family.model_type, "glm5_next")) {
        for (family.formats) |format| if (std.mem.eql(u8, format, "exl3")) return;
    };
    return error.NoExl3Format;
}

test "capability audit: CUDA Qwen3.5 lists BF16 under decisions only, the MoE entry and the others none" {
    for (families) |f| {
        for (f.formats) |format| try std.testing.expect(!std.mem.eql(u8, format, "bf16") or !std.mem.startsWith(u8, f.model_type, "qwen3_5"));
        if (std.mem.eql(u8, f.model_type, "qwen3_5")) {
            try std.testing.expectEqual(@as(usize, 1), f.decision_formats.len);
            try std.testing.expectEqualStrings("bf16", f.decision_formats[0]);
        } else try std.testing.expectEqual(@as(usize, 0), f.decision_formats.len);
    }
}
