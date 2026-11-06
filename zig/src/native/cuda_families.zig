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

/// CUDA families provide metadata, open their lane backend and explain their refusals.
pub const registry = .{ nemotron.native, nemotron.omni, glm.native, minimax.native, qwen35.native, qwen35.native_moe, flashnext.native, kolibri1.native, gemma4.native, gemma4.native31, spark25.native, dory.native };
pub const families: []const api.Family = blk: {
    var out: [registry.len + 4]api.Family = undefined;
    for (registry, 0..) |F, i| out[i] = .{ .model_type = F.model_type, .formats = F.formats };
    out[registry.len] = lfm2.family;
    out[registry.len + 1] = .{ .model_type = "fish_omni", .formats = &.{"fp8-block128"} };
    out[registry.len + 2] = .{ .model_type = "qwen4", .formats = @import("qwen4").family.native_host.formats };
    out[registry.len + 3] = .{ .model_type = "qwen4_text", .formats = @import("qwen4").family.native_host.formats };
    const final = out;
    break :blk &final;
};

test "every registered family and d1's decision engine are listed for capabilities" {
    try std.testing.expectEqual(@as(usize, registry.len + 4), families.len);
    try std.testing.expectEqualStrings("nemotron_h", families[0].model_type);
    try std.testing.expectEqualStrings("lfm2_vl", families[registry.len].model_type);
    try std.testing.expectEqualStrings("qwen4", families[registry.len + 2].model_type);
    try std.testing.expectEqualStrings("qwen4_text", families[registry.len + 3].model_type);
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
