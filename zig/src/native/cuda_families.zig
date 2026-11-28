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
const inkling2 = @import("inkling2");

/// CUDA families provide metadata, open their lane backend and explain their refusals.
pub const registry = .{ nemotron.native, nemotron.omni, nemotron.transcribe, glm.native, glm.native_dsa, minimax.native, qwen35.native, qwen35.native_moe, flashnext.native, kolibri1.native, gemma4.native, gemma4.native31, spark25.native, dory.native, @import("dots_mocr_cuda").native, @import("muse_spark").native, @import("ricursive_cuda").tp2.native, @import("ricursive_cuda").native, @import("lfm3_cuda").native, @import("lfm2_moe_cuda").native, inkling2.native };
/// Model types the CUDA opener also takes (`cuda_also_model_types`): listed after the families, with their formats.
const aliases: usize = blk: {
    var n: usize = 0;
    for (registry) |F| n += if (@hasDecl(F, "cuda_also_model_types")) F.cuda_also_model_types.len else 0;
    break :blk n;
};

// The two runtime variants publish one model type with both formats.
const metadata_count = registry.len - 1;
const ripple_formats = blk: {
    const one = @import("ricursive_cuda").native.formats;
    const two = @import("ricursive_cuda").tp2.native.formats;
    var all: [one.len + two.len][]const u8 = undefined;
    @memcpy(all[0..one.len], one);
    @memcpy(all[one.len..], two);
    break :blk all;
};

pub const families: []const api.Family = blk: {
    var out: [metadata_count + 6 + aliases]api.Family = undefined;
    var count: usize = 0;
    for (registry) |F| {
        if (F == @import("ricursive_cuda").tp2.native) continue;
        out[count] = .{ .model_type = F.model_type, .formats = F.formats, .decision_formats = if (@hasDecl(F, "decision_formats")) F.decision_formats else &.{} };
        if (F == @import("ricursive_cuda").native) out[count].formats = &ripple_formats;
        count += 1;
    }
    std.debug.assert(count == metadata_count);
    out[metadata_count] = lfm2.family;
    out[metadata_count + 1] = .{ .model_type = "fish_omni", .formats = &.{"fp8-block128"} };
    out[metadata_count + 2] = .{ .model_type = "qwen4", .formats = @import("qwen4").family.native_host.formats };
    out[metadata_count + 3] = .{ .model_type = "qwen4_text", .formats = @import("qwen4").family.native_host.formats };
    out[metadata_count + 4] = .{ .model_type = qwen3.native.model_type, .formats = qwen3.native.formats };
    out[metadata_count + 5] = .{ .model_type = "zonos3", .formats = &.{"bf16"} };
    var at: usize = metadata_count + 6;
    for (registry) |F| if (@hasDecl(F, "cuda_also_model_types")) for (F.cuda_also_model_types) |alias| {
        out[at] = .{ .model_type = alias, .formats = F.formats, .decision_formats = if (@hasDecl(F, "decision_formats")) F.decision_formats else &.{} };
        at += 1;
    };
    const final = out;
    break :blk &final;
};

test "every registered family and d1's decision engine are listed for capabilities" {
    try std.testing.expectEqual(@as(usize, metadata_count + 6 + aliases), families.len);
    try std.testing.expectEqualStrings("nemotron_h", families[0].model_type);
    try std.testing.expectEqualStrings("lfm2_moe", families[metadata_count - 2].model_type);
    try std.testing.expectEqualStrings("inkling2_loop", families[metadata_count - 1].model_type);
    try std.testing.expectEqualStrings("qwen4", families[metadata_count + 2].model_type);
    try std.testing.expectEqualStrings("qwen4_text", families[metadata_count + 3].model_type);
    try std.testing.expectEqualStrings("qwen3", families[metadata_count + 4].model_type);
    try std.testing.expectEqualStrings("zonos3", families[metadata_count + 5].model_type);
}

test "CUDA aliases: each is listed for capabilities, model info and benchmarks, with its family's formats" {
    var seen: usize = 0;
    inline for (registry) |F| if (@hasDecl(F, "cuda_also_model_types")) for (F.cuda_also_model_types) |alias| {
        const f = for (families) |f| {
            if (std.mem.eql(u8, f.model_type, alias)) break f;
        } else return error.AliasNotListed;
        try std.testing.expectEqual(F.formats.ptr, f.formats.ptr);
        seen += 1;
    };
    try std.testing.expect(seen >= 1); // kolibri1_1 opens Kolibri 1's family
    for (families) |f| if (std.mem.eql(u8, f.model_type, "kolibri1_1")) return;
    return error.Kolibri11NotListed;
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

test "capability audit: dense Qwen3.5 lists the ModelOpt formats its loader reads" {
    for (families) |f| if (std.mem.eql(u8, f.model_type, "qwen3_5")) {
        var nvfp4 = false;
        var mixed = false;
        for (f.formats) |format| {
            nvfp4 = nvfp4 or std.mem.eql(u8, format, "modelopt-nvfp4");
            mixed = mixed or std.mem.eql(u8, format, "modelopt-mixed");
        }
        try std.testing.expect(nvfp4 and mixed);
        return;
    };
    return error.NoQwen35Family;
}

test "CUDA aliases: Ripple metadata lists both rank formats once" {
    var count: usize = 0;
    var nvfp4 = false;
    var exl3 = false;
    for (families) |f| if (std.mem.eql(u8, f.model_type, "ricursive")) {
        count += 1;
        for (f.formats) |format| {
            nvfp4 = nvfp4 or std.mem.eql(u8, format, "modelopt-nvfp4");
            exl3 = exl3 or std.mem.eql(u8, format, "exl3");
        }
    };
    try std.testing.expectEqual(@as(usize, 1), count);
    try std.testing.expect(nvfp4 and exl3);
    try std.testing.expectEqual(@as(i64, 32768), @import("ricursive_cuda").native.default_context);
    try std.testing.expectEqual(@as(i64, 1048576), @import("ricursive_cuda").tp2.native.default_context);
}
