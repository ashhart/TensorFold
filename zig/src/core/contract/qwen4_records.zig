//! Native dense mixers declare source-derived ceilings; CUDA usage still requires five-SM compiler receipts.
const registry = @import("registry.zig");
const limits = @import("limits.zig");
const suffixes = [_][]const u8{ "", "_bad_nan", "_bad_repeat", "_bad_tail", "_bad_sums" };
pub const cuda = blk: {
    var out: [18]registry.CudaRecord = undefined;
    var at = 0;
    for (.{ .{ "qwen4_dense_act", "qwen4_hc", 256 }, .{ "qwen4_ple_gate", "qwen4_ple", 512 }, .{ "qwen4_ple_conv", "qwen4_ple", 512 }, .{ "qwen4_gdn_chain", "qwen4_gdn", 1024 } }) |item| {
        for (suffixes, 0..) |suffix, i| {
            if ((item[2] == 256 and i == 4) or (item[2] == 1024 and i == 0)) continue;
            out[at] = .{ .launch = .{ .name = item[1] ++ " " ++ item[0] ++ suffix, .symbol = item[0] ++ suffix, .image = item[1], .threads = item[2] }, .usage = registry.none };
            at += 1;
        }
    }
    if (at != out.len) @compileError("Qwen4 launch inventory changed");
    break :blk out;
};
const production = [_]limits.MetalLaunch{
    .{ .name = "qwen4-native hc qwen4_hc_sums", .threads = 256, .reserved = 256, .tg_memory = 8 * 8 * 4 },
    .{ .name = "qwen4-native hc qwen4_hc_normed", .threads = 256, .reserved = 256, .tg_memory = 0 },
    .{ .name = "qwen4-native hc qwen4_hc_mix", .threads = 256, .reserved = 256, .tg_memory = 0 },
    .{ .name = "qwen4-native hc qwen4_dense_act", .threads = 256, .reserved = 256, .tg_memory = 0 },
    .{ .name = "qwen4-native gdn-silu qwen4_gdn_chain", .threads = 1024, .reserved = 1024, .tg_memory = 4 * 16 * 128 * 2 + 2 * 16 * 2 * 4 + 16 * 4 },
    .{ .name = "qwen4-native gdn-sigmoid qwen4_gdn_chain", .threads = 1024, .reserved = 1024, .tg_memory = 4 * 16 * 128 * 2 + 2 * 16 * 2 * 4 + 16 * 4 },
    .{ .name = "qwen4-native qsa qwen4_attn_prep", .threads = 256, .reserved = 256, .tg_memory = (256 / 32 + 256) * 4 },
    .{ .name = "qwen4-native qsa qwen4_idx_pool", .threads = 128, .reserved = 128, .tg_memory = (128 / 32 + 128) * 4 },
    .{ .name = "qwen4-native qsa qwen4_idx_scores", .threads = 256, .reserved = 256, .tg_memory = 0 },
    .{ .name = "qwen4-native qsa qwen4_idx_select", .threads = 1024, .reserved = 1024, .tg_memory = 17680 },
    .{ .name = "qwen4-native qsa qwen4_attn_parts", .threads = 256, .reserved = 256, .tg_memory = (8 + 8 + 8 * 256) * 4 },
    .{ .name = "qwen4-native qsa qwen4_attn_merge", .threads = 256, .reserved = 256, .tg_memory = 0 },
    .{ .name = "qwen4-native ple qwen4_ple_gate", .threads = 256, .reserved = 256, .tg_memory = 3 * 8 * 4 },
    .{ .name = "qwen4-native ple qwen4_ple_conv", .threads = 256, .reserved = 256, .tg_memory = 0 },
};
pub const metal = blk: {
    @setEvalBranchQuota(10000);
    var out: [69]limits.MetalLaunch = undefined;
    var at = 0;
    for (production) |record| {
        for (.{ "", "bad_nan", "bad_repeat", "bad_tail", "bad_sums" }) |control| {
            if (@import("std").mem.indexOf(u8, record.name, "qwen4_idx_select") != null and @import("std").mem.eql(u8, control, "bad_sums")) continue;
            var next = record;
            if (control.len > 0) {
                const space = @import("std").mem.lastIndexOfScalar(u8, record.name, ' ').?;
                next.name = record.name[0..space] ++ " " ++ control ++ record.name[space..];
            }
            out[at] = next;
            at += 1;
        }
    }
    if (at != out.len) @compileError("Qwen4 Metal inventory changed");
    break :blk out;
};
