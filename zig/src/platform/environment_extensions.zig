//! Scoped serving settings extend the shared registry without editing the upstream setting table.
const std = @import("std");
const e = @import("environment.zig");

pub const rules = [_]e.Rule{
    .{ .name = "TF_CUBLAS_BRIDGE", .scope = .cuda, .kind = .path, .forms = "a shared-library path without control characters" },
    .{ .name = "TF_CUBLAS_PREFILL", .scope = .cuda, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_CUDA_WARPS8", .scope = .cuda, .kind = .u32, .max = 7, .forms = "a whole kernel mask from 0 to 7" },
    .{ .name = "TF_DRAFT_BF16", .scope = .cuda, .kind = .u32, .max = 31, .forms = "a whole projection mask from 0 to 31" },
    .{ .name = "TF_DRAFT_BF16_EMBED_DIR", .scope = .cuda, .kind = .path, .forms = "a checkpoint directory path without control characters" },
    .{ .name = "TF_DRAFT_BLOCK", .scope = .cuda, .kind = .size, .only = &.{ 8, 16 }, .forms = "8 or 16" },
    .{ .name = "TF_DRAFT_COPY_MATCH", .scope = .cuda, .kind = .u64, .min = 2, .max = 20, .forms = "a whole copy-match count from 2 to 20" },
    .{ .name = "TF_DRAFT_DATASET_DIR", .scope = .cuda, .kind = .path, .forms = "an output directory path without control characters" },
    .{ .name = "TF_DRAFT_FEEDBACK_TRACE", .scope = .cuda, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_DRAFT_FEEDBACK_WEIGHT", .scope = .cuda, .kind = .real, .max = 1, .forms = "a finite feedback weight from 0 to 1" },
    .{ .name = "TF_DRAFT_GRAPH", .scope = .cuda, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_DRAFT_LANES", .scope = .cuda, .kind = .u32, .only = &.{ 1, 2, 4 }, .forms = "1, 2 or 4" },
    .{ .name = "TF_DRAFT_LANE_BASE_WEIGHT", .scope = .cuda, .kind = .real, .max = 1, .forms = "a finite base-lane weight from 0 to 1" },
    .{ .name = "TF_DRAFT_LATTICE_TRACE", .scope = .cuda, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_DRAFT_MTP_DIR", .scope = .cuda, .kind = .path, .forms = "a checkpoint directory path without control characters" },
    .{ .name = "TF_DRAFT_MTP_GRAFT", .scope = .cuda, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_DRAFT_MTP_PREFIX", .scope = .cuda, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_DRAFT_MTP_QUERIES", .scope = .cuda, .kind = .size, .min = 1, .max = 63, .forms = "a whole query count from 1 to 63 (at least the base tree's node count)" },
    .{ .name = "TF_DRAFT_MTP_WEIGHT", .scope = .cuda, .kind = .real, .max = 1, .forms = "a finite MTP proposal weight from 0 to 1" },
    .{ .name = "TF_DRAFT_QWEN3_NORM", .scope = .cuda, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_DRAFT_REFERENCE_DIR", .scope = .cuda, .kind = .path, .forms = "an output directory path without control characters" },
    .{ .name = "TF_DRAFT_REFINE", .scope = .cuda, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_DRAFT_WIDE_TRACE", .scope = .cuda, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_QWEN_AFFINE_PREPARED", .scope = .cuda, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_QWEN_ATTN_EXACT", .scope = .cuda, .kind = .word, .words = &.{ "0", "1", "2", "3", "4", "5", "6", "7" }, .forms = "one exact-attention selector character from 0 to 7" },
    .{ .name = "TF_QWEN_EXPAND_KV", .scope = .cuda, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_QWEN_FAST_PREFILL", .scope = .cuda, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_QWEN_FP8_ATTENTION", .scope = .cuda, .kind = .word, .words = &.{ "0", "1", "2" }, .forms = "0 (disabled), 1 (single) or 2 (residual)" },
    .{ .name = "TF_QWEN_FP8_TILE", .scope = .cuda, .kind = .word, .words = &.{ "0", "1", "2", "3" }, .forms = "0 (baseline), 1 (128x64), 2 (128x128) or 3 (64x64)" },
    .{ .name = "TF_QWEN_GDN_CHUNKED", .scope = .cuda, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_QWEN_NVFP4_MLP_DIR", .scope = .cuda, .kind = .path, .forms = "an alternate checkpoint directory path without control characters" },
    .{ .name = "TF_QWEN_NVFP4_PREFILL", .scope = .cuda, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_QWEN_NVFP4_SCOPE", .scope = .cuda, .kind = .word, .words = &.{ "mlp", "layers" }, .forms = "mlp or layers" },
    .{ .name = "TF_QWEN_NVFP4_TMA", .scope = .cuda, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_QWEN_PREFILL_GRAPH", .scope = .cuda, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_QWEN_PREFILL_PROFILE", .scope = .cuda, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_QWEN_PREFILL_ROWS", .scope = .cuda, .kind = .size, .min = 64, .forms = "a whole row count in multiples of 64 from 64 to the compiled Qwen prompt limit" },
};

pub fn isQwen(name: []const u8) bool {
    for (rules) |r| if (std.mem.eql(u8, name, r.name))
        return !std.mem.eql(u8, name, "TF_CUDA_WARPS8");
    return false;
}

/// Preserve validation order while composing one registry for all production strict readers.
pub fn extend(comptime core: anytype) [core.len + rules.len]e.Rule {
    const cut = for (core, 0..) |r, i| {
        if (std.mem.eql(u8, r.name, "TF_DSV41_MHC_DEFER")) break i;
    } else @compileError("environment extension anchor is missing");
    var out: [core.len + rules.len]e.Rule = undefined;
    @memcpy(out[0..cut], core[0..cut]);
    @memcpy(out[cut..][0..rules.len], &rules);
    @memcpy(out[cut + rules.len ..], core[cut..]);
    return out;
}

/// The extra row constraint leaves the upstream numeric computation and other settings unchanged.
pub fn checkMultiple(name: []const u8, raw: ?[]const u8) e.Error!void {
    if (!std.mem.eql(u8, name, "TF_QWEN_PREFILL_ROWS")) return;
    const text = raw orelse return;
    if (std.mem.trim(u8, text, " \t\r\n").len == 0) return;
    const n = std.fmt.parseInt(u64, text, 10) catch return error.InvalidEnvironment;
    if (n % 64 != 0) return error.InvalidEnvironment;
}
