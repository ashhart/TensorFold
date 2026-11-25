//! The EXL3 chain's Flash Next rows of the central settings table, and the values each one accepts.
const std = @import("std");
const e = @import("environment.zig");

/// One accepted value, as environment_valid_cases.zig declares it, so both tables concatenate.
pub const Case = @import("environment_valid_cases.zig").Case;

pub const rules = [_]e.Rule{
    .{ .name = "TF_FLASHNEXT_EXL3_MTP_RESIDENT", .scope = .cuda, .kind = .flag, .forms = "0 or 1 (1 keeps the EXL3 MTP experts on device)" },
    .{ .name = "TF_FLASHNEXT_KV_HOST", .scope = .cuda, .kind = .size, .max = std.math.maxInt(usize) / (1 << 20), .forms = "a whole MiB budget fitting a native byte size (0 disables the host tier; blank is unset)" },
    .{ .name = "TF_FLASHNEXT_KV_FRAMES", .scope = .cuda, .kind = .size, .min = 1, .forms = "a positive whole device-frame count, or blank for the computed budget" },
    .{ .name = "TF_FLASHNEXT_KV_SLOTS", .scope = .cuda, .kind = .size, .min = 1, .forms = "a positive whole lent-frame count per stream, or blank for the default" },
    .{ .name = "TF_FLASHNEXT_EXL3_PREFILL", .scope = .cuda, .kind = .flag, .forms = "0 or 1 (default 1)" },
    .{ .name = "TF_FLASHNEXT_EXL3_PREFILL_ROWS", .scope = .cuda, .kind = .size, .forms = "a whole nonnegative row count; unset uses 8192 for general prefill and 4096 for serving, an explicit count applies to both" },
    .{ .name = "TF_FLASHNEXT_EXL3_PREFILL_REDUCED_BRANCH", .scope = .cuda, .kind = .flag, .forms = "0 or 1 (default 0)" },
    .{ .name = "TF_FLASHNEXT_EXL3_PREFILL_TWO_CHUNK", .scope = .cuda, .kind = .flag, .forms = "0 or 1 (default 0)" },
    .{ .name = "TF_FLASHNEXT_EXL3_PREFILL_RING", .scope = .cuda, .kind = .flag, .forms = "0 or 1 (default 1)" },
    .{ .name = "TF_FLASHNEXT_EXL3_PREFILL_RING_AHEAD", .scope = .cuda, .kind = .size, .forms = "a whole nonnegative ring lookahead count (default 1)" },
    .{ .name = "TF_FLASHNEXT_EXL3_PREFILL_OVERLAP", .scope = .cuda, .kind = .flag, .forms = "0 or 1 (default 1)" },
    .{ .name = "TF_FLASHNEXT_CACHE_TELEMETRY", .scope = .cuda, .kind = .flag, .forms = "0 or 1 (1 logs EXL3-CACHE expert-cache counters, with a sync at each boundary)" },
    .{ .name = "TF_FLASHNEXT_EXL3_LAYER_TELEMETRY", .scope = .cuda, .kind = .flag, .forms = "0 or 1 (1 adds per-layer counters when TF_FLASHNEXT_CACHE_TELEMETRY is 1)" },
    .{ .name = "TF_FLASHNEXT_SNAP_HOST", .scope = .cuda, .kind = .flag, .forms = "0 or 1 (kept prompt-cache states in pinned host memory; unset: 1 above 32768 tokens of context on an EXL3 pack, else 0)" },
    .{ .name = "TF_FLASHNEXT_EXL3_ARCH", .scope = .cuda, .kind = .word, .words = &.{"unqualified"}, .forms = "unqualified (serves an EXL3 pack on a device that is not sm_120, without a claim)" },
    .{ .name = "TF_FLASHNEXT_EXL3_LINE_DELTAS", .scope = .cuda, .kind = .path, .forms = "POOL:+N,POOL:-N entries without control characters that move expert lines between decoder pools, as many added as removed" },
    .{ .name = "TF_FLASHNEXT_EXL3_PROFILE_SIZING", .scope = .cuda, .kind = .flag, .forms = "0 or 1 (1 sizes the expert pools from the stored profile)" },
    .{ .name = "TF_FLASHNEXT_EXL3_STREAMS", .scope = .cuda, .kind = .size, .min = 1, .forms = "a positive whole stream count the expert pools leave room for (default 1; --parallel wins)" },
    .{ .name = "TF_FLASHNEXT_EXPERT_LINES", .scope = .cuda, .kind = .size, .forms = "a whole count of expert lines a layer keeps on the device (the pool refuses fewer than 10)" },
    .{ .name = "TF_FLASHNEXT_EXPERT_RESERVE_MB", .scope = .cuda, .kind = .u64, .max = std.math.maxInt(u64) >> 20, .forms = "a whole MiB count from 0 to 17592186044415 left free after the expert pools" },
    .{ .name = "TF_FLASHNEXT_PLE_READ", .scope = .cuda, .kind = .word, .words = &.{ "map", "pread" }, .forms = "map or pread (default map; pread reads the n-gram rows by positional read)" },
    .{ .name = "TF_FLASHNEXT_EXL3_PROFILE_DIR", .scope = .cuda, .kind = .path, .forms = "a directory path without control characters where expert-use profiles persist" },
    .{ .name = "TF_FLASHNEXT_EXL3_ROUTED", .scope = .cuda, .kind = .word, .words = &.{ "host", "device" }, .forms = "host or device (default host; device keeps every routed expert on the device)" },
};

pub const valid = [_]Case{
    .{ .name = "TF_FLASHNEXT_SHARD_HEAD", .raw = "1", .n = 1 },
    .{ .name = "TF_FLASHNEXT_EXL3_MTP_RESIDENT", .raw = "1", .n = 1 },
    .{ .name = "TF_FLASHNEXT_KV_HOST", .raw = "513", .n = 513 },
    .{ .name = "TF_FLASHNEXT_KV_FRAMES", .raw = "7", .n = 7 },
    .{ .name = "TF_FLASHNEXT_KV_SLOTS", .raw = "3", .n = 3 },
    .{ .name = "TF_FLASHNEXT_CHUNK", .raw = "1", .n = 1 },
    .{ .name = "TF_FLASHNEXT_CHUNK", .raw = "16384", .n = 16384 },
    .{ .name = "TF_FLASHNEXT_CHUNK_SOLO", .raw = "16384", .n = 16384 },
    .{ .name = "TF_FLASHNEXT_CACHE_TELEMETRY", .raw = "1", .n = 1 },
    .{ .name = "TF_FLASHNEXT_EXL3_LAYER_TELEMETRY", .raw = "0", .n = 0 },
    .{ .name = "TF_FLASHNEXT_SNAP_HOST", .raw = "0", .n = 0 },
    .{ .name = "TF_FLASHNEXT_EXL3_ARCH", .raw = "unqualified", .n = 0 },
    .{ .name = "TENSORFOLD_MEMORY_RESERVE_GIB", .raw = "0.5", .n = @bitCast(@as(f64, 0.5)) },
    .{ .name = "TF_FLASHNEXT_EXL3_LINE_DELTAS", .raw = "20:+1,15:-1", .n = 0 },
    .{ .name = "TF_FLASHNEXT_EXL3_PROFILE_SIZING", .raw = "1", .n = 1 },
    .{ .name = "TF_FLASHNEXT_EXL3_STREAMS", .raw = "8", .n = 8 },
    .{ .name = "TF_FLASHNEXT_EXL3_PREFILL", .raw = "0", .n = 0 },
    .{ .name = "TF_FLASHNEXT_EXL3_PREFILL_ROWS", .raw = "16385", .n = 16385 },
    .{ .name = "TF_FLASHNEXT_EXL3_PREFILL_REDUCED_BRANCH", .raw = "1", .n = 1 },
    .{ .name = "TF_FLASHNEXT_EXL3_PREFILL_TWO_CHUNK", .raw = "1", .n = 1 },
    .{ .name = "TF_FLASHNEXT_EXL3_PREFILL_RING", .raw = "0", .n = 0 },
    .{ .name = "TF_FLASHNEXT_EXL3_PREFILL_RING_AHEAD", .raw = "3", .n = 3 },
    .{ .name = "TF_FLASHNEXT_EXL3_PREFILL_OVERLAP", .raw = "0", .n = 0 },
    .{ .name = "TF_FLASHNEXT_EXPERT_LINES", .raw = "0", .n = 0 },
    .{ .name = "TF_FLASHNEXT_EXPERT_RESERVE_MB", .raw = "17592186044415", .n = 17592186044415 },
    .{ .name = "TF_FLASHNEXT_PLE_READ", .raw = "pread", .n = 1 },
    .{ .name = "TF_FLASHNEXT_EXL3_PROFILE_DIR", .raw = "/var/lib/exl3 profiles", .n = 0 },
    .{ .name = "TF_FLASHNEXT_EXL3_ROUTED", .raw = "device", .n = 1 },
};
