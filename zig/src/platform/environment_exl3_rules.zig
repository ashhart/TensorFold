//! EXL3 CUDA environment rules, kept at their original position in the platform rule order.
const std = @import("std");
pub const rules = [_]@import("environment.zig").Rule{
    .{ .name = "TF_FLASHNEXT_CACHE_TELEMETRY", .scope = .cuda, .kind = .flag, .forms = "0 or 1 (1 logs EXL3-CACHE expert-cache counters, with a sync at each boundary)" },
    .{ .name = "TF_FLASHNEXT_EXL3_LAYER_TELEMETRY", .scope = .cuda, .kind = .flag, .forms = "0 or 1 (1 adds per-layer counters when TF_FLASHNEXT_CACHE_TELEMETRY is 1)" },
    .{ .name = "TF_FLASHNEXT_SNAP_HOST", .scope = .cuda, .kind = .flag, .forms = "0 or 1 (kept prompt-cache states in pinned host memory; unset: 1 above 32768 tokens of context on an EXL3 pack, else 0)" },
    .{ .name = "TF_FLASHNEXT_EXL3_ARCH", .scope = .cuda, .kind = .word, .words = &.{"unqualified"}, .forms = "unqualified (serves an EXL3 pack on a device that is not sm_120, without a claim)" },
    .{ .name = "TENSORFOLD_MEMORY_RESERVE_GIB", .scope = .cuda, .kind = .real, .trim = " ", .forms = "a finite GiB count of at least 0.5 (2 on unified memory) and at most the memory's size" },
    .{ .name = "TF_FLASHNEXT_EXL3_LINE_DELTAS", .scope = .cuda, .kind = .path, .forms = "POOL:+N,POOL:-N entries without control characters that move expert lines between decoder pools, as many added as removed" },
    .{ .name = "TF_FLASHNEXT_EXL3_PROFILE_SIZING", .scope = .cuda, .kind = .flag, .forms = "0 or 1 (1 sizes the expert pools from the stored profile)" },
    .{ .name = "TF_FLASHNEXT_EXL3_STREAMS", .scope = .cuda, .kind = .size, .min = 1, .forms = "a positive whole stream count the expert pools leave room for (default 1; --parallel wins)" },
    .{ .name = "TF_FLASHNEXT_EXPERT_LINES", .scope = .cuda, .kind = .size, .forms = "a whole count of expert lines a layer keeps on the device (the pool refuses fewer than 10)" },
    .{ .name = "TF_FLASHNEXT_EXPERT_RESERVE_MB", .scope = .cuda, .kind = .u64, .max = std.math.maxInt(u64) >> 20, .forms = "a whole MiB count from 0 to 17592186044415 left free after the expert pools" },
    .{ .name = "TF_FLASHNEXT_PLE_READ", .scope = .cuda, .kind = .word, .words = &.{ "map", "pread" }, .forms = "map or pread (default map; pread reads the n-gram rows by positional read)" },
    .{ .name = "TF_FLASHNEXT_EXL3_PROFILE_DIR", .scope = .cuda, .kind = .path, .forms = "a directory path without control characters where expert-use profiles persist" },
    .{ .name = "TF_FLASHNEXT_EXL3_ROUTED", .scope = .cuda, .kind = .word, .words = &.{ "host", "device" }, .forms = "host or device (default host; device keeps every routed expert on the device)" },
};
