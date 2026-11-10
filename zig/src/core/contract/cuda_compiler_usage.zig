//! Resource receipts for changed CUDA sources, ahead of the registry's unchanged compiler-version fallback.
const r = @import("registry.zig");
const limits = @import("limits.zig");

pub fn sourceUsage(record: r.CudaRecord, release: ?limits.Nvcc) ?[limits.sms.len]limits.Usage {
    if (@import("qwen_root_records.zig").lookup(record, release)) |usage| return usage;
    if (@import("qwen_tap_records.zig").lookup(record, release)) |usage| return usage;
    if (record.first_nvcc != null and (release == null or limits.newer(record.first_nvcc.?, release.?))) return r.none;
    return null;
}
