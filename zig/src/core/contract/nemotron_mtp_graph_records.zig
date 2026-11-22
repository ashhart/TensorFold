//! MTP graph launches are registered; resource measurements stay pending until the real nvcc audit runs.
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "nemotron_keyed tf_nemo_mtp_greedy", .symbol = "tf_nemo_mtp_greedy", .image = "nemotron_keyed", .threads = 256, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "nemotron_keyed tf_nemo_mtp_start", .symbol = "tf_nemo_mtp_start", .image = "nemotron_keyed", .threads = 1, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "nemotron_keyed tf_nemo_mtp_control", .symbol = "tf_nemo_mtp_control", .image = "nemotron_keyed", .threads = 1, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "nemotron_keyed tf_ctl_nemo_mtp_feed", .symbol = "tf_ctl_nemo_mtp_feed", .image = "nemotron_keyed", .threads = 1, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
};
