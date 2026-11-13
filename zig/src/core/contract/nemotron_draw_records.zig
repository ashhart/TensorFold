//! Nemotron's batched draw entries; usage is filled on a GPU (no CUDA compiler in this host cage).
const registry = @import("registry.zig");

pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "cuda sample tf_draw_ids_rows", .symbol = "tf_draw_ids_rows", .image = "sample", .threads = 1024 }, .usage = registry.none },
    .{ .launch = .{ .name = "nemotron_ops tf_nemo_gather_rows", .symbol = "tf_nemo_gather_rows", .image = "nemotron_ops", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "nemotron_ops tf_nemo_gather_last", .symbol = "tf_nemo_gather_last", .image = "nemotron_ops", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "nemotron_ops tf_nemo_scatter_rows", .symbol = "tf_nemo_scatter_rows", .image = "nemotron_ops", .threads = 32 }, .usage = registry.none },
};
