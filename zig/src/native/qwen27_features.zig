//! The 27B's tapped rows for a drafter in front of it (lanes.drafted), retained per stream on the device.
const tf = @import("tensorfold");
const lanes = tf.lanes;
const be = lanes.backend;
const Metal = @import("qwen27_lanes.zig").Metal;
const retained = @import("sensenova_features.zig");

/// The drafter's layers: every pass keeps their rows from now on (one layer: a row is contiguous).
pub fn prepare(ptr: *anyopaque, taps: []const u32) anyerror!void {
    const b = Metal.self(ptr);
    if (b.draft != null or taps.len != 1) return error.BadTapLayers;
    try retained.prepareFeaturesFn(ptr, taps);
    b.external = true;
}

/// Held rows [start, start + count), never a view another stream's pass can overwrite.
pub fn rows(ptr: *anyopaque, s: *lanes.Stream, taps: []const u32, start: u64, count: u32) anyerror!be.Features {
    return retained.featuresFn(ptr, s, taps, start, count);
}
