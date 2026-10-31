//! Actual launch geometry; nvcc usage remains pending until measured.
const r = @import("registry.zig");
pub const records = [_]r.CudaRecord{
    .{ .launch = .{ .name = "lane_gemv GB10 grouped decode and verify", .symbol = "tf_lane_gemv_many", .image = "lane_gemv", .threads = 128, .dynamic = 35328, .min_sm = 121 }, .usage = r.none },
};
