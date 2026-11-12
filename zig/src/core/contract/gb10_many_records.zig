//! Actual launch geometry and the measured sm_121 live resource attributes.
const r = @import("registry.zig");
const measured = blk: {
    var usage = r.none;
    usage[4] = .{ .regs = 64, .static = 0, .bound = 128 };
    break :blk usage;
};
pub const records = [_]r.CudaRecord{
    .{ .launch = .{ .name = "lane_gemv GB10 grouped decode and verify", .symbol = "tf_lane_gemv_many", .image = "lane_gemv", .threads = 128, .dynamic = 35328, .min_sm = 121 }, .usage = measured },
};
