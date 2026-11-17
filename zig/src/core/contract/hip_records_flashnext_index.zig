//! Gate-5 record for the Flash Next indexer's tf_fn_idx_select (series 6).
//! fn_cut_held is a __device__ helper called inside this one launch, so it gets no launch record.
//! Static LDS: hist[256] + waves[32] + whist[32][256] + cut_s + need_s = 33,928 bytes.
const std = @import("std");
const limits = @import("limits.zig");
const Record = @import("hip_records.zig").Record;

const U = limits.HipUsage;

fn row(vgprs: u32, sgprs: u32) U {
    return .{ .vgprs = vgprs, .sgprs = sgprs, .lds = 33928, .private = 0, .bound = 1024, .wave = 32 };
}

pub const records = [1]Record{
    .{
        .launch = .{ .name = "flashnext tf_fn_idx_select", .symbol = "tf_fn_idx_select", .image = "flashnext", .threads = 1024 },
        .usage = .{ row(64, 48), row(64, 48), row(63, 48), row(60, 50) },
    },
};
