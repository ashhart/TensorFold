//! topk_rows.cu's and sparse_index.cu's launches: every family's row top-k, indexer scores and block summaries.
const std = @import("std");
const registry = @import("registry.zig");
const R = registry.CudaRecord;
const U = @import("limits.zig").Usage;

const narrow_name = "topk rows tf_topk_rows";
const scores_name = "sparse index tf_index_scores (64 heads)";
const wide_name = "topk rows tf_topk_rows_wide";

fn with(regs: [5]u32, static: u32, bound: u32) [5]U {
    var out: [5]U = undefined;
    for (&out, regs) |*u, r| u.* = .{ .regs = r, .static = static, .bound = bound };
    return out;
}

/// nvcc 13.0, sm_80 / 86 / 89 / 120 / 121 (--ftz=false); the scores' shared memory is 64 heads' at most.
pub const cuda = [_]R{
    .{ .launch = .{ .name = scores_name, .symbol = "tf_index_scores", .image = "sparse_index", .threads = 64, .dynamic = 64 * 128 * 2 + 64 * 4 }, .usage = with(.{ 40, 64, 48, 46, 46 }, 0, 64) },
    .{ .launch = .{ .name = "sparse index tf_index_blocks", .symbol = "tf_index_blocks", .image = "sparse_index", .threads = 128 }, .usage = with(.{ 30, 34, 34, 26, 26 }, 16, 128) },
    .{ .launch = .{ .name = "sparse index tf_index_open_blocks", .symbol = "tf_index_open_blocks", .image = "sparse_index", .threads = 128 }, .usage = with(.{ 25, 37, 37, 32, 32 }, 16, 128) },
    .{ .launch = .{ .name = narrow_name, .symbol = "tf_topk_rows", .image = "topk_rows", .threads = 128 }, .usage = with(.{ 56, 54, 54, 40, 40 }, 1064, 128) },
    .{ .launch = .{ .name = wide_name, .symbol = "tf_topk_rows_wide", .image = "topk_rows", .threads = 1024 }, .usage = with(.{ 32, 57, 57, 60, 64 }, 1288, 1024) },
};

/// nvcc 13.2 where it differs.
pub const nvcc132 = [_]registry.Override{
    .{ .name = scores_name, .usage = with(.{ 40, 64, 48, 56, 56 }, 0, 64) },
    .{ .name = wide_name, .usage = with(.{ 32, 57, 57, 64, 64 }, 1288, 1024) },
};

test "sparse index records: the launches fit every SM by their measured records" {
    for (cuda) |r| try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &.{r}, true));
}
