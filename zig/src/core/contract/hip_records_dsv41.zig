//! Gate 5's records of DeepSeek-V4.1's indexer top-512 on HIP (deepseek_v41/topk512.hip's tf_dsv41_topk512).
const limits = @import("limits.zig");
const Record = @import("hip_records.zig").Record;

const U = limits.HipUsage;

/// A topk512.hip launch and its (VGPRs, SGPRs, LDS bytes) on each chip, wave32 with no scratch.
fn launch(comptime symbol: []const u8, threads: u32, u: [limits.rdna.len][3]u32) Record {
    var usage: [limits.rdna.len]U = undefined;
    for (&usage, u) |*x, v| x.* = .{ .vgprs = v[0], .sgprs = v[1], .lds = v[2], .bound = threads };
    const l: limits.HipLaunch = .{ .name = "dsv41 " ++ symbol, .symbol = symbol, .image = "dsv41", .threads = threads, .dynamic = 0 };
    return .{ .launch = l, .usage = usage };
}

pub const records = [_]Record{
    // a 1,024-thread block a row: 2,048 bins, 34 words of picks and two 32-word halves of wave totals in LDS
    launch("tf_dsv41_topk512", 1024, .{ .{ 47, 56, 8584 }, .{ 47, 56, 8584 }, .{ 47, 56, 8584 }, .{ 47, 56, 8584 } }),
};
