//! Gate 5's records of EXL3's decode-row kernels (exl3.hip's tf_exl3_rot_in and tf_exl3_linear_k*_w*).
const std = @import("std");
const limits = @import("limits.zig");
const Record = @import("hip_records.zig").Record;

const U = limits.HipUsage;

/// An exl3.hip launch and its (VGPRs, SGPRs, LDS bytes) on each chip, wave32 with no scratch.
fn launch(comptime symbol: []const u8, threads: u32, dynamic: u32, u: [limits.rdna.len][3]u32) Record {
    var usage: [limits.rdna.len]U = undefined;
    for (&usage, u) |*x, v| x.* = .{ .vgprs = v[0], .sgprs = v[1], .lds = v[2], .bound = threads };
    const l: limits.HipLaunch = .{
        .name = "exl3 " ++ symbol,
        .symbol = symbol,
        .image = "exl3",
        .threads = threads,
        .dynamic = dynamic,
    };
    return .{ .launch = l, .usage = usage };
}

/// The linear at `k2` half-bits and `wk` waves, its waves' sums for 8 rows in dynamic LDS: (VGPRs, SGPRs) a chip.
fn linear(comptime k2: u32, comptime wk: u32, u: [limits.rdna.len][2]u32) Record {
    var with: [limits.rdna.len][3]u32 = undefined;
    for (&with, u) |*x, v| x.* = .{ v[0], v[1], wk * k2 * 128 };
    return launch(std.fmt.comptimePrint("tf_exl3_linear_k{d}_w{d}", .{ k2, wk }), wk * 32, wk * 8 * 128 * 4, with);
}

pub const records = [_]Record{
    launch("tf_exl3_rot_in", 128, 0, .{ .{ 16, 12, 0 }, .{ 16, 18, 0 }, .{ 16, 10, 0 }, .{ 16, 9, 0 } }),
    linear(4, 4, .{ .{ 206, 45 }, .{ 205, 45 }, .{ 205, 45 }, .{ 205, 46 } }),
    linear(4, 8, .{ .{ 214, 43 }, .{ 213, 43 }, .{ 213, 43 }, .{ 213, 44 } }),
    linear(6, 4, .{ .{ 209, 45 }, .{ 208, 45 }, .{ 208, 45 }, .{ 208, 46 } }),
    linear(6, 8, .{ .{ 217, 43 }, .{ 216, 43 }, .{ 216, 43 }, .{ 216, 44 } }),
    linear(8, 4, .{ .{ 209, 45 }, .{ 208, 45 }, .{ 208, 45 }, .{ 208, 46 } }),
    linear(8, 8, .{ .{ 217, 43 }, .{ 216, 43 }, .{ 216, 43 }, .{ 216, 44 } }),
    linear(10, 4, .{ .{ 213, 45 }, .{ 212, 45 }, .{ 212, 45 }, .{ 212, 46 } }),
    linear(10, 8, .{ .{ 221, 43 }, .{ 220, 43 }, .{ 220, 43 }, .{ 220, 44 } }),
    linear(12, 4, .{ .{ 214, 45 }, .{ 213, 45 }, .{ 213, 45 }, .{ 213, 46 } }),
    linear(12, 8, .{ .{ 222, 43 }, .{ 221, 43 }, .{ 221, 43 }, .{ 221, 44 } }),
    linear(16, 4, .{ .{ 219, 45 }, .{ 218, 45 }, .{ 218, 45 }, .{ 218, 46 } }),
    linear(16, 8, .{ .{ 227, 43 }, .{ 226, 43 }, .{ 226, 43 }, .{ 226, 44 } }),
};
