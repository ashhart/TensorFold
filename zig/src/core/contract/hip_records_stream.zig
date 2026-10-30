//! Gate 5 for every plain and SwiGLU stream specialization; group size is a runtime argument.
const std = @import("std");
const limits = @import("limits.zig");

pub fn tile(comptime bits: u32, comptime rows: u32, comptime cb: u32, comptime pair: bool) limits.HipLaunch {
    @setEvalBranchQuota(300_000);
    const fmt = "_ZN2tf4rocm18affine_dot2_streamINS0_{s}ELi{d}ELi{d}ELi{d}ELb{d}EEEvNS0_6AffineENS0_11StreamSidesEii";
    const half = std.fmt.comptimePrint(fmt, .{ "6DotF16", bits, rows, cb, @intFromBool(pair) });
    const brain = std.fmt.comptimePrint(fmt, .{ "7DotBF16", bits, rows, cb, @intFromBool(pair) });
    return .{
        .name = std.fmt.comptimePrint("{s} stream tile {d}-bit {d}x{d}", .{
            if (pair) "swiglu" else "plain", bits, rows, cb,
        }),
        .symbol = brain,
        .image = "affine_tiles",
        .threads = 128,
        .symbols = .{ half, brain, brain, brain },
    };
}

pub const records = @import("hip_records_stream_plain.zig").records ++
    @import("hip_records_stream_swiglu.zig").records;
