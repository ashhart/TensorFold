//! Gate 5's records of the planned walk's short-span score (plan.hip's tf_plan_score2_*, two keys a warp).
const limits = @import("limits.zig");
const Record = @import("hip_records.zig").Record;

const U = limits.HipUsage;

/// The planned walk's score at two keys a warp: 128 threads, the block's queries in dynamic LDS.
fn shortScore(comptime symbol: []const u8, usage: [limits.rdna.len]U) Record {
    const name = "planned walk " ++ symbol;
    return .{ .launch = .{ .name = name, .symbol = symbol, .image = "plan", .threads = 128, .dynamic = 16 * 128 * 4 }, .usage = usage };
}

pub const records = [4]Record{
    shortScore("tf_plan_score2_f16", .{
        .{ .vgprs = 31, .sgprs = 34, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        .{ .vgprs = 31, .sgprs = 33, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        .{ .vgprs = 31, .sgprs = 34, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        .{ .vgprs = 31, .sgprs = 30, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
    }),
    shortScore("tf_plan_score2_bf16", .{
        .{ .vgprs = 31, .sgprs = 34, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        .{ .vgprs = 31, .sgprs = 33, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        .{ .vgprs = 31, .sgprs = 34, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        .{ .vgprs = 31, .sgprs = 30, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
    }),
    shortScore("tf_plan_score2_act_f16", .{
        .{ .vgprs = 31, .sgprs = 34, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        .{ .vgprs = 31, .sgprs = 32, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        .{ .vgprs = 31, .sgprs = 32, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        .{ .vgprs = 31, .sgprs = 30, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
    }),
    shortScore("tf_plan_score2_act_bf16", .{
        .{ .vgprs = 31, .sgprs = 34, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        .{ .vgprs = 31, .sgprs = 32, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        .{ .vgprs = 31, .sgprs = 32, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        .{ .vgprs = 31, .sgprs = 30, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
    }),
};
