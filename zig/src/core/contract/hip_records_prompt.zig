//! Gate 5's records of the 64-row prompt attention tiles (prefill.hip's tf_fa_wide_* and tf_fa_paged_*).
const limits = @import("limits.zig");
const Record = @import("hip_records.zig").Record;

const U = limits.HipUsage;

/// The 64-row prompt attention tile (prefill.hip), flat or over pages, at 256 threads and no dynamic LDS.
fn promptTile(comptime symbol: []const u8, usage: [limits.rdna.len]U) Record {
    const name = "prompt attention " ++ symbol;
    return .{ .launch = .{ .name = name, .symbol = symbol, .image = "prefill", .threads = 256 }, .usage = usage };
}

pub const records = [_]Record{
    promptTile("tf_fa_wide_f16", .{
        .{ .vgprs = 231, .sgprs = 45, .lds = 40960, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 231, .sgprs = 45, .lds = 40960, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 231, .sgprs = 45, .lds = 40960, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 231, .sgprs = 47, .lds = 40960, .private = 0, .bound = 256, .wave = 32 },
    }),
    promptTile("tf_fa_wide_bf16", .{
        .{ .vgprs = 247, .sgprs = 45, .lds = 40960, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 231, .sgprs = 45, .lds = 40960, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 231, .sgprs = 45, .lds = 40960, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 231, .sgprs = 47, .lds = 40960, .private = 0, .bound = 256, .wave = 32 },
    }),
    promptTile("tf_fa_paged_f16", .{
        .{ .vgprs = 232, .sgprs = 51, .lds = 40960, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 232, .sgprs = 51, .lds = 40960, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 232, .sgprs = 51, .lds = 40960, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 232, .sgprs = 51, .lds = 40960, .private = 0, .bound = 256, .wave = 32 },
    }),
    promptTile("tf_fa_paged_bf16", .{
        .{ .vgprs = 250, .sgprs = 50, .lds = 40960, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 232, .sgprs = 51, .lds = 40960, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 232, .sgprs = 51, .lds = 40960, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 232, .sgprs = 51, .lds = 40960, .private = 0, .bound = 256, .wave = 32 },
    }),
};
