//! Gate 5's records of Flash Next's index scores on HIP.
const Record = @import("hip_records.zig").Record;

pub const records = [_]Record{
    .{
        .launch = .{ .name = "Flash Next index scores, 4 lanes a score", .symbol = "tf_fn_idx_scores_quad", .image = "flashnext", .threads = 128 },
        .usage = .{
            .{ .vgprs = 184, .sgprs = 22, .lds = 32772, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 184, .sgprs = 22, .lds = 32772, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 184, .sgprs = 20, .lds = 32772, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 184, .sgprs = 20, .lds = 32772, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "Flash Next index scores, 8 rows a wave", .symbol = "tf_fn_idx_scores8", .image = "flashnext", .threads = 256 },
        .usage = .{
            .{ .vgprs = 79, .sgprs = 26, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 93, .sgprs = 26, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 93, .sgprs = 26, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 93, .sgprs = 44, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
};
