//! Gate 5's records of the MoE route and pick launches: a prompt's plan in three launches and its picks.
const Record = @import("hip_records.zig").Record;

pub const records = [_]Record{
    .{
        .launch = .{ .name = "MoE plan count tf_route_count", .symbol = "tf_route_count", .image = "ops", .threads = 256 },
        .usage = .{
            .{ .vgprs = 7, .sgprs = 13, .lds = 4096, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 7, .sgprs = 18, .lds = 4096, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 7, .sgprs = 13, .lds = 4096, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 7, .sgprs = 12, .lds = 4096, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "MoE plan scan tf_route_scan", .symbol = "tf_route_scan", .image = "ops", .threads = 1024 },
        .usage = .{
            .{ .vgprs = 11, .sgprs = 15, .lds = 20480, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 11, .sgprs = 15, .lds = 20480, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 10, .sgprs = 16, .lds = 20480, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 10, .sgprs = 16, .lds = 20480, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "MoE plan scatter tf_route_scatter", .symbol = "tf_route_scatter", .image = "ops", .threads = 512 },
        .usage = .{
            .{ .vgprs = 14, .sgprs = 14, .lds = 2048, .private = 0, .bound = 512, .wave = 32 },
            .{ .vgprs = 14, .sgprs = 18, .lds = 2048, .private = 0, .bound = 512, .wave = 32 },
            .{ .vgprs = 14, .sgprs = 14, .lds = 2048, .private = 0, .bound = 512, .wave = 32 },
            .{ .vgprs = 14, .sgprs = 12, .lds = 2048, .private = 0, .bound = 512, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "MoE prompt picks a wave a row tf_select_rows", .symbol = "tf_select_rows", .image = "decode", .threads = 128 },
        .usage = .{
            .{ .vgprs = 135, .sgprs = 49, .lds = 5428, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 135, .sgprs = 46, .lds = 5428, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 135, .sgprs = 50, .lds = 5428, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 135, .sgprs = 52, .lds = 5428, .private = 0, .bound = 128, .wave = 32 },
        },
    },
};
