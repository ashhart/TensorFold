//! Gate 5's records of Flash Next's hyper-connection launches: the dense mix's gate table.
const limits = @import("limits.zig");
const Record = @import("hip_records.zig").Record;

pub const records = [_]Record{
    .{
        .launch = .{ .name = "Flash Next HC gate table tf_hc_gate_fill", .symbol = "tf_hc_gate_fill", .image = "affine_tiles", .threads = 256, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 7, .sgprs = 18, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 7, .sgprs = 4, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
};
