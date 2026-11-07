//! Gate 5's record of the decode tail (decode.hip: tf_tail): a row a block of 512 threads, the row held in LDS.
const Record = @import("hip_records.zig").Record;

pub const records = [1]Record{.{
    .launch = .{ .name = "decode tail tf_tail", .symbol = "tf_tail", .image = "decode", .threads = 512 },
    .usage = .{
        .{ .vgprs = 125, .sgprs = 58, .lds = 32832, .private = 0, .bound = 512, .wave = 32 },
        .{ .vgprs = 120, .sgprs = 58, .lds = 32832, .private = 0, .bound = 512, .wave = 32 },
        .{ .vgprs = 120, .sgprs = 58, .lds = 32832, .private = 0, .bound = 512, .wave = 32 },
        .{ .vgprs = 120, .sgprs = 58, .lds = 32832, .private = 0, .bound = 512, .wave = 32 },
    },
}};
