//! Gate 5's record of the unquantized embedding gather (elementwise.hpp's tf_embed_dense, fp32 or bf16 tables).
const limits = @import("limits.zig");
const Record = @import("hip_records.zig").Record;

pub const records = [_]Record{.{
    .launch = .{ .name = "embedding gather tf_embed_dense", .symbol = "tf_embed_dense", .image = "ops", .threads = 256 },
    .usage = .{
        .{ .vgprs = 13, .sgprs = 21, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
        .{ .vgprs = 13, .sgprs = 21, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
        .{ .vgprs = 13, .sgprs = 21, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
        .{ .vgprs = 13, .sgprs = 23, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
    },
}};
