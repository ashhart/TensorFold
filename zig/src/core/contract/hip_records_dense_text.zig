//! Measured resources of dense BF16 text operators, in limits.rdna order.
const Record = @import("hip_records.zig").Record;

pub const records = [2]Record{
    .{
        .launch = .{ .name = "Dense BF16 embedding", .symbol = "tf_embed_bf16", .image = "ops", .threads = 256 },
        .usage = .{
            .{ .vgprs = 13, .sgprs = 22, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 13, .sgprs = 22, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 13, .sgprs = 22, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 13, .sgprs = 24, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "Torch BF16 RoPE", .symbol = "tf_rope_torch_bf16", .image = "ops", .threads = 256 },
        .usage = .{
            .{ .vgprs = 22, .sgprs = 25, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 28, .sgprs = 25, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 22, .sgprs = 25, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 22, .sgprs = 28, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
};
