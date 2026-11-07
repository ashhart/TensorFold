//! Measured resources of the shared vision kernels, in limits.rdna order.
const Record = @import("hip_records.zig").Record;

pub const records = [_]Record{
    .{
        .launch = .{ .name = "BF16 LayerNorm", .symbol = "tf_layer_norm_bf16", .image = "ops", .threads = 256 },
        .usage = .{
            .{ .vgprs = 14, .sgprs = 28, .lds = 1024, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 14, .sgprs = 28, .lds = 1024, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 14, .sgprs = 28, .lds = 1024, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 13, .sgprs = 28, .lds = 1024, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "BF16 exact GELU", .symbol = "tf_gelu_bf16", .image = "ops", .threads = 256 },
        .usage = .{
            .{ .vgprs = 8, .sgprs = 9, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 8, .sgprs = 18, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 8, .sgprs = 8, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 8, .sgprs = 7, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "BF16 axial vision RoPE", .symbol = "tf_vision_rope_bf16", .image = "ops", .threads = 256 },
        .usage = .{
            .{ .vgprs = 32, .sgprs = 29, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 33, .sgprs = 29, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 32, .sgprs = 29, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 32, .sgprs = 30, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "FP32 attention to BF16 rows", .symbol = "tf_attention_rows_bf16", .image = "ops", .threads = 256 },
        .usage = .{
            .{ .vgprs = 17, .sgprs = 25, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 25, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 25, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 27, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "BF16 full image attention", .symbol = "tf_fa_full_bf16", .image = "prefill", .threads = 256 },
        .usage = .{
            .{ .vgprs = 245, .sgprs = 45, .lds = 40960, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 229, .sgprs = 46, .lds = 40960, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 229, .sgprs = 46, .lds = 40960, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 229, .sgprs = 48, .lds = 40960, .private = 0, .bound = 256, .wave = 32 },
        },
    },
};
