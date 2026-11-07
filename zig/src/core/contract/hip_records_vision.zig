//! Measured resources of the shared vision kernels, in limits.rdna order.
const Record = @import("hip_records.zig").Record;

pub const records = [12]Record{
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
    .{
        .launch = .{ .name = "vision tf_vl_embed", .symbol = "tf_vl_embed", .image = "ops", .threads = 256 },
        .usage = .{
            .{ .vgprs = 13, .sgprs = 23, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 13, .sgprs = 23, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 13, .sgprs = 23, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 13, .sgprs = 25, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "vision tf_vl_linear", .symbol = "tf_vl_linear", .image = "ops", .threads = 256 },
        .usage = .{
            .{ .vgprs = 26, .sgprs = 26, .lds = 5120, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 26, .sgprs = 26, .lds = 5120, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 26, .sgprs = 26, .lds = 5120, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 26, .sgprs = 26, .lds = 5120, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "vision tf_vl_layernorm", .symbol = "tf_vl_layernorm", .image = "ops", .threads = 256 },
        .usage = .{
            .{ .vgprs = 17, .sgprs = 29, .lds = 1024, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 16, .sgprs = 30, .lds = 1024, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 16, .sgprs = 30, .lds = 1024, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 16, .sgprs = 30, .lds = 1024, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "vision tf_vl_pointwise", .symbol = "tf_vl_pointwise", .image = "ops", .threads = 256 },
        .usage = .{
            .{ .vgprs = 10, .sgprs = 18, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 10, .sgprs = 18, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 10, .sgprs = 14, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 10, .sgprs = 14, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "vision tf_vl_layout", .symbol = "tf_vl_layout", .image = "ops", .threads = 256 },
        .usage = .{
            .{ .vgprs = 15, .sgprs = 27, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 15, .sgprs = 27, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 15, .sgprs = 27, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 15, .sgprs = 32, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "vision tf_vl_resize", .symbol = "tf_vl_resize", .image = "ops", .threads = 256 },
        .usage = .{
            .{ .vgprs = 27, .sgprs = 31, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 27, .sgprs = 31, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 27, .sgprs = 31, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 27, .sgprs = 38, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "vision tf_vl_position", .symbol = "tf_vl_position", .image = "ops", .threads = 256 },
        .usage = .{
            .{ .vgprs = 24, .sgprs = 26, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 25, .sgprs = 26, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 24, .sgprs = 26, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 24, .sgprs = 30, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
};
