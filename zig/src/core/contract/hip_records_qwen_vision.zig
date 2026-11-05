//! Compiler-derived Qwen vision and rounding-control usage on gfx1030, gfx1100, gfx1151 and gfx1201.
const Record = @import("hip_records.zig").Record;
pub const records = [_]Record{
    .{
        .launch = .{ .name = "Qwen vision tf_qvh_linear", .symbol = "tf_qvh_linear", .image = "qwen_vision", .threads = 32 },
        .usage = .{
            .{ .vgprs = 9, .sgprs = 16, .lds = 0, .private = 0, .bound = 32, .wave = 32 },
            .{ .vgprs = 9, .sgprs = 18, .lds = 0, .private = 0, .bound = 32, .wave = 32 },
            .{ .vgprs = 9, .sgprs = 16, .lds = 0, .private = 0, .bound = 32, .wave = 32 },
            .{ .vgprs = 9, .sgprs = 18, .lds = 0, .private = 0, .bound = 32, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "Qwen vision tf_qvh_embed", .symbol = "tf_qvh_embed", .image = "qwen_vision", .threads = 256 },
        .usage = .{
            .{ .vgprs = 17, .sgprs = 31, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 31, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 29, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 13, .sgprs = 34, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "Qwen vision tf_qvh_layernorm", .symbol = "tf_qvh_layernorm", .image = "qwen_vision", .threads = 256 },
        .usage = .{
            .{ .vgprs = 20, .sgprs = 22, .lds = 36, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 20, .sgprs = 22, .lds = 36, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 19, .sgprs = 22, .lds = 36, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 19, .sgprs = 24, .lds = 36, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "Qwen vision tf_qvh_qkv_rope", .symbol = "tf_qvh_qkv_rope", .image = "qwen_vision", .threads = 64 },
        .usage = .{
            .{ .vgprs = 25, .sgprs = 26, .lds = 0, .private = 0, .bound = 64, .wave = 32 },
            .{ .vgprs = 25, .sgprs = 26, .lds = 0, .private = 0, .bound = 64, .wave = 32 },
            .{ .vgprs = 25, .sgprs = 26, .lds = 0, .private = 0, .bound = 64, .wave = 32 },
            .{ .vgprs = 25, .sgprs = 24, .lds = 0, .private = 0, .bound = 64, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "Qwen vision rounding control tf_ctl_qvh_qkv_unrounded", .symbol = "tf_ctl_qvh_qkv_unrounded", .image = "qwen_vision", .threads = 64 },
        .usage = .{
            .{ .vgprs = 25, .sgprs = 22, .lds = 0, .private = 0, .bound = 64, .wave = 32 },
            .{ .vgprs = 25, .sgprs = 26, .lds = 0, .private = 0, .bound = 64, .wave = 32 },
            .{ .vgprs = 25, .sgprs = 22, .lds = 0, .private = 0, .bound = 64, .wave = 32 },
            .{ .vgprs = 25, .sgprs = 24, .lds = 0, .private = 0, .bound = 64, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "Qwen vision tf_qvh_attention64", .symbol = "tf_qvh_attention64", .image = "qwen_vision", .threads = 64 },
        .usage = .{
            .{ .vgprs = 223, .sgprs = 24, .lds = 32768, .private = 0, .bound = 64, .wave = 32 },
            .{ .vgprs = 223, .sgprs = 24, .lds = 32768, .private = 0, .bound = 64, .wave = 32 },
            .{ .vgprs = 223, .sgprs = 24, .lds = 32768, .private = 0, .bound = 64, .wave = 32 },
            .{ .vgprs = 219, .sgprs = 24, .lds = 32768, .private = 0, .bound = 64, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "Qwen vision tf_qvh_bias_gelu", .symbol = "tf_qvh_bias_gelu", .image = "qwen_vision", .threads = 256 },
        .usage = .{
            .{ .vgprs = 10, .sgprs = 19, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 9, .sgprs = 19, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 9, .sgprs = 19, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 9, .sgprs = 19, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "Qwen vision tf_qvh_bias_residual", .symbol = "tf_qvh_bias_residual", .image = "qwen_vision", .threads = 256 },
        .usage = .{
            .{ .vgprs = 9, .sgprs = 14, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 8, .sgprs = 18, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 8, .sgprs = 14, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 8, .sgprs = 17, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "Qwen vision tf_qvh_bias_cast", .symbol = "tf_qvh_bias_cast", .image = "qwen_vision", .threads = 256 },
        .usage = .{
            .{ .vgprs = 8, .sgprs = 14, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 8, .sgprs = 18, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 8, .sgprs = 14, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 8, .sgprs = 17, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "Qwen vision tf_qvh_text_rope", .symbol = "tf_qvh_text_rope", .image = "qwen_vision", .threads = 256 },
        .usage = .{
            .{ .vgprs = 23, .sgprs = 27, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 30, .sgprs = 27, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 23, .sgprs = 27, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 23, .sgprs = 28, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "Qwen vision rounding control tf_ctl_qvh_text_trig_unrounded", .symbol = "tf_ctl_qvh_text_trig_unrounded", .image = "qwen_vision", .threads = 256 },
        .usage = .{
            .{ .vgprs = 24, .sgprs = 27, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 32, .sgprs = 27, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 24, .sgprs = 27, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 24, .sgprs = 28, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "Qwen vision rounding control tf_ctl_qvh_text_products_unrounded", .symbol = "tf_ctl_qvh_text_products_unrounded", .image = "qwen_vision", .threads = 256 },
        .usage = .{
            .{ .vgprs = 23, .sgprs = 27, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 30, .sgprs = 27, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 23, .sgprs = 27, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 23, .sgprs = 28, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
};
