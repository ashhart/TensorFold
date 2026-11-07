//! Compiler-derived GLM-OCR launch usage on gfx1030, gfx1100, gfx1151 and gfx1201.
const Record = @import("hip_records.zig").Record;
pub const records = [12]Record{
    .{ .launch = .{ .name = "GLM-OCR tf_glmocr_bias", .symbol = "tf_glmocr_bias", .image = "glm_ocr", .threads = 256 }, .usage = .{
        .{ .vgprs = 12, .sgprs = 17, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 12, .sgprs = 18, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 12, .sgprs = 17, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 12, .sgprs = 21, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
    } },
    .{ .launch = .{ .name = "GLM-OCR tf_glmocr_bias_fill", .symbol = "tf_glmocr_bias_fill", .image = "glm_ocr", .threads = 256 }, .usage = .{
        .{ .vgprs = 10, .sgprs = 19, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 10, .sgprs = 19, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 10, .sgprs = 19, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 10, .sgprs = 23, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
    } },
    .{ .launch = .{ .name = "GLM-OCR tf_glmocr_cast", .symbol = "tf_glmocr_cast", .image = "glm_ocr", .threads = 256 }, .usage = .{
        .{ .vgprs = 5, .sgprs = 9, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 5, .sgprs = 18, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 5, .sgprs = 8, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 5, .sgprs = 7, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
    } },
    .{ .launch = .{ .name = "GLM-OCR tf_glmocr_add", .symbol = "tf_glmocr_add", .image = "glm_ocr", .threads = 256 }, .usage = .{
        .{ .vgprs = 6, .sgprs = 18, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 6, .sgprs = 18, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 6, .sgprs = 14, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 6, .sgprs = 11, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
    } },
    .{ .launch = .{ .name = "GLM-OCR tf_glmocr_silu_mul", .symbol = "tf_glmocr_silu_mul", .image = "glm_ocr", .threads = 256 }, .usage = .{
        .{ .vgprs = 9, .sgprs = 18, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 9, .sgprs = 18, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 9, .sgprs = 14, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 9, .sgprs = 11, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
    } },
    .{ .launch = .{ .name = "GLM-OCR tf_glmocr_gelu", .symbol = "tf_glmocr_gelu", .image = "glm_ocr", .threads = 256 }, .usage = .{
        .{ .vgprs = 8, .sgprs = 10, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 8, .sgprs = 18, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 8, .sgprs = 10, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 8, .sgprs = 7, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
    } },
    .{ .launch = .{ .name = "GLM-OCR tf_glmocr_silu_mul_packed", .symbol = "tf_glmocr_silu_mul_packed", .image = "glm_ocr", .threads = 256 }, .usage = .{
        .{ .vgprs = 11, .sgprs = 20, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 11, .sgprs = 20, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 11, .sgprs = 20, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 11, .sgprs = 23, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
    } },
    .{ .launch = .{ .name = "GLM-OCR tf_glmocr_merge_layout", .symbol = "tf_glmocr_merge_layout", .image = "glm_ocr", .threads = 256 }, .usage = .{
        .{ .vgprs = 13, .sgprs = 20, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 13, .sgprs = 20, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 13, .sgprs = 20, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 13, .sgprs = 25, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
    } },
    .{ .launch = .{ .name = "GLM-OCR tf_glmocr_qkv_split", .symbol = "tf_glmocr_qkv_split", .image = "glm_ocr", .threads = 256 }, .usage = .{
        .{ .vgprs = 11, .sgprs = 24, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 12, .sgprs = 24, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 11, .sgprs = 24, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 11, .sgprs = 27, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
    } },
    .{ .launch = .{ .name = "GLM-OCR tf_glmocr_norm", .symbol = "tf_glmocr_norm", .image = "glm_ocr", .threads = 256 }, .usage = .{
        .{ .vgprs = 16, .sgprs = 30, .lds = 1024, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 15, .sgprs = 30, .lds = 1024, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 15, .sgprs = 30, .lds = 1024, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 15, .sgprs = 30, .lds = 1024, .private = 0, .bound = 256, .wave = 32 },
    } },
    .{ .launch = .{ .name = "GLM-OCR tf_glmocr_vision_rope", .symbol = "tf_glmocr_vision_rope", .image = "glm_ocr", .threads = 256 }, .usage = .{
        .{ .vgprs = 18, .sgprs = 14, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 26, .sgprs = 18, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 18, .sgprs = 10, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 18, .sgprs = 10, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
    } },
    .{ .launch = .{ .name = "GLM-OCR tf_glmocr_text_rope", .symbol = "tf_glmocr_text_rope", .image = "glm_ocr", .threads = 256 }, .usage = .{
        .{ .vgprs = 17, .sgprs = 21, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 24, .sgprs = 21, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 17, .sgprs = 21, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 17, .sgprs = 25, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
    } },
};
