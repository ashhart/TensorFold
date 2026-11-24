const std = @import("std");
const DType = @import("tensor_metal").DType;

pub const copy = "sam3_copy";
pub const bias_rows = "sam3_bias_rows";
pub const columns = "sam3_columns";
pub const transpose = "sam3_transpose";
pub const scale = "sam3_scale";
pub const relu = "sam3_relu";
pub const sigmoid = "sam3_sigmoid";
pub const channel_fill = "sam3_channel_fill";
pub const tiles = "sam3_tiles";
pub const resample = "sam3_resample";
pub const conv_pack = "sam3_conv_pack";
pub const deconv_pack = "sam3_deconv_pack";
pub const deconv_scatter = "sam3_deconv_scatter";
pub const unfold = "sam3_unfold";
pub const depthwise = "sam3_depthwise";
pub const group_norm = "sam3_group_norm";
pub const rotary_table = "sam3_rotary_table";
pub const rotate = "sam3_rotate";
pub const rotate_split_f32 = "sam3_rotate_split_f32";
pub const rotate_split_f32_flag = "sam3_rotate_split_f32_flag";
pub const rotate_split_f16 = "sam3_rotate_split_f16";
pub const tile_project = "sam3_tile_project";
pub const tile_project_gelu = "sam3_tile_project_gelu";
pub const tile_project_residual = "sam3_tile_project_residual";
pub const tile_norm_f16 = "sam3_tile_norm_f16";
pub const attention_rows = "sam3_attention_rows";
pub const mask_squash = "sam3_mask_squash";
pub const mask_first_conv = "sam3_mask_first_conv";

pub fn flash(dtype: DType, head_width: usize) error{UnsupportedFlashShape}![]const u8 {
    return switch (dtype) {
        .f32 => switch (head_width) {
            32 => "sam3_flash_f32_d32_k32",
            64 => "sam3_flash_f32_d64_k32",
            else => error.UnsupportedFlashShape,
        },
        .f16 => switch (head_width) {
            32 => "sam3_flash_f16_d32_k64",
            64 => "sam3_flash_f16_d64_k32",
            else => error.UnsupportedFlashShape,
        },
        .bf16 => error.UnsupportedFlashShape,
    };
}

pub const Scale = extern struct { count: u32, factor: f32 };
pub const ChannelFill = extern struct { pixels: u32, channels: u32, first: u32, count: u32, bits: u32 };
pub const Tiles = extern struct { channels: u32, grid: u32, tile: u32, direction: u32 };
pub const ConvShape = extern struct {
    in_h: u32,
    in_w: u32,
    in_c: u32,
    out_c: u32,
    kernel: u32,
    out_h: u32,
    out_w: u32,
    stride: u32,
    pad: u32,
    first_row: u32,
    row_count: u32,
};
pub const Rotary = extern struct {
    rows: u32,
    heads: u32,
    head_width: u32,
    rotated_rows: u32,
    positions: u32,
    side: u32,
    scale: f32,
};
pub const RotarySplit = extern struct { rows: u32, width: u32, head_width: u32, positions: u32 };
pub const TileProject = extern struct { rows: u32, cols: u32, depth: u32, scatter: u32, grid: u32, tile: u32 };
pub const TileNorm = extern struct { width: u32, gather: u32, grid: u32, tile: u32, eps: f32 };
pub const Flash = extern struct { rows: u32, keys: u32, heads: u32, batch: u32 };
pub const RowsAttention = extern struct { rows: u32, keys: u32, heads: u32, head_width: u32, has_bias: u32 };
pub const Squash = extern struct {
    pixels: u32,
    stride: u32,
    slots: u32,
    sources: u32,
    active: u32,
    threshold: u32,
    accurate: u32,
    gain: f32,
    offset: f32,
};

test "SAM kernel parameter blocks have the Metal layout" {
    const sizes = .{
        .{ Scale, 8 },   .{ ChannelFill, 20 },   .{ Tiles, 16 },       .{ ConvShape, 44 },
        .{ Rotary, 28 }, .{ RotarySplit, 16 },   .{ TileProject, 24 }, .{ TileNorm, 20 },
        .{ Flash, 16 },  .{ RowsAttention, 20 }, .{ Squash, 36 },
    };
    inline for (sizes) |entry| {
        try std.testing.expectEqual(@as(usize, entry[1]), @sizeOf(entry[0]));
        inline for (@typeInfo(entry[0]).@"struct".field_names, 0..) |name, i|
            try std.testing.expectEqual(4 * i, @offsetOf(entry[0], name));
    }
    try std.testing.expectEqual(@as(usize, 40), @offsetOf(ConvShape, "row_count"));
    try std.testing.expectEqual(@as(usize, 32), @offsetOf(Squash, "offset"));
}

test "SAM flash kernel names cover exactly the four tile shapes" {
    try std.testing.expectEqualStrings("sam3_flash_f32_d32_k32", try flash(.f32, 32));
    try std.testing.expectEqualStrings("sam3_flash_f32_d64_k32", try flash(.f32, 64));
    try std.testing.expectEqualStrings("sam3_flash_f16_d32_k64", try flash(.f16, 32));
    try std.testing.expectEqualStrings("sam3_flash_f16_d64_k32", try flash(.f16, 64));
    try std.testing.expectError(error.UnsupportedFlashShape, flash(.f32, 16));
    try std.testing.expectError(error.UnsupportedFlashShape, flash(.f16, 128));
    try std.testing.expectError(error.UnsupportedFlashShape, flash(.bf16, 32));
}
