//! MiniMax-H3-Image's edit encoder kernels, compiled at load from source on any Apple GPU.
pub const source = @embedFile("h3_image_encoder.metal");
pub const names = [_][:0]const u8{ "hie_conv", "hie_group_part", "hie_group_finish", "hie_norm_silu" };
