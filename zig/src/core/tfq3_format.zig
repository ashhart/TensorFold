//! TFQ3's on-disk records and wire constants, one copy for every backend; each backend keeps its checks and loaders.
const std = @import("std");
pub const tfq4 = @import("tfq4_format.zig");

pub const magic_layer: u32 = 0x57504654; // "TFPW": layer_NN.tffpw and mtp_00.tffpw, routed experts
pub const magic_nonmoe: u32 = 0x4E504654; // "TFPN": nonmoe.tffpw and dense shards
pub const magic_tables: u32 = 0x33565750; // "PWV3": pw_v3_tables.bin
pub const format_version: u32 = 3;
/// The boundary a GPU section (scales, weights) starts on in a published file.
pub const section_align: usize = 16384;
/// The kernels' whole table: magic, version, six affine floats, padding, three LUTs and three c64 tables.
pub const tables_bytes: usize = 10816;
/// A published pw_v3_tables.bin: the table padded to one section.
pub const tables_file_bytes: usize = 16384;

/// A TFPW file's header: one layer's routed experts, their sections and the converter's tag.
pub const LayerHeader = extern struct {
    magic: u32,
    version: u32,
    layer_idx: u32,
    num_experts: u32,
    num_tensors: u32,
    hidden_size: u32,
    moe_inter: u32,
    reserved0: u32,
    raw_fp8_bytes: u64,
    stream_words: u64,
    scale_halfs: u64,
    off_meta: u64,
    off_scales: u64,
    off_weights: u64,
    reserved1: [4]u64,
    meta_tag: [32]u8,
};

/// One routed expert's record in a TFPW file: its width and the gate, up and down offsets into both sections.
pub const ExpertMeta = extern struct {
    b_bits: u32,
    heat_rank: u32,
    gate_word_off: u64,
    up_word_off: u64,
    down_word_off: u64,
    gate_sc_off: u32,
    up_sc_off: u32,
    down_sc_off: u32,
    gate_snr_db: f32,
    up_snr_db: f32,
    down_snr_db: f32,
    reserved: [2]u32,
};

/// A TFPN file's header: its dense tensor records and both sections' extents.
pub const NonMoeHeader = extern struct {
    magic: u32,
    version: u32,
    num_tensors: u32,
    pad0: u32,
    off_meta: u64,
    off_scales: u64,
    off_weights: u64,
    total_scale_halfs: u64,
    total_weight_words: u64,
    pad1: u64,
};

comptime {
    std.debug.assert(@sizeOf(LayerHeader) == 144);
    std.debug.assert(@sizeOf(ExpertMeta) == 64);
    std.debug.assert(@sizeOf(NonMoeHeader) == 64);
}

/// Whether `bits` is valid in a hybrid `TFPW` v3 MoE shard (`2`/`225` TFQ4 E8P or `3`/`4`/`5` TFQ3).
pub fn validTfpwBits(bits: u32) bool {
    return tfq4.isValidHybridBits(bits);
}

/// Packed `u32` weight word count for a hybrid `TFPW` `[r, c]` matrix (`b_bits in {2, 225, 3, 4, 5}`).
pub fn matWords(r: u32, c: u32, bits: u32) tfq4.Error!u64 {
    return tfq4.hybridMatWords(r, c, bits);
}

test {
    _ = @import("tfq3_format_test.zig");
    _ = tfq4;
}
