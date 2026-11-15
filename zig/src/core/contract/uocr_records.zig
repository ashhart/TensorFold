//! Unlimited-OCR-2's Metal launches, one file a kernel group; the contract's limits gate checks them in all_metal.
pub const records = @import("uocr_vision_glue_records.zig").records ++ @import("uocr_vision_maps_records.zig").records ++ @import("uocr_vision_attn_records.zig").records ++ @import("uocr_decoder_records.zig").records;
