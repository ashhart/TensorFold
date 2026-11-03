//! Unlimited-OCR-2's Metal launches, one file a kernel group; the contract's limits gate checks them in all_metal.
pub const records = @import("uocr_rows_records.zig").records ++ @import("uocr_moves_records.zig").records ++ @import("uocr_attention_records.zig").records ++ @import("uocr_route_records.zig").records ++ @import("uocr_matvec_records.zig").records;
