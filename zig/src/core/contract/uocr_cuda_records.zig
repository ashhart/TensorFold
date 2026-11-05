//! Unlimited-OCR-2's CUDA launches (uocr.cu), group by group as zig/tests/cuda/contract_uocr_*.zig registers them.

pub const cuda = @import("uocr_cuda_rows_records.zig").cuda ++ @import("uocr_cuda_moves_records.zig").cuda ++ @import("uocr_cuda_route_records.zig").cuda ++ @import("uocr_cuda_attention_records.zig").cuda ++ @import("uocr_cuda_matvec_records.zig").cuda;
