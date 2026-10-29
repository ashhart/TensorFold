//! Family sizing constants and the CUDA core copy-source owner.
pub const chunk: usize = 128;
pub const upload_chunk: usize = 64 << 20;
pub const Stage = @import("cuda").copy_source.Stage;
