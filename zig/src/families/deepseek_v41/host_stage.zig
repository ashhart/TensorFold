//! Family sizing constants and the CUDA core copy-source owner.
pub const chunk: usize = 128;
pub const Stage = @import("cuda").copy_source.Stage;
