//! The learned change's sizes and settings, the same on every backend: a block of ranks a lesson, its sketches, Adam.

pub const block = 16; // ranks a lesson adds at each layer
pub const max_rank = 512; // thirty-two lessons' blocks
pub const max_blocks = max_rank / block;
pub const scale: f32 = 10;
pub const avoid_dims = 192; // directions sketched from the inputs a new block must leave alone
pub const candidates = 64; // directions sketched from the fact's inputs, which a new block's are chosen among

/// Adam's settings for a block's outputs (its gate keeps them off what must stay): rate, moments' decays, epsilon.
pub const hyper = [4]f32{ 3e-4, 0.9, 0.999, 1e-8 };

/// Each new input direction's length, as MLX's LoRA draws its down factor's columns (uniform within 1 / sqrt(in)).
pub const a_norm: f32 = 0.577;

/// A block's directions' squared length: a row's cosine with one is (x . a) / sqrt(unit |x|^2).
pub const unit = a_norm * a_norm;

/// A block's gate before its lesson sets one: above any cosine, so the block stays shut.
pub const shut: f32 = 2;
