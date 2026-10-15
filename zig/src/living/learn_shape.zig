//! The learned change's shape, shared by the Metal and CUDA learners: blocks of ranks, their scale and gates.

pub const block = 16; // ranks a lesson adds at each layer
pub const max_rank = 2048; // a hundred and twenty-eight lessons' blocks, each layer's buffers grown to it as lessons come
pub const max_blocks = max_rank / block;
pub const scale: f32 = 10;
pub const avoid_dims = 192; // directions sketched from the inputs a new block must leave alone
pub const candidates = 64; // directions sketched from the fact's inputs, which a new block's are chosen among

/// Adam's settings for a block's outputs (its gate keeps them off what must stay): rate, moments' decays, epsilon.
pub const hyper = [4]f32{ 3e-4, 0.9, 0.999, 1e-8 };

/// Adam's settings at step `step`: the rate halves every `settle` steps, three times at most, so later rounds refine.
pub fn hyperAt(step: u64) [4]f32 {
    var h = hyper;
    h[0] /= @as(f32, @floatFromInt(@as(u32, 1) << @intCast(@min(step / settle, 3))));
    return h;
}
pub const settle = 60;

/// Each new input direction's length, as MLX's LoRA draws its down factor's columns (uniform within 1 / sqrt(in)).
pub const a_norm: f32 = 0.577;

/// A block's directions' squared length: a row's cosine with one is (x . a) / sqrt(unit |x|^2).
pub const unit = a_norm * a_norm;
