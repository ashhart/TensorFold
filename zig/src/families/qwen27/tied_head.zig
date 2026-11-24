//! A tied head: the output projection reads the embedding's tensors and format, never a stored `lm_head`.
const std = @import("std");

pub const flag = "TF_QWEN27_TIED_HEAD";
pub const embedding = "model.embed_tokens";
pub const stored_head = "lm_head";

/// The module the head loads from; a tied config with its own `lm_head` is ambiguous and refused.
pub fn module(tied: bool, stored: bool) error{TiedHeadInCheckpoint}![]const u8 {
    if (!tied) return stored_head;
    if (stored) return error.TiedHeadInCheckpoint;
    return embedding;
}

/// Bytes of the head's load-time repack, which a tied checkpoint's files do not count.
pub fn bytes(vocab: usize, hidden: usize, bits: usize, group: usize) usize {
    return vocab * (hidden * bits / 8 + hidden / group * 4);
}
