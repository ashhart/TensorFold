//! No reply or counter claims more cached tokens than its prompt holds (issue #1899).
/// The engine's count clamped to the prompt: the reply, its done line and the counter use one value.
pub fn within(cached: ?u32, prompt_len: usize) usize {
    return @min(@as(usize, cached orelse 0), prompt_len);
}
