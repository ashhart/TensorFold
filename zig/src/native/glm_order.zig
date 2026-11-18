//! GLM's default prompt order on the lane host: shortest-remaining-first on one Mac, as Qwen3.8-27B does.

/// Whether the host starts with shortest-remaining-first: one Mac, no decode share, no prompt share below 1.
pub fn srptByDefault(pair: bool, decode_share: ?f64, prompt_share: f64) bool {
    return !pair and decode_share == null and prompt_share >= 1;
}
