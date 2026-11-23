//! The DeepSeek draft lanes' switch settings: TF_DSV41_TAIL_JOIN and the pieces' run flags.

/// TF_DSV41_TAIL_JOIN=1: a prompt in pieces leaves its last token to its first round's shared window (default off).
pub fn joinTail() !bool {
    return envOn("TF_DSV41_TAIL_JOIN");
}

/// TF_DSV41_PIECE_RUNS=1: a round's prompt pieces of several slots in one forward (default off).
pub fn envOn(name: [:0]const u8) !bool {
    return @import("sys").environment.read(bool, name, false);
}
