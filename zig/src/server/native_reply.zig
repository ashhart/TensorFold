//! The common result of a model family's token-aware full-reply parser.
const json = @import("json");

pub const Result = struct {
    content: []const u8,
    reasoning: []const u8,
    calls: ?[]json.Value,
    reasoning_tokens: usize,
    rejected: bool = false,
    rejected_calls: usize = 0, // calls a parser refused and dropped; only Inkling 2 reports them
    open_think: bool = false,
};
