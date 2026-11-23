//! The common result of a model family's token-aware full-reply parser.
const json = @import("json");

pub const Result = struct {
    content: []const u8,
    reasoning: []const u8,
    calls: ?[]json.Value,
    reasoning_tokens: usize,
    rejected: bool = false,
    open_think: bool = false,
};
