//! The common result of a model family's token-aware full-reply parser.
const json = @import("json");

pub const Result = struct {
    content: []const u8,
    reasoning: []const u8,
    calls: ?[]json.Value,
    reasoning_tokens: usize,
    rejected: bool = false,
    rejected_calls: usize = 0, // calls a parser refused and dropped; only Inkling 2 reports them
    kept: ?@import("tool_parse.zig").Parsed = null, // a bare Inkling 2 reply's envelopes kept as content
    bare: bool = false, // Inkling 2: no message header before the reply's end or its stop, as a grammar writes
    cut_calls: bool = false, // the calls were cut by the token cap: the reply keeps finish_reason length
    cut_call: ?*json.Object = null, // the last call came from open-envelope cap recovery, not a closed envelope
    open_think: bool = false,
};
