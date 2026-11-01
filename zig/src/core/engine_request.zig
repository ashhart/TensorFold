//! The shapes a request carries: its stop strings, its tool gate and its guided output.

const lanes = @import("lanes");

/// The grammar the guided fields compile against, as the lane core's own.
pub const grammar = lanes.grammar;

/// The request's stop strings, checked on the engine's thread after each committed token (the server decodes the tail).
pub const Stop = struct {
    ctx: *anyopaque,
    check: *const fn (ctx: *anyopaque, emitted: []const u32) bool,
};

/// tool_choice "required" or a named function: outside a think block the answer opens a call to an offered tool.
pub const CallGate = struct {
    /// The token that opens a call, the template text before the name, and the mark that ends the name.
    opener: u32,
    lead: []const u8 = "",
    tail: []const u8 = "",
    /// Offered names (empty: any); the gate completes a name its prefix starts.
    names: []const []const u8 = &.{},
    think_open: ?u32 = null,
    think_end: ?u32 = null,
};

/// response_format and the guided_* fields: every token keeps the reply inside this grammar.
pub const Structure = struct {
    kind: enum { json, json_schema, regex, choice, grammar, structural_tag },
    /// The schema (JSON text), regex, choices (a JSON array), EBNF or structural tag; empty for any JSON object.
    text: []const u8 = "",
    /// With thinking on, the grammar starts after this token.
    after: ?u32 = null,
    /// The server's compiler, which has compiled `text` once already (the engine's compile is then a cache hit).
    compiler: ?*grammar.Compiler = null,
    /// That compile, which the engine reads instead of compiling again; the server frees it after `finished`.
    compiled: ?grammar.Compiled = null,
    /// grammar_budget: mask tokens after which the reply could no longer complete within its token limits.
    budget: bool = false,

    pub fn free(st: Structure) void {
        if (st.compiled) |g| g.free();
    }
};
