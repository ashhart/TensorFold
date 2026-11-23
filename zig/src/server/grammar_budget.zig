//! Request-time budget refusals retain resource, deadline and slot causes instead of calling them uncovered rules.
const api = @import("engine_api");
const errors = @import("errors.zig");
pub fn require(cx: *errors.Cx, g: api.grammar.Compiled) errors.Refused!void {
    if (g.f.budget_ready == null and g.f.budget_status == null)
        return cx.refuse("grammar_budget needs a current libtfgrammar: rebuild with zig build grammar or use a release archive's lib/");
    const ready = g.budgetReadiness() catch
        return cx.refuse("grammar_budget exploration failed; retry or send the request without grammar_budget");
    if (message(ready)) |why| return cx.refuse(why);
}
pub fn message(ready: api.grammar.Readiness) ?[]const u8 {
    return switch (ready) {
        .ready => null,
        .over_work => "grammar_budget exploration exceeded its work limit; send the request without grammar_budget",
        .over_edges => "grammar_budget exploration exceeded its successor-edge limit; send the request without grammar_budget",
        .over_bytes => "grammar_budget exploration exceeded its table-memory limit; send the request without grammar_budget",
        .timeout => "grammar_budget exploration exceeded its time limit; retry or send the request without grammar_budget",
        .slots => "grammar_budget timed out waiting for a compile slot; retry or send the request without grammar_budget",
        .uncovered => "grammar_budget cannot be kept for this grammar: it uses token-level, budgeted, captured or left-recursive rules",
    };
}
