//! Readout requests: an answer the backend computes from a prompt's final-norm rows into request-owned BF16 logits.
const std = @import("std");

pub const Kind = enum { joint_schema };

/// Question type ids the joint schema head knows: noul 0, choice 1, score 2.
pub const types: u8 = 3;

/// What an engine's readouts take: the kinds it computes and the bounds its memory plan holds.
pub const Info = struct {
    kinds: []const Kind = &.{.joint_schema},
    max_rows: u32,
    max_questions: u32,
    max_options: u32,
};

/// One readout: token spans [start, end) of the prompt, each option's question, and one logit out per option.
pub const Spec = struct {
    kind: Kind = .joint_schema,
    question_spans: []const [2]u32,
    question_types: []const u8,
    option_spans: []const [2]u32,
    option_question: []const u16,
    logits: []u16, // BF16 bits, the reference's logit dtype, in option order; borrowed until the request finishes

    pub fn validate(s: Spec, info: Info, prompt: []const u32, context: u32) !void {
        if (std.mem.indexOfScalar(Kind, info.kinds, s.kind) == null) return error.UnsupportedReadoutKind;
        if (info.max_rows == 0 or info.max_questions == 0 or info.max_options == 0 or prompt.len == 0 or
            prompt.len > info.max_rows or (context > 0 and prompt.len > context)) return error.InvalidReadoutPrompt;
        const q = s.question_spans.len;
        const o = s.option_spans.len;
        if (q == 0 or q > info.max_questions or s.question_types.len != q) return error.InvalidReadoutQuestions;
        if (o == 0 or o > info.max_options or s.option_question.len != o) return error.InvalidReadoutOptions;
        if (s.logits.len != o) return error.InvalidReadoutLogits;
        for (s.question_spans) |span| if (!inside(span, prompt.len)) return error.InvalidReadoutSpan;
        for (s.option_spans) |span| if (!inside(span, prompt.len)) return error.InvalidReadoutSpan;
        for (s.question_types) |t| if (t >= types) return error.UnknownReadoutType;
        // each question in turn owns one non-empty run of options: the ids go 0, 1, ... q - 1, never back
        var next: usize = 0;
        for (s.option_question, 0..) |k, j| {
            if (j > 0 and k == s.option_question[j - 1]) continue;
            if (k != next) return error.InvalidReadoutGroups;
            next += 1;
        }
        if (next != q) return error.InvalidReadoutGroups;
        const out = std.mem.sliceAsBytes(s.logits);
        const inputs = [_][]const u8{ std.mem.sliceAsBytes(prompt), std.mem.sliceAsBytes(s.question_spans), s.question_types, std.mem.sliceAsBytes(s.option_spans), std.mem.sliceAsBytes(s.option_question) };
        for (inputs) |in| if (overlap(out, in)) return error.ReadoutOutputAliasesInput;
    }
};

/// A readout request's own fields: nothing generated, drafted, sampled, constrained or shown; then its spec.
pub fn admit(engine: anytype, request: anytype) !void {
    const cap = engine.readouts orelse return error.ReadoutsUnsupported;
    const spec = request.readout orelse return error.NotAReadout;
    if (engine.text_generation or request.embedding != null or request.max_tokens != 0 or request.drafts or
        request.sampling != null or request.structure != null or request.logprobs != null or
        request.prompt_logprobs != null or request.images.len != 0 or request.decode_spans.len != 0) return error.ReadoutGenerates;
    try spec.validate(cap, request.prompt, engine.context_window);
}

fn inside(span: [2]u32, len: usize) bool {
    return span[0] < span[1] and span[1] <= len;
}

fn overlap(a: []const u8, b: []const u8) bool {
    if (a.len == 0 or b.len == 0) return false;
    const ap = @intFromPtr(a.ptr);
    const bp = @intFromPtr(b.ptr);
    return if (ap <= bp) bp - ap < a.len else ap - bp < b.len;
}
