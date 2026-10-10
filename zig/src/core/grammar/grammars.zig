//! A checkpoint's grammar compilers over two tokenizer views (`text`: no added tokens; `tools`: call markup kept).

const std = @import("std");
const engine = @import("engine.zig");
const xgr = @import("xgr.zig");
const vocab_mod = @import("vocab.zig");
const tags = @import("tags.zig");
const constraint = @import("constraint.zig");

/// The request kinds (their order is the packed form's).
pub const Kind = enum(u8) { json, json_schema, regex, choice, grammar, tools };

/// A request's structured output: its kind and canonical text (json: empty; choice: a JSON array; tools: tags.zig).
pub const Spec = struct { kind: Kind, text: []const u8 = "" };

/// json_object: any JSON object (OpenAI's contract), not an array.
pub const object_schema = "{\"type\": \"object\"}";
/// The most blank characters between JSON tokens: pretty-printing, not endless.
pub const blanks = 32;
/// Compiled grammars each view's compiler keeps, by their text.
pub const cache_bytes: i64 = 128 << 20;

pub const Options = struct {
    /// the logits' columns (config.json's vocab_size)
    vocab_size: u32,
    /// the engine's eos ids: allowed where the grammar may end
    stops: []const u32,
    /// the tools view's kept added tokens (the model's call markup) and its structural tag
    tool_tokens: []const []const u8,
    tag_model: tags.Model,
};

pub const Grammars = struct {
    gpa: std.mem.Allocator,
    vocab_size: u32,
    words: u32,
    think_open: ?u32,
    think_end: ?u32,
    views: [2]View,

    const View = struct {
        tok: xgr.Tokenizer,
        compiler: xgr.Compiler,
        /// tokens no grammar of this view takes (special, not a stop token)
        never: std.DynamicBitSetUnmanaged,

        fn deinit(v: *View, gpa: std.mem.Allocator) void {
            v.compiler.deinit();
            v.tok.deinit();
            v.never.deinit(gpa);
        }
    };

    /// The compilers for a tokenizer.json text.
    pub fn init(gpa: std.mem.Allocator, tokenizer_json: []const u8, o: Options) !*Grammars {
        const g = try gpa.create(Grammars);
        errdefer gpa.destroy(g);
        const meta = try xgr.detectMetadata(gpa, tokenizer_json);
        const stops = try gpa.alloc(i32, o.stops.len);
        defer gpa.free(stops);
        for (stops, o.stops) |*s, t| s.* = @intCast(t);
        g.* = .{ .gpa = gpa, .vocab_size = o.vocab_size, .words = xgr.bitmaskWords(o.vocab_size), .think_open = null, .think_end = null, .views = undefined };
        var made: usize = 0;
        errdefer for (g.views[0..made]) |*v| v.deinit(gpa);
        for (&g.views, [_][]const []const u8{ &.{}, o.tool_tokens }) |*v, keep| {
            var voc = try vocab_mod.build(gpa, tokenizer_json, o.vocab_size, keep);
            defer voc.deinit();
            g.think_open = voc.think_open;
            g.think_end = voc.think_end;
            const tok = try xgr.Tokenizer.init(gpa, voc.tokens, meta, stops);
            errdefer tok.deinit();
            var never = try std.DynamicBitSetUnmanaged.initEmpty(gpa, o.vocab_size);
            errdefer never.deinit(gpa);
            const special = try tok.special(gpa);
            defer gpa.free(special);
            for (special) |t| if (t >= 0 and t < o.vocab_size and std.mem.indexOfScalar(u32, o.stops, @intCast(t)) == null) never.set(@intCast(t));
            v.* = .{ .tok = tok, .compiler = try xgr.Compiler.init(tok, o.vocab_size, cache_bytes), .never = never };
            made += 1;
        }
        return g;
    }

    pub fn deinit(g: *Grammars) void {
        for (&g.views) |*v| v.deinit(g.gpa);
        g.gpa.destroy(g);
    }

    fn view(g: *Grammars, kind: Kind) *View {
        return &g.views[if (kind == .tools) 1 else 0];
    }

    /// Why a spec does not compile: xgrammar's trimmed message or the tools spec's problem.
    pub const Why = struct {
        buf: [1024]u8 = undefined,
        len: usize = 0,

        pub fn text(w: *const Why) []const u8 {
            return w.buf[0..w.len];
        }
        fn set(w: *Why, s: []const u8) void {
            w.len = @min(s.len, w.buf.len);
            @memcpy(w.buf[0..w.len], s[0..w.len]);
        }
    };

    /// The compiled grammar of `spec` (cached by text); error.Invalid with `why` when it cannot be enforced.
    pub fn compile(g: *Grammars, spec: Spec, why: *Why) !engine.Compiled {
        const c = g.view(spec.kind).compiler.compiler();
        var problem: tags.Problem = .{};
        return switch (spec.kind) {
            .json => compiled(c.compile(.json_schema, object_schema, blanks), why),
            .json_schema => compiled(c.compile(.json_schema, spec.text, blanks), why),
            .regex => compiled(c.compile(.regex, spec.text, null), why),
            .grammar => compiled(c.compile(.ebnf, spec.text, null), why),
            .choice => {
                const ebnf = choiceGrammar(g.gpa, spec.text) catch |e| switch (e) {
                    error.OutOfMemory => return error.OutOfMemory,
                    else => {
                        why.set("choice must be a JSON array of strings");
                        return error.Invalid;
                    },
                };
                defer g.gpa.free(ebnf);
                return compiled(c.compile(.ebnf, ebnf, null), why);
            },
            .tools => {
                const tag = tags.build(g.gpa, .deepseek_v4_1, spec.text, &problem) catch |e| switch (e) {
                    error.InvalidTools => {
                        why.set(problem.message);
                        return error.Invalid;
                    },
                    else => |x| return x,
                };
                defer g.gpa.free(tag);
                return compiled(c.compile(.structural_tag, tag, null), why);
            },
        };
    }

    fn compiled(r: engine.Error!engine.Compiled, why: *Why) !engine.Compiled {
        return r catch |e| switch (e) {
            error.OutOfMemory => error.OutOfMemory,
            error.Xgrammar => {
                why.set(message(xgr.lastError()));
                return error.Invalid;
            },
        };
    }

    /// A fresh reply's grammar state for `compiled`, bound to its prompt's think state.
    pub fn constraint_(g: *Grammars, kind: Kind, compiled_: engine.Compiled, prompt: []const u32) !constraint.Constraint {
        const active = constraint.thinkActive(prompt, g.think_open, g.think_end);
        return g.bound(kind, compiled_, active);
    }

    /// A fresh reply's state at a given think state (a follower's, from the leader's packed grammar).
    pub fn bound(g: *Grammars, kind: Kind, compiled_: engine.Compiled, active: bool) !constraint.Constraint {
        return constraint.Constraint.init(compiled_, &g.view(kind).never, g.think_end orelse std.math.maxInt(u32), active, g.words);
    }

    pub fn tokenizer(g: *Grammars, kind: Kind) xgr.Tokenizer {
        return g.view(kind).tok;
    }
};

/// xgrammar's message without its timestamp and source location: the first line, ``^\[[^\]]*\]\s*\S+:\d+:\s*`` removed.
pub fn message(full: []const u8) []const u8 {
    const trimmed = std.mem.trim(u8, full, " \t\r\n");
    var s = trimmed[0 .. std.mem.indexOfScalar(u8, trimmed, '\n') orelse trimmed.len];
    if (s.len == 0) return "XGrammarError";
    if (s[0] != '[') return s;
    const close = std.mem.indexOfScalar(u8, s, ']') orelse return s;
    var i = close + 1;
    while (i < s.len and std.ascii.isWhitespace(s[i])) i += 1;
    const loc = i;
    while (i < s.len and !std.ascii.isWhitespace(s[i])) i += 1;
    // \S+:\d+: — the location's last ':<digits>:' ends it (\S+ is greedy, backtracking to the last such match)
    const word = s[loc..i];
    var end: ?usize = null;
    var k: usize = word.len;
    while (k > 0) : (k -= 1) {
        if (word[k - 1] != ':') continue;
        var d = k - 1;
        while (d > 0 and std.ascii.isDigit(word[d - 1])) d -= 1;
        if (d < k - 1 and d > 1 and word[d - 1] == ':') {
            end = k;
            break;
        }
    }
    if (end == null) {
        // the location may be followed by the message in the same word (no space after the colon)
        return s;
    }
    var j = loc + end.?;
    while (j < s.len and std.ascii.isWhitespace(s[j])) j += 1;
    s = s[j..];
    return s;
}

/// ``root ::= "a" | "b"``: each option a JSON string, non-ASCII written raw.
pub fn choiceGrammar(gpa: std.mem.Allocator, choices_json: []const u8) ![]u8 {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, choices_json, .{});
    defer parsed.deinit();
    if (parsed.value != .array) return error.InvalidChoices;
    var out: std.Io.Writer.Allocating = .init(gpa);
    errdefer out.deinit();
    try out.writer.writeAll("root ::= ");
    for (parsed.value.array.items, 0..) |v, i| {
        if (v != .string) return error.InvalidChoices;
        if (i > 0) try out.writer.writeAll(" | ");
        try pyString(&out.writer, v.string);
    }
    return out.toOwnedSlice();
}

/// A JSON string with only `"`, `\` and control characters escaped (non-ASCII raw).
pub fn pyString(w: *std.Io.Writer, s: []const u8) !void {
    try tags.string(w, s);
}
