//! The tokenizer and chat template a server renders prompts and replies with (the tok and jinja branches plug in here).
const std = @import("std");
const json = @import("json");
const Allocator = std.mem.Allocator;

pub const RenderOptions = struct {
    /// The offered tools (a JSON array), or null.
    tools: ?json.Value = null,
    add_generation_prompt: bool = true,
    /// Also ``thinking_mode`` ("thinking" or "chat") for templates that read that switch.
    enable_thinking: bool = false,
    reasoning_effort: ?[]const u8 = null,
};

pub const Error = error{ Template, OutOfMemory };
pub const Span = [2]usize;

/// The real rendered bytes and the half-open request ranges within them.
pub const Rendered = struct {
    text: []const u8,
    spans: []const Span = &.{},

    pub fn isTemplate(r: Rendered, start: usize, end: usize) bool {
        if (start == end) return true;
        var lo: usize = 0;
        var hi = r.spans.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (r.spans[mid][1] <= start) lo = mid + 1 else hi = mid;
        }
        return lo == r.spans.len or r.spans[lo][0] >= end;
    }

    pub fn lastTemplate(r: Rendered, word: []const u8) ?usize {
        var end = r.text.len;
        while (std.mem.lastIndexOf(u8, r.text[0..end], word)) |at| {
            if (r.isTemplate(at, at + word.len)) return at;
            end = at;
        }
        return null;
    }

    pub fn slice(r: Rendered, a: Allocator, start: usize, end: usize) Allocator.Error!Rendered {
        if (start == end) return .{ .text = r.text[start..end] };
        var spans: std.ArrayList(Span) = .empty;
        for (r.spans) |p| {
            if (p[0] >= end) break;
            if (p[1] > start) try spans.append(a, .{ @max(start, p[0]) - start, @min(end, p[1]) - start });
        }
        return .{ .text = r.text[start..end], .spans = spans.items };
    }

    pub fn encode(r: Rendered, t: Text, a: Allocator, problem: *[]const u8) Error![]u32 {
        return t.encodeMasked(a, r.text, r.spans, problem);
    }

    pub fn appendPlain(r: Rendered, a: Allocator, label: []const u8) Allocator.Error!Rendered {
        if (label.len == 0) return r;
        const spans = try a.alloc(Span, r.spans.len + 1);
        errdefer a.free(spans);
        @memcpy(spans[0..r.spans.len], r.spans);
        spans[r.spans.len] = .{ r.text.len, r.text.len + label.len };
        return .{ .text = try std.mem.concat(a, u8, &.{ r.text, label }), .spans = spans };
    }

    /// Media replacement is limited to template-written occurrences, with every other range retained.
    pub fn replaceTemplate(r: Rendered, a: Allocator, old: []const u8, new: []const u8) Allocator.Error!Rendered {
        if (old.len == 0) return r;
        var text: std.ArrayList(u8) = .empty;
        var spans: std.ArrayList(Span) = .empty;
        var cursor: usize = 0;
        var done: usize = 0;
        while (std.mem.indexOfPos(u8, r.text, done, old)) |hit| {
            try r.appendRange(a, &text, &spans, &cursor, done, hit);
            while (cursor < r.spans.len and r.spans[cursor][1] <= hit) cursor += 1;
            if (cursor < r.spans.len and r.spans[cursor][0] < hit + old.len) {
                try r.appendRange(a, &text, &spans, &cursor, hit, hit + old.len);
            } else try text.appendSlice(a, new);
            done = hit + old.len;
        }
        try r.appendRange(a, &text, &spans, &cursor, done, r.text.len);
        return .{ .text = text.items, .spans = spans.items };
    }

    fn appendRange(r: Rendered, a: Allocator, text: *std.ArrayList(u8), spans: *std.ArrayList(Span), cursor: *usize, start: usize, end: usize) Allocator.Error!void {
        const offset = text.items.len;
        try text.appendSlice(a, r.text[start..end]);
        while (cursor.* < r.spans.len and r.spans[cursor.*][1] <= start) cursor.* += 1;
        for (r.spans[cursor.*..]) |p| {
            if (p[0] >= end) break;
            if (p[1] > start) try spans.append(a, .{ offset + @max(start, p[0]) - start, offset + @min(end, p[1]) - start });
        }
    }
};

pub const Text = struct {
    ctx: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Ids of ``text``, special-token text included as its token; ``add_special`` adds what the tokenizer prepends.
        encode: *const fn (ctx: *anyopaque, a: Allocator, text: []const u8, add_special: bool) Error![]u32,
        /// UTF-8 of ``ids`` with special tokens; bytes that end mid-character decode as U+FFFD, as Python's replace does.
        decode: *const fn (ctx: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8,
        /// The id of a token the vocabulary holds whole (``convert_tokens_to_ids``), else null.
        token_id: *const fn (ctx: *anyopaque, piece: []const u8) ?u32,
        /// A token's vocabulary string (``convert_ids_to_tokens``).
        token_string: *const fn (ctx: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8,
        /// Tokens including added ones.
        vocab_size: *const fn (ctx: *anyopaque) u32,
        eos_ids: *const fn (ctx: *anyopaque) []const u32,
        /// The chat template on ``messages`` (a JSON array) as text; on ``Template`` failure ``problem`` says why.
        render: *const fn (ctx: *anyopaque, a: Allocator, messages: json.Value, options: RenderOptions, problem: *[]const u8) Error!Rendered,
        /// The template source, read for the reasoning efforts it names.
        template_source: *const fn (ctx: *anyopaque) []const u8,
        /// Whether an id is a special token (an added token flagged special); null: none are known.
        special: ?*const fn (ctx: *anyopaque, id: u32) bool = null,
        /// Ids of ``text`` with no added token matched (a spelled special token is plain text); null: ``encode``.
        encode_plain: ?*const fn (ctx: *anyopaque, a: Allocator, text: []const u8) Error![]u32 = null,
        /// Required when rendering request ranges; absent providers fail closed.
        encode_masked: ?*const fn (ctx: *anyopaque, a: Allocator, text: []const u8, spans: []const Span, problem: *[]const u8) Error![]u32 = null,
        /// A token's own bytes (a partial UTF-8 sequence kept whole) for logprob rows; null: the tokenizer can't say.
        token_bytes: ?*const fn (ctx: *anyopaque, a: Allocator, id: u32) Allocator.Error!?[]u8 = null,
    };

    pub fn encode(t: Text, a: Allocator, text: []const u8, add_special: bool) Error![]u32 {
        return t.vtable.encode(t.ctx, a, text, add_special);
    }

    pub fn encodePlain(t: Text, a: Allocator, text: []const u8) Error![]u32 {
        const f = t.vtable.encode_plain orelse return t.encode(a, text, false);
        return f(t.ctx, a, text);
    }

    pub fn decode(t: Text, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        return t.vtable.decode(t.ctx, a, ids);
    }

    pub fn tokenId(t: Text, piece: []const u8) ?u32 {
        return t.vtable.token_id(t.ctx, piece);
    }

    pub fn tokenString(t: Text, a: Allocator, id: u32) Allocator.Error![]u8 {
        return t.vtable.token_string(t.ctx, a, id);
    }

    pub fn vocabSize(t: Text) u32 {
        return t.vtable.vocab_size(t.ctx);
    }

    /// A token's own bytes, or null when this text cannot give them.
    pub fn tokenBytes(t: Text, a: Allocator, id: u32) Allocator.Error!?[]u8 {
        const f = t.vtable.token_bytes orelse return null;
        return f(t.ctx, a, id);
    }

    pub fn eosIds(t: Text) []const u32 {
        return t.vtable.eos_ids(t.ctx);
    }

    pub fn render(t: Text, a: Allocator, messages: json.Value, options: RenderOptions, problem: *[]const u8) Error!Rendered {
        return t.vtable.render(t.ctx, a, messages, options, problem);
    }

    /// The chat template's ids, as ``apply_chat_template(tokenize=True)`` gives them.
    pub fn renderIds(t: Text, a: Allocator, messages: json.Value, options: RenderOptions, problem: *[]const u8) Error![]u32 {
        const rendered = try t.render(a, messages, options, problem);
        return rendered.encode(t, a, problem);
    }

    pub fn encodeMasked(t: Text, a: Allocator, text: []const u8, spans: []const Span, problem: *[]const u8) Error![]u32 {
        if (spans.len == 0) return t.encode(a, text, false);
        const encode_masked = t.vtable.encode_masked orelse {
            problem.* = "the tokenizer does not support request byte provenance";
            return error.Template;
        };
        return encode_masked(t.ctx, a, text, spans, problem);
    }

    pub fn templateSource(t: Text) []const u8 {
        return t.vtable.template_source(t.ctx);
    }

    pub fn isSpecial(t: Text, id: u32) bool {
        const f = t.vtable.special orelse return false;
        return f(t.ctx, id);
    }
};
