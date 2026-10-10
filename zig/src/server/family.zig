//! A model family's own chat rules (prompt encoding, effort budgets, call markup) in place of the chat template.
const std = @import("std");
const json = @import("json.zig");
const errors = @import("errors.zig");
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

/// A request's thinking switch and effort as it asked for them (null: the server's default).
pub const Thinking = struct { enable: ?bool = null, effort: ?[]const u8 = null };

/// What a chat prompt is rendered from: the request's own messages and tools, as the client sent them.
pub const Prompt = struct {
    messages: Value,
    tools: []const Value,
    thinking: bool,
    /// The effort label (``Family.thinking``'s, else the default).
    effort: []const u8,
    generation: bool = true,
    /// The request body, for fields the family reads itself (``response_format``).
    body: Value = .null,
};

pub const Rendered = struct { ids: []const u32, images_omitted: u32 = 0, images: ?Images = null };

/// A prompt's prepared images: ``held`` lives until ``release``; ``token`` for /tokenize.
pub const Images = struct {
    held: *const anyopaque,
    token: u32,
    ctx: *anyopaque,
    release_fn: *const fn (ctx: *anyopaque) void,

    pub fn release(im: Images) void {
        im.release_fn(im.ctx);
    }
};

/// A finished reply's parts, calls as OpenAI tool-call objects.
pub const Parsed = struct { reasoning: []const u8, content: []const u8, calls: []Value };

/// One reply as its tokens arrive: the text so far, and the deltas it adds.
pub const Reader = struct {
    ctx: *anyopaque,
    vt: *const VTable,

    pub const VTable = struct {
        /// Tokens committed (end-of-sentence ids already left out).
        push: *const fn (ctx: *anyopaque, ids: []const u32) Allocator.Error!void,
        /// A character is split across tokens: nothing new is readable yet.
        pending: *const fn (ctx: *anyopaque) bool,
        /// Deltas since the last feed (content a string, reasoning and calls delta objects); ``finished`` flushes.
        feed: *const fn (ctx: *anyopaque, finished: bool, out: *std.ArrayList(Value)) Allocator.Error!void,
        /// The whole reply parsed (after the last ``push``; a split character at the end reads as U+FFFD).
        parse: *const fn (ctx: *anyopaque) Allocator.Error!Parsed,
        /// The calls the stream sent, in order, as OpenAI tool-call objects.
        calls: *const fn (ctx: *anyopaque) Allocator.Error![]Value,
        /// The finished reply's tool markup that yielded no call (a reason, at most 300 bytes) for the log; null: none
        markup: ?*const fn (ctx: *anyopaque) Allocator.Error!?Markup = null,
    };

    pub fn push(r: Reader, ids: []const u32) Allocator.Error!void {
        return r.vt.push(r.ctx, ids);
    }
    pub fn pending(r: Reader) bool {
        return r.vt.pending(r.ctx);
    }
    pub fn feed(r: Reader, finished: bool, out: *std.ArrayList(Value)) Allocator.Error!void {
        return r.vt.feed(r.ctx, finished, out);
    }
    pub fn parse(r: Reader) Allocator.Error!Parsed {
        return r.vt.parse(r.ctx);
    }
    pub fn calls(r: Reader) Allocator.Error![]Value {
        return r.vt.calls(r.ctx);
    }
    pub fn markup(r: Reader) Allocator.Error!?Markup {
        const f = r.vt.markup orelse return null;
        return f(r.ctx);
    }
};

/// A reply's tool markup that the parser turned into no call (``Reader.markup``).
pub const Markup = struct { at: usize, reason: []const u8, window: []const u8 };

/// Where a reply's reasoning goes: ``reasoning_content`` (OpenAI's), ``reasoning`` (vLLM's), or both.
pub const ReasoningFields = enum { both, reasoning_content, reasoning };

pub const Family = struct {
    ctx: *anyopaque,
    vt: *const VTable,

    pub const VTable = struct {
        name: []const u8,
        /// ``chat_template_kwargs`` and ``reasoning_effort`` read; a bad value refuses the request.
        thinking: *const fn (ctx: *anyopaque, cx: *Cx, body: Value) errors.Refused!Thinking,
        /// The effort a request that names none gets.
        default_effort: *const fn (ctx: *anyopaque) []const u8,
        /// The tools the prompt offers: ``tool_choice`` none offers none; otherwise all of them.
        tools: *const fn (ctx: *anyopaque, cx: *Cx, body: Value) errors.Refused![]const Value,
        render: *const fn (ctx: *anyopaque, cx: *Cx, p: Prompt) errors.Refused!Rendered,
        reader: *const fn (ctx: *anyopaque, a: Allocator, thinking: bool, tools: []const Value, single: bool) Allocator.Error!Reader,
        /// A finished reply's reasoning, kept for a later request whose history comes back without it.
        remember: *const fn (ctx: *anyopaque, reasoning: []const u8, content: []const u8, calls: []const Value) void,
        reasoning_fields: ReasoningFields = .both,
        /// Whether the request's ``min_p`` reaches the sampler (some families ignore it).
        reads_min_p: bool = true,
    };

    pub fn thinking(f: Family, cx: *Cx, body: Value) errors.Refused!Thinking {
        return f.vt.thinking(f.ctx, cx, body);
    }
    pub fn defaultEffort(f: Family) []const u8 {
        return f.vt.default_effort(f.ctx);
    }
    pub fn tools(f: Family, cx: *Cx, body: Value) errors.Refused![]const Value {
        return f.vt.tools(f.ctx, cx, body);
    }
    pub fn render(f: Family, cx: *Cx, p: Prompt) errors.Refused!Rendered {
        return f.vt.render(f.ctx, cx, p);
    }
    pub fn reader(f: Family, a: Allocator, think: bool, offered: []const Value, single: bool) Allocator.Error!Reader {
        return f.vt.reader(f.ctx, a, think, offered, single);
    }
    pub fn remember(f: Family, reasoning: []const u8, content: []const u8, calls: []const Value) void {
        f.vt.remember(f.ctx, reasoning, content, calls);
    }

    /// A reasoning delta under the family's field names.
    pub fn reasoningDelta(f: Family, a: Allocator, text: []const u8) Allocator.Error!Value {
        const o = try json.newObject(a);
        if (f.vt.reasoning_fields != .reasoning) try o.put(a, "reasoning_content", .{ .string = text });
        if (f.vt.reasoning_fields != .reasoning_content) try o.put(a, "reasoning", .{ .string = text });
        return .{ .object = o };
    }
};
