//! The tokenizer and chat template a server renders prompts and replies with (the tok and jinja branches plug in here).
const std = @import("std");
const json = @import("json.zig");
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
        render: *const fn (ctx: *anyopaque, a: Allocator, messages: json.Value, options: RenderOptions, problem: *[]const u8) Error![]u8,
        /// The template source, read for the reasoning efforts it names.
        template_source: *const fn (ctx: *anyopaque) []const u8,
        /// Whether an id is a special token (an added token flagged special); null: none are known.
        special: ?*const fn (ctx: *anyopaque, id: u32) bool = null,
        /// Ids of ``text`` with no added token matched (text that spells a special token is ordinary text); null: ``encode``.
        encode_plain: ?*const fn (ctx: *anyopaque, a: Allocator, text: []const u8) Error![]u32 = null,
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

    pub fn render(t: Text, a: Allocator, messages: json.Value, options: RenderOptions, problem: *[]const u8) Error![]u8 {
        return t.vtable.render(t.ctx, a, messages, options, problem);
    }

    /// The chat template's ids, as ``apply_chat_template(tokenize=True)`` gives them.
    pub fn renderIds(t: Text, a: Allocator, messages: json.Value, options: RenderOptions, problem: *[]const u8) Error![]u32 {
        const text = try t.render(a, messages, options, problem);
        return t.encode(a, text, false);
    }

    pub fn templateSource(t: Text) []const u8 {
        return t.vtable.template_source(t.ctx);
    }

    pub fn isSpecial(t: Text, id: u32) bool {
        const f = t.vtable.special orelse return false;
        return f(t.ctx, id);
    }
};
