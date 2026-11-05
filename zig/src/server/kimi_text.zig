//! Kimi's release tiktoken and XTML behind the server's provenance-aware text interface.

// Copyright (c) 2026 Moonshot AI; release adaptation notice: LICENSES/Kimi-K3.1-Pro.txt.
const std = @import("std");
const tokenizer = @import("tokenizer");
const json = @import("json");
const mt = @import("model_text.zig");
const prompt = @import("kimi_prompt.zig");
const A = std.mem.Allocator;
const TikToken = tokenizer.kimi_tiktoken.TikToken;

pub const KimiText = struct {
    gpa: A,
    arena: std.heap.ArenaAllocator,
    tok: TikToken,
    eos: []const u32 = &.{},
    special_flags: [256]bool = @splat(false),
    vocab_lookup: std.StringHashMapUnmanaged(u32) = .empty,

    pub fn load(gpa: A, io: std.Io, dir: []const u8) !*KimiText {
        const t = try gpa.create(KimiText);
        errdefer gpa.destroy(t);
        t.* = .{ .gpa = gpa, .arena = .init(gpa), .tok = undefined };
        errdefer t.arena.deinit();
        const a = t.arena.allocator();
        const tc = try readJson(a, io, dir, "tokenizer_config.json", false);
        const named_tokens = field(tc, "added_tokens_decoder") orelse return error.BadKimiTokenizer;
        if (named_tokens != .object) return error.BadKimiTokenizer;
        for (named_tokens.object.values()) |entry| {
            if (entry != .object) return error.BadKimiTokenizer;
            const content = entry.object.get("content") orelse return error.BadKimiTokenizer;
            if (content != .string or content.string.len == 0) return error.BadKimiTokenizer;
            if (entry.object.get("special")) |s| if (s != .bool) return error.BadKimiTokenizer;
        }
        t.tok = try TikToken.load(io, gpa, dir);
        errdefer t.tok.deinit();
        var unique: std.StringHashMapUnmanaged(void) = .empty;
        for (t.tok.specials.items) |piece| {
            const entry = try unique.getOrPut(a, piece);
            if (entry.found_existing) return error.BadKimiTokenizer;
        }
        for ([_][]const u8{ "<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>" }) |control|
            if (!unique.contains(control)) return error.BadKimiTokenizer;

        for (t.tok.tokens.items, 0..) |_, id|
            try t.vocab_lookup.put(a, try t.tok.tokenString(a, @intCast(id)), @intCast(id));
        for (named_tokens.object.keys()) |key| {
            const id = std.fmt.parseInt(u32, key, 10) catch return error.BadKimiTokenizer;
            if (id < t.tok.first_special or id - t.tok.first_special >= 256) return error.BadKimiTokenizer;
        }
        for (0..256) |i| {
            const key = try std.fmt.allocPrint(a, "{d}", .{t.tok.first_special + i});
            if (named_tokens.object.get(key)) |entry| if (entry.object.get("special")) |s| {
                t.special_flags[i] = s.bool;
            };
        }
        const model = try readJson(a, io, dir, "config.json", false);
        const text_cfg = field(model, "text_config") orelse model;
        const generation = try readJson(a, io, dir, "generation_config.json", true);
        const eos = field(generation, "eos_token_id") orelse field(model, "eos_token_id") orelse
            field(text_cfg, "eos_token_id") orelse return error.MissingKimiEos;
        var ids: std.ArrayList(u32) = .empty;
        if (eos == .array) {
            for (eos.array.items) |v| try ids.append(a, try t.eosId(v));
        } else try ids.append(a, try t.eosId(eos));
        if (ids.items.len == 0) return error.MissingKimiEos;
        t.eos = ids.items;
        const vocab_value = field(text_cfg, "vocab_size") orelse return error.BadKimiTokenizer;
        if (vocab_value != .integer or vocab_value.integer != t.tok.first_special + 256) return error.KimiVocabMismatch;
        return t;
    }

    fn field(v: std.json.Value, key: []const u8) ?std.json.Value {
        return if (v == .object) v.object.get(key) else null;
    }

    fn eosId(t: *const KimiText, v: std.json.Value) !u32 {
        if (v != .integer or v.integer < 0 or v.integer >= t.tok.first_special + 256) return error.BadKimiEos;
        return @intCast(v.integer);
    }

    fn readJson(a: A, io: std.Io, dir: []const u8, file: []const u8, optional: bool) !std.json.Value {
        const path = try std.fs.path.join(a, &.{ dir, file });
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20)) catch |err| {
            if (optional and err == error.FileNotFound) return .null;
            return err;
        };
        const result = try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
        if (result != .object) return error.BadKimiTokenizer;
        return result;
    }

    pub fn deinit(t: *KimiText) void {
        t.tok.deinit();
        t.arena.deinit();
        t.gpa.destroy(t);
    }

    pub fn identity(t: *const KimiText) [32]u8 {
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        h.update("kimi-tiktoken-xtml-segments-v1");
        for (t.tok.tokens.items) |piece| {
            var size: [8]u8 = undefined;
            std.mem.writeInt(u64, &size, piece.len, .little);
            h.update(&size);
            h.update(piece);
        }
        for (t.tok.specials.items, t.special_flags) |piece, flagged| {
            var size: [8]u8 = undefined;
            std.mem.writeInt(u64, &size, piece.len, .little);
            h.update(&size);
            h.update(piece);
            h.update(&.{ 0, @intFromBool(flagged) });
        }
        h.update(std.mem.sliceAsBytes(t.eos));
        return h.finalResult();
    }

    pub fn text(t: *KimiText) mt.Text {
        return .{ .ctx = t, .vtable = &.{
            .format = .kimi,
            .encode = encode,
            .encode_plain = plain,
            .encode_masked = masked,
            .decode = decode,
            .token_id = tokenId,
            .token_string = tokenString,
            .token_bytes = tokenBytes,
            .vocab_size = vocab,
            .eos_ids = eosIds,
            .render = render,
            .template_source = source,
            .special = special,
            .added = added,
        } };
    }

    fn self(ctx: *anyopaque) *KimiText {
        return @ptrCast(@alignCast(ctx));
    }
    fn encoded(ctx: *anyopaque, a: A, s: []const u8, allow: bool) mt.Error![]u32 {
        if (!std.unicode.utf8ValidateSlice(s)) return error.Template;
        return self(ctx).tok.encodeAlloc(a, s, allow) catch |err|
            if (err == error.OutOfMemory) error.OutOfMemory else error.Template;
    }
    fn encode(ctx: *anyopaque, a: A, s: []const u8, _: bool) mt.Error![]u32 {
        return encoded(ctx, a, s, true);
    }
    fn plain(ctx: *anyopaque, a: A, s: []const u8) mt.Error![]u32 {
        return encoded(ctx, a, s, false);
    }
    fn masked(ctx: *anyopaque, a: A, s: []const u8, spans: []const mt.Span, problem: *[]const u8) mt.Error![]u32 {
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(a);
        var pos: usize = 0;
        for (spans) |span| {
            if (span[0] < pos or span[1] <= span[0] or span[1] > s.len) {
                problem.* = "invalid Kimi request byte provenance";
                return error.Template;
            }
            const control = try encoded(ctx, a, s[pos..span[0]], true);
            defer a.free(control);
            try out.appendSlice(a, control);
            const data = try encoded(ctx, a, s[span[0]..span[1]], false);
            defer a.free(data);
            try out.appendSlice(a, data);
            pos = span[1];
        }
        const rest = try encoded(ctx, a, s[pos..], true);
        defer a.free(rest);
        try out.appendSlice(a, rest);
        return out.toOwnedSlice(a);
    }
    fn decode(ctx: *anyopaque, a: A, ids: []const u32) A.Error![]u8 {
        return self(ctx).tok.decodeAlloc(a, ids) catch |err|
            if (err == error.OutOfMemory) error.OutOfMemory else a.dupe(u8, "\u{fffd}");
    }
    fn tokenId(ctx: *anyopaque, piece: []const u8) ?u32 {
        const t = self(ctx);
        for (t.tok.specials.items, 0..) |s, i| if (std.mem.eql(u8, s, piece)) return t.tok.first_special + @as(u32, @intCast(i));
        return t.vocab_lookup.get(piece);
    }
    fn tokenString(ctx: *anyopaque, a: A, id: u32) A.Error![]u8 {
        return self(ctx).tok.tokenString(a, id);
    }
    fn tokenBytes(ctx: *anyopaque, a: A, id: u32) A.Error!?[]u8 {
        return if (self(ctx).tok.tokenBytes(id)) |b| try a.dupe(u8, b) else null;
    }
    fn vocab(ctx: *anyopaque) u32 {
        return self(ctx).tok.first_special + 256;
    }
    fn eosIds(ctx: *anyopaque) []const u32 {
        return self(ctx).eos;
    }
    fn added(ctx: *anyopaque, id: u32) bool {
        const first = self(ctx).tok.first_special;
        return id >= first and id - first < 256;
    }
    fn special(ctx: *anyopaque, id: u32) bool {
        return added(ctx, id) and self(ctx).special_flags[id - self(ctx).tok.first_special];
    }
    fn source(_: *anyopaque) []const u8 {
        return "enable_thinking reasoning_effort [\"low\",\"medium\",\"high\",\"max\"] native Kimi XTML";
    }
    fn render(_: *anyopaque, a: A, messages: json.Value, options: mt.RenderOptions, problem: *[]const u8) mt.Error!mt.Rendered {
        const segments = prompt.render(a, messages, .{
            .thinking = options.enable_thinking,
            .add_generation_prompt = options.add_generation_prompt,
            .thinking_effort = if (options.enable_thinking) options.reasoning_effort orelse "max" else null,
            .tools = options.tools orelse .null,
            .tool_choice = options.tool_choice orelse .null,
            .response_format = options.response_format orelse .null,
        }) catch |err| {
            if (err == error.OutOfMemory) return error.OutOfMemory;
            problem.* = try std.fmt.allocPrint(a, "Kimi XTML cannot render this request ({s})", .{@errorName(err)});
            return error.Template;
        };
        var bytes: std.ArrayList(u8) = .empty;
        var spans: std.ArrayList(mt.Span) = .empty;
        const cuts = try a.alloc(mt.Segment, segments.len);
        for (segments, cuts) |s, *cut| {
            const start = bytes.items.len;
            try bytes.appendSlice(a, s.text);
            cut.* = .{ .end = bytes.items.len, .allow_special = s.allow_special };
            if (s.request) try spans.append(a, .{ start, bytes.items.len });
        }
        return .{ .text = bytes.items, .spans = spans.items, .segments = cuts };
    }
};
