//! Clean-room native Zig ByteLevel BPE tokenizer for HuggingFace tokenizer.json (Qwen3.5 / Qwen3.8).
//!
//! Supports direct token-ID decoding (O(1) per token), exact special-token splitting,
//! Qwen pre-tokenization + priority-rank BPE merging, and Qwen3.5/3.8 chat formatting.
const std = @import("std");
const st = @import("safetensors.zig");

pub const SpecialToken = struct {
    content: []const u8,
    id: u32,
};

pub const ChatStyle = enum { qwen, llama3, mistral, gemma, gemma4, deepseek, glm, lfm2 };

pub const Tokenizer = struct {
    a: std.mem.Allocator,
    vocab_size: usize,
    max_digit_run: u8 = 1,
    is_metaspace: bool = false,
    add_prefix_space: bool = true,
    chat_style: ChatStyle = .qwen,
    has_think_token: bool = false,
    bos_id: ?u32 = null,
    // id -> raw UTF-8 bytes (decoded from ByteLevel or Metaspace)
    id_to_bytes: [][]u8,
    // ByteLevel unicode piece string -> id
    piece_to_id: std.StringHashMapUnmanaged(u32) = .empty,
    // (left_id << 32) | right_id -> merge rank (lower = higher priority)
    merges: std.AutoHashMapUnmanaged(u64, u32) = .empty,
    specials: std.ArrayList(SpecialToken) = .empty,
    // Byte (0..255) -> UTF-8 bytes of its ByteLevel unicode char (1 or 2 bytes)
    byte_to_u8: [256][2]u8 = undefined,
    byte_to_len: [256]u8 = undefined,
    // Single-byte token IDs (byte 0..255 -> vocab id)
    byte_token_id: [256]u32 = @splat(0),
    // Reverse map: unicode codepoint (0..512) -> original byte
    cp_to_byte: [512]u8 = @splat(0),
    cp_valid: [512]bool = @splat(false),

    pub fn load(a: std.mem.Allocator, model_dir: []const u8) !Tokenizer {
        const path = try std.fs.path.join(a, &.{ model_dir, "tokenizer.json" });
        defer a.free(path);
        const raw = try st.mapFile(a, path);
        defer st.unmapFile(raw);

        const parsed = try std.json.parseFromSlice(std.json.Value, a, raw, .{});
        defer parsed.deinit();

        const root = parsed.value.object;
        const model = root.get("model").?.object;
        const vocab_obj = model.get("vocab").?.object;

        var max_id: usize = vocab_obj.count();
        if (root.get("added_tokens")) |at| {
            for (at.array.items) |item| {
                const id: usize = @intCast(item.object.get("id").?.integer);
                if (id + 1 > max_id) max_id = id + 1;
            }
        }

        var tok = Tokenizer{
            .a = a,
            .vocab_size = max_id,
            .max_digit_run = if (std.mem.indexOf(u8, raw, "\\p{N}{1,3}") != null) 3 else 1,
            .is_metaspace = (std.mem.indexOf(u8, raw, "\"Metaspace\"") != null or std.mem.indexOf(u8, raw, "\xE2\x96\x81") != null and std.mem.indexOf(u8, raw, "\"ByteLevel\"") == null),
            .id_to_bytes = try a.alloc([]u8, max_id),
        };
        for (tok.id_to_bytes) |*slot| slot.* = &.{};
        errdefer tok.deinit();

        tok.initByteTables();

        try tok.piece_to_id.ensureTotalCapacity(a, @intCast(max_id));

        // 1. Load vocab
        var vit = vocab_obj.iterator();
        while (vit.next()) |entry| {
            const piece = entry.key_ptr.*;
            const id: u32 = @intCast(entry.value_ptr.integer);
            const piece_copy = try a.dupe(u8, piece);
            try tok.piece_to_id.put(a, piece_copy, id);
            if (id < max_id) {
                tok.id_to_bytes[id] = if (tok.is_metaspace)
                    try tok.decodeMetaspace(piece)
                else
                    try tok.decodeByteLevel(piece);
            }
        }

        // Map single bytes to their vocab IDs (ByteLevel or SentencePiece <0xXX> byte fallback)
        if (tok.is_metaspace) {
            for (0..256) |b| {
                var hx: [6]u8 = undefined;
                const tag = std.fmt.bufPrint(&hx, "<0x{X:0>2}>", .{b}) catch unreachable;
                if (tok.piece_to_id.get(tag)) |id| {
                    tok.byte_token_id[b] = id;
                } else {
                    const one = [1]u8{@intCast(b)};
                    if (tok.piece_to_id.get(&one)) |id| tok.byte_token_id[b] = id;
                }
            }
        } else {
            for (0..256) |b| {
                const s = tok.byte_to_u8[b][0..tok.byte_to_len[b]];
                if (tok.piece_to_id.get(s)) |id| {
                    tok.byte_token_id[b] = id;
                }
            }
        }

        // 2. Load added/special tokens & detect chat style
        if (root.get("added_tokens")) |at| {
            for (at.array.items) |item| {
                const obj = item.object;
                const id: u32 = @intCast(obj.get("id").?.integer);
                const content = try a.dupe(u8, obj.get("content").?.string);
                try tok.specials.append(a, .{ .content = content, .id = id });
                if (std.mem.eql(u8, content, "<|start_header_id|>")) tok.chat_style = .llama3;
                if (std.mem.eql(u8, content, "<start_of_turn>")) tok.chat_style = .gemma;
                if (std.mem.eql(u8, content, "<|turn>")) tok.chat_style = .gemma4;
                if (std.mem.eql(u8, content, "[INST]")) tok.chat_style = .mistral;
                if (std.mem.eql(u8, content, "<\xEF\xBD\x9CUser\xEF\xBD\x9C>")) tok.chat_style = .deepseek;
                if (std.mem.eql(u8, content, "<|user|>")) tok.chat_style = .glm;
                if (std.mem.eql(u8, content, "<|startoftext|>") and tok.chat_style == .qwen) tok.chat_style = .lfm2;
                if (std.mem.eql(u8, content, "<think>")) tok.has_think_token = true;
                if (std.mem.eql(u8, content, "<|begin_of_text|>") or std.mem.eql(u8, content, "<s>") or std.mem.eql(u8, content, "<bos>") or std.mem.eql(u8, content, "<|startoftext|>") or std.mem.eql(u8, content, "<\xEF\xBD\x9Cbegin\xE2\x96\x81of\xE2\x96\x81sentence\xEF\xBD\x9C>")) {
                    tok.bos_id = id;
                }
                if (id < max_id) {
                    if (tok.id_to_bytes[id].len > 0) a.free(tok.id_to_bytes[id]);
                    tok.id_to_bytes[id] = try a.dupe(u8, content);
                }
            }
            // Sort longest special token first for greedy matching
            std.mem.sort(SpecialToken, tok.specials.items, {}, struct {
                fn lessThan(_: void, lhs: SpecialToken, rhs: SpecialToken) bool {
                    return lhs.content.len > rhs.content.len;
                }
            }.lessThan);
        }

        // 3. Load merges
        const merges_arr = model.get("merges").?.array.items;
        try tok.merges.ensureTotalCapacity(a, @intCast(merges_arr.len));
        for (merges_arr, 0..) |item, rank| {
            var left_s: []const u8 = undefined;
            var right_s: []const u8 = undefined;
            switch (item) {
                .array => |pair| {
                    left_s = pair.items[0].string;
                    right_s = pair.items[1].string;
                },
                .string => |line| {
                    const sp = std.mem.indexOfScalar(u8, line, ' ') orelse continue;
                    left_s = line[0..sp];
                    right_s = line[sp + 1 ..];
                },
                else => continue,
            }
            const lid = tok.piece_to_id.get(left_s) orelse continue;
            const rid = tok.piece_to_id.get(right_s) orelse continue;
            const key: u64 = (@as(u64, lid) << 32) | @as(u64, rid);
            try tok.merges.put(a, key, @intCast(rank));
        }

        return tok;
    }

    pub fn deinit(self: *Tokenizer) void {
        var it = self.piece_to_id.keyIterator();
        while (it.next()) |k| self.a.free(k.*);
        self.piece_to_id.deinit(self.a);
        self.merges.deinit(self.a);
        for (self.specials.items) |s| self.a.free(s.content);
        self.specials.deinit(self.a);
        for (self.id_to_bytes) |b| if (b.len > 0) self.a.free(b);
        self.a.free(self.id_to_bytes);
    }

    fn initByteTables(self: *Tokenizer) void {
        var direct: [256]bool = @splat(false);
        for (33..127) |b| direct[b] = true;
        for (161..173) |b| direct[b] = true;
        for (174..256) |b| direct[b] = true;

        var extra: u21 = 256;
        for (0..256) |b| {
            const cp: u21 = if (direct[b]) @intCast(b) else blk: {
                const c = extra;
                extra += 1;
                break :blk c;
            };
            self.cp_to_byte[cp] = @intCast(b);
            self.cp_valid[cp] = true;
            var tmp: [4]u8 = undefined;
            const n = std.unicode.utf8Encode(cp, &tmp) catch unreachable;
            self.byte_to_len[b] = @intCast(n);
            self.byte_to_u8[b][0] = tmp[0];
            self.byte_to_u8[b][1] = if (n > 1) tmp[1] else 0;
        }
    }

    fn decodeByteLevel(self: *const Tokenizer, piece: []const u8) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(self.a);
        var i: usize = 0;
        while (i < piece.len) {
            const seq_len = std.unicode.utf8ByteSequenceLength(piece[i]) catch {
                try out.append(self.a, piece[i]);
                i += 1;
                continue;
            };
            if (i + seq_len > piece.len) break;
            const cp = std.unicode.utf8Decode(piece[i .. i + seq_len]) catch {
                try out.append(self.a, piece[i]);
                i += 1;
                continue;
            };
            if (cp < 512 and self.cp_valid[cp]) {
                try out.append(self.a, self.cp_to_byte[cp]);
            } else {
                try out.appendSlice(self.a, piece[i .. i + seq_len]);
            }
            i += seq_len;
        }
        return out.toOwnedSlice(self.a);
    }

    fn decodeMetaspace(self: *const Tokenizer, piece: []const u8) ![]u8 {
        if (piece.len == 6 and std.mem.startsWith(u8, piece, "<0x") and piece[5] == '>') {
            if (std.fmt.parseInt(u8, piece[3..5], 16)) |bv| {
                const out = try self.a.alloc(u8, 1);
                out[0] = bv;
                return out;
            } else |_| {}
        }
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(self.a);
        var i: usize = 0;
        while (i < piece.len) {
            if (i + 3 <= piece.len and piece[i] == 0xE2 and piece[i + 1] == 0x96 and piece[i + 2] == 0x81) {
                try out.append(self.a, ' ');
                i += 3;
            } else {
                try out.append(self.a, piece[i]);
                i += 1;
            }
        }
        return out.toOwnedSlice(self.a);
    }

    /// O(1) decode of one token ID to raw UTF-8 bytes.
    pub fn decodeToken(self: *const Tokenizer, id: u32) []const u8 {
        if (id >= self.id_to_bytes.len) return "";
        return self.id_to_bytes[id];
    }

    /// Return the turn-boundary token string for prefix-cache checkpointing.
    pub fn turnEndToken(self: *const Tokenizer) []const u8 {
        return switch (self.chat_style) {
            .llama3 => "<|eot_id|>",
            .gemma => "<end_of_turn>",
            .gemma4 => "<turn|>",
            .mistral => "[/INST]",
            .deepseek => "<\xEF\xBD\x9Cend\xE2\x96\x81of\xE2\x96\x81sentence\xEF\xBD\x9C>",
            .glm => "<|user|>",
            .qwen, .lfm2 => "<|im_end|>",
        };
    }

    /// Auto-detected model-native single-user chat template across all 8 architecture families.
    pub fn formatChat(self: *const Tokenizer, a: std.mem.Allocator, user_prompt: []const u8, enable_thinking: bool) ![]u8 {
        const one = [1]ChatMessage{.{ .role = "user", .content = user_prompt }};
        return self.formatChatStyleMessages(a, &one, true, enable_thinking);
    }

    /// Render a single-user Qwen3.5/3.8 chat prompt (matching mlx_lm's chat_template.jinja).
    pub fn formatChatPrompt(a: std.mem.Allocator, user_prompt: []const u8, enable_thinking: bool) ![]u8 {
        if (enable_thinking) {
            return std.fmt.allocPrint(a, "<|im_start|>user\n{s}<|im_end|>\n<|im_start|>assistant\n<think>\n", .{user_prompt});
        }
        return std.fmt.allocPrint(a, "<|im_start|>user\n{s}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", .{user_prompt});
    }

    pub const ChatMessage = struct {
        role: []const u8,
        content: []const u8,
    };

    /// Render a multi-turn chat message list using this model's auto-detected native template
    /// (Qwen2.5/3/3.5/3.8, Llama 3, Mistral/Mixtral, Gemma 2/3, Gemma 4, DeepSeek, GLM-4/5, LFM2).
    pub fn formatChatStyleMessages(
        self: *const Tokenizer,
        a: std.mem.Allocator,
        msgs: []const ChatMessage,
        add_gen_prompt: bool,
        enable_thinking: bool,
    ) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(a);

        switch (self.chat_style) {
            .llama3 => {
                try out.appendSlice(a, "<|begin_of_text|>");
                for (msgs) |msg| {
                    try out.appendSlice(a, "<|start_header_id|>");
                    try out.appendSlice(a, msg.role);
                    try out.appendSlice(a, "<|end_header_id|>\n\n");
                    try out.appendSlice(a, msg.content);
                    try out.appendSlice(a, "<|eot_id|>");
                }
                if (add_gen_prompt) {
                    try out.appendSlice(a, "<|start_header_id|>assistant<|end_header_id|>\n\n");
                }
            },
            .mistral => {
                try out.appendSlice(a, "<s>");
                var sys_prefix: []const u8 = "";
                var sys_used = false;
                for (msgs) |msg| {
                    if (std.mem.eql(u8, msg.role, "system")) {
                        sys_prefix = msg.content;
                    } else if (std.mem.eql(u8, msg.role, "user")) {
                        try out.appendSlice(a, "[INST] ");
                        if (sys_prefix.len > 0 and !sys_used) {
                            try out.appendSlice(a, sys_prefix);
                            try out.appendSlice(a, "\n\n");
                            sys_used = true;
                        }
                        try out.appendSlice(a, msg.content);
                        try out.appendSlice(a, " [/INST]");
                    } else {
                        try out.appendSlice(a, " ");
                        try out.appendSlice(a, msg.content);
                        try out.appendSlice(a, "</s>");
                    }
                }
            },
            .gemma => {
                try out.appendSlice(a, "<bos>");
                var sys_prefix: []const u8 = "";
                var sys_used = false;
                for (msgs) |msg| {
                    if (std.mem.eql(u8, msg.role, "system")) {
                        sys_prefix = msg.content;
                        continue;
                    }
                    const role_s = if (std.mem.eql(u8, msg.role, "assistant")) "model" else msg.role;
                    try out.appendSlice(a, "<start_of_turn>");
                    try out.appendSlice(a, role_s);
                    try out.append(a, '\n');
                    if (sys_prefix.len > 0 and !sys_used and std.mem.eql(u8, role_s, "user")) {
                        try out.appendSlice(a, sys_prefix);
                        try out.appendSlice(a, "\n\n");
                        sys_used = true;
                    }
                    try out.appendSlice(a, msg.content);
                    try out.appendSlice(a, "<end_of_turn>\n");
                }
                if (add_gen_prompt) try out.appendSlice(a, "<start_of_turn>model\n");
            },
            .gemma4 => {
                try out.appendSlice(a, "<bos>");
                for (msgs) |msg| {
                    const role_s = if (std.mem.eql(u8, msg.role, "assistant")) "model" else msg.role;
                    try out.appendSlice(a, "<|turn>");
                    try out.appendSlice(a, role_s);
                    try out.append(a, '\n');
                    try out.appendSlice(a, msg.content);
                    try out.appendSlice(a, "<turn|>\n");
                }
                if (add_gen_prompt) try out.appendSlice(a, "<|turn>model\n");
            },
            .deepseek => {
                try out.appendSlice(a, "<\xEF\xBD\x9Cbegin\xE2\x96\x81of\xE2\x96\x81sentence\xEF\xBD\x9C>");
                for (msgs) |msg| {
                    if (std.mem.eql(u8, msg.role, "system")) {
                        try out.appendSlice(a, msg.content);
                        try out.appendSlice(a, "\n\n");
                    } else if (std.mem.eql(u8, msg.role, "user")) {
                        try out.appendSlice(a, "<\xEF\xBD\x9CUser\xEF\xBD\x9C>");
                        try out.appendSlice(a, msg.content);
                    } else {
                        try out.appendSlice(a, "<\xEF\xBD\x9CAssistant\xEF\xBD\x9C>");
                        try out.appendSlice(a, msg.content);
                        try out.appendSlice(a, "<\xEF\xBD\x9Cend\xE2\x96\x81of\xE2\x96\x81sentence\xEF\xBD\x9C>");
                    }
                }
                if (add_gen_prompt) {
                    try out.appendSlice(a, "<\xEF\xBD\x9CAssistant\xEF\xBD\x9C>");
                    if (self.has_think_token) {
                        if (enable_thinking) {
                            try out.appendSlice(a, "<think>\n");
                        } else {
                            try out.appendSlice(a, "<think>\n\n</think>\n\n");
                        }
                    }
                }
            },
            .glm => {
                try out.appendSlice(a, "[gMASK]<sop>");
                for (msgs) |msg| {
                    if (std.mem.eql(u8, msg.role, "system")) {
                        try out.appendSlice(a, "<|system|>\n");
                    } else if (std.mem.eql(u8, msg.role, "assistant")) {
                        try out.appendSlice(a, "<|assistant|>\n");
                    } else {
                        try out.appendSlice(a, "<|user|>\n");
                    }
                    try out.appendSlice(a, msg.content);
                }
                if (add_gen_prompt) try out.appendSlice(a, "<|assistant|>\n");
            },
            .lfm2 => {
                try out.appendSlice(a, "<|startoftext|>");
                for (msgs) |msg| {
                    try out.appendSlice(a, "<|im_start|>");
                    try out.appendSlice(a, msg.role);
                    try out.append(a, '\n');
                    try out.appendSlice(a, msg.content);
                    try out.appendSlice(a, "<|im_end|>\n");
                }
                if (add_gen_prompt) try out.appendSlice(a, "<|im_start|>assistant\n");
            },
            .qwen => {
                for (msgs) |msg| {
                    try out.appendSlice(a, "<|im_start|>");
                    try out.appendSlice(a, msg.role);
                    try out.append(a, '\n');
                    try out.appendSlice(a, msg.content);
                    try out.appendSlice(a, "<|im_end|>\n");
                }
                if (add_gen_prompt) {
                    if (self.has_think_token) {
                        if (enable_thinking) {
                            try out.appendSlice(a, "<|im_start|>assistant\n<think>\n");
                        } else {
                            try out.appendSlice(a, "<|im_start|>assistant\n<think>\n\n</think>\n\n");
                        }
                    } else {
                        try out.appendSlice(a, "<|im_start|>assistant\n");
                    }
                }
            },
        }
        return out.toOwnedSlice(a);
    }

    /// Render a multi-turn Qwen3.5/3.8 chat message list (optionally appending the assistant generation prompt).
    pub fn formatChatMessages(a: std.mem.Allocator, msgs: []const ChatMessage, add_gen_prompt: bool, enable_thinking: bool) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(a);
        for (msgs) |msg| {
            try out.appendSlice(a, "<|im_start|>");
            try out.appendSlice(a, msg.role);
            try out.append(a, '\n');
            try out.appendSlice(a, msg.content);
            try out.appendSlice(a, "<|im_end|>\n");
        }
        if (add_gen_prompt) {
            if (enable_thinking) {
                try out.appendSlice(a, "<|im_start|>assistant\n<think>\n");
            } else {
                try out.appendSlice(a, "<|im_start|>assistant\n<think>\n\n</think>\n\n");
            }
        }
        return out.toOwnedSlice(a);
    }

    /// Encode UTF-8 text (including special tokens) into token IDs.
    pub fn encode(self: *const Tokenizer, a: std.mem.Allocator, text: []const u8) ![]u32 {
        var ids: std.ArrayList(u32) = .empty;
        errdefer ids.deinit(a);

        var pos: usize = 0;
        var plain_start: usize = 0;
        while (pos < text.len) {
            var matched: ?SpecialToken = null;
            if (text[pos] == '<' or text[pos] == '[') {
                for (self.specials.items) |sp| {
                    if (std.mem.startsWith(u8, text[pos..], sp.content)) {
                        matched = sp;
                        break;
                    }
                }
            }
            if (matched) |sp| {
                if (pos > plain_start) {
                    try self.encodePlain(a, text[plain_start..pos], &ids);
                }
                try ids.append(a, sp.id);
                pos += sp.content.len;
                plain_start = pos;
            } else {
                pos += 1;
            }
        }
        if (plain_start < text.len) {
            try self.encodePlain(a, text[plain_start..], &ids);
        }
        return ids.toOwnedSlice(a);
    }

    fn encodePlain(self: *const Tokenizer, a: std.mem.Allocator, text: []const u8, out: *std.ArrayList(u32)) !void {
        if (self.is_metaspace) {
            try self.encodeMetaspaceBpe(a, text, out);
            return;
        }
        var i: usize = 0;
        while (i < text.len) {
            const end = nextPreToken(text, i, self.max_digit_run);
            try self.encodeChunkBpe(a, text[i..end], out);
            i = end;
        }
    }

    fn encodeMetaspaceBpe(self: *const Tokenizer, a: std.mem.Allocator, text: []const u8, out: *std.ArrayList(u32)) !void {
        if (text.len == 0) return;
        const meta = "\xE2\x96\x81";
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(a);
        if (self.add_prefix_space and text[0] != ' ') try buf.appendSlice(a, meta);
        for (text) |c| {
            if (c == ' ') {
                try buf.appendSlice(a, meta);
            } else {
                try buf.append(a, c);
            }
        }
        const Sym = struct { id: u32, start: usize, len: usize };
        var syms: std.ArrayList(Sym) = .empty;
        defer syms.deinit(a);
        var i: usize = 0;
        while (i < buf.items.len) {
            const clen = std.unicode.utf8ByteSequenceLength(buf.items[i]) catch 1;
            const slen = @min(clen, buf.items.len - i);
            const slice = buf.items[i .. i + slen];
            if (self.piece_to_id.get(slice)) |id| {
                try syms.append(a, .{ .id = id, .start = i, .len = slen });
            } else {
                for (0..slen) |bi| {
                    try syms.append(a, .{ .id = self.byte_token_id[buf.items[i + bi]], .start = i + bi, .len = 1 });
                }
            }
            i += slen;
        }
        while (syms.items.len >= 2) {
            var best_rank: u32 = std.math.maxInt(u32);
            var best_idx: usize = std.math.maxInt(usize);
            for (0..syms.items.len - 1) |k| {
                const key = (@as(u64, syms.items[k].id) << 32) | @as(u64, syms.items[k + 1].id);
                if (self.merges.get(key)) |rank| {
                    if (rank < best_rank) {
                        best_rank = rank;
                        best_idx = k;
                    }
                }
            }
            if (best_idx == std.math.maxInt(usize)) break;
            const merged_start = syms.items[best_idx].start;
            const merged_len = syms.items[best_idx].len + syms.items[best_idx + 1].len;
            const merged_id = self.piece_to_id.get(buf.items[merged_start .. merged_start + merged_len]) orelse break;
            syms.items[best_idx] = .{ .id = merged_id, .start = merged_start, .len = merged_len };
            _ = syms.orderedRemove(best_idx + 1);
        }
        for (syms.items) |s| try out.append(a, s.id);
    }

    fn encodeChunkBpe(self: *const Tokenizer, a: std.mem.Allocator, chunk: []const u8, out: *std.ArrayList(u32)) !void {
        if (chunk.len == 0) return;

        // Fast path: check if entire ByteLevel chunk is already in vocab
        var bl_buf: std.ArrayList(u8) = .empty;
        defer bl_buf.deinit(a);
        try bl_buf.ensureTotalCapacity(a, chunk.len * 2);
        for (chunk) |b| {
            try bl_buf.appendSlice(a, self.byte_to_u8[b][0..self.byte_to_len[b]]);
        }
        if (self.piece_to_id.get(bl_buf.items)) |whole_id| {
            try out.append(a, whole_id);
            return;
        }

        // Start from individual byte tokens and iteratively merge lowest-rank pair
        const Sym = struct { id: u32, start: usize, len: usize };
        var syms: std.ArrayList(Sym) = .empty;
        defer syms.deinit(a);
        try syms.ensureTotalCapacity(a, chunk.len);

        var off: usize = 0;
        for (chunk) |b| {
            const l = self.byte_to_len[b];
            try syms.append(a, .{ .id = self.byte_token_id[b], .start = off, .len = l });
            off += l;
        }

        while (syms.items.len >= 2) {
            var best_rank: u32 = std.math.maxInt(u32);
            var best_idx: usize = std.math.maxInt(usize);

            for (0..syms.items.len - 1) |k| {
                const key = (@as(u64, syms.items[k].id) << 32) | @as(u64, syms.items[k + 1].id);
                if (self.merges.get(key)) |rank| {
                    if (rank < best_rank) {
                        best_rank = rank;
                        best_idx = k;
                    }
                }
            }
            if (best_idx == std.math.maxInt(usize)) break;

            const merged_start = syms.items[best_idx].start;
            const merged_len = syms.items[best_idx].len + syms.items[best_idx + 1].len;
            const merged_str = bl_buf.items[merged_start .. merged_start + merged_len];
            const merged_id = self.piece_to_id.get(merged_str) orelse break;

            syms.items[best_idx] = .{ .id = merged_id, .start = merged_start, .len = merged_len };
            _ = syms.orderedRemove(best_idx + 1);
        }

        for (syms.items) |s| try out.append(a, s.id);
    }
};

fn isAsciiLetter(c: u8) bool {
    return (c >= 'A' and c <= 'Z') or (c >= 'a' and c <= 'z');
}

fn isLetterByte(c: u8) bool {
    return isAsciiLetter(c) or c >= 0x80; // treat UTF-8 multibyte letters as word chars
}

fn isDigit(c: u8) bool {
    return c >= '0' and c <= '9';
}

fn isWs(c: u8) bool {
    return c == ' ' or c == '\t' or c == '\r' or c == '\n' or c == 0x0B or c == 0x0C;
}

fn isNewline(c: u8) bool {
    return c == '\r' or c == '\n';
}

/// Pre-tokenizer splitter supporting both Qwen (`\p{N}`, max_digit_run=1) and Llama 3 (`\p{N}{1,3}`, max_digit_run=3).
fn nextPreToken(s: []const u8, start: usize, max_digit_run: u8) usize {
    const rem = s[start..];
    if (rem.len == 0) return start;

    // 1. English contractions: 's 't 're 've 'm 'll 'd (case-insensitive)
    if (rem[0] == '\'' and rem.len >= 2) {
        const c1 = std.ascii.toLower(rem[1]);
        if (c1 == 's' or c1 == 't' or c1 == 'm' or c1 == 'd') return start + 2;
        if (rem.len >= 3) {
            const c2 = std.ascii.toLower(rem[2]);
            if ((c1 == 'r' and c2 == 'e') or (c1 == 'v' and c2 == 'e') or (c1 == 'l' and c2 == 'l')) return start + 3;
        }
    }

    // 2. [^\r\n\p{L}\p{N}]?\p{L}+
    if (isLetterByte(rem[0])) {
        var i: usize = 1;
        while (i < rem.len and isLetterByte(rem[i])) : (i += 1) {}
        return start + i;
    }
    if (!isNewline(rem[0]) and !isDigit(rem[0]) and rem.len >= 2 and isLetterByte(rem[1])) {
        var i: usize = 2;
        while (i < rem.len and isLetterByte(rem[i])) : (i += 1) {}
        return start + i;
    }

    // 3. Digit run \p{N}{1,max_digit_run}
    if (isDigit(rem[0])) {
        var i: usize = 1;
        while (i < rem.len and i < max_digit_run and isDigit(rem[i])) : (i += 1) {}
        return start + i;
    }

    // 4. Optional single space + non-ws/non-letter/non-digit run + trailing [\r\n]*
    if ((!isWs(rem[0]) and !isLetterByte(rem[0]) and !isDigit(rem[0])) or
        (rem[0] == ' ' and rem.len >= 2 and !isWs(rem[1]) and !isLetterByte(rem[1]) and !isDigit(rem[1])))
    {
        var i: usize = if (rem[0] == ' ') 2 else 1;
        while (i < rem.len and !isWs(rem[i]) and !isLetterByte(rem[i]) and !isDigit(rem[i])) : (i += 1) {}
        while (i < rem.len and isNewline(rem[i])) : (i += 1) {}
        return start + i;
    }

    // 5. Whitespace runs (\s*[\r\n]+ | \s+(?!\S) | \s+)
    var ws_end: usize = 0;
    var last_nl: usize = 0;
    while (ws_end < rem.len and isWs(rem[ws_end])) : (ws_end += 1) {
        if (isNewline(rem[ws_end])) last_nl = ws_end + 1;
    }
    if (last_nl > 0) return start + last_nl;
    if (ws_end > 1 and ws_end < rem.len) return start + ws_end - 1; // \s+(?!\S) keeps last space for next word
    return start + @max(ws_end, 1);
}
