//! GGUF gpt2/qwen35 metadata becomes assets for the existing native tokenizer and template engines.
const std = @import("std");
const gg = @import("gguf.zig");
const V = std.json.Value;
const A = std.mem.Allocator;
pub const source = @embedFile("gguf_tokenizer.zig");
pub const pattern = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
pub const Assets = struct { tokenizer: V, config: V };
const Token = struct { text: []const u8, kind: i32 };
fn object() V {
    return .{ .object = .empty };
}
fn string(s: []const u8) V {
    return .{ .string = s };
}
fn put(a: A, v: *V, key: []const u8, value: V) !void {
    try v.object.put(a, key, value);
}
fn array(a: A, items: []const V) !V {
    var v: V = .{ .array = std.array_list.Managed(V).init(a) };
    try v.array.appendSlice(items);
    return v;
}
fn idToken(tokens: []const Token, id: u64) !Token {
    if (id >= tokens.len or tokens[@intCast(id)].kind == 5) return error.BadGgufSpecialToken;
    return tokens[@intCast(id)];
}
// GPT-2 maps the 68 nonprinting bytes to U+0100..U+0143; the rest keep their Latin-1 codepoints.
fn byteCodepoint(b: u21) u21 {
    return if (b <= 32) 256 + b else if (b >= 127 and b <= 160) 162 + b else if (b == 173) 323 else b;
}
fn byteSymbol(cp: u21) bool {
    return (cp >= 33 and cp <= 126) or (cp >= 161 and cp <= 172) or (cp >= 174 and cp <= 323);
}
/// All output storage belongs to the caller's arena; token strings can borrow the document until serialized.
pub fn assets(a: A, d: *const gg.Document) !Assets {
    const view = try d.tokenizer();
    if (!std.mem.eql(u8, view.model orelse "", "gpt2") or
        !std.mem.eql(u8, view.pre orelse "", "qwen35")) return error.UnsupportedGgufTokenizer;
    const words = view.tokens orelse return error.MissingGgufTokenizerTokens;
    const types = view.types orelse return error.MissingGgufTokenizerTypes;
    const merges = view.merges orelse return error.MissingGgufTokenizerMerges;
    if (words.len == 0 or words.len > std.math.maxInt(u32)) return error.BadGgufVocabulary;
    const tokens = try a.alloc(Token, words.len);
    var lookup: std.StringHashMapUnmanaged(usize) = .empty;
    var vocab = object();
    var added: V = .{ .array = std.array_list.Managed(V).init(a) };
    var wi = words.iterator();
    var ti = types.iterator();
    for (tokens, 0..) |*token, id| {
        const text = (try wi.next()).?.string;
        const kind = (try ti.next()).?.int32;
        if (text.len == 0 or !std.unicode.utf8ValidateSlice(text)) return error.BadGgufVocabulary;
        const entry = try lookup.getOrPut(a, text);
        if (entry.found_existing) return error.DuplicateGgufToken;
        entry.value_ptr.* = id;
        token.* = .{ .text = text, .kind = kind };
        if (kind != 1 and kind != 3 and kind != 4 and kind != 5) return error.UnsupportedGgufTokenType;
        if (kind == 5) continue;
        if (kind == 1) {
            var it = (try std.unicode.Utf8View.init(text)).iterator();
            while (it.nextCodepoint()) |cp| if (!byteSymbol(cp)) return error.BadGgufByteToken;
        }
        // Keeping added spellings in vocab preserves even sparse ids with the native JSON reader.
        try put(a, &vocab, text, .{ .integer = @intCast(id) });
        if (kind == 3 or kind == 4) {
            var value = object();
            try put(a, &value, "id", .{ .integer = @intCast(id) });
            try put(a, &value, "content", string(text));
            try put(a, &value, "special", .{ .bool = kind == 3 });
            for ([_][]const u8{ "single_word", "lstrip", "rstrip", "normalized" }) |key| try put(a, &value, key, .{ .bool = false });
            try added.array.append(value);
        }
    }
    for (0..256) |b| {
        var buf: [4]u8 = undefined;
        const len = try std.unicode.utf8Encode(byteCodepoint(@intCast(b)), &buf);
        const id = lookup.get(buf[0..len]) orelse return error.IncompleteGgufByteAlphabet;
        if (tokens[id].kind != 1) return error.IncompleteGgufByteAlphabet;
    }
    var ranks: V = .{ .array = std.array_list.Managed(V).init(a) };
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    var mi = merges.iterator();
    while (try mi.next()) |value| {
        const merge = value.string;
        const split = std.mem.indexOfScalar(u8, merge, ' ') orelse return error.BadGgufMerge;
        if (split == 0 or split + 1 == merge.len or std.mem.indexOfScalar(u8, merge[split + 1 ..], ' ') != null) return error.BadGgufMerge;
        const joined = try std.mem.concat(a, u8, &.{ merge[0..split], merge[split + 1 ..] });
        for ([_][]const u8{ merge[0..split], merge[split + 1 ..], joined }) |part| {
            const id = lookup.get(part) orelse return error.BadGgufMerge;
            if (tokens[id].kind != 1) return error.BadGgufMerge;
        }
        if ((try seen.getOrPut(a, merge)).found_existing) return error.DuplicateGgufMerge;
        try ranks.array.append(string(merge));
    }
    var model = object();
    try put(a, &model, "type", string("BPE"));
    try put(a, &model, "vocab", vocab);
    try put(a, &model, "merges", ranks);
    try put(a, &model, "dropout", .null);
    try put(a, &model, "unk_token", .null);
    try put(a, &model, "continuing_subword_prefix", string(""));
    try put(a, &model, "end_of_word_suffix", string(""));
    for ([_][]const u8{ "fuse_unk", "byte_fallback", "ignore_merges" }) |key| try put(a, &model, key, .{ .bool = false });
    var split = object();
    var regex = object();
    try put(a, &regex, "Regex", string(pattern));
    try put(a, &split, "type", string("Split"));
    try put(a, &split, "pattern", regex);
    try put(a, &split, "behavior", string("Isolated"));
    try put(a, &split, "invert", .{ .bool = false });
    const bytelevel = try byteLevel(a);
    var pre = object();
    try put(a, &pre, "type", string("Sequence"));
    try put(a, &pre, "pretokenizers", try array(a, &.{ split, bytelevel }));
    var root = object();
    try put(a, &root, "version", string("1.0"));
    try put(a, &root, "truncation", .null);
    try put(a, &root, "padding", .null);
    var normal = object();
    try put(a, &normal, "type", string("NFC"));
    try put(a, &root, "normalizer", normal);
    try put(a, &root, "pre_tokenizer", pre);
    try put(a, &root, "decoder", bytelevel);
    try put(a, &root, "model", model);
    try put(a, &root, "added_tokens", added);
    var config = object();
    const pad = if (d.get("tokenizer.ggml.padding_token_id")) |v| try v.unsigned() else null;
    const ids = [_]?u64{ view.bos, view.eos, pad };
    for ([_][]const u8{ "bos_token", "eos_token", "pad_token" }, ids) |key, maybe| if (maybe) |id| {
        const token = try idToken(tokens, id);
        try put(a, &config, key, string(token.text));
        try put(a, &config, try std.fmt.allocPrint(a, "{s}_id", .{key}), .{ .integer = @intCast(id) });
    };
    const flags = [_]bool{ view.add_bos orelse false, view.add_eos orelse false };
    try put(a, &config, "add_bos_token", .{ .bool = flags[0] });
    try put(a, &config, "add_eos_token", .{ .bool = flags[1] });
    if (d.get("tokenizer.chat_template")) |v| {
        if (v != .string or !std.unicode.utf8ValidateSlice(v.string)) return error.BadGgufChatTemplate;
        try put(a, &config, "chat_template", string(v.string));
    }
    const post = if (flags[0] or flags[1]) try processor(a, tokens, ids, flags) else bytelevel;
    try put(a, &root, "post_processor", post);
    return .{ .tokenizer = root, .config = config };
}
fn byteLevel(a: A) !V {
    var v = object();
    try put(a, &v, "type", string("ByteLevel"));
    for ([_][]const u8{ "add_prefix_space", "trim_offsets", "use_regex" }) |key| try put(a, &v, key, .{ .bool = false });
    return v;
}
fn sequence(a: A, name: []const u8, type_id: i64) !V {
    var details = object();
    try put(a, &details, "id", string(name));
    try put(a, &details, "type_id", .{ .integer = type_id });
    var out = object();
    try put(a, &out, "Sequence", details);
    return out;
}
fn special(a: A, name: []const u8, type_id: i64) !V {
    var v = try sequence(a, name, type_id);
    const details = v.object.get("Sequence").?;
    _ = v.object.swapRemove("Sequence");
    try put(a, &v, "SpecialToken", details);
    return v;
}
fn processor(a: A, tokens: []const Token, ids: [3]?u64, flags: [2]bool) !V {
    var single: V = .{ .array = std.array_list.Managed(V).init(a) };
    var pair: V = .{ .array = std.array_list.Managed(V).init(a) };
    var specials = object();
    if (flags[0]) {
        const id = ids[0] orelse return error.MissingGgufSpecialToken;
        const token = try idToken(tokens, id);
        try single.array.append(try special(a, token.text, 0));
        try pair.array.append(try special(a, token.text, 0));
    }
    try single.array.append(try sequence(a, "A", 0));
    try pair.array.append(try sequence(a, "A", 0));
    try pair.array.append(try sequence(a, "B", 1));
    if (flags[1]) {
        const id = ids[1] orelse return error.MissingGgufSpecialToken;
        const token = try idToken(tokens, id);
        try single.array.append(try special(a, token.text, 0));
        try pair.array.append(try special(a, token.text, 1));
    }
    for (flags, ids[0..2]) |flag, maybe| if (flag) {
        const id = maybe.?;
        const name = (try idToken(tokens, id)).text;
        var v = object();
        try put(a, &v, "id", string(name));
        try put(a, &v, "ids", try array(a, &.{.{ .integer = @intCast(id) }}));
        try put(a, &v, "tokens", try array(a, &.{string(name)}));
        try put(a, &specials, name, v);
    };
    var out = object();
    try put(a, &out, "type", string("TemplateProcessing"));
    try put(a, &out, "single", single);
    try put(a, &out, "pair", pair);
    try put(a, &out, "special_tokens", specials);
    return out;
}
pub fn write(_: ?*anyopaque, a: A, io: std.Io, d: *const gg.Document, dir: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const value = try assets(arena.allocator(), d);
    for ([_][]const u8{ "tokenizer.json", "tokenizer_config.json" }, [_]V{ value.tokenizer, value.config }) |name, v| {
        const text = try std.json.Stringify.valueAlloc(arena.allocator(), v, .{});
        const path = try std.fs.path.join(arena.allocator(), &.{ dir, name });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = text });
    }
}
