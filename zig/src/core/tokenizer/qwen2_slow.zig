//! The tokenizer.json a Qwen2 checkpoint ships when it has only a slow tokenizer's files.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Error = error{ InvalidTokenizer, UnsupportedModel, MissingMergeToken, InvalidAddedTokenId };

/// Reads `dir`'s config, vocab and merges, and writes the tokenizer.json they describe.
pub fn converted(io: std.Io, a: Allocator, dir: []const u8) ![]u8 {
    const cwd = std.Io.Dir.cwd();
    const config_path = try std.fs.path.join(a, &.{ dir, "tokenizer_config.json" });
    defer a.free(config_path);
    const config_text = try cwd.readFileAlloc(io, config_path, a, .limited(1 << 26));
    defer a.free(config_text);
    const config = std.json.parseFromSlice(std.json.Value, a, config_text, .{}) catch return error.InvalidTokenizer;
    defer config.deinit();
    if (config.value != .object) return error.InvalidTokenizer;
    const class = config.value.object.get("tokenizer_class") orelse return error.InvalidTokenizer;
    if (class != .string or !std.mem.startsWith(u8, class.string, "Qwen2Tokenizer")) return error.UnsupportedModel;
    const vocab_path = try std.fs.path.join(a, &.{ dir, "vocab.json" });
    defer a.free(vocab_path);
    const merges_path = try std.fs.path.join(a, &.{ dir, "merges.txt" });
    defer a.free(merges_path);
    const vocab_text = try cwd.readFileAlloc(io, vocab_path, a, .limited(1 << 28));
    defer a.free(vocab_text);
    const merges = try cwd.readFileAlloc(io, merges_path, a, .limited(1 << 28));
    defer a.free(merges);
    const vocab = try parseVocab(a, vocab_text);
    defer vocab.deinit();
    var out: std.Io.Writer.Allocating = .init(a);
    errdefer out.deinit();
    const w = &out.writer;
    try w.writeAll("{\"model\":{\"type\":\"BPE\",\"dropout\":null,\"unk_token\":null,\"continuing_subword_prefix\":\"\",\"end_of_word_suffix\":\"\",\"fuse_unk\":false,\"byte_fallback\":false,\"ignore_merges\":false,\"vocab\":{");
    var first = true;
    var entries = vocab.value.object.iterator();
    while (entries.next()) |entry| {
        if (!first) try w.writeByte(',');
        first = false;
        try std.json.Stringify.encodeJsonString(entry.key_ptr.*, .{}, w);
        try w.print(":{d}", .{try idOf(entry.value_ptr.*)});
    }
    try w.writeAll("},\"merges\":[");
    first = true;
    var lines = std.mem.splitScalar(u8, merges, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trimEnd(u8, raw, "\r");
        if (line.len == 0 or std.mem.startsWith(u8, line, "#version")) continue;
        const space = std.mem.indexOfScalar(u8, line, ' ') orelse return error.InvalidTokenizer;
        const left = line[0..space];
        const right = line[space + 1 ..];
        if (right.len == 0 or std.mem.indexOfScalar(u8, right, ' ') != null) return error.InvalidTokenizer;
        if (!vocab.value.object.contains(left) or !vocab.value.object.contains(right)) return error.MissingMergeToken;
        if (!first) try w.writeByte(',');
        first = false;
        try std.json.Stringify.encodeJsonString(line, .{}, w);
    }
    try w.writeAll("]},\"added_tokens\":[");
    if (config.value.object.get("added_tokens_decoder")) |added| if (added == .object) {
        var it = added.object.iterator();
        var n: usize = 0;
        while (it.next()) |entry| : (n += 1) {
            if (entry.value_ptr.* != .object) return error.InvalidTokenizer;
            const o = entry.value_ptr.object;
            const content = o.get("content") orelse return error.InvalidTokenizer;
            if (content != .string) return error.InvalidTokenizer;
            const id = try parseAddedId(entry.key_ptr.*);
            if (n > 0) try w.writeByte(',');
            try w.print("{{\"id\":{d},\"content\":", .{id});
            try std.json.Stringify.encodeJsonString(content.string, .{}, w);
            inline for (.{ "single_word", "lstrip", "rstrip", "normalized", "special" }) |flag_name| {
                const v = o.get(flag_name);
                try w.print(",\"{s}\":{s}", .{ flag_name, if (v != null and v.? == .bool and v.?.bool) "true" else "false" });
            }
            try w.writeByte('}');
        }
    };
    try w.writeAll("],\"normalizer\":{\"type\":\"NFC\"},\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":[{\"type\":\"Split\",\"pattern\":{\"Regex\":");
    try std.json.Stringify.encodeJsonString("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+", .{}, w);
    try w.writeAll("},\"behavior\":\"Isolated\",\"invert\":false},{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":true,\"use_regex\":false}]},\"post_processor\":{\"type\":\"ByteLevel\",\"add_prefix_space\":true,\"trim_offsets\":false,\"use_regex\":true},\"decoder\":{\"type\":\"ByteLevel\",\"add_prefix_space\":true,\"trim_offsets\":true,\"use_regex\":true}}");
    return out.toOwnedSlice();
}

fn parseVocab(a: Allocator, text: []const u8) !std.json.Parsed(std.json.Value) {
    const parsed = std.json.parseFromSlice(std.json.Value, a, text, .{}) catch return error.InvalidTokenizer;
    errdefer parsed.deinit();
    if (parsed.value != .object) return error.InvalidTokenizer;
    var it = parsed.value.object.iterator();
    while (it.next()) |entry| _ = try idOf(entry.value_ptr.*);
    return parsed;
}

fn idOf(value: std.json.Value) !u32 {
    if (value != .integer or value.integer < 0) return error.InvalidTokenizer;
    return std.math.cast(u32, value.integer) orelse error.InvalidTokenizer;
}

fn parseAddedId(text: []const u8) !u32 {
    if (text.len == 0) return error.InvalidAddedTokenId;
    for (text) |c| if (c < '0' or c > '9') return error.InvalidAddedTokenId;
    return std.fmt.parseInt(u32, text, 10) catch error.InvalidAddedTokenId;
}
