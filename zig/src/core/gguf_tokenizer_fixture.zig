//! Small byte-complete Qwen35 metadata with independently specified JSON assets for host parity checks.
const std = @import("std");
const Writer = @import("gguf_test_fixture.zig").Fixture;
const A = std.mem.Allocator;
pub const extras = [_][]const u8{ "ab", "bc", "Ġa", "Ġab", "Ã©", "12", "'s" };
pub const controls = [_][]const u8{ "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>" };
pub const merges = [_][]const u8{ "a b", "b c", "Ġ a", "Ġa b", "Ã ©", "1 2", "' s" };
pub const normal_count = 256 + extras.len;
pub const count = normal_count + controls.len + 1;
pub const bos = normal_count;
pub const eos = normal_count + 2;
pub const chat = "{{ bos_token }}{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
pub fn words(a: A) ![][]const u8 {
    const out = try a.alloc([]const u8, count);
    var n: u21 = 256;
    // This fixture computes the published byte alphabet without importing the adapter or tokenizer's table.
    for (0..256) |b| {
        const printable = (33 <= b and b <= 126) or (161 <= b and b <= 172) or b >= 174;
        const cp: u21 = if (printable) @intCast(b) else n;
        if (!printable) n += 1;
        var buf: [4]u8 = undefined;
        out[b] = try a.dupe(u8, buf[0..try std.unicode.utf8Encode(cp, &buf)]);
    }
    @memcpy(out[256..normal_count], &extras);
    @memcpy(out[normal_count..][0..controls.len], &controls);
    out[count - 1] = "[PAD_UNUSED]";
    return out;
}
pub fn metadata(h: *Writer, a: A, add_bos: bool, add_eos: bool) !void {
    const tokens = try words(a);
    h.kv("tokenizer.ggml.model", .{ .string = "gpt2" });
    h.kv("tokenizer.ggml.pre", .{ .string = "qwen35" });
    h.array("tokenizer.ggml.tokens", .string, count);
    for (tokens) |s| h.string(s);
    h.array("tokenizer.ggml.token_type", .int32, count);
    for (0..count) |i| h.int(i32, if (i < normal_count) 1 else if (i == count - 1) 5 else if (i == count - 2) 4 else 3);
    h.array("tokenizer.ggml.merges", .string, merges.len);
    for (merges) |s| h.string(s);
    h.kv("tokenizer.ggml.bos_token_id", .{ .uint32 = bos });
    h.kv("tokenizer.ggml.eos_token_id", .{ .uint32 = eos });
    h.kv("tokenizer.ggml.padding_token_id", .{ .uint32 = bos });
    h.kv("tokenizer.ggml.add_bos_token", .{ .boolean = add_bos });
    h.kv("tokenizer.ggml.add_eos_token", .{ .boolean = add_eos });
    h.kv("tokenizer.chat_template", .{ .string = chat });
}
pub const metadata_count = 11;
pub fn make(a: A, add_bos: bool, add_eos: bool) !Writer {
    var h = Writer.init(0, metadata_count);
    try metadata(&h, a, add_bos, add_eos);
    h.finish(32, 0);
    return h;
}
/// A separate JSON path, with controls outside model.vocab as in a Hugging Face tokenizer.json.
pub fn reference(a: A) !std.json.Value {
    const base =
        \\{"version":"1.0","normalizer":{"type":"NFC"},"pre_tokenizer":{"type":"Sequence","pretokenizers":[
        \\{"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},
        \\{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]},
        \\"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
        \\"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
        \\"model":{"type":"BPE","vocab":{},"merges":[],"unk_token":null,"ignore_merges":false,"byte_fallback":false},"added_tokens":[]}
    ;
    var root = try std.json.parseFromSliceLeaky(std.json.Value, a, base, .{});
    const tokens = try words(a);
    const model = root.object.getPtr("model").?;
    for (tokens[0..normal_count], 0..) |s, id| try model.object.getPtr("vocab").?.object.put(a, s, .{ .integer = @intCast(id) });
    for (merges) |s| try model.object.getPtr("merges").?.array.append(.{ .string = s });
    for (controls, 0..) |s, i| {
        const bytes = try std.json.Stringify.valueAlloc(a, .{ .id = normal_count + i, .content = s, .special = i != 3, .normalized = false, .single_word = false, .lstrip = false, .rstrip = false }, .{});
        try root.object.getPtr("added_tokens").?.array.append(try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}));
    }
    return root;
}
