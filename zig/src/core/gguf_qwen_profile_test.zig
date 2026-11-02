//! The qwen35 profile follows the checkpoint's regex, added-token policy and complete generation stop set.
const std = @import("std");
const t = std.testing;
const gg = @import("gguf.zig");
const adapter = @import("gguf_tokenizer.zig");
const f = @import("gguf_tokenizer_fixture.zig");
const tokenizer = @import("tokenizer");
fn parse(a: std.mem.Allocator, v: std.json.Value) !tokenizer.Tokenizer {
    return tokenizer.parse(a, try std.json.Stringify.valueAlloc(a, v, .{}));
}
test "GGUF Qwen: letter-only words split combining marks before BPE, unlike the old regex control" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bytes = try f.make(a, false, false);
    var d = try gg.Document.parse(t.allocator, bytes.slice(), .{});
    defer d.deinit();
    const assets = try adapter.assets(a, &d);
    var good = try parse(a, assets.tokenizer);
    defer good.deinit();
    var json = try parse(a, try f.reference(a));
    defer json.deinit();
    const want = [_]u32{ 264, 263, 191 };
    try t.expectEqualSlices(u32, &want, try good.encode(a, "कि"));
    try t.expectEqualSlices(u32, &want, try json.encode(a, "कि"));
    var old = try f.reference(a);
    const pre = old.object.getPtr("pre_tokenizer").?.object.getPtr("pretokenizers").?.array.items;
    pre[0].object.getPtr("pattern").?.object.getPtr("Regex").?.string = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    var bad = try parse(a, old);
    defer bad.deinit();
    try t.expectEqualSlices(u32, &.{267}, try bad.encode(a, "कि"));
    const steps = assets.tokenizer.object.get("pre_tokenizer").?.object.get("pretokenizers").?.array.items;
    const byte = steps[1].object;
    try t.expect(!byte.get("add_prefix_space").?.bool and byte.get("trim_offsets").?.bool and !byte.get("use_regex").?.bool);
    const decoder = assets.tokenizer.object.get("decoder").?.object;
    try t.expect(decoder.get("add_prefix_space").?.bool and decoder.get("trim_offsets").?.bool and decoder.get("use_regex").?.bool);
}
test "GGUF Qwen: six ordinary added tokens survive special-skipping decode but remain masked in request text" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bytes = try f.make(a, false, false);
    var d = try gg.Document.parse(t.allocator, bytes.slice(), .{});
    defer d.deinit();
    const v = (try adapter.assets(a, &d)).tokenizer;
    var good = try parse(a, v);
    defer good.deinit();
    for (f.controls[4..], 4..) |text, i| {
        const ids = [_]u32{@intCast(f.normal_count + i)};
        try t.expectEqualSlices(u32, &ids, try good.encode(a, text));
        try t.expectEqualStrings(text, try good.decode(a, &ids, true));
        for (try good.encodePlain(a, text)) |id| try t.expect(id < f.normal_count);
    }
    // Reinstating the GGUF's control flags must fail the same skip-special decoding check.
    var old = (try adapter.assets(a, &d)).tokenizer;
    for (old.object.getPtr("added_tokens").?.array.items[4..]) |*item| item.object.getPtr("special").?.bool = true;
    var bad = try parse(a, old);
    defer bad.deinit();
    try t.expectEqualStrings("", try bad.decode(a, &.{f.normal_count + 4}, true));
}
test "GGUF Qwen: generation config and effective eosIds include both declared end and end-of-text control" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bytes = try @import("../families/qwen27/gguf_fixture.zig").makeTokenized(a);
    var d = try gg.Document.parse(t.allocator, bytes, .{});
    defer d.deinit();
    const cvalue = try @import("../families/qwen27/gguf_config.zig").build(a, &d);
    const c = try @import("../families/qwen27/gguf_config.zig").parse(a, cvalue);
    try t.expectEqualSlices(u32, &.{ f.eos, f.bos }, c.eos[0..c.eos_count]);
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    const image = @import("../families/qwen27/gguf_image.zig");
    try adapter.write(null, t.allocator, t.io, &d, root);
    try image.jsonFile(t.allocator, t.io, root, "config.json", cvalue);
    try image.jsonFile(t.allocator, t.io, root, "generation_config.json", try image.sampling(a, &d, c));
    const HfText = @import("../server/hf_text.zig").HfText;
    var problem: []const u8 = "";
    const text = try HfText.load(t.allocator, t.io, root, null, a, &problem);
    defer text.deinit();
    // The independent MLX generation config declares the same two stopping meanings.
    const eos = @import("tokenizer").eos;
    const mlx = try std.json.parseFromSliceLeaky(std.json.Value, a, try std.fmt.allocPrint(a, "{{\"eos_token_id\":[{d},{d}]}}", .{ f.eos, f.bos }), .{});
    try t.expectEqualSlices(u32, try eos.resolve(a, .null, mlx, f.count), text.text().eosIds());
    try t.expect(text.text().eosIds().len != 1); // The old image's singleton stop set is the planted control.
    const kinds = &d.metadata[d.keys.get("tokenizer.ggml.token_type").?].value.array;
    const raw = try a.dupe(u8, kinds.bytes);
    kinds.bytes = raw;
    std.mem.writeInt(i32, raw[f.bos * 4 ..][0..4], 4, .little);
    try t.expectError(error.BadGgufEndOfTextControl, adapter.eosIds(a, &d));
    std.mem.writeInt(i32, raw[f.bos * 4 ..][0..4], 3, .little);
    d.metadata[d.keys.get("tokenizer.ggml.eos_token_id").?].value.uint32 = f.bos;
    try t.expectEqual(@as(usize, 1), (try adapter.eosIds(a, &d)).array.items.len);
}
