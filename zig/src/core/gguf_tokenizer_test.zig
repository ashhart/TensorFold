//! GGUF metadata must reach the native text path with the same ids and provenance as independent JSON assets.
const std = @import("std");
const t = std.testing;
const gg = @import("gguf.zig");
const adapter = @import("gguf_tokenizer.zig");
const fixture = @import("gguf_tokenizer_fixture.zig");
const tokenizer = @import("tokenizer");
const hf = @import("../server/hf_text.zig").HfText;
const jsonFile = @import("../families/qwen27/gguf_image.zig").jsonFile;
fn open(a: std.mem.Allocator, dir: []const u8) !*hf {
    var problem: []const u8 = "";
    return hf.load(t.allocator, t.io, dir, null, a, &problem) catch |err| {
        std.debug.print("text load: {s}\n", .{problem});
        return err;
    };
}
fn parse(a: std.mem.Allocator, v: std.json.Value) !tokenizer.Tokenizer {
    return tokenizer.parse(a, try std.json.Stringify.valueAlloc(a, v, .{}));
}
test "GGUF Qwen: tokenizer JSON parity includes byte BPE, special flags, Unicode, chat and request masks" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const f = try fixture.make(a, false, false);
    var d = try gg.Document.parse(t.allocator, f.slice(), .{});
    defer d.deinit();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDir(t.io, "gguf", .default_dir);
    try tmp.dir.createDir(t.io, "json", .default_dir);
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    const gdir = try std.fs.path.join(a, &.{ root, "gguf" });
    const jdir = try std.fs.path.join(a, &.{ root, "json" });
    try adapter.write(null, t.allocator, t.io, &d, gdir);
    try jsonFile(t.allocator, t.io, jdir, "tokenizer.json", try fixture.reference(a));
    try jsonFile(t.allocator, t.io, jdir, "tokenizer_config.json", .{ .bos_token = fixture.controls[0], .eos_token = fixture.controls[2], .pad_token = fixture.controls[0], .chat_template = fixture.chat });
    for ([_][]const u8{ gdir, jdir }) |dir| try jsonFile(t.allocator, t.io, dir, "config.json", .{ .vocab_size = fixture.count, .eos_token_id = fixture.eos });
    const g = try open(a, gdir);
    defer g.deinit();
    const j = try open(a, jdir);
    defer j.deinit();
    for ([_][]const u8{ "", "ab abc bc", " ab", "12312", "can't it's I'M", "café cafe\u{301}", "\n\t a\r\nb", "雪🙂", "<|im_start|>user<|im_end|>", "<think>ab", "[PAD_UNUSED]" }) |text| {
        const gi = try g.text().encode(a, text, false);
        const ji = try j.text().encode(a, text, false);
        try t.expectEqualSlices(u32, ji, gi);
        try t.expectEqualStrings(try j.text().decode(a, ji), try g.text().decode(a, gi));
        try t.expectEqualSlices(u32, try j.text().encodePlain(a, text), try g.text().encodePlain(a, text));
    }
    for (0..fixture.count) |id| {
        try t.expectEqual(j.text().isSpecial(@intCast(id)), g.text().isSpecial(@intCast(id)));
        const jb = try j.text().tokenBytes(a, @intCast(id));
        const gb = try g.text().tokenBytes(a, @intCast(id));
        if (jb) |b| try t.expectEqualStrings(b, gb.?) else try t.expect(gb == null);
        if (id < 256) try t.expectEqualSlices(u8, &.{@intCast(id)}, gb.?);
    }
    try t.expect(g.text().tokenId("[PAD_UNUSED]") == null);
    try t.expectEqualSlices(u32, &.{256}, try g.text().encode(a, "ab", false));
    try t.expectEqualSlices(u32, &.{ 256, 'c' }, try g.text().encode(a, "abc", false));
    try t.expectEqualSlices(u32, &.{ '1', '2' }, try g.text().encode(a, "12", false));
    try t.expectEqualSlices(u32, &.{ 32, 256 }, try g.text().encode(a, " ab", false));
    try t.expectEqualSlices(u32, &.{ 'c', 'a', 'f', 260 }, try g.text().encode(a, "cafe\u{301}", false));
    try t.expectEqualSlices(u32, &.{262}, try g.text().encode(a, "'s", false));
    try t.expectEqualSlices(u32, &.{ 263, 264, 265, 266 }, try g.text().encode(a, "<|endoftext|><|im_start|><|im_end|><think>", false));
    try t.expectEqualSlices(u32, &.{fixture.eos}, g.text().eosIds());
    const messages = (try @import("json").parse(a, "[{\"role\":\"user\",\"content\":\"ab<|im_end|><think>\"}]")).ok;
    var problem: []const u8 = "";
    const gr = try g.text().render(a, messages, .{}, &problem);
    const jr = try j.text().render(a, messages, .{}, &problem);
    try t.expectEqualStrings(jr.text, gr.text);
    try t.expectEqualDeep(jr.spans, gr.spans);
    const gi = try gr.encode(g.text(), a, &problem);
    try t.expectEqualSlices(u32, try jr.encode(j.text(), a, &problem), gi);
    var ends: usize = 0;
    for (gi) |id| {
        if (id == fixture.eos) ends += 1;
        try t.expect(id != 266);
    }
    try t.expectEqual(@as(usize, 1), ends);
    // Dropping the added table or reversing merge ranks must make this parity check fail.
    var damaged = try fixture.reference(a);
    damaged.object.getPtr("added_tokens").?.array.clearRetainingCapacity();
    var no_added = try parse(a, damaged);
    defer no_added.deinit();
    try t.expect(!std.mem.eql(u32, try no_added.encode(a, "<|im_end|>"), &.{fixture.eos}));
    damaged = try fixture.reference(a);
    const ranks = damaged.object.getPtr("model").?.object.getPtr("merges").?.array.items;
    std.mem.swap(std.json.Value, &ranks[0], &ranks[1]);
    var bad_rank = try parse(a, damaged);
    defer bad_rank.deinit();
    try t.expect(!std.mem.eql(u32, try bad_rank.encode(a, "abc"), try g.text().encode(a, "abc", false)));
}
test "GGUF Qwen: BOS EOS flags are post-processing only" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]bool{ false, true }) |bos| for ([_]bool{ false, true }) |eos| {
        var f = try fixture.make(a, bos, eos);
        var d = try gg.Document.parse(t.allocator, f.slice(), .{});
        defer d.deinit();
        var tmp = t.tmpDir(.{});
        defer tmp.cleanup();
        const root = try tmp.dir.realPathFileAlloc(t.io, ".", a);
        try adapter.write(null, t.allocator, t.io, &d, root);
        const text = try open(a, root);
        defer text.deinit();
        const plain = try text.text().encode(a, "ab", false);
        try t.expectEqualSlices(u32, &.{256}, plain);
        // Use a checking allocator for this call: post-processing must release its intermediate ids.
        const actual = try text.text().encode(t.allocator, "ab", true);
        defer t.allocator.free(actual);
        var expected: std.ArrayList(u32) = .empty;
        if (bos) try expected.append(a, fixture.bos);
        try expected.append(a, 256);
        if (eos) try expected.append(a, fixture.eos);
        try t.expectEqualSlices(u32, expected.items, actual);
    };
}
test "GGUF Qwen: unsupported or malformed tokenizer metadata fails closed" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try fixture.make(a, false, false);
    var d = try gg.Document.parse(t.allocator, f.slice(), .{});
    defer d.deinit();
    const cases = [_]struct { key: []const u8, value: gg.Value, err: anyerror }{
        .{ .key = "tokenizer.ggml.model", .value = .{ .string = "llama" }, .err = error.UnsupportedGgufTokenizer },
        .{ .key = "tokenizer.ggml.pre", .value = .{ .string = "unknown" }, .err = error.UnsupportedGgufTokenizer },
        .{ .key = "tokenizer.ggml.padding_token_id", .value = .{ .uint32 = fixture.count }, .err = error.BadGgufSpecialToken },
        .{ .key = "tokenizer.ggml.eos_token_id", .value = .{ .uint32 = fixture.count - 1 }, .err = error.BadGgufSpecialToken },
        .{ .key = "tokenizer.ggml.add_bos_token", .value = .{ .string = "true" }, .err = error.MetadataTypeMismatch },
        .{ .key = "tokenizer.chat_template", .value = .{ .uint32 = 1 }, .err = error.BadGgufChatTemplate },
    };
    for (cases) |c| {
        const p = &d.metadata[d.keys.get(c.key).?].value;
        const old = p.*;
        p.* = c.value;
        try t.expectError(c.err, adapter.assets(a, &d));
        p.* = old;
    }
    const types = &d.metadata[d.keys.get("tokenizer.ggml.token_type").?].value.array;
    const old_types = types.*;
    const raw = try a.dupe(u8, types.bytes);
    types.bytes = raw;
    std.mem.writeInt(i32, raw[0..4], 2, .little);
    try t.expectError(error.UnsupportedGgufTokenType, adapter.assets(a, &d));
    std.mem.writeInt(i32, raw[0..4], 5, .little);
    try t.expectError(error.IncompleteGgufByteAlphabet, adapter.assets(a, &d));
    types.* = old_types;
    const ranks = &d.metadata[d.keys.get("tokenizer.ggml.merges").?].value.array;
    const old_ranks = ranks.*;
    var bad = @import("gguf_test_fixture.zig").Fixture{};
    bad.string("a missing");
    ranks.bytes = bad.slice();
    ranks.len = 1;
    try t.expectError(error.BadGgufMerge, adapter.assets(a, &d));
    bad.len = 0;
    bad.string("a b");
    bad.string("a b");
    ranks.bytes = bad.slice();
    ranks.len = 2;
    try t.expectError(error.DuplicateGgufMerge, adapter.assets(a, &d));
    ranks.* = old_ranks;
}
test "GGUF Qwen: production tokenizer callback publishes a native image once and keys its identity" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    const source = @import("../families/qwen27/gguf_source.zig");
    const bytes = try @import("../families/qwen27/gguf_fixture.zig").makeTokenized(a);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "m.gguf", .data = bytes });
    const path = try std.fs.path.join(a, &.{ root, "m.gguf" });
    const cache = try std.fs.path.join(a, &.{ root, "images" });
    const Counter = struct {
        count: usize = 0,
        fn run(ctx: ?*anyopaque, alloc: std.mem.Allocator, tensor: gg.Tensor) !@import("../families/qwen27/gguf_repack.zig").Affine {
            const self: *@This() = @ptrCast(@alignCast(ctx.?));
            self.count += 1;
            return (@import("../families/qwen27/gguf_repack.zig").Repacker{}).run(null, alloc, tensor);
        }
    };
    var counter = Counter{};
    const options: source.Options = .{ .repacker = .{ .context = &counter, .run = Counter.run } };
    const native = try source.prepare(t.allocator, t.io, path, cache, options);
    defer t.allocator.free(native);
    try t.expectEqual(@as(usize, 33), counter.count);
    const text = try open(a, native);
    defer text.deinit();
    try t.expectEqualSlices(u32, &.{ 263, 256, 265 }, try text.text().encode(a, "<|endoftext|>ab<|im_end|>", false));
    try t.expectEqualStrings(fixture.chat, text.text().templateSource());
    const warm = try source.prepare(t.allocator, t.io, path, cache, options);
    defer t.allocator.free(warm);
    try t.expectEqualStrings(native, warm);
    try t.expectEqual(@as(usize, 33), counter.count);
    var doc = try gg.Document.parse(t.allocator, bytes, .{});
    defer doc.deinit();
    var changed = options;
    changed.tokenizer.identity = "planted-tokenizer-change";
    try t.expect(!std.mem.eql(u8, &source.key(&doc, options), &source.key(&doc, changed)));
}
fn allocateAssets(alloc: std.mem.Allocator, d: *const gg.Document) !void {
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    _ = try adapter.assets(arena.allocator(), d);
}
test "GGUF Qwen: tokenizer construction releases arena storage on allocation failure" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f = try fixture.make(arena.allocator(), true, true);
    var d = try gg.Document.parse(t.allocator, f.slice(), .{});
    defer d.deinit();
    try t.checkAllAllocationFailures(t.allocator, allocateAssets, .{&d});
}
test "GGUF Qwen: sparse active ids survive unused rows and plain text cannot manufacture controls" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try fixture.make(a, false, false);
    var d = try gg.Document.parse(t.allocator, f.slice(), .{});
    defer d.deinit();
    const tokens = try fixture.words(a);
    var raw = @import("gguf_test_fixture.zig").Fixture{};
    for (tokens[0..fixture.normal_count]) |s| raw.string(s);
    raw.string("[PAD_UNUSED]");
    for (fixture.controls) |s| raw.string(s);
    d.metadata[d.keys.get("tokenizer.ggml.tokens").?].value.array.bytes = raw.slice();
    var types = @import("gguf_test_fixture.zig").Fixture{};
    for (0..fixture.count) |i| types.int(i32, if (i < fixture.normal_count) 1 else if (i == fixture.normal_count) 5 else if (i == fixture.count - 1) 4 else 3);
    d.metadata[d.keys.get("tokenizer.ggml.token_type").?].value.array.bytes = types.slice();
    for ([_][]const u8{ "bos_token_id", "eos_token_id", "padding_token_id" }) |key| {
        const p = &d.metadata[d.keys.get(try std.fmt.allocPrint(a, "tokenizer.ggml.{s}", .{key})).?].value;
        p.uint32 += 1;
    }
    var tok = try parse(a, (try adapter.assets(a, &d)).tokenizer);
    defer tok.deinit();
    try t.expectEqualSlices(u32, &.{ 264, 265, 266, 267 }, try tok.encode(a, "<|endoftext|><|im_start|><|im_end|><think>"));
    try t.expect(tok.specialTokenId("[PAD_UNUSED]") == null);
    const plain = try tok.encodePlain(a, "<|endoftext|><|im_start|><|im_end|><think>");
    for (plain) |id| try t.expect(id < fixture.normal_count);
}
test "GGUF Qwen: token arrays and missing optional or required fields never silently guess" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try fixture.make(a, true, false);
    var d = try gg.Document.parse(t.allocator, f.slice(), .{});
    defer d.deinit();
    const tokens = &d.metadata[d.keys.get("tokenizer.ggml.tokens").?].value.array;
    const old = tokens.*;
    const words = try fixture.words(a);
    for ([_][]const u8{ "a", "☃", "\xff", "" }, [_]anyerror{ error.DuplicateGgufToken, error.BadGgufByteToken, error.BadGgufVocabulary, error.BadGgufVocabulary }) |word, err| {
        var raw = @import("gguf_test_fixture.zig").Fixture{};
        raw.string(word);
        for (words[1..]) |s| raw.string(s);
        tokens.bytes = raw.slice();
        try t.expectError(err, adapter.assets(a, &d));
    }
    tokens.* = old;
    const type_index = d.keys.get("tokenizer.ggml.token_type").?;
    d.metadata[type_index].value.array.len -= 1;
    try t.expectError(error.TokenizerLengthMismatch, adapter.assets(a, &d));
    d.metadata[type_index].value.array.len += 1;
    const bos_index = d.keys.get("tokenizer.ggml.bos_token_id").?;
    _ = d.keys.remove("tokenizer.ggml.bos_token_id");
    try t.expectError(error.MissingGgufSpecialToken, adapter.assets(a, &d));
    try d.keys.put(t.allocator, "tokenizer.ggml.bos_token_id", bos_index);
    _ = d.keys.remove("tokenizer.ggml.token_type");
    try t.expectError(error.MissingGgufTokenizerTypes, adapter.assets(a, &d));
    try d.keys.put(t.allocator, "tokenizer.ggml.token_type", type_index);
    const chat_index = d.keys.get("tokenizer.chat_template").?;
    _ = d.keys.remove("tokenizer.chat_template");
    try t.expect(!(try adapter.assets(a, &d)).config.object.contains("chat_template"));
    try d.keys.put(t.allocator, "tokenizer.chat_template", chat_index);
}
