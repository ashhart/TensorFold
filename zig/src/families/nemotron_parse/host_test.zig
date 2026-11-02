//! Device-free Parse admission and processor checks; the optional checkpoint must already exist.
const std = @import("std");
const core = @import("core");
const family = @import("root.zig");
const c = family.config;
const schema = family.schema;
const processor = @import("processor.zig");
const a = std.testing.allocator;
const io = std.testing.io;

const valid =
    \\{"model_type":"nemotron_parse","legacy_causal_vision_prefix":true,"tie_word_embeddings":true,
    \\"navit_position_embedding_mode":"content_normalized_2d_interpolate","navit_coordinate_frame":"content_normalized",
    \\"navit_packing":true,"max_sequence_length":16384,"num_extra_heads":1,"vision_spatial_dim":1152,
    \\"vision_summary_dim":2304,"decoder_start_token_id":2,"eos_token_id":2,"pad_token_id":1,
    \\"decoder":{"model_type":"nemotron_parse_text","activation_function":"gelu","scale_embedding":true,
    \\"tie_word_embeddings":true,"causal_vision_prefix":true,"add_cross_attention":false,
    \\"d_model":1024,"decoder_layers":10,"decoder_ffn_dim":4096,"vocab_size":72512,
    \\"decoder_attention_heads":16,"self_attention_kv_heads":8,"cross_attention_kv_heads":0},
    \\"encoder":{"version":"c-radio_v4-so400m","patch_size":16,"max_resolution":2048,
    \\"args":{"model":"vit_so400m_patch16_224","cpe_max_size":2048,"register_multiple":10}}}
;

test "Parse config accepts its graph and refuses wrong arithmetic and conflicting aliases" {
    try c.validate(a, valid);
    for ([_]struct { []const u8, []const u8 }{
        .{ "\"d_model\":1024", "\"d_model\":2048" },
        .{ "\"self_attention_kv_heads\":8", "\"self_attention_kv_heads\":16" },
        .{ "\"gelu\"", "\"gelu_new\"" },
        .{ "\"scale_embedding\":true", "\"scale_embedding\":false" },
        .{ "\"max_sequence_length\":16384", "\"max_sequence_length\":32768" },
        .{ "\"num_extra_heads\":1", "\"num_extra_heads\":0" },
        .{ "\"model_type\":\"nemotron_parse\"", "\"model_type\":\"nemotron_h\"" },
        .{ "\"pad_token_id\":1", "\"pad_token_id\":1,\"hidden_size\":2048" },
        .{ "\"pad_token_id\":1", "\"pad_token_id\":1,\"quantization\":{}" },
    }) |change| {
        const bad = try replace(valid, change[0], change[1]);
        defer a.free(bad);
        try std.testing.expectError(error.UnsupportedParseConfig, c.validate(a, bad));
    }
    try std.testing.expectError(error.BadParseConfig, c.validate(a, "[]"));
}

fn replace(text: []const u8, old: []const u8, new: []const u8) ![]u8 {
    const at = std.mem.indexOf(u8, text, old) orelse return error.MissingTestField;
    return std.mem.concat(a, u8, &.{ text[0..at], new, text[at + old.len ..] });
}

const Header = struct {
    arena: std.heap.ArenaAllocator,
    names: core.safetensors.Header = .empty,

    pub fn tensor(h: *Header, name: []const u8, dtype: core.safetensors.DType, shape: []const usize) !void {
        const ar = h.arena.allocator();
        var dims: [core.safetensors.max_rank]usize = @splat(1);
        @memcpy(dims[0..shape.len], shape);
        try h.names.put(ar, try ar.dupe(u8, name), .{ .dtype = dtype, .rank = @intCast(shape.len), .shape = dims, .begin = 0, .end = 0 });
    }
};

test "Parse tensor admission rejects missing, wrong-shape, mixed-dtype and extra tensors" {
    var fixture: Header = .{ .arena = .init(a) };
    defer fixture.arena.deinit();
    try schema.each(&fixture);
    try schema.validate(&fixture.names);
    try std.testing.expectEqual(@as(usize, 510), fixture.names.count());
    const name = "encoder.conv2.weight";
    const original = fixture.names.get(name).?;
    fixture.names.getPtr(name).?.shape[3] = 2; // a 2-by-2 shuffle is a different model
    try std.testing.expectError(error.BadParseTensor, schema.validate(&fixture.names));
    fixture.names.getPtr(name).?.* = original;
    fixture.names.getPtr(name).?.dtype = .bf16;
    try std.testing.expectError(error.BadParseTensor, schema.validate(&fixture.names));
    fixture.names.getPtr(name).?.* = original;
    try fixture.tensor("decoder.embed_positions.weight", .f32, &.{ 16384, 1024 });
    try std.testing.expectError(error.UnexpectedParseTensor, schema.validate(&fixture.names));
    _ = fixture.names.swapRemove("decoder.embed_positions.weight");
    _ = fixture.names.swapRemove(name);
    try std.testing.expectError(error.MissingParseTensor, schema.validate(&fixture.names));
}

test "Parse real checkpoint validates all 510 tensors without GPU initialization" {
    const dir = @import("parse_options").model_dir orelse return error.SkipZigTest;
    var ck = try family.checkpoint.Checkpoint.open(a, io, dir);
    defer ck.close();
    try std.testing.expectEqual(@as(usize, 510), ck.file.names.count());
    const embed = try ck.get("decoder.embed_tokens.weight");
    try std.testing.expect(embed.is(.f32, &.{ c.vocab, c.hidden }));
    try std.testing.expectEqual(@as(usize, 4 * c.vocab * c.hidden), embed.bytes.len);
}

test "Parse geometry keeps small pages and counts prefix slots and context edges" {
    const small = try processor.geometry(17, 65);
    try std.testing.expectEqualDeep(processor.Size{ .h = 17, .w = 65 }, small.content);
    try std.testing.expectEqualDeep(processor.Size{ .h = 32, .w = 128 }, small.padded);
    try std.testing.expectEqual(@as(usize, 5), small.tokens());
    const large = try processor.geometry(4096, 3328);
    try std.testing.expectEqualDeep(processor.Size{ .h = 2048, .w = 1664 }, large.content);
    try std.testing.expectEqual(@as(usize, 3329), large.tokens());
    try large.fits(c.context - 3329, c.context);
    try std.testing.expectError(error.ContextExceeded, large.fits(c.context - 3328, c.context));
    try std.testing.expectError(error.ContextExceeded, large.fits(1, 3329));
    try std.testing.expectError(error.Invalid, processor.geometry(0, 7));
    try std.testing.expectError(error.ImageTooSmall, processor.geometry(8192, 1));
}

test "Parse patches preserve channel-row-column order and white padding" {
    var pixels: [17 * 65 * 3]u8 = undefined;
    for (&pixels, 0..) |*v, i| v.* = @truncate(i * 37);
    var p = try processor.prepareRgb(a, .{ .height = 17, .width = 65, .pixels = &pixels });
    defer p.deinit(a);
    try std.testing.expectEqual(@as(usize, 16 * 768), p.pixels.len);
    // Each channel has its own contiguous 16x16 patch, not interleaved RGB pixels.
    try std.testing.expectEqual((@as(f32, pixels[3]) / 255 - c.mean[0]) / c.deviation[0], p.pixels[1]);
    try std.testing.expectEqual((@as(f32, pixels[1]) / 255 - c.mean[1]) / c.deviation[1], p.pixels[256]);
    const pad = 7 * 768 + 256 + 15; // rightmost patch, row 0, channel 1, padded column
    try std.testing.expectEqual((1 - c.mean[1]) / c.deviation[1], p.pixels[pad]);
    try std.testing.expectError(error.BadPixels, processor.prepareRgb(a, .{ .height = 17, .width = 65, .pixels = pixels[0..3] }));
}

test "Parse processor matches independent Pillow and NumPy fingerprints" {
    const Case = struct { height: usize, width: usize, content_h: usize, content_w: usize, padded_h: usize, padded_w: usize, tokens: usize, sha256: []const u8 };
    const Fixture = struct { source_revision: []const u8, cases: []const Case };
    // Generated by `python tools/zig/nemotron_parse_reference.py processor`.
    const doc = try std.json.parseFromSlice(Fixture, a, @embedFile("fixtures/processor.json"), .{});
    defer doc.deinit();
    for (doc.value.cases) |cell| {
        const pixels = try a.alloc(u8, cell.height * cell.width * 3);
        defer a.free(pixels);
        for (0..cell.height) |y| for (0..cell.width) |x| for (0..3) |ch| {
            pixels[(y * cell.width + x) * 3 + ch] = @truncate(y * 17 + x * 29 + ch * 83);
        };
        var p = try processor.prepareRgb(a, .{ .height = cell.height, .width = cell.width, .pixels = pixels });
        defer p.deinit(a);
        try std.testing.expectEqualDeep(processor.Size{ .h = cell.content_h, .w = cell.content_w }, p.geometry.content);
        try std.testing.expectEqualDeep(processor.Size{ .h = cell.padded_h, .w = cell.padded_w }, p.geometry.padded);
        try std.testing.expectEqual(cell.tokens, p.geometry.tokens());
        var hash: std.crypto.hash.sha2.Sha256 = .init(.{});
        for (p.pixels) |v| {
            var bytes: [4]u8 = undefined;
            std.mem.writeInt(u32, &bytes, @bitCast(v), .little);
            hash.update(&bytes);
        }
        var want_bytes: [32]u8 = undefined;
        const want = try std.fmt.hexToBytes(&want_bytes, cell.sha256);
        try std.testing.expectEqualSlices(u8, want, &hash.finalResult());
    }
}

test "Parse processor config rejects an effective canvas that conflicts with size metadata" {
    const text =
        \\{"navit_packing":true,"do_normalize":true,"do_rescale":true,
        \\"navit_padding_mode":"top_left_white","navit_coordinate_frame":"content_normalized",
        \\"size":{"height":2048,"width":1664},"final_size":[2048,1664],
        \\"image_mean":[0.48145466,0.4578275,0.40821073],"image_std":[0.26862954,0.26130258,0.27577711],
        \\"rescale_factor":0.00392156862745098}
    ;
    try c.validateProcessor(a, text);
    for ([_]struct { []const u8, []const u8 }{
        .{ "[2048,1664]", "[512,512]" },
        .{ "top_left_white", "center_white" },
        .{ "0.48145466", "0.5" },
        .{ "0.00392156862745098", "1.0" },
    }) |change| {
        const bad = try replace(text, change[0], change[1]);
        defer a.free(bad);
        try std.testing.expectError(error.UnsupportedParseConfig, c.validateProcessor(a, bad));
    }
}

test "Parse image identity includes content dimensions when normalized padded pixels are equal" {
    var first: [17 * 65 * 3]u8 = @splat(255);
    var second: [18 * 66 * 3]u8 = @splat(255);
    var p = try processor.prepareRgb(a, .{ .height = 17, .width = 65, .pixels = &first });
    defer p.deinit(a);
    var q = try processor.prepareRgb(a, .{ .height = 18, .width = 66, .pixels = &second });
    defer q.deinit(a);
    try std.testing.expectEqualSlices(f32, p.pixels, q.pixels);
    try std.testing.expect(p.hash != q.hash);
    try std.testing.expectError(error.Oversized, processor.geometry(std.math.maxInt(usize), 7));
}

test "Parse decoded image matches the actual Hugging Face example processor output" {
    const dir = @import("parse_options").reference_dir orelse return error.SkipZigTest;
    const image = try readReference(dir, "input.image", 16 << 20);
    defer a.free(image);
    const expected = try readReference(dir, "input.pixel_values.bin", 64 << 20);
    defer a.free(expected);
    var p = try processor.prepare(io, a, image);
    defer p.deinit(a);
    try std.testing.expectEqual(p.pixels.len * 4, expected.len);
    for (p.pixels, 0..) |v, i| {
        const bits = std.mem.readInt(u32, expected[i * 4 ..][0..4], .little);
        try std.testing.expectEqual(bits, @as(u32, @bitCast(v)));
    }
    const sizes = try readReference(dir, "input.content_sizes.bin", 17);
    defer a.free(sizes);
    try std.testing.expectEqual(@as(usize, 16), sizes.len);
    try std.testing.expectEqual(p.geometry.content.h, std.mem.readInt(u64, sizes[0..8], .little));
    try std.testing.expectEqual(p.geometry.content.w, std.mem.readInt(u64, sizes[8..16], .little));
}

fn readReference(dir: []const u8, name: []const u8, limit: usize) ![]u8 {
    const path = try std.fs.path.join(a, &.{ dir, name });
    defer a.free(path);
    return std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(limit));
}

test "Parse media admission preserves unpadded content and refuses malformed visual prefixes" {
    const Image = struct { pixels: []const f32, gh: u32, gw: u32, at: u32, tokens: u32, content_size: ?[2]u32 };
    const pixels = try a.alloc(f32, 2 * 8 * 768);
    defer a.free(pixels);
    var image: Image = .{ .pixels = pixels, .gh = 2, .gw = 8, .at = 0, .tokens = 5, .content_size = .{ 17, 65 } };
    const media = @import("media.zig");
    const valid_geometry = try media.geometry(image);
    try std.testing.expectEqualDeep(processor.Size{ .h = 17, .w = 65 }, valid_geometry.content);
    image.at = 1;
    try std.testing.expectError(error.ParseImageMustBePrefix, media.geometry(image));
    image.at = 0;
    image.content_size = null;
    try std.testing.expectError(error.ParseContentSizeRequired, media.geometry(image));
    for ([_][2]u32{ .{ 0, 65 }, .{ 33, 65 }, .{ 17, 129 }, .{ 17, 64 } }) |bad| {
        image.content_size = bad;
        try std.testing.expectError(error.BadParseImage, media.geometry(image));
    }
    image.content_size = .{ 17, 65 };
    image.tokens = 4;
    try std.testing.expectError(error.BadParseImage, media.geometry(image));
    image.tokens = 5;
    image.pixels = pixels[0..3];
    try std.testing.expectError(error.BadParseImage, media.geometry(image));
    image.pixels = pixels;
    image.gw = 7;
    try std.testing.expectError(error.BadParseImage, media.geometry(image));
}

test "Parse native tokenizer matches HF task IDs and full serialized OCR output" {
    const model = @import("parse_options").model_dir orelse return error.SkipZigTest;
    const reference = @import("parse_options").reference_dir orelse return error.SkipZigTest;
    const text = try readReference(reference, "manifest.json", 4 << 20);
    defer a.free(text);
    const Manifest = struct { prompt_ids: []u32, output_ids: []u32, text: []const u8 };
    const doc = try std.json.parseFromSlice(Manifest, a, text, .{ .ignore_unknown_fields = true });
    defer doc.deinit();
    var tokenizer = try core.tokenizer.loadTokenizer(io, a, model);
    defer tokenizer.deinit();
    const prompt = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>";
    const ids = try tokenizer.encode(a, prompt);
    defer a.free(ids);
    try std.testing.expectEqualSlices(u32, doc.value.prompt_ids, ids);
    const decoded = try tokenizer.decode(a, doc.value.output_ids, false);
    defer a.free(decoded);
    try std.testing.expectEqualStrings(doc.value.text, decoded);
    try std.testing.expect(tokenizer.isAdded(1)); // the visual prefix uses the checkpoint's added <pad> marker
}
