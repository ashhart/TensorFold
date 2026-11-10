//! MiniMax-H3-Image's prompt rules against hand-worked cases, each failing without its rule.
const std = @import("std");
const pp = @import("picture_prompt.zig");
const picture = @import("picture");
const testing = std.testing;

const marks: pp.Marks = .{ .start = 151652, .image = 151655, .end = 151653 };

/// One id per segment: the hash of the whole text, so a segment merged with its neighbour would show.
const Whole = struct {
    pub fn encode(_: Whole, a: std.mem.Allocator, text: []const u8) ![]u32 {
        const out = try a.alloc(u32, 1);
        out[0] = @truncate(std.hash.Wyhash.hash(0, text));
        return out;
    }
};

/// One id per byte.
const Bytes = struct {
    pub fn encode(_: Bytes, a: std.mem.Allocator, text: []const u8) ![]u32 {
        const out = try a.alloc(u32, text.len);
        for (out, text) |*o, c| o.* = c;
        return out;
    }
};

fn hash(text: []const u8) u32 {
    return @truncate(std.hash.Wyhash.hash(0, text));
}

test "h3 prompt: t2i's ids are the prompt's alone, every row tagged text" {
    const a = testing.allocator;
    var p = try pp.present(a, Bytes{}, marks, &.{}, "A red fox");
    defer p.deinit(a);
    try testing.expectEqualSlices(u32, &.{ 'A', ' ', 'r', 'e', 'd', ' ', 'f', 'o', 'x' }, p.ids);
    for (p.tags) |t| try testing.expectEqual(pp.tag_text, t);
    try testing.expectError(error.H3EmptyPrompt, pp.present(a, Bytes{}, marks, &.{}, ""));
}

test "h3 prompt: each reference's label and the prompt tokenised alone, vision blocks between, tags 0 inside" {
    const a = testing.allocator;
    var p = try pp.present(a, Whole{}, marks, &.{ 2, 3 }, "Put <Picture 1> on <Picture 2>");
    defer p.deinit(a);
    const s = marks.start;
    const i = marks.image;
    const e = marks.end;
    try testing.expectEqualSlices(u32, &.{ hash("Picture 1: "), s, i, i, e, hash("Picture 2: "), s, i, i, i, e, hash("Put <Picture 1> on <Picture 2>") }, p.ids);
    try testing.expectEqualSlices(i32, &.{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1 }, p.tags);
    try testing.expectEqualSlices(u32, &.{ 2, 7 }, p.at);
    try testing.expectError(error.H3BadVisionGrid, pp.present(a, Whole{}, marks, &.{0}, "x"));
}

test "h3 prompt: the image pads' rows, each reference's run in order" {
    const a = testing.allocator;
    var p = try pp.present(a, Whole{}, marks, &.{ 2, 3 }, "x");
    defer p.deinit(a);
    const rows = try pp.pads(a, p, marks, &.{ 2, 3 });
    defer a.free(rows);
    try testing.expectEqualSlices(u32, &.{ 2, 3, 7, 8, 9 }, rows);
    try testing.expectError(error.H3BadVisionGrid, pp.pads(a, p, marks, &.{ 3, 3 }));
}

test "h3 prompt: each segment cut to its first 4,096 tokens on its own" {
    const a = testing.allocator;
    const long = try a.alloc(u8, 5000);
    defer a.free(long);
    for (long, 0..) |*c, k| c.* = @intCast('a' + k % 26);
    var p = try pp.present(a, Bytes{}, marks, &.{1}, long);
    defer p.deinit(a);
    const label = "Picture 1: ".len;
    try testing.expectEqual(label + 3 + 4096, p.ids.len);
    try testing.expectEqualSlices(u32, &.{ 'P', 'i', 'c' }, p.ids[0..3]);
    try testing.expectEqual(@as(u32, 'a' + 4095 % 26), p.ids[p.ids.len - 1]);
    try testing.expectEqual(p.ids.len, p.tags.len);
}

test "h3 prompt: a reference's vision tokens are grid_h x grid_w / 4" {
    try testing.expectEqual(@as(usize, 1024), try pp.visionTokens(64, 64));
    try testing.expectEqual(@as(usize, 6), try pp.visionTokens(4, 6));
    try testing.expectEqual(@as(usize, 2 * 51), try pp.visionTokens(4, 102));
    try testing.expectError(error.H3BadVisionGrid, pp.visionTokens(3, 4));
    try testing.expectError(error.H3BadVisionGrid, pp.visionTokens(0, 4));
}

test "h3 prompt: positions count up through text and sit on a reference's merged grid" {
    const a = testing.allocator;
    var t2i = try pp.present(a, Bytes{}, marks, &.{}, "abc");
    defer t2i.deinit(a);
    const flat = try pp.positions(a, t2i, marks, &.{});
    defer a.free(flat);
    try testing.expectEqualSlices([3]i32, &.{ .{ 0, 0, 0 }, .{ 1, 1, 1 }, .{ 2, 2, 2 } }, flat);
    var edit = try pp.present(a, Whole{}, marks, &.{6}, "x");
    defer edit.deinit(a);
    const pos = try pp.positions(a, edit, marks, &.{.{ 4, 6 }});
    defer a.free(pos);
    // label 0, start 1, the 2 x 3 merged grid at base 2, end at 2 + max(2, 3), the prompt after it
    const want = [_][3]i32{ .{ 0, 0, 0 }, .{ 1, 1, 1 }, .{ 2, 2, 2 }, .{ 2, 2, 3 }, .{ 2, 2, 4 }, .{ 2, 3, 2 }, .{ 2, 3, 3 }, .{ 2, 3, 4 }, .{ 5, 5, 5 }, .{ 6, 6, 6 } };
    try testing.expectEqualSlices([3]i32, &want, pos);
    try testing.expectError(error.H3BadVisionGrid, pp.positions(a, edit, marks, &.{.{ 4, 4 }}));
}

test "h3 prompt: rotary frequency i turns by its interleaved axis' position, rounded to bf16" {
    const a = testing.allocator;
    const pos = [_][3]i32{ .{ 3, 3, 3 }, .{ 5, 7, 9 } };
    const cos = try a.alloc(f32, 2 * 64);
    defer a.free(cos);
    const sin = try a.alloc(f32, 2 * 64);
    defer a.free(sin);
    pp.rotary(cos, sin, &pos, 5e6, .{ 24, 20, 20 });
    const want = struct {
        fn at(p: i32, i: usize, s: bool) f32 {
            const freq: f32 = 1.0 / std.math.pow(f32, 5e6, @as(f32, @floatFromInt(2 * i)) / 128.0);
            const angle: f64 = @as(f32, @floatFromInt(p)) * freq;
            return @import("bf16_round").round32(@floatCast(if (s) @sin(angle) else @cos(angle)));
        }
    };
    for (0..64) |i| try testing.expectEqual(want.at(3, i, false), cos[i]);
    // frequencies 1, 4, ... take h and 2, 5, ... take w below 60; the rest, and 0, 3, ..., take t
    const axis = [_]struct { usize, i32 }{ .{ 0, 5 }, .{ 1, 7 }, .{ 2, 9 }, .{ 3, 5 }, .{ 58, 7 }, .{ 59, 9 }, .{ 60, 5 }, .{ 61, 5 }, .{ 62, 5 }, .{ 63, 5 } };
    for (axis) |c| {
        try testing.expectEqual(want.at(c[1], c[0], false), cos[64 + c[0]]);
        try testing.expectEqual(want.at(c[1], c[0], true), sin[64 + c[0]]);
    }
    try testing.expect(cos[64 + 1] != want.at(5, 1, false));
    for (cos) |v| try testing.expectEqual(v, @import("bf16_round").round32(v));
}

test "h3 prompt: the Qwen bound keeps a 1,024 short edge and snaps scaled sides to 32, ties to even" {
    const S = pp.Size;
    try testing.expectEqual(S{ .width = 1024, .height = 1536 }, try pp.qwenBound(512, 768));
    try testing.expectEqual(S{ .width = 1376, .height = 1024 }, try pp.qwenBound(2048, 1536));
    try testing.expectEqual(S{ .width = 1024, .height = 1050 }, try pp.qwenBound(1024, 1050));
    // 2080 x 0.5 = 1040 = 32 x 32.5 rounds down to even; 2144 x 0.5 = 33.5 x 32 rounds up to even
    try testing.expectEqual(S{ .width = 1024, .height = 1024 }, try pp.qwenBound(2048, 2080));
    try testing.expectEqual(S{ .width = 1088, .height = 1024 }, try pp.qwenBound(2144, 2048));
    try testing.expectError(error.H3ReferenceTooSmall, pp.qwenBound(31, 1024));
    // the processor's smart resize then snaps a kept odd side: 1050 to 1056
    try testing.expectEqual(S{ .width = 1024, .height = 1056 }, try pp.qwenProcessed(.{ .width = 1024, .height = 1050 }, 65536, 16777216));
    try testing.expectEqual(S{ .width = 1376, .height = 1024 }, try pp.qwenProcessed(.{ .width = 1376, .height = 1024 }, 65536, 16777216));
}

test "h3 prompt: the Qwen copy keeps a 1,024 short edge's pixels and resamples others with Pillow's LANCZOS" {
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(31);
    const kept_px = try a.alloc(u8, 1024 * 1040 * 3);
    defer a.free(kept_px);
    prng.random().bytes(kept_px);
    var kept = try pp.qwenCopy(a, .{ .width = 1040, .height = 1024, .pixels = kept_px });
    defer kept.deinit(a);
    try testing.expectEqual(@as(usize, 1040), kept.width);
    try testing.expectEqualSlices(u8, kept_px, kept.pixels);
    const small = try a.alloc(u8, 40 * 48 * 3);
    defer a.free(small);
    prng.random().bytes(small);
    var grown = try pp.qwenCopy(a, .{ .width = 40, .height = 48, .pixels = small });
    defer grown.deinit(a);
    try testing.expectEqual(pp.Size{ .width = 1024, .height = 1216 }, pp.Size{ .width = grown.width, .height = grown.height });
    const view = picture.resample.View.of(small, 40, 48);
    const lanczos = try picture.resample.resizeWith(a, .pillow, .lanczos, view, 1024, 1216);
    defer a.free(lanczos);
    const cubic = try picture.resample.resizeWith(a, .pillow, .bicubic, view, 1024, 1216);
    defer a.free(cubic);
    try testing.expectEqualSlices(u8, lanczos, grown.pixels);
    try testing.expect(!std.mem.eql(u8, cubic, grown.pixels));
}

const encoder_config =
    \\{"model_type": "qwen3_vl", "image_token_id": 151655, "vision_start_token_id": 151652, "vision_end_token_id": 151653,
    \\ "text_config": {"attention_bias": false, "head_dim": 128, "hidden_act": "silu", "hidden_size": 5120,
    \\  "intermediate_size": 25600, "num_attention_heads": 64, "num_hidden_layers": 64, "num_key_value_heads": 8,
    \\  "rms_norm_eps": 1e-06, "rope_scaling": {"mrope_interleaved": true, "mrope_section": [24, 20, 20],
    \\  "rope_type": "default"}, "rope_theta": 5000000, "vocab_size": 151936},
    \\ "vision_config": {"out_hidden_size": 5120, "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2}}
;

fn refused(comptime want: anyerror, json: []const u8, from: []const u8, to: []const u8, check: anytype) !void {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const changed = try std.mem.replaceOwned(u8, a, json, from, to);
    try testing.expect(!std.mem.eql(u8, changed, json));
    try testing.expectError(want, check(a, changed));
}

test "h3 prompt: only the released Qwen3-VL text encoder's config passes" {
    const log_level = testing.log_level;
    testing.log_level = .err;
    defer testing.log_level = log_level;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const e = try pp.checkEncoder(arena.allocator(), encoder_config);
    try testing.expectEqual(pp.released_text, e.text);
    try testing.expectEqual(marks, e.marks);
    const no = error.H3UnsupportedPromptModel;
    try refused(no, encoder_config, "\"qwen3_vl\"", "\"qwen3_5\"", pp.checkEncoder);
    try refused(no, encoder_config, "\"num_key_value_heads\": 8", "\"num_key_value_heads\": 4", pp.checkEncoder);
    try refused(no, encoder_config, "1e-06", "1e-05", pp.checkEncoder);
    try refused(no, encoder_config, "5000000", "1000000", pp.checkEncoder);
    try refused(no, encoder_config, "[24, 20, 20]", "[16, 24, 24]", pp.checkEncoder);
    try refused(no, encoder_config, "\"mrope_interleaved\": true", "\"mrope_interleaved\": false", pp.checkEncoder);
    try refused(no, encoder_config, "\"attention_bias\": false", "\"attention_bias\": true", pp.checkEncoder);
    try refused(no, encoder_config, "\"spatial_merge_size\": 2", "\"spatial_merge_size\": 4", pp.checkEncoder);
    try refused(error.H3BadPromptConfig, encoder_config, "\"image_token_id\"", "\"video_token_id\"", pp.checkEncoder);
}

const transformer_config =
    \\{"hidden_size": 5376, "num_layers": 50, "token_refiner_num_layers": 2, "num_attention_heads": 56,
    \\ "attention_head_dim": 128, "ffn_hidden_size": 14336, "text_dim": 5120, "norm_eps": 1e-05,
    \\ "qk_norm_eps": 1e-05, "final_norm_eps": 1e-05, "audio_enabled": false}
;

test "h3 prompt: the transformer's text front end at the released geometry, its epsilons read" {
    const log_level = testing.log_level;
    testing.log_level = .err;
    defer testing.log_level = log_level;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testing.expectEqual(pp.released_refiner, try pp.checkRefiner(arena.allocator(), transformer_config));
    const other = try std.mem.replaceOwned(u8, arena.allocator(), transformer_config, "\"qk_norm_eps\": 1e-05", "\"qk_norm_eps\": 1e-06");
    try testing.expectEqual(@as(f32, 1e-6), (try pp.checkRefiner(arena.allocator(), other)).qk_eps);
    const no = error.H3UnsupportedPromptModel;
    try refused(no, transformer_config, "\"text_dim\": 5120", "\"text_dim\": 4096", pp.checkRefiner);
    try refused(no, transformer_config, "\"token_refiner_num_layers\": 2", "\"token_refiner_num_layers\": 3", pp.checkRefiner);
    try refused(no, transformer_config, "\"ffn_hidden_size\": 14336", "\"ffn_hidden_size\": 14335", pp.checkRefiner);
    try refused(error.H3BadPromptConfig, transformer_config, "\"final_norm_eps\": 1e-05", "\"final_norm_eps\": 0", pp.checkRefiner);
}

test "h3 prompt: the text model's tensor prefix, the vision tower's names left out" {
    const names = [_][]const u8{ "model.visual.blocks.0.norm1.weight", "model.visual.layers.0.input_layernorm.weight", "lm_head.weight", "model.language_model.layers.0.input_layernorm.weight", "model.language_model.layers.1.input_layernorm.weight" };
    try testing.expectEqualStrings("model.language_model.", try pp.textPrefix(&names));
    try testing.expectEqualStrings("", try pp.textPrefix(&.{"layers.0.input_layernorm.weight"}));
    try testing.expectError(error.H3NoTextModel, pp.textPrefix(names[0..3]));
    try testing.expectError(error.H3AmbiguousTextModel, pp.textPrefix(&.{ "a.layers.0.input_layernorm.weight", "b.layers.0.input_layernorm.weight" }));
}

test "h3 prompt: the tensor tables follow their structs' fields, at the released shapes" {
    const layer = pp.layerTensors(pp.released_text);
    inline for (comptime std.meta.fieldNames(pp.Layer(u8)), 0..) |f, i| try testing.expectEqualStrings(f, layer[i][0]);
    const block = pp.blockTensors(pp.released_refiner);
    inline for (comptime std.meta.fieldNames(pp.Block(u8)), 0..) |f, i| try testing.expectEqualStrings(f, block[i][0]);
    try testing.expectEqual(pp.Shape{ 8192, 5120 }, layer[2][2]);
    try testing.expectEqual(pp.Shape{ 1024, 5120 }, layer[3][2]);
    try testing.expectEqual(pp.Shape{ 5120, 25600 }, layer[10][2]);
    try testing.expectEqual(pp.Shape{ 21504, 5376 }, block[2][2]);
    try testing.expectEqual(pp.Shape{ 5376, 7168 }, block[5][2]);
    try testing.expectEqual(pp.Shape{ 28672, 5376 }, block[6][2]);
    try testing.expectEqual(pp.Shape{ 5376, 5120 }, pp.frontTensors(pp.released_refiner)[0][2]);
}
