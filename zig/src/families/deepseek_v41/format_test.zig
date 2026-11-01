//! The two checkpoint formats the family admits: the EXL3 trellises and the release's own FP8 blocks and MXFP4.

const std = @import("std");
const Config = @import("config.zig").Config;
const Format = @import("config.zig").Format;
const Pack = @import("pack.zig").Pack;
const Spec = @import("pack.zig").Spec;
const image = @import("pack.zig").image;
const projection = @import("projection.zig");
const names = @import("names.zig");

const testing = std.testing;

/// The release's own quantization_config, as the release's config.json carries it.
const release_quant =
    \\"quantization_config": {
    \\  "quant_method": "fp8",
    \\  "activation_scheme": "dynamic",
    \\  "weight_block_size": [32, 32],
    \\  "scale_fmt": "ue8m0",
    \\  "expert_dtype": "fp4"
    \\}
;

/// The fixture's quantization_config, from its first quote to its last brace.
fn quant_span(text: []const u8) struct { start: usize, end: usize } {
    const key = "\"quantization_config\"";
    const start = std.mem.indexOf(u8, text, key).?;
    const open = std.mem.indexOfScalarPos(u8, text, start, '{').?;
    var depth: usize = 0;
    var i = open;
    while (i < text.len) : (i += 1) {
        if (text[i] == '{') depth += 1;
        if (text[i] == '}') {
            depth -= 1;
            if (depth == 0) return .{ .start = start, .end = i + 1 };
        }
    }
    unreachable;
}

/// The release's config: the fixture's text_config with the release's own quantization_config.
fn releaseConfig(a: std.mem.Allocator) ![]u8 {
    const good = @embedFile("fixtures/config.json");
    const span = quant_span(good);
    return std.mem.replaceOwned(u8, a, good, good[span.start..span.end], release_quant);
}

test "an EXL3 config still parses and names the EXL3 format" {
    const a = testing.allocator;
    const good = @embedFile("fixtures/config.json");
    const c = try Config.parse(a, good);
    try testing.expectEqual(Format.exl3, c.format);
}

test "the release's own config parses and names the release format" {
    const a = testing.allocator;
    const json = try releaseConfig(a);
    defer a.free(json);
    const c = try Config.parse(a, json);
    try testing.expectEqual(Format.release_fp8, c.format);
    // the release's text_config is the admitted one, key for key
    try testing.expectEqual(@as(u32, 5120), c.hidden);
    try testing.expectEqual(@as(u32, 40), c.layers);
}

test "a release config with 128x128 blocks, a wrong scale format or a wrong expert dtype is refused" {
    const a = testing.allocator;
    const json = try releaseConfig(a);
    defer a.free(json);
    const bad_block = try std.mem.replaceOwned(u8, a, json, "[32, 32]", "[128, 128]");
    defer a.free(bad_block);
    try testing.expectError(error.UnsupportedCheckpoint, Config.parse(a, bad_block));
    const bad_scale = try std.mem.replaceOwned(u8, a, json, "\"ue8m0\"", "\"mxfp8\"");
    defer a.free(bad_scale);
    try testing.expectError(error.UnsupportedCheckpoint, Config.parse(a, bad_scale));
    const bad_expert = try std.mem.replaceOwned(u8, a, json, "\"expert_dtype\": \"fp4\"", "\"expert_dtype\": \"fp8\"");
    defer a.free(bad_expert);
    try testing.expectError(error.UnsupportedCheckpoint, Config.parse(a, bad_expert));
    const bad_method = try std.mem.replaceOwned(u8, a, json, "\"quant_method\": \"fp8\"", "\"quant_method\": \"int8\"");
    defer a.free(bad_method);
    try testing.expectError(error.UnsupportedCheckpoint, Config.parse(a, bad_method));
}

/// A pack of the given tensors, read from a header dump.
fn packOf(a: std.mem.Allocator, specs: []const Spec) !Pack {
    const bytes = try image(a, specs);
    defer a.free(bytes);
    var p = Pack.init(a);
    errdefer p.deinit();
    try p.addHeaderDump("model.safetensors", bytes);
    return p;
}

/// A pack of the given tensors, read from their header alone: the release's tensors are too big to hold.
fn headerPackOf(a: std.mem.Allocator, specs: []const Spec) !Pack {
    const bytes = try @import("pack.zig").headerOnly(a, specs);
    defer a.free(bytes);
    var p = Pack.init(a);
    errdefer p.deinit();
    try p.addHeaderDump("model.safetensors", bytes);
    return p;
}

test "the release's own tensor shapes are admitted as they stand" {
    const a = testing.allocator;
    // the release's own shapes, as RELEASE-LAYOUTS.md measures them from the drafter shards
    var p = try headerPackOf(a, &.{
        // dense FP8: E4M3 with one E8M0 scale per 32x32 block
        .{ .name = "wq_a.weight", .dtype = "F8_E4M3", .shape = &.{ 1280, 5120 } },
        .{ .name = "wq_a.scale", .dtype = "F8_E8M0", .shape = &.{ 40, 160 } },
        .{ .name = "wo_b.weight", .dtype = "F8_E4M3", .shape = &.{ 5120, 8192 } },
        .{ .name = "wo_b.scale", .dtype = "F8_E8M0", .shape = &.{ 160, 256 } },
        .{ .name = "shared.w2.weight", .dtype = "F8_E4M3", .shape = &.{ 5120, 2304 } },
        .{ .name = "shared.w2.scale", .dtype = "F8_E8M0", .shape = &.{ 160, 72 } },
        // routed experts: I8 holding two E2M1 codes a byte, one E8M0 scale per 32 along K
        .{ .name = "w1.weight", .dtype = "I8", .shape = &.{ 2304, 2560 } },
        .{ .name = "w1.scale", .dtype = "F8_E8M0", .shape = &.{ 2304, 160 } },
        .{ .name = "w2.weight", .dtype = "I8", .shape = &.{ 5120, 1152 } },
        .{ .name = "w2.scale", .dtype = "F8_E8M0", .shape = &.{ 5120, 72 } },
    });
    defer p.deinit();
    const wq_a = try projection.read(&p, "wq_a");
    try testing.expect(wq_a == .fp8_block);
    try testing.expectEqual(@as(usize, 5120), wq_a.fp8_block.k);
    try testing.expectEqual(@as(usize, 1280), wq_a.fp8_block.n);
    const wo_b = try projection.read(&p, "wo_b");
    try testing.expect(wo_b == .fp8_block);
    try testing.expectEqual(@as(usize, 8192), wo_b.fp8_block.k);
    const shared = try projection.read(&p, "shared.w2");
    try testing.expect(shared == .fp8_block);
    try testing.expectEqual(@as(usize, 2304), shared.fp8_block.k);
    // an expert's stored width is half its logical K: 2560 bytes hold 5120 codes
    const w1 = try projection.read(&p, "w1");
    try testing.expect(w1 == .mxfp4);
    try testing.expectEqual(@as(usize, 5120), w1.mxfp4.k);
    try testing.expectEqual(@as(usize, 2304), w1.mxfp4.n);
    const w2 = try projection.read(&p, "w2");
    try testing.expect(w2 == .mxfp4);
    try testing.expectEqual(@as(usize, 2304), w2.mxfp4.k);
    try testing.expectEqual(@as(usize, 5120), w2.mxfp4.n);
}

test "the release's dense FP8 blocks and its MXFP4 experts are admitted with their scales" {
    const a = testing.allocator;
    // a 64x96 E4M3 matrix with one E8M0 scale per 32x32 block: 2 x 3
    var p = try packOf(a, &.{
        .{ .name = "dense.weight", .dtype = "F8_E4M3", .shape = &.{ 64, 96 } },
        .{ .name = "dense.scale", .dtype = "F8_E8M0", .shape = &.{ 2, 3 } },
        // an expert's 32x256 logical matrix as I8 [32,128], one E8M0 scale per 32 of K
        .{ .name = "expert.weight", .dtype = "I8", .shape = &.{ 32, 128 } },
        .{ .name = "expert.scale", .dtype = "F8_E8M0", .shape = &.{ 32, 8 } },
    });
    defer p.deinit();
    const dense = try projection.read(&p, "dense");
    try testing.expect(dense == .fp8_block);
    try testing.expectEqual(@as(usize, 96), dense.fp8_block.k);
    try testing.expectEqual(@as(usize, 64), dense.fp8_block.n);
    try testing.expectEqual(p.get("dense.scale").?, dense.fp8_block.scale);
    const expert = try projection.read(&p, "expert");
    try testing.expect(expert == .mxfp4);
    try testing.expectEqual(@as(usize, 256), expert.mxfp4.k);
    try testing.expectEqual(@as(usize, 32), expert.mxfp4.n);
    try testing.expectEqual(p.get("expert.scale").?, expert.mxfp4.scale);
    // the CUDA rank image executes only EXL3 trellises
    try testing.expectError(error.UnsupportedProjection, dense.cudaGroup());
    try testing.expectError(error.UnsupportedProjection, expert.cudaGroup());
}

test "the release's Engram table is scaled a row at a time" {
    const a = testing.allocator;
    var p = try packOf(a, &.{
        .{ .name = "embed.weight", .dtype = "F8_E4M3", .shape = &.{ 64, 256 } },
        .{ .name = "embed.scale", .dtype = "F8_E8M0", .shape = &.{ 64, 8 } },
    });
    defer p.deinit();
    const embed = try projection.read(&p, "embed");
    try testing.expect(embed == .fp8_rows);
    try testing.expectEqual(@as(usize, 256), embed.fp8_rows.k);
    try testing.expectEqual(@as(usize, 64), embed.fp8_rows.n);
    try testing.expectError(error.UnsupportedProjection, embed.cudaGroup());
}

test "a release weight with a bad scale shape or the wrong dtype is refused" {
    const a = testing.allocator;
    // a 64x64 matrix wants 2 x 2 blocked scales or 64 x 2 row scales; 4 x 2 is neither
    var p = try packOf(a, &.{
        .{ .name = "dense.weight", .dtype = "F8_E4M3", .shape = &.{ 64, 64 } },
        .{ .name = "dense.scale", .dtype = "F8_E8M0", .shape = &.{ 4, 2 } },
    });
    defer p.deinit();
    try testing.expectError(error.BadScales, projection.read(&p, "dense"));
    // a missing scale is missing, not bad
    var q = try packOf(a, &.{.{ .name = "dense.weight", .dtype = "F8_E4M3", .shape = &.{ 64, 64 } }});
    defer q.deinit();
    try testing.expectError(error.MissingTensor, projection.read(&q, "dense"));
    // an expert's scales run one per 32 of K, not one per 32x32 block
    var r = try packOf(a, &.{
        .{ .name = "expert.weight", .dtype = "I8", .shape = &.{ 32, 128 } },
        .{ .name = "expert.scale", .dtype = "F8_E8M0", .shape = &.{ 1, 8 } },
    });
    defer r.deinit();
    try testing.expectError(error.BadScales, projection.read(&r, "expert"));
    // an odd stored width cannot hold two codes a byte
    var s = try packOf(a, &.{
        .{ .name = "expert.weight", .dtype = "I8", .shape = &.{ 32, 7 } },
        .{ .name = "expert.scale", .dtype = "F8_E8M0", .shape = &.{ 32, 1 } },
    });
    defer s.deinit();
    try testing.expectError(error.BadScales, projection.read(&s, "expert"));
    // a dtype the release does not use is refused
    var t = try packOf(a, &.{
        .{ .name = "dense.weight", .dtype = "F32", .shape = &.{ 64, 64 } },
        .{ .name = "dense.scale", .dtype = "F8_E8M0", .shape = &.{ 2, 2 } },
    });
    defer t.deinit();
    try testing.expectError(error.UnsupportedProjection, projection.read(&t, "dense"));
}

test "the release's names are the ones the release stores" {
    var buf: [128]u8 = undefined;
    try testing.expectEqualStrings("layers.1.attn.wq_a.weight", try names.releaseName("layers.1", "attn.wq_a", &buf));
    try testing.expectEqualStrings("layers.1.ffn.experts.3.w2.weight", try names.releaseExpertName("layers.1", 3, .w2, &buf));
    try testing.expectEqualStrings("layers.14.engram.embed.weight", try names.releaseName("layers.14", "engram.embed", &buf));
    const c: Config = .{};
    const r = try names.releaseBlocks(&c, 14, true);
    var seen_embed = false;
    var seen_wkv = false;
    for (r.items()) |s| {
        if (s.proj == .engram_embed) seen_embed = true;
        if (s.proj == .engram_wkv) seen_wkv = true;
    }
    try testing.expect(seen_embed and seen_wkv);
    // a layer without Engram carries neither
    const r2 = try names.releaseBlocks(&c, 2, true);
    for (r2.items()) |s| try testing.expect(s.proj != .engram_embed and s.proj != .engram_wkv);
}
