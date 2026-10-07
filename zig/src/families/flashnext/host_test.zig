//! Host-only contracts exercise strict text configuration, affine six-bit packing and borrowed-view shapes.
const std = @import("std");
const posix = std.posix;
const family = @import("flashnext.zig");
const st = @import("../../core/safetensors.zig");

test "pinned public FlashNext config preserves MTP EOS and indexer declarations" {
    const bytes = @embedFile("fixtures/config.json");
    const expected = "f5574e3431b94b6297e0490778e6f528928593e2b0b545bc9da09f951dc981db";
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &hash, .{});
    try std.testing.expectEqualStrings(expected, &std.fmt.bytesToHex(hash, .lower));
    var c = try family.config.parse(std.testing.allocator, bytes);
    defer c.deinit();
    try std.testing.expectEqual(@as(usize, 2560), c.hidden);
    try std.testing.expectEqual(@as(usize, 10_240), try c.wide());
    try std.testing.expectEqual(@as(usize, 36), std.mem.count(family.config.Kind, c.kinds[0..c.layers], &.{.linear_attention}));
    try std.testing.expectEqual(@as(usize, 10_240), try c.convWidth());
    try std.testing.expectEqual(@as(usize, 1), c.index_kv_heads);
    try std.testing.expectEqual(@as(usize, 1), c.mtp.layers);
    try std.testing.expectEqual(family.config.Kind.sparse_attention, c.mtp.kinds[0]);
    try std.testing.expect(c.mtp.hybrid and !c.mtp.dedicated_embeddings);
    try std.testing.expect(c.mtp.source_layer == null);
    try std.testing.expectEqualSlices(u32, &.{ 248046, 248044 }, c.eos[0..c.eos_count]);
    try std.testing.expectEqualSlices(usize, &.{ 11, 11, 10 }, &c.mrope_section);
    try std.testing.expect(c.mrope_interleaved);
    try std.testing.expectEqual(@as(u8, 6), c.global_affine.bits);
    try std.testing.expectEqual(@as(usize, 32), c.global_affine.group);
}

test "pinned public MTP and rotary metadata cannot be silently discarded" {
    const bytes = @embedFile("fixtures/config.json");
    for ([_]struct { old: []const u8, replacement: []const u8 }{
        .{ .old = "\"num_hidden_layers\": 1,", .replacement = "\"num_hidden_layers\": 17," },
        .{ .old = "\"indexer_kv_heads\": 1,", .replacement = "\"indexer_kv_heads\": 3," },
        .{ .old = "\"type\": \"default\"", .replacement = "\"type\": \"unsupported\"" },
    }) |change| {
        const changed = try std.mem.replaceOwned(u8, std.testing.allocator, bytes, change.old, change.replacement);
        defer std.testing.allocator.free(changed);
        if (family.config.parse(std.testing.allocator, changed)) |valid| {
            var c = valid;
            c.deinit();
            return error.InvalidConfigAccepted;
        } else |_| {}
    }
}

test "public index admits every configured text MTP and PLE tensor name without claiming headers" {
    const a = std.testing.allocator;
    var c = try family.config.parse(a, @embedFile("fixtures/config.json"));
    defer c.deinit();
    const bytes = @embedFile("fixtures/index.json");
    const inventory = try family.index.admit(a, bytes, &c);
    try std.testing.expectEqual(@as(usize, 3747), inventory.tensor_names);
    try std.testing.expectEqual(@as(usize, 30), inventory.shards);
    try std.testing.expect(inventory.required_names > 3300 and !inventory.headers_verified);
    const dotted = try std.mem.replaceOwned(u8, a, bytes, "ngram_embedding.shard_", "ngram_embedding.shards.");
    defer a.free(dotted);
    try std.testing.expectEqual(inventory, try family.index.admit(a, dotted, &c));
    for ([_]struct { old: []const u8, replacement: []const u8, err: anyerror }{
        .{ .old = "\"language_model.lm_head.weight\"", .replacement = "\"missing.weight\"", .err = error.MissingFlashTensor },
        .{ .old = "\"language_model.model.layers.0.linear_attn.in_proj_qkv.scales\"", .replacement = "\"missing.scales\"", .err = error.IncompleteFlashAffine },
        .{ .old = "model-00001-of-00030.safetensors", .replacement = "../model-00001-of-00030.safetensors", .err = error.UnsafeFlashShard },
        .{ .old = "ngram_embedding.shard_0.", .replacement = "ngram_embedding.shards.0.", .err = error.MissingFlashTensor },
    }) |change| {
        const altered = try std.mem.replaceOwned(u8, a, bytes, change.old, change.replacement);
        defer a.free(altered);
        try std.testing.expectError(change.err, family.index.admit(a, altered, &c));
    }
}

fn spellingOf(a: std.mem.Allocator, bytes: []const u8) ![]const u8 {
    const p = try std.json.parseFromSlice(std.json.Value, a, bytes, .{});
    defer p.deinit();
    return family.index.ngramSpelling(p.value.object.get("weight_map").?.object);
}

test "n-gram tables load under the spelling their index lists" {
    const a = std.testing.allocator;
    const bytes = @embedFile("fixtures/index.json");
    try std.testing.expectEqualStrings("shard_", try spellingOf(a, bytes));
    const dotted = try std.mem.replaceOwned(u8, a, bytes, "ngram_embedding.shard_", "ngram_embedding.shards.");
    defer a.free(dotted);
    try std.testing.expectEqualStrings("shards.", try spellingOf(a, dotted));
}

const toy =
    \\{"model_type":"qwen4_exp","hidden_size":64,"vocab_size":128,"num_hidden_layers":2,
    \\"layer_types":["linear_attention","full_attention"],"rms_norm_eps":0.00001,
    \\"num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,
    \\"rope_parameters":{"rope_theta":10000000,"partial_rotary_factor":0.5},
    \\"linear_num_key_heads":2,"linear_num_value_heads":4,"linear_key_head_dim":16,
    \\"linear_value_head_dim":16,"linear_conv_kernel_dim":4,"num_experts":8,"num_experts_per_tok":2,
    \\"moe_intermediate_size":32,"shared_expert_intermediate_size":64,"hc_count":4,"hc_lowrank":8,
    \\"ple_embed_dim":32,"heads_per_ngram":2,"ngram_size":3,"ple_layer_ids":[1,1],"eos_token_id":[127],
    \\"quantization":{"bits":6,"group_size":32,"model.layers.0.raw":false,
    \\"language_model.model.layers.0.override":{"group_size":64},"layers.0.inherit":true}}
;

test "FlashNext derived dimensions retain independent HC attention and recurrence widths" {
    var c = try family.config.parse(std.testing.allocator, toy);
    defer c.deinit();
    try std.testing.expectEqual(@as(usize, 256), try c.wide());
    try std.testing.expectEqual(@as(usize, 128), try c.convWidth());
    try std.testing.expectEqual(@as(usize, 4), try c.ngramHeadCount());
    try std.testing.expectEqual(@as(usize, 1), c.ple_count);
    try std.testing.expectEqual(family.config.Kind.sparse_attention, try c.kind(1));
    try std.testing.expectError(error.LayerOutOfBounds, c.kind(2));
    try std.testing.expectEqual(@as(usize, 8), c.rotary_dim);
    try std.testing.expectEqual(@as(u32, 127), c.eos[0]);
}

test "module quantization preserves false true and dictionary-default override semantics" {
    var c = try family.config.parse(std.testing.allocator, toy);
    defer c.deinit();
    try std.testing.expect((try c.quantization("language_model.model.layers.0.raw.weight")) == null);
    const override = (try c.quantization("model.layers.0.override")).?;
    try std.testing.expectEqual(@as(u8, 4), override.bits);
    try std.testing.expectEqual(@as(usize, 64), override.group);
    try std.testing.expectEqual(@as(u8, 6), (try c.quantization("layers.0.inherit")).?.bits);
    try std.testing.expectEqual(@as(usize, 32), (try c.quantization("unknown.projection")).?.group);
    try std.testing.expectError(error.UnsupportedFlashAffineKernel, override.checkFlashKernel());
    try c.global_affine.checkFlashKernel();
}

test "malformed FlashNext configuration fields are refused" {
    for ([_]struct { old: []const u8, replacement: []const u8 }{
        .{ .old = "\"num_key_value_heads\":2", .replacement = "\"num_key_value_heads\":3" },
        .{ .old = "\"linear_conv_kernel_dim\":4", .replacement = "\"linear_conv_kernel_dim\":1" },
        .{ .old = "\"hc_count\":4", .replacement = "\"hc_count\":0" },
        .{ .old = "\"ngram_size\":3", .replacement = "\"ngram_size\":1" },
        .{ .old = "\"ple_layer_ids\":[1,1]", .replacement = "\"ple_layer_ids\":[3]" },
    }) |change| {
        const changed = try std.mem.replaceOwned(u8, std.testing.allocator, toy, change.old, change.replacement);
        defer std.testing.allocator.free(changed);
        if (family.config.parse(std.testing.allocator, changed)) |valid| {
            var c = valid;
            c.deinit();
            return error.InvalidConfigAccepted;
        } else |_| {}
    }
}

test "six-bit codes cross word boundaries without changing neighboring codes" {
    var words: [6]u32 = @splat(0);
    for (0..32) |i| try family.affine.setCode6(&words, i, @intCast((i * 13 + 7) % 64));
    for (0..32) |i| try std.testing.expectEqual(@as(u8, @intCast((i * 13 + 7) % 64)), try family.affine.code6(&words, i));
    for ([_]usize{ 5, 10, 21, 26 }) |i| {
        const previous = try family.affine.code6(&words, i - 1);
        const next = try family.affine.code6(&words, i + 1);
        try family.affine.setCode6(&words, i, 63);
        try std.testing.expectEqual(previous, try family.affine.code6(&words, i - 1));
        try std.testing.expectEqual(next, try family.affine.code6(&words, i + 1));
    }
    try std.testing.expectError(error.PackedCodeOutOfBounds, family.affine.code6(&words, 32));
    try std.testing.expectError(error.PackedCodeOutOfBounds, family.affine.setCode6(&words, 0, 64));
    for (0..32) |index| {
        var reference: u8 = 0;
        for (0..6) |bit| {
            const at = index * 6 + bit;
            const set = (words[at / 32] >> @as(u5, @intCast(at % 32))) & 1;
            reference |= @as(u8, @intCast(set)) << @as(u3, @intCast(bit));
        }
        try std.testing.expectEqual(reference, try family.affine.code6(&words, index));
    }
}

fn tensor(dtype: st.DType, dimensions: []const usize) st.Entry {
    var shape: [4]usize = @splat(1);
    @memcpy(shape[0..dimensions.len], dimensions);
    var count: usize = dtype.size();
    for (dimensions) |n| count *= n;
    return .{ .dtype = dtype, .rank = @intCast(dimensions.len), .shape = shape, .begin = 0, .end = count };
}

test "affine metadata validates matrix and expert stack storage without tensor payloads" {
    const spec = try family.affine.Spec.init(6, 32);
    const plain = try family.affine.matrix(tensor(.u32, &.{ 8, 12 }), tensor(.bf16, &.{ 8, 2 }), tensor(.bf16, &.{ 8, 2 }), spec);
    try std.testing.expectEqual(@as(usize, 64), plain.k);
    const expert = try family.affine.matrix(tensor(.u32, &.{ 4, 8, 12 }), tensor(.f32, &.{ 4, 8, 2 }), tensor(.f32, &.{ 4, 8, 2 }), spec);
    try std.testing.expectEqual(@as(usize, 4), expert.experts);
    try std.testing.expectEqual(st.DType.f32, expert.dtype);
    try std.testing.expectError(error.InvalidAffineShape, family.affine.matrix(tensor(.u32, &.{ 8, 8 }), tensor(.bf16, &.{ 8, 2 }), tensor(.bf16, &.{ 8, 2 }), spec));
    try std.testing.expectError(error.InvalidAffineMetadataPrecision, family.affine.matrix(tensor(.u32, &.{ 8, 12 }), tensor(.bf16, &.{ 8, 2 }), tensor(.f32, &.{ 8, 2 }), spec));
    var truncated = tensor(.u32, &.{ 8, 12 });
    truncated.end -= 4;
    try std.testing.expectError(error.InvalidAffineByteCount, family.affine.matrix(truncated, tensor(.bf16, &.{ 8, 2 }), tensor(.bf16, &.{ 8, 2 }), spec));
}

test "affine bit group and dimension admission refuses unsupported geometry" {
    try std.testing.expectError(error.UnsupportedAffineBits, family.affine.Spec.init(7, 32));
    try std.testing.expectError(error.UnsupportedAffineGroup, family.affine.Spec.init(6, 16));
    const spec = try family.affine.Spec.init(6, 32);
    try std.testing.expectError(error.InvalidAffineWidth, spec.words(48));
    try std.testing.expectError(error.Overflow, spec.words(std.math.maxInt(usize) - 31));
}

test "the PLE reference values derive from the pinned config as the Python embedding does" {
    const bytes = @embedFile("fixtures/config.json");
    var c = try family.config.parse(std.testing.allocator, bytes);
    defer c.deinit();
    const ref = try family.config.pleRef(&c);
    try std.testing.expectEqual(@as(i64, 248044), ref.eos);
    try std.testing.expectEqualSlices(i64, &.{ 23703573157769, 20109073645365, 8052911324071 }, &ref.multipliers);
    try std.testing.expectEqualSlices(i64, &.{
        20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
        20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171,
    }, &ref.sizes);
    try std.testing.expectEqualSlices(i64, &.{
        0,         20000003,  40000026,  60000059,  80000106,  100000165, 120000228, 140000297,
        160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275,
    }, &ref.offsets);
    const scale = 1.0 / std.math.sqrt(@as(f64, @floatFromInt(c.head_dim)));
    try std.testing.expectEqual(@as(f64, 0.0625), scale);
}

test "the checked-in role table resolves every width serve asks for, with its source checked in" {
    const table = @import("roles_gen.zig");
    const sources = @import("kernel_sources").flashnext_gen;
    try std.testing.expect(table.entries.len > 0);
    // every entry's file and function must exist in the embedded sources, with the kernel inside it
    for (table.entries) |e| {
        var found = false;
        for (sources.sources) |s| {
            if (std.mem.eql(u8, s.name, e.file)) {
                found = true;
                try std.testing.expect(std.mem.indexOf(u8, s.text, e.function) != null);
            }
        }
        try std.testing.expect(found);
    }
    // every stem the table carries resolves at every width serve asks for, 7, 9 and 10-16 included
    var stems: std.StringArrayHashMapUnmanaged([16]bool) = .empty;
    defer stems.deinit(std.testing.allocator);
    for (table.entries) |e| {
        const at = std.mem.lastIndexOf(u8, e.site, "|") orelse return error.BadSite;
        const w = try std.fmt.parseInt(u16, e.site[at + 1 ..], 10);
        if (w == 0 or w > 16) return error.BadWidth;
        const gop = try stems.getOrPut(std.testing.allocator, e.site[0..at]);
        if (!gop.found_existing) gop.value_ptr.* = @splat(false);
        gop.value_ptr.*[w - 1] = true;
    }
    for (stems.values()) |set| {
        for (0..16) |w| try std.testing.expect(set[w]);
    }
    // The indexer trio is in the table, so block selection binds those kernels directly when there is no dump.
    const trio = [_][]const u8{ "q4_idx_pool_", "q4_idx_scores_", "q4_idx_select_" };
    for (trio) |t| {
        var hit = false;
        for (table.entries) |e| {
            if (std.mem.indexOf(u8, e.function, t) != null) hit = true;
        }
        try std.testing.expect(hit);
    }
    // Every checked-in entry names an embedded source, so load never opens a recorded path.
    for (table.entries) |e| {
        try std.testing.expect(std.mem.indexOfScalar(u8, e.file, '/') == null);
    }
    // Prompt.init patches the checked-in q4_gdn@gdn|8 text in place: its file must be one of the embeds
    for (table.entries) |e| {
        if (std.mem.eql(u8, e.site, "q4_gdn@gdn|8")) {
            var found = false;
            for (sources.sources) |s| {
                if (std.mem.eql(u8, s.name, e.file)) found = true;
            }
            try std.testing.expect(found);
        }
    }
}

test "the checked-in lane sources have a no-tensor-unit twin with the same kernel names" {
    const sources = @import("kernel_sources").flashnext_gen;
    for (sources.sources) |s| {
        if (std.mem.indexOf(u8, s.name, "lane_qmm_bytes_grouped") == null) continue;
        if (std.mem.endsWith(u8, s.name, "-lanes.metal")) continue;
        var twin: [128]u8 = undefined;
        const at = std.mem.indexOf(u8, s.name, ".metal").?;
        const lanes_name = try std.fmt.bufPrint(&twin, "{s}-lanes{s}", .{ s.name[0..at], s.name[at..] });
        var found = false;
        for (sources.sources) |t| {
            if (std.mem.eql(u8, t.name, lanes_name)) {
                found = true;
                // the rewrites keep every kernel's name and the constexpr shape block
                try std.testing.expect(std.mem.indexOf(u8, t.text, "constexpr int N = ") != null);
                try std.testing.expect(std.mem.indexOf(u8, t.text, "colok[f][0]") != null);
            }
        }
        try std.testing.expect(found);
    }
}

test "the prompt's checked-in text resolves and patches from another working directory" {
    // variantText resolves each checked-in file, and the gdn patch applies with no kernel file beside the process.
    const replay = @import("replay.zig");
    const table = @import("roles_gen.zig");
    const sources = @import("kernel_sources").flashnext_gen;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const old_fd = std.c.open(".", .{}); // O_RDONLY via the flags struct's default
    defer _ = std.c.close(old_fd);
    const tmpz = try std.fmt.allocPrintSentinel(std.testing.allocator, ".zig-cache/tmp/{s}", .{tmp.sub_path}, 0);
    defer std.testing.allocator.free(tmpz);
    try std.testing.expect(std.c.chdir(tmpz) == 0);
    defer _ = std.c.fchdir(old_fd);
    for (table.entries) |e| {
        var text: []const u8 = "";
        for (sources.sources) |s| {
            if (std.mem.eql(u8, s.name, e.file)) text = s.text;
        }
        try std.testing.expect(text.len != 0);
        const v = replay.Variant{ .inputs = &.{}, .outputs = &.{}, .meta = &.{}, .pipe = undefined, .file = e.file, .name = e.function, .text = text };
        const got = try replay.Run.variantText(std.testing.allocator, &v);
        try std.testing.expectEqualStrings(text, got);
    }
    // the gdn site Prompt.init patches carries the patch string exactly once in the resolved text
    const gs = for (table.entries) |e| {
        if (std.mem.eql(u8, e.site, "q4_gdn@gdn|8")) break e;
    } else return error.NoSite;
    const from = "SO[((size_t(r) * NV + hv)";
    var count: usize = 0;
    for (sources.sources) |s| {
        if (std.mem.eql(u8, s.name, gs.file)) count = std.mem.count(u8, s.text, from);
    }
    try std.testing.expectEqual(@as(usize, 1), count);
}
