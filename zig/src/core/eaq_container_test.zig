//! Host tests of eaq_container.zig on synthetic packs written to a temporary directory: no GPU and no real weights.

const std = @import("std");
const Sha256 = std.crypto.hash.sha2.Sha256;
const eaq = @import("eaq_container.zig");
const exl3_format = @import("exl3_format.zig");
const fixture = @import("eaq_fixture.zig");
const Tamper = fixture.Tamper;
const hex = fixture.hex;
const sha = fixture.sha;

fn openIn(t: Tamper, o: eaq.Options) !eaq.Pack {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try fixture.write(gpa, io, tmp.dir, t, .{});
    var path: [96]u8 = undefined;
    return eaq.open(gpa, io, try std.fmt.bufPrint(&path, ".zig-cache/tmp/{s}", .{tmp.sub_path}), o);
}

fn expectRefused(t: Tamper, want: anyerror, o: eaq.Options) !void {
    var why: eaq.Why = .{};
    var opts = o;
    opts.why = &why;
    if (openIn(t, opts)) |p| {
        var pack = p;
        pack.deinit();
        return error.TestUnexpectedSuccess;
    } else |err| try std.testing.expectEqual(want, err);
}

test "a good pack verifies and exposes its entries" {
    var p = try openIn(.none, .{ .family = "qwen3_5" });
    defer p.deinit();
    try std.testing.expectEqualStrings("qwen3_5", p.family);
    try std.testing.expectEqualStrings("eaq.pack/0", p.version);
    try std.testing.expectEqual(@as(usize, 2), p.shards.len);
    try std.testing.expectEqual(@as(usize, 4), p.n_tensors);
    const head = p.entry("lm_head").?;
    try std.testing.expectEqual(eaq.Format.trellis, head.format);
    try std.testing.expectEqual(@as(usize, 3), head.parts.len);
    try std.testing.expectEqual(@as(i64, 64), head.params.get("K").?.integer);
    try std.testing.expectEqualStrings("model-00002.safetensors", p.entry("model.norm.weight").?.shard);
    try std.testing.expect(p.entry("nope") == null);
}

test "entries kept in the shard headers read as inline ones do; none at all is refused" {
    var p = try openIn(.entries_in_header, .{});
    defer p.deinit();
    try std.testing.expectEqual(@as(usize, 2), p.entries.len);
    try std.testing.expectEqualStrings("model-00001.safetensors", p.entry("lm_head").?.shard);
    try std.testing.expectEqualStrings("model-00002.safetensors", p.entry("model.norm.weight").?.shard);
    try expectRefused(.entries_absent, error.BadManifest, .{});
}

test "config.json decides what is an EAQ pack" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path: [96]u8 = undefined;
    const dir = try std.fmt.bufPrint(&path, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    try std.testing.expect(!try eaq.isEaq(gpa, io, dir)); // no config.json
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"quantization_config\":{\"quant_method\":\"exl3\",\"bits\":3.0}}" });
    try std.testing.expect(!try eaq.isEaq(gpa, io, dir));
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"quantization\":{\"bits\":4,\"group_size\":64}}" });
    try std.testing.expect(!try eaq.isEaq(gpa, io, dir));
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"text_config\":{\"quantization_config\":{\"quant_method\":\"EAQ\"}}}" });
    try std.testing.expect(try eaq.isEaq(gpa, io, dir));
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"quantization_config\":{\"quant_method\":\"exl3\"}}" });
    try std.testing.expectError(error.NotEaq, eaq.open(gpa, io, dir, .{}));
}

test "a config that does not name the manifest and its hash is refused" {
    try expectRefused(.config_name, error.BadConfig, .{});
    try expectRefused(.config_sha_short, error.BadConfig, .{});
    try expectRefused(.config_hash, error.ManifestHashMismatch, .{});
    try expectRefused(.manifest_edited, error.ManifestHashMismatch, .{});
}

test "a pack the writer did not finish, or another pack's marker, is refused" {
    try expectRefused(.no_marker, error.MarkerMissing, .{});
    try expectRefused(.marker_other, error.MarkerMismatch, .{});
}

test "layout, pack version, family and required features are gates" {
    try expectRefused(.layout, error.UnsupportedLayout, .{});
    try expectRefused(.pack_version, error.UnsupportedPackVersion, .{});
    try expectRefused(.family, error.FamilyMismatch, .{ .family = "qwen3_5" });
    try expectRefused(.feature, error.MissingFeature, .{});
    var p = try openIn(.family, .{}); // no family asked for: any is served
    p.deinit();
    var q = try openIn(.feature, .{ .features = &.{"eaq.future/9"} });
    defer q.deinit();
    try std.testing.expectEqualStrings("eaq.future/9", q.features[0]);
}

test "a dictionary must be one of TensorFold's own tables, with the same hash" {
    try expectRefused(.dictionary_id, error.UnknownDictionary, .{});
    try expectRefused(.dictionary_hash, error.DictionaryMismatch, .{});
    var d: [32]u8 = undefined;
    eaq.codebookSha(.mcg, &d);
    var e: [32]u8 = undefined;
    eaq.codebookSha(.mul1, &e);
    try std.testing.expect(!std.mem.eql(u8, &d, &e));
}

test "an unknown entry format is refused" {
    try expectRefused(.format, error.UnknownFormat, .{});
}

test "shard bytes: a flipped byte fails the hash, a short or missing file fails earlier" {
    try expectRefused(.shard_flip, error.ShardHashMismatch, .{});
    try expectRefused(.shard_flip, error.ShardHashMismatch, .{ .threads = 1 });
    try expectRefused(.shard_truncated, error.ShardSizeMismatch, .{});
    try expectRefused(.shard_truncated, error.ShardSizeMismatch, .{ .hash_shards = false });
    try expectRefused(.shard_missing, error.ShardMissing, .{});
    var p = try openIn(.shard_flip, .{ .hash_shards = false }); // the sizes and the inventory still hold
    p.deinit();
}

test "the shard headers must hold exactly the manifest's tensors" {
    try expectRefused(.tensor_extra, error.TensorExtra, .{});
    try expectRefused(.tensor_missing, error.TensorMissing, .{});
    try expectRefused(.tensor_dtype, error.TensorMismatch, .{});
    try expectRefused(.tensor_shape, error.TensorMismatch, .{});
    try expectRefused(.tensor_duplicate, error.DuplicateTensor, .{});
}

test "a refusal names what it is about" {
    var why: eaq.Why = .{};
    try std.testing.expectError(error.TensorExtra, openIn(.tensor_extra, .{ .why = &why }));
    try std.testing.expectEqualStrings("stray.weight", why.text());
    try std.testing.expectError(error.ShardHashMismatch, openIn(.shard_flip, .{ .why = &why }));
    try std.testing.expectEqualStrings("model-00001.safetensors", why.text());
}

test "the codebook table hashes follow the decoder, not a stored constant" {
    var a: [32]u8 = undefined;
    eaq.codebookSha(.inst3, &a);
    var h = Sha256.init(.{});
    for (0..65536) |s| {
        var two: [2]u8 = undefined;
        std.mem.writeInt(u16, &two, @bitCast(exl3_format.codebookValue(.inst3, @intCast(s))), .little);
        h.update(&two);
    }
    var b: [32]u8 = undefined;
    h.final(&b);
    try std.testing.expectEqualSlices(u8, &b, &a);
}

test "the three tables hash to the values an independent numpy writer recorded" {
    const want = [_]struct { exl3_format.Codebook, []const u8 }{
        .{ .inst3, "ad88de2dcff5bb05fff2faec37cceceb43aa1c2e8203f9d81a6fbd6dd3393eea" },
        .{ .mcg, "95383563929baa9d1ae5de0be22490284382673620bdd85929feea7bdbef04a9" },
        .{ .mul1, "bc48d02cb1c14939dc47b90f870dd689d63e3b1ab69a157e65538db68677a6c8" },
    };
    for (want) |w| {
        var d: [32]u8 = undefined;
        eaq.codebookSha(w[0], &d);
        const have = hex(d);
        try std.testing.expectEqualStrings(w[1], &have);
    }
}
