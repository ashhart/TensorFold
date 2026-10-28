//! Public dsq-1 config admission through the serving reader, without weights or devices.
const std = @import("std");
const Config = @import("config.zig").Config;
const t = std.testing;
const public_config = @embedFile("fixtures/flash1017-dsq-config.json");
const public_version = "dsq-1 (exllamav3 1.5.4 quantize_exl3)";

fn refusal(comptime expected: anyerror, before: []const u8, after: []const u8) !void {
    const changed = try std.mem.replaceOwned(u8, t.allocator, public_config, before, after);
    defer t.allocator.free(changed);
    try t.expect(!std.mem.eql(u8, public_config, changed));
    try t.expectError(expected, Config.parse(t.allocator, changed));
}

test "dsq admission: exact public config passes the serving Config.read path on CPU" {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(public_config, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    try t.expectEqualStrings("8716a12d34f798e991bcc8e6689b25c006dd8a9675de5653e685094503c5d864", &hex);
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var file = try tmp.dir.createFile(t.io, "config.json", .{});
    defer file.close(t.io);
    try file.writePositionalAll(t.io, public_config, 0);
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(dir);
    const admitted = try Config.read(t.allocator, t.io, dir);
    const metadata = try Config.parseMetadata(t.allocator, public_config);
    try t.expectEqual(@as(u32, 5120), admitted.hidden);
    try t.expectEqual(@as(u32, 384), admitted.experts);
    try t.expectEqual(metadata.engram_compressed_vocab, admitted.engram_compressed_vocab);
    try t.expectEqualSlices(u64, metadata.engram_rows.items(), admitted.engram_rows.items());
    try admitted.validateRelease();
}

test "dsq admission: format identity is independent of producer version and pack annotations" {
    for ([_][]const u8{ "dsq-1", "dsq-1 (other-exporter 9.0)", "1", "1.4.2", "1.future" }) |version| {
        const changed = try std.mem.replaceOwned(u8, t.allocator, public_config, public_version, version);
        defer t.allocator.free(changed);
        _ = try Config.parse(t.allocator, changed);
    }
    for ([_][]const u8{ "inst3", "mcg", "mul1" }) |codebook| {
        const field = try std.fmt.allocPrint(t.allocator, "\"codebook\": \"{s}\"", .{codebook});
        defer t.allocator.free(field);
        const changed = try std.mem.replaceOwned(u8, t.allocator, public_config, "\"codebook\": \"mul1\"", field);
        defer t.allocator.free(changed);
        _ = try Config.parse(t.allocator, changed);
    }
}

test "dsq admission: unknown or malformed metadata and release geometry remain refused" {
    for ([_][]const u8{ "dsq-2", "dsq-10", "dsq-1broken", "dsq-1 ()", "dsq-1 (producer", "2.0" }) |version|
        try refusal(error.UnsupportedCheckpoint, public_version, version);
    try refusal(error.UnsupportedCheckpoint, "\"quant_method\": \"exl3\"", "\"quant_method\": \"fp8\"");
    try refusal(error.UnsupportedCheckpoint, "\"quant_method\": \"exl3\"", "\"quant_method\": false");
    try refusal(error.MissingField, "\"quant_method\":", "\"missing_quant_method\":");
    try refusal(error.UnsupportedCheckpoint, "\"version\": \"" ++ public_version ++ "\"", "\"version\": 1");
    try refusal(error.BadConfig, "\"quantization_config\": {", "\"quantization_config\": null, \"ignored_quant\": {");
    try refusal(error.MissingField, "\"version\":", "\"missing_version\":");
    try refusal(error.MissingField, "\"codebook\":", "\"missing_codebook\":");
    try refusal(error.UnsupportedCheckpoint, "\"codebook\": \"mul1\"", "\"codebook\": \"unknown\"");
    try refusal(error.UnsupportedCheckpoint, "\"out_scales\": \"always\"", "\"out_scales\": \"unknown\"");
    try refusal(error.MissingField, "\"out_scales\":", "\"missing_out_scales\":");
    try refusal(error.MissingField, "\"index_topk\":", "\"missing_index_topk\":");
    try refusal(error.UnsupportedArchitecture, "\"hidden_size\": 5120", "\"hidden_size\": 5248");
    try refusal(error.UnsupportedArchitecture, "\"num_key_value_heads\": 1", "\"num_key_value_heads\": 2");
    try refusal(error.UnsupportedCheckpoint, "\"model_type\": \"deepseek_v41\"", "\"model_type\": \"other\"");
}

test "dsq admission: legacy 1.x optional metadata remains backward compatible" {
    const legacy = try std.mem.replaceOwned(u8, t.allocator, public_config, public_version, "1.4.2");
    defer t.allocator.free(legacy);
    const without_codebook = try std.mem.replaceOwned(u8, t.allocator, legacy, "\"codebook\":", "\"unused_codebook\":");
    defer t.allocator.free(without_codebook);
    const without_scales = try std.mem.replaceOwned(u8, t.allocator, without_codebook, "\"out_scales\":", "\"unused_out_scales\":");
    defer t.allocator.free(without_scales);
    _ = try Config.parse(t.allocator, without_scales);
}
