//! A synthetic EAQ pack written to a directory, with one chosen defect: the tests' fixture (no GPU, no real weights).

const std = @import("std");
const Sha256 = std.crypto.hash.sha2.Sha256;
const eaq = @import("eaq_container.zig");
const exl3_format = @import("exl3_format.zig");

pub const Tamper = enum {
    none,
    config_hash, // config.json states another manifest hash
    config_name, // config.json names another manifest file
    config_sha_short,
    manifest_edited, // the manifest changes after config.json hashed it
    no_marker,
    marker_other, // the marker vouches for other counts
    layout,
    pack_version,
    family,
    feature,
    dictionary_id,
    dictionary_hash,
    format,
    shard_flip, // one byte of a shard differs
    shard_truncated,
    shard_missing,
    tensor_extra,
    tensor_missing,
    tensor_dtype,
    tensor_shape,
    tensor_duplicate,
    entries_in_header, // entries_inline false: the entries sit in the shard headers' metadata
    entries_absent, // entries_inline false and no metadata
    entry_shard_outside, // an inline entry names a shard outside the folder
    entry_shard_other, // an inline entry names a listed shard that does not hold its tensor
    dictionary_unused, // the trellis entry names one of TensorFold's tables that the manifest does not declare
    dictionaries_empty, // no dictionary declared; the trellis entry still names one
    entry_duplicate, // two entries share a name
    empty, // no shards and no entries: a consistent, empty pack
};

const Spec = struct { name: []const u8, dtype: []const u8, shape: []const u8, bytes: usize };

const extra = Spec{ .name = "stray.weight", .dtype = "BF16", .shape = "[2]", .bytes = 4 };
const shard2 = [_]Spec{.{ .name = "model.norm.weight", .dtype = "BF16", .shape = "[64]", .bytes = 128 }};

fn image(gpa: std.mem.Allocator, specs: []const Spec, meta: ?[]const u8) ![]u8 {
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(gpa);
    var at: usize = 0;
    try header.append(gpa, '{');
    if (meta) |m| try header.print(gpa, "\"__metadata__\":{{\"eaq.entries\":{f}}}{s}", .{ std.json.fmt(m, .{}), if (specs.len > 0) "," else "" });
    for (specs, 0..) |s, i| {
        try header.print(gpa, "{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":{s},\"data_offsets\":[{d},{d}]}}", .{ if (i > 0) "," else "", s.name, s.dtype, s.shape, at, at + s.bytes });
        at += s.bytes;
    }
    try header.append(gpa, '}');
    const out = try gpa.alloc(u8, 8 + header.items.len + at);
    std.mem.writeInt(u64, out[0..8], header.items.len, .little);
    @memcpy(out[8..][0..header.items.len], header.items);
    for (out[8 + header.items.len ..], 0..) |*b, i| b.* = @intCast(i % 251);
    return out;
}

pub fn hex(bytes: [32]u8) [64]u8 {
    return std.fmt.bytesToHex(bytes, .lower);
}

pub fn sha(data: []const u8) [64]u8 {
    var d: [32]u8 = undefined;
    Sha256.hash(data, &d, .{});
    return hex(d);
}

/// What the pack says about itself, where a test needs more than the default.
pub const Shape = struct {
    family: []const u8 = "qwen3_5",
    codebook: exl3_format.Codebook = .mul1,
    dictionary: []const u8 = "exl3.mul1.v1",
    bits: u8 = 6,
};

/// Writes a pack in `dir` (a 64-wide, 128-token toy head) with one defect, or none.
pub fn write(gpa_in: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, t: Tamper, shape: Shape) !void {
    var arena = std.heap.ArenaAllocator.init(gpa_in);
    defer arena.deinit();
    const gpa = arena.allocator();
    if (t == .empty) return writeEmpty(gpa, io, dir, shape);
    const words = 16 * @as(usize, shape.bits);
    const trellis_shape = try std.fmt.allocPrint(gpa, "[4,8,{d}]", .{words});
    var s1 = [_]Spec{
        .{ .name = "lm_head.trellis", .dtype = "I16", .shape = trellis_shape, .bytes = 2 * 4 * 8 * words },
        .{ .name = "lm_head.suh", .dtype = "F16", .shape = "[64]", .bytes = 128 },
        .{ .name = "lm_head.svh", .dtype = "F16", .shape = "[128]", .bytes = 256 },
        extra,
    };
    const specs1: []const Spec = if (t == .tensor_extra) &s1 else s1[0..3];
    var s2 = shard2;
    if (t == .tensor_dtype) s2[0].dtype = "F16";
    if (t == .tensor_shape) s2[0].shape = "[2,32]";
    if (t == .tensor_duplicate) s2[0].name = "lm_head.suh";
    var table: [32]u8 = undefined;
    eaq.codebookSha(shape.codebook, &table);
    const dict_hash = if (t == .dictionary_hash) sha("x") else hex(table);
    const head_entry = try std.fmt.allocPrint(gpa,
        \\{{"name":"lm_head","format":"{s}","params":{{"bits":{d}.0,"K":64,"N":128,"dictionary":"{s}"}},"shard":"{s}",
        \\"parts":{{"trellis":{{"tensor":"lm_head.trellis","dtype":"I16","shape":[4,8,{d}]}},"suh":{{"tensor":"lm_head.suh","dtype":"F16","shape":[64]}},
        \\"svh":{{"tensor":"lm_head.svh","dtype":"F16","shape":[128]}}}}}}
    , .{
        if (t == .format) "eaq.mystery/0" else "eaq.trellis/0",
        shape.bits,
        if (t == .dictionary_unused) "exl3.mcg.v1" else shape.dictionary,
        if (t == .entry_shard_outside) "../outside.safetensors" else "model-00001.safetensors",
        words,
    });
    const norm_entry = try std.fmt.allocPrint(gpa,
        \\{{"name":"{s}","format":"eaq.dense/0","params":{{}},"shard":"{s}",
        \\"parts":{{"data":{{"tensor":"{s}","dtype":"BF16","shape":[64]}}}}}}
    , .{
        if (t == .entry_duplicate) "lm_head" else "model.norm.weight",
        if (t == .entry_shard_other) "model-00001.safetensors" else "model-00002.safetensors",
        if (t == .tensor_missing) "model.norm.gone" else "model.norm.weight",
    });
    const entries = try std.fmt.allocPrint(gpa, "[{s},{s}]", .{ head_entry, norm_entry });
    const inline_entries = t != .entries_in_header and t != .entries_absent;
    const img1 = try image(gpa, specs1, if (t == .entries_in_header) try std.fmt.allocPrint(gpa, "[{s}]", .{head_entry}) else null);
    const img2 = try image(gpa, &s2, if (t == .entries_in_header) try std.fmt.allocPrint(gpa, "[{s}]", .{norm_entry}) else null);
    const dict_name = if (t == .dictionary_id) "exl3.made.up" else shape.dictionary;
    const dicts = if (t == .dictionaries_empty) "{}" else try std.fmt.allocPrint(gpa, "{{\"{s}\":{{\"table_sha256\":\"{s}\"}}}}", .{ dict_name, dict_hash });
    const manifest = try std.fmt.allocPrint(gpa,
        \\{{"eaq_layout":"{s}","eaq_pack":"{s}","model":{{"adapter":{{"family":"{s}"}}}},"required_features":[{s}],
        \\"dictionaries":{s},"n_entries":2,
        \\"shards":[{{"file":"model-00001.safetensors","bytes":{d},"sha256":"{s}"}},{{"file":"model-00002.safetensors","bytes":{d},"sha256":"{s}"}}]
        \\{s}}}
    , .{
        if (t == .layout) "eaq-container/v9" else eaq.layout,
        if (t == .pack_version) "eaq.pack/9" else "eaq.pack/0",
        if (t == .family) "glm5_next" else shape.family,
        if (t == .feature) "\"eaq.future/9\"" else "",
        dicts,
        img1.len,
        sha(img1),
        img2.len,
        sha(img2),
        if (inline_entries) try std.fmt.allocPrint(gpa, ",\"entries_inline\":true,\"entries\":{s}", .{entries}) else "",
    });
    var h: [64]u8 = sha(manifest);
    if (t == .config_hash) h = sha("other");
    const config = try std.fmt.allocPrint(gpa,
        \\{{"vocab_size":128,"quantization_config":{{"quant_method":"eaq","manifest":"{s}","manifest_sha256":"{s}","average_bits_trellis":5.0}}}}
    , .{ if (t == .config_name) "../eaq_manifest.json" else eaq.manifest_name, if (t == .config_sha_short) h[0..60] else h[0..] });
    const marker = try std.fmt.allocPrint(gpa, "{{\"manifest_sha256\":\"{s}\",\"eaq_layout\":\"{s}\",\"n_shards\":2,\"n_entries\":{d}}}", .{ h, if (t == .layout) "eaq-container/v9" else eaq.layout, @as(u8, if (t == .marker_other) 3 else 2) });

    try dir.writeFile(io, .{ .sub_path = "config.json", .data = config });
    try dir.writeFile(io, .{ .sub_path = eaq.manifest_name, .data = if (t == .manifest_edited) try std.mem.concat(gpa, u8, &.{ manifest, " " }) else manifest });
    if (t != .no_marker) try dir.writeFile(io, .{ .sub_path = eaq.marker_name, .data = marker });
    if (t == .shard_flip) img1[img1.len - 1] ^= 1;
    const out1 = if (t == .shard_truncated) img1[0 .. img1.len - 4] else img1;
    if (t != .shard_missing) try dir.writeFile(io, .{ .sub_path = "model-00001.safetensors", .data = out1 });
    try dir.writeFile(io, .{ .sub_path = "model-00002.safetensors", .data = img2 });
}

/// A pack with no shards and no entries, its config.json and marker consistent with it.
fn writeEmpty(gpa: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, shape: Shape) !void {
    const manifest = try std.fmt.allocPrint(gpa,
        \\{{"eaq_layout":"{s}","eaq_pack":"eaq.pack/0","model":{{"adapter":{{"family":"{s}"}}}},"dictionaries":{{}},
        \\"n_entries":0,"shards":[],"entries_inline":true,"entries":[]}}
    , .{ eaq.layout, shape.family });
    const h = sha(manifest);
    const config = try std.fmt.allocPrint(gpa, "{{\"quantization_config\":{{\"quant_method\":\"eaq\",\"manifest\":\"{s}\",\"manifest_sha256\":\"{s}\"}}}}", .{ eaq.manifest_name, h });
    const marker = try std.fmt.allocPrint(gpa, "{{\"manifest_sha256\":\"{s}\",\"eaq_layout\":\"{s}\",\"n_shards\":0,\"n_entries\":0}}", .{ h, eaq.layout });
    try dir.writeFile(io, .{ .sub_path = "config.json", .data = config });
    try dir.writeFile(io, .{ .sub_path = eaq.manifest_name, .data = manifest });
    try dir.writeFile(io, .{ .sub_path = eaq.marker_name, .data = marker });
}
