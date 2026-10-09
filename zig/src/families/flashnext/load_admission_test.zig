//! Load admission reads config and index fixtures without any device or checkpoint payload.
const std = @import("std");
const admission = @import("load_admission.zig");
const config = @import("config.zig");
const a = std.testing.allocator;
const io = std.testing.io;
const checked_config = @embedFile("fixtures/config.json");
const checked_index = @embedFile("fixtures/index.json");

fn fixture(tmp: std.testing.TmpDir, bytes: []const u8, index: []const u8) ![:0]u8 {
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = bytes });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index });
    return tmp.dir.realPathFileAlloc(io, ".", a);
}

fn overridden(path: []const u8, value: []const u8) ![]u8 {
    const declaration = try std.fmt.allocPrint(a, "\"mode\": \"affine\", \"{s}\": {s}", .{ path, value });
    defer a.free(declaration);
    return std.mem.replaceOwned(u8, a, checked_config, "\"mode\": \"affine\"", declaration);
}

test "Flash Next load refuses a four-bit group-32 checkpoint before touching any weights or cache" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const bytes = try std.mem.replaceOwned(u8, a, checked_config, "\"bits\": 6", "\"bits\": 4");
    defer a.free(bytes);
    var metadata = try config.parse(a, bytes);
    defer metadata.deinit();
    try std.testing.expectEqual(@as(u8, 4), metadata.global_affine.bits);
    try std.testing.expectEqual(@as(usize, 32), metadata.global_affine.group);
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = bytes });
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    try std.testing.expectError(error.UnsupportedFlashAffineKernel, admission.check(a, io, dir));
    try std.testing.expectError(error.FileNotFound, tmp.dir.openFile(io, "model.safetensors.index.json", .{}));
    try std.testing.expectError(error.FileNotFound, tmp.dir.openDir(io, "zig-pack", .{}));
}

test "load admission accepts the supported text and MTP matrices without reading any weight shard" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try fixture(tmp, checked_config, checked_index);
    defer a.free(dir);
    try admission.check(a, io, dir);
    try std.testing.expectError(error.FileNotFound, tmp.dir.openFile(io, "model-00001-of-00030.safetensors", .{}));
    try std.testing.expectError(error.FileNotFound, tmp.dir.openDir(io, "zig-pack", .{}));
}

test "global unsupported affine groups refuse before the index is opened" {
    for ([_][]const u8{ "64", "128" }) |group| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const declaration = try std.fmt.allocPrint(a, "\"group_size\": {s}", .{group});
        defer a.free(declaration);
        const bytes = try std.mem.replaceOwned(u8, a, checked_config, "\"group_size\": 32", declaration);
        defer a.free(bytes);
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = bytes });
        const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
        defer a.free(dir);
        try std.testing.expectError(error.UnsupportedFlashAffineKernel, admission.check(a, io, dir));
    }
}

test "required text and MTP affine overrides cannot bypass the runtime format" {
    for ([_]struct { path: []const u8, value: []const u8 }{
        .{ .path = "language_model.model.layers.0.linear_attn.in_proj_qkv", .value = "{\"bits\":4,\"group_size\":32}" },
        .{ .path = "model.layers.0.linear_attn.in_proj_qkv.weight", .value = "{\"bits\":6,\"group_size\":64}" },
        .{ .path = "language_model.mtp.layers.0.self_attn.q_proj", .value = "{\"bits\":4,\"group_size\":32}" },
        .{ .path = "language_model.lm_head", .value = "{}" },
        .{ .path = "language_model.model.embed_tokens", .value = "false" },
    }) |override| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const bytes = try overridden(override.path, override.value);
        defer a.free(bytes);
        const dir = try fixture(tmp, bytes, checked_index);
        defer a.free(dir);
        var c = try config.parse(a, bytes);
        defer c.deinit();
        _ = try @import("index.zig").admit(a, checked_index, &c);
        try std.testing.expectError(error.UnsupportedFlashAffineKernel, admission.check(a, io, dir));
    }
}

test "unquantized norms and irrelevant vision overrides do not reject supported text kernels" {
    const vision_index = try std.mem.replaceOwned(u8, a, checked_index, "\"weight_map\":{", "\"weight_map\":{\"vision_tower.extra.weight\":\"model-00001-of-00030.safetensors\",\"vision_tower.extra.scales\":\"model-00001-of-00030.safetensors\",\"vision_tower.extra.biases\":\"model-00001-of-00030.safetensors\",");
    defer a.free(vision_index);
    for ([_]struct { path: []const u8, value: []const u8 }{
        .{ .path = "language_model.model.layers.0.attn_hyper_connection.hc_norm", .value = "false" },
        .{ .path = "language_model.model.layers.0.attn_hyper_connection.hc_norm.weight", .value = "{\"bits\":4,\"group_size\":64}" },
        .{ .path = "vision_tower.extra", .value = "{\"bits\":4,\"group_size\":64}" },
        .{ .path = "language_model.model.layers.0.linear_attn.in_proj_qkv", .value = "{\"bits\":6,\"group_size\":32}" },
        .{ .path = "language_model.model.layers.0.linear_attn.in_proj_qkv", .value = "true" },
    }) |override| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const bytes = try overridden(override.path, override.value);
        defer a.free(bytes);
        const dir = try fixture(tmp, bytes, vision_index);
        defer a.free(dir);
        try admission.check(a, io, dir);
    }
}

test "generic four-bit config and index metadata remain supported outside runtime admission" {
    const bytes = try std.mem.replaceOwned(u8, a, checked_config, "\"bits\": 6", "\"bits\": 4");
    defer a.free(bytes);
    var c = try config.parse(a, bytes);
    defer c.deinit();
    const inventory = try @import("index.zig").admit(a, checked_index, &c);
    try std.testing.expect(inventory.required_names > 3300);
}

test "normal and dump load paths admit before any engine device or pack work and render the human refusal" {
    const source = @embedFile("engine.zig");
    const entry = std.mem.indexOf(u8, source, "pub fn loadWith(").?;
    const body = source[entry + std.mem.indexOf(u8, source[entry..], "{").? + 1 ..];
    try std.testing.expect(std.mem.startsWith(u8, std.mem.trimStart(u8, body, " \t\r\n"), "try load_admission.check(gpa, io, model_dir);"));
    const native = @embedFile("../../native/metal.zig");
    const open = std.mem.indexOf(u8, native, "fn openFlashNext(").?;
    const mapping = std.mem.indexOfPos(u8, native, open, "tf.flashnext_engine.quantizationProblem(e)");
    try std.testing.expect(mapping != null);
    const problem = admission.problemFor(error.UnsupportedFlashAffineKernel).?;
    try std.testing.expect(std.mem.indexOf(u8, problem, "supported 6-bit affine, group_size 32") != null);
    try std.testing.expect(admission.problemFor(error.OutOfMemory) == null);
}
