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

test "Flash Next load admits a four-bit group-32 checkpoint without touching any weights or cache" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const bytes = try std.mem.replaceOwned(u8, a, checked_config, "\"bits\": 6", "\"bits\": 4");
    defer a.free(bytes);
    var metadata = try config.parse(a, bytes);
    defer metadata.deinit();
    try std.testing.expectEqual(@as(u8, 4), metadata.global_affine.bits);
    try std.testing.expectEqual(@as(usize, 32), metadata.global_affine.group);
    const dir = try fixture(tmp, bytes, checked_index);
    defer a.free(dir);
    try admission.check(a, io, dir);
    try std.testing.expectError(error.FileNotFound, tmp.dir.openFile(io, "model-00001-of-00030.safetensors", .{}));
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

test "a global group of 64 or 128 is admitted for every matrix, and the n-gram tables' 160-wide rows refuse it" {
    for ([_][]const u8{ "64", "128" }) |group| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const declaration = try std.fmt.allocPrint(a, "\"group_size\": {s}", .{group});
        defer a.free(declaration);
        const bytes = try std.mem.replaceOwned(u8, a, checked_config, "\"group_size\": 32", declaration);
        defer a.free(bytes);
        const dir = try fixture(tmp, bytes, checked_index);
        defer a.free(dir);
        var refusal: @import("index.zig").Refusal = .{};
        try std.testing.expectError(error.UnsupportedFlashAffineKernel, admission.checkWith(a, io, dir, &refusal));
        try std.testing.expect(std.mem.indexOf(u8, refusal.name(), ".ple.ple_embedding.ngram_embedding.shard_0") != null);
    }
}

test "text and MTP overrides at other widths are admitted, and a stored matrix declared plain is refused" {
    for ([_]struct { path: []const u8, value: []const u8 }{
        .{ .path = "language_model.model.layers.0.linear_attn.in_proj_qkv", .value = "{\"bits\":4,\"group_size\":32}" },
        .{ .path = "model.layers.0.linear_attn.in_proj_qkv.weight", .value = "{\"bits\":6,\"group_size\":64}" },
        .{ .path = "language_model.mtp.layers.0.self_attn.q_proj", .value = "{\"bits\":3,\"group_size\":128}" },
    }) |override| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const bytes = try overridden(override.path, override.value);
        defer a.free(bytes);
        const dir = try fixture(tmp, bytes, checked_index);
        defer a.free(dir);
        try admission.check(a, io, dir);
    }
    for ([_]struct { path: []const u8, value: []const u8 }{
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
    const engine = @embedFile("engine.zig");
    try std.testing.expect(std.mem.indexOf(u8, engine, "pub const loadWith = @import(\"engine_load.zig\").loadWith;") != null);
    try std.testing.expect(std.mem.indexOf(u8, engine, "pub const loadConfigured = @import(\"engine_load.zig\").loadConfigured;") != null);
    const source = @embedFile("engine_load.zig");
    const wrapper = entryBody(source, "pub fn loadWith(").?;
    try std.testing.expect(std.mem.startsWith(u8, wrapper, "return loadConfigured(gpa, io, model_dir, dump_dir, speed_up, .{});"));
    try std.testing.expect(admissionFirst(source));
    const native = @embedFile("../../native/metal.zig");
    const open = std.mem.indexOf(u8, native, "fn openFlashNext(").?;
    const mapping = std.mem.indexOfPos(u8, native, open, "tf.flashnext_engine.quantizationProblem(e)");
    try std.testing.expect(mapping != null);
    const problem = admission.problemFor(error.UnsupportedFlashAffineKernel).?;
    try std.testing.expect(std.mem.indexOf(u8, problem, "2- to 8-bit weights in groups of 32, 64 or 128") != null);
    try std.testing.expect(admission.problemFor(error.OutOfMemory) == null);
}

test "a module the device kernels cannot read is refused under its checkpoint tensor name, whatever its key's prefix" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const bytes = try overridden("model.layers.1.ple.ple_embedding.ngram_embedding.shard_3.weight", "{\"bits\":8,\"group_size\":64}");
    defer a.free(bytes);
    const dir = try fixture(tmp, bytes, checked_index);
    defer a.free(dir);
    var refusal: @import("index.zig").Refusal = .{};
    try std.testing.expectError(error.UnsupportedFlashAffineKernel, admission.checkWith(a, io, dir, &refusal));
    try std.testing.expectEqualStrings("language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_3", refusal.name());
    try std.testing.expectEqual(@as(u8, 8), refusal.width.?.bits);
    try std.testing.expectEqual(@as(u16, 64), refusal.width.?.group);
}

test "an index keeping the MTP head at top-level mtp (oMLX) is admitted as one keeping it under language_model" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const index = try std.mem.replaceOwned(u8, a, checked_index, "\"language_model.mtp.", "\"mtp.");
    defer a.free(index);
    try std.testing.expect(std.mem.indexOf(u8, index, "\"mtp.fc_embedding.weight\"") != null);
    const dir = try fixture(tmp, checked_config, index);
    defer a.free(dir);
    try admission.check(a, io, dir);
    const missing = try std.mem.replaceOwned(u8, a, index, "\"mtp.fc_hidden.weight\"", "\"mtp.fc_hidden.other\"");
    defer a.free(missing);
    var tmp2 = std.testing.tmpDir(.{});
    defer tmp2.cleanup();
    const dir2 = try fixture(tmp2, checked_config, missing);
    defer a.free(dir2);
    try std.testing.expectError(error.MissingFlashTensor, admission.check(a, io, dir2));
}

test "the width and layout refusals past admission map to text a user can act on" {
    for ([_]anyerror{ error.InvalidAffineShape, error.ExpertGateUpWidths, error.NgramTableWidths, error.LaneRowsNotFours, error.TooManyLaneMembers }) |e|
        try std.testing.expectEqualStrings(admission.stored_problem, admission.problemFor(e).?);
    try std.testing.expectEqualStrings(admission.speed_up_problem, admission.problemFor(error.SpeedUpNeeds6Bit32).?);
    try std.testing.expectEqualStrings(admission.scale_problem, admission.problemFor(error.NgramTableScale).?);
}

fn entryBody(source: []const u8, name: []const u8) ?[]const u8 {
    const entry = std.mem.indexOf(u8, source, name) orelse return null;
    const brace = std.mem.indexOfScalarPos(u8, source, entry, '{') orelse return null;
    return std.mem.trimStart(u8, source[brace + 1 ..], " \t\r\n");
}

fn admissionFirst(source: []const u8) bool {
    var body = entryBody(source, "pub fn loadConfigured(") orelse return false;
    const validation = "try window.validate(speed_up != null or platform.c.getenv(\"TF_FLASHNEXT_TP\") != null);";
    if (!std.mem.startsWith(u8, body, validation)) return false;
    body = std.mem.trimStart(u8, body[validation.len..], " \t\r\n");
    return std.mem.startsWith(u8, body, "try load_admission.check(gpa, io, model_dir);");
}
