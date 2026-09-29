//! Complete metadata contracts for the four supported, fixed checkpoint recipes.
const std = @import("std");
const mx = @import("mlx.zig");
const DType = @import("safetensors.zig").DType;
pub const Kind = enum { qwen, dflash, nemotron, flash, gemma, deepseek, glm };
const Spec = struct { name: []const u8, dtype: DType, shape: []const i32 };
const Metadata = struct { dtype: DType, shape: []const i32 };
fn source(kind: Kind) []const u8 {
    return switch (kind) {
        .qwen => @embedFile("schemas/qwen.json"),
        .dflash => @embedFile("schemas/dflash.json"),
        .nemotron => @embedFile("schemas/nemotron.json"),
        .flash => @embedFile("schemas/flash.json"),
        .gemma => @embedFile("schemas/gemma.json"),
        .deepseek => @embedFile("schemas/deepseek.json"),
        .glm => @embedFile("schemas/glm.json"),
    };
}
fn check(spec: Spec, actual: ?Metadata) !void {
    const value = actual orelse return if (std.mem.startsWith(u8, spec.name, "mtp.")) error.MissingDraftHead else error.MissingWeight;
    if (value.dtype != spec.dtype) return error.InvalidTensorDType;
    if (!std.mem.eql(i32, value.shape, spec.shape)) return error.InvalidTensorShape;
}
fn required(spec: Spec, drafts: bool) bool {
    return drafts or !std.mem.startsWith(u8, spec.name, "mtp.");
}
pub fn validate(kind: Kind, arrays: *const std.StringHashMap(mx.Array), drafts: bool) !void {
    return validateConfig(kind, arrays, drafts, null);
}
fn adjusted(spec: Spec, config: ?std.json.Value, shape: *[8]i32, actual_dtype: ?DType) !?Spec {
    const cfg = config orelse return spec;
    const scale = std.mem.endsWith(u8, spec.name, ".scales");
    const bias = std.mem.endsWith(u8, spec.name, ".biases");
    const kind = cfg.object.get("model_type");
    const bonsai = kind != null and kind.? == .string and std.mem.eql(u8, kind.?.string, "prism_hadamard_qwen35");
    if (spec.dtype != .U32 and !scale and !bias) {
        var result = spec;
        if (bonsai) result.dtype = try floating(actual_dtype);
        return result;
    }
    const name = spec.name[0 .. spec.name.len - 7];
    const format = if (bonsai and (std.mem.endsWith(u8, name, ".in_proj_a") or std.mem.endsWith(u8, name, ".in_proj_b"))) null else try @import("quantization.zig").resolve(cfg, name);
    if (spec.shape.len < 2 or spec.shape.len > shape.len) return error.InvalidTensorShape;
    @memcpy(shape[0..spec.shape.len], spec.shape);
    const last = spec.shape.len - 1;
    const base_bits: i64 = if (std.mem.endsWith(u8, name, ".router.proj")) 8 else 4;
    var result = spec;
    if (format) |f| {
        const width = @as(i64, spec.shape[last]) * (if (scale or bias) @as(i64, 64) else @divExact(32, base_bits));
        if (@mod(width, f.group_size) != 0 or @mod(width * f.bits, 32) != 0) return error.InvalidTensorShape;
        const elements = if (scale or bias) @divExact(width, f.group_size) else @divExact(width * f.bits, 32);
        shape[last] = std.math.cast(i32, elements) orelse return error.InvalidTensorShape;
        if (scale or bias) result.dtype = try floating(actual_dtype);
    } else {
        if (scale or bias) return null;
        shape[last] = std.math.mul(i32, shape[last], @intCast(@divExact(32, base_bits))) catch return error.InvalidTensorShape;
        result.dtype = try floating(actual_dtype);
    }
    result.shape = shape[0..spec.shape.len];
    return result;
}
fn floating(dtype: ?DType) !DType {
    return switch (dtype orelse return error.MissingWeight) {
        .BF16, .F16, .F32 => dtype.?,
        else => error.InvalidTensorDType,
    };
}
pub fn validateConfig(kind: Kind, arrays: *const std.StringHashMap(mx.Array), drafts: bool, config: ?std.json.Value) !void {
    const specs = try std.json.parseFromSlice([]const Spec, mx.allocator, source(kind), .{});
    defer specs.deinit();
    for (specs.value) |spec| {
        if (!required(spec, drafts)) continue;
        const array = arrays.get(spec.name);
        const actual: ?Metadata = if (array) |a| .{
            .dtype = switch (mx.dtype(a)) {
                mx.c.MLX_BFLOAT16 => .BF16,
                mx.c.MLX_FLOAT16 => .F16,
                mx.c.MLX_FLOAT32 => .F32,
                mx.c.MLX_UINT32 => .U32,
                mx.c.MLX_UINT8 => .U8,
                mx.c.MLX_INT32 => .I32,
                mx.c.MLX_INT64 => .I64,
                else => return error.InvalidTensorDType,
            },
            .shape = mx.shape(a),
        } else null;
        var shape: [8]i32 = undefined;
        const expected = (try adjusted(spec, config, &shape, if (actual) |v| v.dtype else null)) orelse continue;
        check(expected, actual) catch |err| {
            std.debug.print("Checkpoint schema: {s}: {s}; expected {s} {any}\n", .{ spec.name, @errorName(err), @tagName(spec.dtype), spec.shape });
            return err;
        };
    }
}
pub fn checkCheckpoint(kind: Kind, io: std.Io, dir: []const u8) !void {
    const safe = @import("safetensors.zig");
    const a = mx.allocator;
    const specs = try std.json.parseFromSlice([]const Spec, a, source(kind), .{});
    defer specs.deinit();
    var path: [4096]u8 = undefined;
    const config: ?std.json.Parsed(std.json.Value) = if (kind == .qwen or kind == .gemma) blk: {
        const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&path, "{s}/config.json", .{dir}));
        defer a.free(bytes);
        const parsed = try std.json.parseFromSlice(std.json.Value, a, bytes, .{ .allocate = .alloc_always });
        errdefer parsed.deinit();
        if (kind == .qwen) try @import("config.zig").target(parsed.value);
        break :blk parsed;
    } else null;
    defer if (config) |parsed| parsed.deinit();
    const cfg: ?std.json.Value = if (config) |parsed| parsed.value else null;
    const index: ?std.json.Parsed(std.json.Value) = if (kind == .dflash) null else blk: {
        const bytes = @import("weights.zig").readFile(io, try std.fmt.bufPrint(&path, "{s}/model.safetensors.index.json", .{dir})) catch |err| {
            if ((kind == .qwen or kind == .gemma or kind == .flash) and err == error.FileNotFound) break :blk null;
            return err;
        };
        defer a.free(bytes);
        break :blk try std.json.parseFromSlice(std.json.Value, a, bytes, .{ .allocate = .alloc_always });
    };
    defer if (index) |parsed| parsed.deinit();
    if (kind == .flash and index == null) return checkFlashCheckpoint(io, dir);
    var files: std.ArrayList(safe.File) = .empty;
    defer {
        for (files.items) |*file| file.deinit();
        files.deinit(a);
    }
    var names = std.StringHashMap(usize).init(a);
    defer names.deinit();
    for (specs.value) |spec| {
        var geometry: [8]i32 = undefined;
        if (try adjusted(spec, cfg, &geometry, .BF16) == null) continue;
        var buffer: [512]u8 = undefined;
        const mtp = kind == .nemotron and std.mem.startsWith(u8, spec.name, "mtp.");
        var key = if (mtp) spec.name[4..] else try std.fmt.bufPrint(&buffer, "{s}{s}", .{ if (kind == .qwen or kind == .flash or kind == .gemma) "language_model." else "", spec.name });
        var alias: [512]u8 = undefined;
        const filename = if (mtp) "mtp-4bit.safetensors" else if (index == null) "model.safetensors" else blk: {
            const root = index.?.value;
            if (root != .object) return error.InvalidWeightIndex;
            const map = root.object.get("weight_map") orelse return error.InvalidWeightIndex;
            if (map != .object) return error.InvalidWeightIndex;
            if (kind == .flash) key = try @import("flash_names.zig").resolve(map.object, &alias, key);
            const value = map.object.get(key) orelse return error.MissingWeight;
            if (value != .string) return error.InvalidWeightIndex;
            break :blk value.string;
        };
        try safe.shardName(filename);
        const number = names.get(filename) orelse blk: {
            var file = safe.File.open(a, io, try std.fmt.bufPrint(&path, "{s}/{s}", .{ dir, filename })) catch |err| return if (mtp and err == error.FileNotFound) error.MissingDraftHead else err;
            files.append(a, file) catch |err| {
                file.deinit();
                return err;
            };
            const id = files.items.len - 1;
            try names.put(filename, id);
            break :blk id;
        };
        const tensor = files.items[number].header.tensors.get(key);
        const expected = (try adjusted(spec, cfg, &geometry, if (tensor) |t| t.dtype else null)).?;
        check(expected, if (tensor) |*t| .{ .dtype = t.dtype, .shape = t.shape() } else null) catch |err| {
            std.debug.print("Checkpoint schema: {s}: {s}\n", .{ spec.name, @errorName(err) });
            return err;
        };
    }
    std.debug.print("PASS: {s}: all {d} tensor names, shapes, dtypes and index references match the native recipe\n", .{ @tagName(kind), specs.value.len });
}

fn checkFlashCheckpoint(io: std.Io, dir: []const u8) !void {
    var checkpoint = try @import("safetensors.zig").Checkpoint.open(mx.allocator, io, dir);
    defer checkpoint.deinit();
    const specs = try std.json.parseFromSlice([]const Spec, mx.allocator, source(.flash), .{});
    defer specs.deinit();
    for (specs.value) |spec| {
        var name: [512]u8 = undefined;
        var alias: [512]u8 = undefined;
        const key = @import("flash_names.zig").resolve(checkpoint.tensors, &alias, try std.fmt.bufPrint(&name, "language_model.{s}", .{spec.name})) catch |err| {
            if (err == error.MissingWeight and std.mem.startsWith(u8, spec.name, "mtp.")) return error.MissingDraftHead;
            return err;
        };
        const file = checkpoint.tensors.get(key) orelse return error.MissingWeight;
        const tensor = checkpoint.files.items[file].header.tensors.get(key) orelse return error.MissingWeight;
        try check(spec, .{ .dtype = tensor.dtype, .shape = tensor.shape() });
    }
    std.debug.print("PASS: flash: all {d} tensor names, shapes and dtypes match the native recipe\n", .{specs.value.len});
}
test "mixed affine schema follows config and rejects malformed quantization tensors" {
    const config = try std.json.parseFromSlice(std.json.Value, std.testing.allocator,
        \\{"model_type":"qwen3_5","quantization":{"bits":3,"group_size":128,"model.layers.0.q":false}}
    , .{});
    defer config.deinit();
    var shape: [8]i32 = undefined;
    const weight = Spec{ .name = "model.embed_tokens.weight", .dtype = .U32, .shape = &.{ 248320, 640 } };
    const adjusted_weight = (try adjusted(weight, config.value, &shape, .U32)).?;
    try check(adjusted_weight, .{ .dtype = .U32, .shape = &.{ 248320, 480 } });
    try std.testing.expectError(error.InvalidTensorShape, check(adjusted_weight, .{ .dtype = .U32, .shape = weight.shape }));
    const scale = Spec{ .name = "model.embed_tokens.scales", .dtype = .BF16, .shape = &.{ 248320, 80 } };
    try check((try adjusted(scale, config.value, &shape, .F16)).?, .{ .dtype = .F16, .shape = &.{ 248320, 40 } });
    try std.testing.expectError(error.InvalidTensorDType, adjusted(scale, config.value, &shape, .U32));
    const dense = Spec{ .name = "model.layers.0.q.weight", .dtype = .U32, .shape = &.{ 48, 640 } };
    try check((try adjusted(dense, config.value, &shape, .F32)).?, .{ .dtype = .F32, .shape = &.{ 48, 5120 } });
    const dense_scale = Spec{ .name = "model.layers.0.q.scales", .dtype = .BF16, .shape = &.{ 48, 80 } };
    try std.testing.expectEqual(@as(?Spec, null), try adjusted(dense_scale, config.value, &shape, null));
}
test "Gemma schema preserves expert axes and eight-bit router packing" {
    const config = try std.json.parseFromSlice(std.json.Value, std.testing.allocator,
        \\{"quantization":{"bits":4,"group_size":32,"model.layers.0.router.proj":{"bits":8,"group_size":128}}}
    , .{});
    defer config.deinit();
    var shape: [8]i32 = undefined;
    const expert = Spec{ .name = "model.layers.0.experts.switch_glu.gate_proj.scales", .dtype = .BF16, .shape = &.{ 128, 704, 44 } };
    try check((try adjusted(expert, config.value, &shape, .BF16)).?, .{ .dtype = .BF16, .shape = &.{ 128, 704, 88 } });
    const router = Spec{ .name = "model.layers.0.router.proj.weight", .dtype = .U32, .shape = &.{ 128, 704 } };
    try check((try adjusted(router, config.value, &shape, .U32)).?, .{ .dtype = .U32, .shape = router.shape });
    const scales = Spec{ .name = "model.layers.0.router.proj.scales", .dtype = .BF16, .shape = &.{ 128, 44 } };
    try check((try adjusted(scales, config.value, &shape, .BF16)).?, .{ .dtype = .BF16, .shape = &.{ 128, 22 } });
}

test "checkpoint schemas are complete metadata sets with unique tensor names" {
    const counts = [_]usize{ 1847, 81, 763, 3414, 1339, 2481, 114160 };
    for (std.enums.values(Kind), counts) |kind, count| {
        const specs = try std.json.parseFromSlice([]const Spec, std.testing.allocator, source(kind), .{});
        defer specs.deinit();
        try std.testing.expectEqual(count, specs.value.len);
        var names = std.StringHashMap(void).init(std.testing.allocator);
        defer names.deinit();
        var head: usize = 0;
        for (specs.value) |spec| {
            const entry = try names.getOrPut(spec.name);
            try std.testing.expect(!entry.found_existing);
            try std.testing.expect(spec.name.len > 0 and spec.shape.len > 0 and spec.shape.len <= 8);
            for (spec.shape) |dim| try std.testing.expect(dim > 0);
            if (!required(spec, false)) head += 1;
        }
        try std.testing.expectEqual(kind == .nemotron or kind == .flash, head > 0);
    }
}
test "missing MTP, incompatible dtype, rank and tensor geometry fail before kernels" {
    const weight = Spec{ .name = "model.embed_tokens.weight", .dtype = .U32, .shape = &.{ 248320, 640 } };
    const head = Spec{ .name = "mtp.fc_hidden.weight", .dtype = .U32, .shape = &.{ 2560, 320 } };
    try check(weight, .{ .dtype = .U32, .shape = &.{ 248320, 640 } });
    try std.testing.expectError(error.MissingWeight, check(weight, null));
    try std.testing.expectError(error.MissingDraftHead, check(head, null));
    try std.testing.expectError(error.InvalidTensorDType, check(weight, .{ .dtype = .BF16, .shape = weight.shape }));
    try std.testing.expectError(error.InvalidTensorShape, check(weight, .{ .dtype = .U32, .shape = &.{ 248320, 320 } }));
    try std.testing.expectError(error.InvalidTensorShape, check(weight, .{ .dtype = .U32, .shape = &.{ 1, 248320, 640 } }));
    try std.testing.expectError(error.InvalidTensorShape, check(weight, .{ .dtype = .U32, .shape = &.{ 248319, 640 } }));
    try std.testing.expect(!required(head, false));
    try std.testing.expect(required(head, true));
    try std.testing.expect(required(weight, false));
}
