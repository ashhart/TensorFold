//! Complete metadata contracts for the four supported, fixed checkpoint recipes.
const std = @import("std");
const mx = @import("mlx.zig");
const DType = @import("safetensors.zig").DType;
pub const Kind = enum { qwen, dflash, nemotron, flash };
const Spec = struct { name: []const u8, dtype: DType, shape: []const i32 };
const Metadata = struct { dtype: DType, shape: []const i32 };
fn source(kind: Kind) []const u8 {
    return switch (kind) {
        .qwen => @embedFile("schemas/qwen.json"),
        .dflash => @embedFile("schemas/dflash.json"),
        .nemotron => @embedFile("schemas/nemotron.json"),
        .flash => @embedFile("schemas/flash.json"),
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
    const specs = try std.json.parseFromSlice([]const Spec, mx.allocator, source(kind), .{});
    defer specs.deinit();
    for (specs.value) |spec| {
        if (!required(spec, drafts)) continue;
        const array = arrays.get(spec.name);
        const actual: ?Metadata = if (array) |a| .{
            .dtype = switch (mx.dtype(a)) {
                mx.c.MLX_BFLOAT16 => .BF16,
                mx.c.MLX_FLOAT32 => .F32,
                mx.c.MLX_UINT32 => .U32,
                mx.c.MLX_INT64 => .I64,
                else => return error.InvalidTensorDType,
            },
            .shape = mx.shape(a),
        } else null;
        check(spec, actual) catch |err| {
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
    const index: ?std.json.Parsed(std.json.Value) = if (kind == .dflash) null else blk: {
        const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&path, "{s}/model.safetensors.index.json", .{dir}));
        defer a.free(bytes);
        break :blk try std.json.parseFromSlice(std.json.Value, a, bytes, .{ .allocate = .alloc_always });
    };
    defer if (index) |parsed| parsed.deinit();
    var files: std.ArrayList(safe.File) = .empty;
    defer {
        for (files.items) |*file| file.deinit();
        files.deinit(a);
    }
    var names = std.StringHashMap(usize).init(a);
    defer names.deinit();
    for (specs.value) |spec| {
        var buffer: [512]u8 = undefined;
        const mtp = kind == .nemotron and std.mem.startsWith(u8, spec.name, "mtp.");
        const key = if (mtp) spec.name[4..] else try std.fmt.bufPrint(&buffer, "{s}{s}", .{ if (kind == .qwen or kind == .flash) "language_model." else "", spec.name });
        const filename = if (mtp) "mtp-4bit.safetensors" else if (kind == .dflash) "model.safetensors" else blk: {
            const root = index.?.value;
            if (root != .object) return error.InvalidWeightIndex;
            const map = root.object.get("weight_map") orelse return error.InvalidWeightIndex;
            if (map != .object) return error.InvalidWeightIndex;
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
        check(spec, if (tensor) |*t| .{ .dtype = t.dtype, .shape = t.shape() } else null) catch |err| {
            std.debug.print("Checkpoint schema: {s}: {s}\n", .{ spec.name, @errorName(err) });
            return err;
        };
    }
    std.debug.print("PASS: {s}: all {d} tensor names, shapes, dtypes and index references match the native recipe\n", .{ @tagName(kind), specs.value.len });
}
test "checkpoint schemas are complete metadata sets with unique tensor names" {
    const counts = [_]usize{ 1847, 81, 763, 3414 };
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
