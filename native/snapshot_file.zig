const std = @import("std");
const mx = @import("mlx.zig");
const safe = @import("safetensors.zig");
const kv = @import("kv_buffer.zig");
const V = std.json.Value;

// Capacity buffers and forward-pass writes are rebuilt from the logical cache arrays.
fn transient(comptime T: type) bool {
    return T == kv.Buffer or T == kv.Write;
}

fn encode(a: std.mem.Allocator, arrays: mx.c.mlx_map_string_to_array, path: [:0]const u8, value: anytype) anyerror!V {
    const T = @TypeOf(value);
    if (comptime transient(T)) return .null;
    if (T == mx.Array) {
        if (value.ctx == null) return .null;
        try mx.check(mx.c.mlx_map_string_to_array_insert(arrays, path, value));
        return .{ .string = path };
    }
    return switch (@typeInfo(T)) {
        .void => .null,
        .bool => .{ .bool = value },
        .int => .{ .number_string = try std.fmt.allocPrint(a, "{d}", .{value}) },
        .optional => if (value) |present| try encode(a, arrays, path, present) else .null,
        .@"struct" => blk: {
            var fields: std.json.ObjectMap = .empty;
            inline for (comptime std.meta.fieldNames(T)) |field| {
                const key = try std.fmt.allocPrintSentinel(a, "{s}.{s}", .{ path, field }, 0);
                try fields.put(a, field, try encode(a, arrays, key, @field(value, field)));
            }
            break :blk .{ .object = fields };
        },
        .array, .pointer => blk: {
            if (@typeInfo(T) == .pointer and @typeInfo(T).pointer.size != .slice) @compileError("Snapshot pointers must be owned slices");
            var items = std.array_list.Managed(V).init(a);
            for (value, 0..) |item, i| {
                const key = try std.fmt.allocPrintSentinel(a, "{s}.{d}", .{ path, i }, 0);
                try items.append(try encode(a, arrays, key, item));
            }
            break :blk .{ .array = items };
        },
        else => @compileError("Unsupported snapshot field: " ++ @typeName(T)),
    };
}

fn release(value: anytype) void {
    const T = @TypeOf(value.*);
    if (comptime transient(T)) return;
    if (T == mx.Array) return mx.free(value.*);
    switch (@typeInfo(T)) {
        .@"struct" => inline for (comptime std.meta.fieldNames(T)) |field| release(&@field(value, field)),
        .array => for (value) |*item| release(item),
        .pointer => {
            for (value.*) |*item| release(item);
            mx.allocator.free(value.*);
        },
        .optional => if (value.*) |*item| release(item),
        .void, .bool, .int => {},
        else => @compileError("Unsupported snapshot field: " ++ @typeName(T)),
    }
}

fn decode(comptime T: type, file: *safe.File, a: std.mem.Allocator, path: []const u8, value: V, count: *usize) anyerror!T {
    if (comptime transient(T)) {
        if (value != .null) return error.InvalidSnapshotState;
        return .{};
    }
    if (T == mx.Array) {
        if (value == .null) return mx.empty;
        if (value != .string or !std.mem.eql(u8, path, value.string)) return error.InvalidSnapshotArray;
        const tensor = file.header.tensors.get(path) orelse return error.MissingSnapshotArray;
        const dtype: mx.c.mlx_dtype = switch (tensor.dtype) {
            .BOOL => mx.c.MLX_BOOL,
            .U8 => mx.c.MLX_UINT8,
            .I8 => mx.c.MLX_INT8,
            .U16 => mx.c.MLX_UINT16,
            .I16 => mx.c.MLX_INT16,
            .U32 => mx.c.MLX_UINT32,
            .I32 => mx.c.MLX_INT32,
            .U64 => mx.c.MLX_UINT64,
            .I64 => mx.c.MLX_INT64,
            .F16 => mx.c.MLX_FLOAT16,
            .BF16 => mx.c.MLX_BFLOAT16,
            .F32 => mx.c.MLX_FLOAT32,
            else => return error.InvalidSnapshotDType,
        };
        const bytes = try mx.allocator.alloc(u8, @intCast(tensor.len));
        defer mx.allocator.free(bytes);
        if (try file.file.readPositionalAll(file.io, bytes, file.data_offset + tensor.offset) != bytes.len) return error.TruncatedSafetensors;
        const out = mx.c.mlx_array_new_data(bytes.ptr, tensor.dims[0..tensor.rank].ptr, @intCast(tensor.rank), dtype);
        errdefer mx.free(out);
        if (out.ctx == null) return error.InvalidSnapshotArray;
        try mx.eval(out);
        count.* += 1;
        return out;
    }
    return switch (@typeInfo(T)) {
        .void => if (value == .null) {} else error.InvalidSnapshotState,
        .bool => if (value == .bool) value.bool else error.InvalidSnapshotState,
        .int => switch (value) {
            .integer => std.math.cast(T, value.integer) orelse error.InvalidSnapshotState,
            .number_string => std.fmt.parseInt(T, value.number_string, 10) catch error.InvalidSnapshotState,
            else => error.InvalidSnapshotState,
        },
        .optional => |info| if (value == .null) null else try decode(info.child, file, a, path, value, count),
        .@"struct" => blk: {
            if (value != .object or value.object.count() != (comptime std.meta.fieldNames(T)).len) return error.InvalidSnapshotState;
            var out = std.mem.zeroes(T);
            errdefer release(&out);
            inline for (comptime std.meta.fieldNames(T)) |field| {
                const item = value.object.get(field) orelse return error.InvalidSnapshotState;
                const key = try std.fmt.allocPrint(a, "{s}.{s}", .{ path, field });
                @field(out, field) = try decode(@FieldType(T, field), file, a, key, item, count);
            }
            break :blk out;
        },
        .array, .pointer => blk: {
            if (value != .array) return error.InvalidSnapshotState;
            const child = if (@typeInfo(T) == .array) @typeInfo(T).array.child else @typeInfo(T).pointer.child;
            var out: T = if (@typeInfo(T) == .array) std.mem.zeroes(T) else try mx.allocator.alloc(child, value.array.items.len);
            if (@typeInfo(T) == .array) {
                if (out.len != value.array.items.len) return error.InvalidSnapshotState;
            } else @memset(out, std.mem.zeroes(child));
            errdefer release(&out);
            for (value.array.items, 0..) |item, i| {
                const key = try std.fmt.allocPrint(a, "{s}.{d}", .{ path, i });
                out[i] = try decode(child, file, a, key, item, count);
            }
            break :blk out;
        },
        else => @compileError("Unsupported snapshot field: " ++ @typeName(T)),
    };
}

const Metadata = struct {
    format: u32,
    identity: []const u8,
    dependencies: []const u8,
    tensor_backend: bool,
    state_type: []const u8,
    tokens: []const i32,
    state: V,
};

/// identity must identify the checkpoint, drafter and native arithmetic revision.
pub fn save(io: std.Io, path: []const u8, identity: []const u8, tokens: []const i32, state: anytype) !void {
    var arena = std.heap.ArenaAllocator.init(mx.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const arrays = mx.c.mlx_map_string_to_array_new();
    defer _ = mx.c.mlx_map_string_to_array_free(arrays);
    const document = Metadata{ .format = 1, .dependencies = @embedFile("dependencies.json"), .identity = identity, .tensor_backend = mx.tensor_units, .state_type = @typeName(@TypeOf(state)), .tokens = tokens, .state = try encode(a, arrays, "state", state) };
    const json = try std.json.Stringify.valueAlloc(a, document, .{});
    const metadata = mx.c.mlx_map_string_to_string_new();
    defer _ = mx.c.mlx_map_string_to_string_free(metadata);
    try mx.check(mx.c.mlx_map_string_to_string_insert(metadata, "tensorfold_native", try a.dupeSentinel(u8, json, 0)));
    try std.Io.Dir.cwd().createDirPath(io, std.fs.path.dirname(path) orelse ".");
    var nonce: [16]u8 = undefined;
    io.random(&nonce);
    const partial = try std.fmt.allocPrintSentinel(a, "{s}.{x}.partial.safetensors", .{ path, nonce }, 0);
    const reservation = try std.Io.Dir.cwd().createFile(io, partial, .{ .exclusive = true });
    reservation.close(io);
    defer std.Io.Dir.cwd().deleteFile(io, partial) catch {};
    try mx.check(mx.c.mlx_save_safetensors(partial, arrays, metadata));
    try safe.validateFile(io, partial);
    try std.Io.Dir.cwd().rename(partial, .cwd(), path, io);
}

pub const Reader = struct {
    file: safe.File,
    metadata: std.json.Parsed(Metadata),

    pub fn open(io: std.Io, path: []const u8, identity: []const u8) !Reader {
        var file = try safe.File.open(mx.allocator, io, path);
        errdefer file.deinit();
        const meta = file.header.parsed.value.object.get("__metadata__") orelse return error.IncompatibleSnapshot;
        if (meta != .object) return error.IncompatibleSnapshot;
        const json = meta.object.get("tensorfold_native") orelse return error.IncompatibleSnapshot;
        if (json != .string) return error.IncompatibleSnapshot;
        const metadata = try std.json.parseFromSlice(Metadata, mx.allocator, json.string, .{ .allocate = .alloc_always });
        errdefer metadata.deinit();
        const v = metadata.value;
        if (v.format != 1 or !std.mem.eql(u8, v.identity, identity) or !std.mem.eql(u8, v.dependencies, @embedFile("dependencies.json")) or v.tensor_backend != mx.tensor_units) return error.IncompatibleSnapshot;
        return .{ .file = file, .metadata = metadata };
    }

    pub fn deinit(r: *Reader) void {
        r.metadata.deinit();
        r.file.deinit();
    }

    pub fn load(r: *Reader, comptime T: type) !T {
        if (!std.mem.eql(u8, r.metadata.value.state_type, @typeName(T))) return error.IncompatibleSnapshot;
        var arena = std.heap.ArenaAllocator.init(mx.allocator);
        defer arena.deinit();
        var count: usize = 0;
        var out = try decode(T, &r.file, arena.allocator(), "state", r.metadata.value.state, &count);
        errdefer release(&out);
        if (count != r.file.header.tensors.count()) return error.UnexpectedSnapshotArray;
        return out;
    }
};

const Fixture = struct {
    arrays: []mx.Array,
    optional: ?[2]u64,
    buffer: kv.Buffer,
    write: kv.Write,
    position: i32,
};

pub fn exercise(io: std.Io) !void {
    var scope = mx.Scope{};
    defer scope.deinit();
    const original = try scope.ints(&.{ 17, 31, 67, 89 });
    const sliced = try scope.slice(try scope.reshape(original, &.{ 2, 2 }), 1, 0, 1);
    const bf16 = try scope.cast(original, mx.bf16);
    var arrays = [_]mx.Array{ original, sliced, bf16, mx.empty };
    const state = Fixture{ .arrays = &arrays, .optional = .{ 0, std.math.maxInt(u64) }, .buffer = .{ .current = original, .offset = 4 }, .write = .{}, .position = 4 };
    const path = "build/native-checks/snapshot-fixture.safetensors";
    try save(io, path, "snapshot-fixture", &.{ 1, 2, 3, 4 }, state);
    var reader = try Reader.open(io, path, "snapshot-fixture");
    defer reader.deinit();
    try std.testing.expectEqual(@as(usize, 3), reader.file.header.tensors.count());
    var loaded = try reader.load(Fixture);
    defer release(&loaded);
    try std.testing.expectEqualDeep(state.optional, loaded.optional);
    try std.testing.expectEqual(state.position, loaded.position);
    try std.testing.expectEqual(@as(usize, 4), loaded.arrays.len);
    try std.testing.expect(loaded.buffer.current.ctx == null);
    try std.testing.expect(loaded.arrays[3].ctx == null);
    for (state.arrays[0..3], loaded.arrays[0..3]) |before, after| {
        try std.testing.expectEqualSlices(i32, mx.shape(before), mx.shape(after));
        try std.testing.expectEqual(mx.dtype(before), mx.dtype(after));
        try @import("sampling_checks.zig").equal(&scope, before, after);
    }
}

pub fn check(io: std.Io) !void {
    try exercise(io);
    const path = "build/native-checks/snapshot-fixture.safetensors";
    try std.testing.expectError(error.IncompatibleSnapshot, Reader.open(io, path, "other-model"));
    const backend = mx.tensor_units;
    mx.tensor_units = !backend;
    const wrong_backend = Reader.open(io, path, "snapshot-fixture");
    mx.tensor_units = backend;
    try std.testing.expectError(error.IncompatibleSnapshot, wrong_backend);
    var reader = try Reader.open(io, path, "snapshot-fixture");
    defer reader.deinit();
    try std.testing.expectError(error.IncompatibleSnapshot, reader.load(struct { position: i32 }));
    const position = reader.metadata.value.state.object.getPtr("position").?;
    const saved_position = position.*;
    for ([_]V{ .null, .{ .integer = std.math.maxInt(i64) }, .{ .string = "4" } }) |invalid| {
        position.* = invalid;
        try std.testing.expectError(error.InvalidSnapshotState, reader.load(Fixture));
    }
    position.* = saved_position;
    const first = &reader.metadata.value.state.object.getPtr("arrays").?.array.items[0];
    const saved_first = first.*;
    first.* = .{ .string = "state.arrays.1" };
    try std.testing.expectError(error.InvalidSnapshotArray, reader.load(Fixture));
    first.* = .null;
    try std.testing.expectError(error.UnexpectedSnapshotArray, reader.load(Fixture));
    first.* = saved_first;
    const count = reader.metadata.value.state.object.getPtr("arrays").?.array.items.len;
    reader.metadata.value.state.object.getPtr("arrays").?.array.items.len = 1;
    try std.testing.expectError(error.UnexpectedSnapshotArray, reader.load(Fixture));
    reader.metadata.value.state.object.getPtr("arrays").?.array.items.len = count;
    var loaded = try reader.load(Fixture);
    defer release(&loaded);
    // A file shortened after metadata inspection is still rejected by positional reads.
    const writer = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = false });
    defer writer.close(io);
    try writer.setLength(io, reader.file.data_offset + 1);
    try std.testing.expectError(error.TruncatedSafetensors, reader.load(Fixture));
    std.debug.print("PASS: snapshot identity/backend/type isolation, noncontiguous and BF16 arrays, malformed state, orphan arrays and truncated reads\n", .{});
}
