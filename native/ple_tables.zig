//! Read only the packed PLE rows selected by n-gram IDs, not entire 250 MB shards.
const std = @import("std");
const mx = @import("mlx.zig");
const safe = @import("safetensors.zig");
const Ref = struct { file: usize, tensor: safe.Tensor };
pub const Tables = struct {
    resident: ?@import("ple_resident.zig").Resident = null,
    files: std.ArrayList(safe.File) = .empty,
    rows: [128][3]Ref = undefined,
    starts: [129]i64 = undefined,
    pub fn deinit(t: *Tables) void {
        if (t.resident) |*r| r.deinit();
        for (t.files.items) |*file| file.deinit();
        t.files.deinit(mx.allocator);
    }
    pub fn makeResident(t: *Tables) !void {
        if (t.resident != null) return;
        t.resident = try @import("ple_resident.zig").Resident.init(t);
    }
    pub fn init(io: std.Io, dir: []const u8) !Tables {
        var t = Tables{};
        errdefer t.deinit();
        var path: [4096]u8 = undefined;
        const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&path, "{s}/model.safetensors.index.json", .{dir}));
        defer mx.allocator.free(bytes);
        const parsed = try std.json.parseFromSlice(std.json.Value, mx.allocator, bytes, .{});
        defer parsed.deinit();
        if (parsed.value != .object) return error.InvalidWeightIndex;
        const index = parsed.value.object.get("weight_map") orelse return error.InvalidWeightIndex;
        if (index != .object) return error.InvalidWeightIndex;
        var files = std.StringHashMap(usize).init(mx.allocator);
        defer files.deinit();
        t.starts[0] = 0;
        for (0..128) |shard| {
            var name: [256]u8 = undefined;
            inline for (.{ "weight", "scales", "biases" }, 0..) |suffix, part| {
                const key = try std.fmt.bufPrint(&name, "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_{d}.{s}", .{ shard, suffix });
                const filename = index.object.get(key) orelse return error.MissingPLEWeight;
                if (filename != .string) return error.InvalidWeightIndex;
                try safe.shardName(filename.string);
                const file_index = files.get(filename.string) orelse blk: {
                    var file = try safe.File.open(mx.allocator, io, try std.fmt.bufPrint(&path, "{s}/{s}", .{ dir, filename.string }));
                    t.files.append(mx.allocator, file) catch |err| {
                        file.deinit();
                        return err;
                    };
                    const number = t.files.items.len - 1;
                    try files.put(filename.string, number);
                    break :blk number;
                };
                const tensor = t.files.items[file_index].header.tensors.get(key) orelse return error.MissingPLEWeight;
                if (tensor.rank != 2 or tensor.dims[0] <= 0 or tensor.dims[1] != (if (part == 0) @as(i32, 20) else 5)) return error.InvalidTensorShape;
                if (tensor.dtype != (if (part == 0) safe.DType.U32 else safe.DType.BF16)) return error.InvalidTensorDType;
                t.rows[shard][part] = .{ .file = file_index, .tensor = tensor };
                if (part > 0 and tensor.dims[0] != t.rows[shard][0].tensor.dims[0]) return error.InvalidTensorShape;
            }
            t.starts[shard + 1] = t.starts[shard] + t.rows[shard][0].tensor.dims[0];
        }
        const n = @import("ngram.zig").NGram.init();
        const total = std.mem.alignForward(i64, n.offsets[15] + n.sizes[15], 128);
        if (t.starts[128] != total) return error.InvalidTensorShape;
        return t;
    }
    pub fn locate(t: *const Tables, id: i64) !struct { shard: usize, row: usize } {
        if (id < 0 or id >= t.starts[128]) return error.InvalidToken;
        var lo: usize = 0;
        var hi: usize = 128;
        while (lo + 1 < hi) {
            const mid = (lo + hi) / 2;
            if (id < t.starts[mid]) hi = mid else lo = mid;
        }
        return .{ .shard = lo, .row = @intCast(id - t.starts[lo]) };
    }
    pub fn gather(t: *const Tables, s: *mx.Scope, ids: []const i64) !mx.Array {
        if (ids.len == 0 or ids.len > 16 * 16) return error.InvalidLaneWidth;
        var weights: [16 * 16 * 20]u32 = undefined;
        var scales: [16 * 16 * 5]u16 = undefined;
        var biases: [16 * 16 * 5]u16 = undefined;
        for (ids, 0..) |id, i| {
            const loc = try t.locate(id);
            const refs = t.rows[loc.shard];
            inline for (.{ &weights, &scales, &biases }, 0..) |buffer, part| {
                const width = if (part == 0) 20 else 5;
                try t.files.items[refs[part].file].readRow(refs[part].tensor, loc.row, std.mem.sliceAsBytes(buffer[i * width ..][0..width]));
            }
        }
        const count: i32 = @intCast(ids.len);
        return @import("checkpoint.zig").dequantize(s, .{
            try s.data(&weights, &.{ count, 20 }, mx.c.MLX_UINT32),
            try s.data(&scales, &.{ count, 5 }, mx.bf16),
            try s.data(&biases, &.{ count, 5 }, mx.bf16),
        }, 32);
    }
    pub fn check(io: std.Io, dir: []const u8) !void {
        try mx.init();
        defer mx.shutdown();
        var tables = try Tables.init(io, dir);
        defer tables.deinit();
        const cp = @import("checkpoint.zig");
        for (0..128) |shard| {
            var scope = mx.Scope{};
            defer scope.deinit();
            var oracle = cp.Store.init(32);
            defer oracle.deinit();
            // MLX independently parses/loads the complete packed tensor; it is freed
            // after this shard so this diagnostic also has a bounded working set.
            var name: [256]u8 = undefined;
            const key = try std.fmt.bufPrint(&name, "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_{d}", .{shard});
            var files: [3]usize = @splat(std.math.maxInt(usize));
            for (tables.rows[shard], 0..) |ref, part| {
                if (std.mem.indexOfScalar(usize, files[0..part], ref.file) == null) try oracle.loadFile(io, tables.files.items[ref.file].path, "", "");
                files[part] = ref.file;
            }
            const end = tables.starts[shard + 1] - tables.starts[shard];
            const rows = [_]i32{ 0, 1, @intCast(@divTrunc(end, 2)), @intCast(end - 2), @intCast(end - 1) };
            var ids: [5]i64 = undefined;
            for (rows, &ids) |row, *id| id.* = tables.starts[shard] + row;
            try @import("sampling_checks.zig").equal(&scope, try oracle.embed(&scope, key, &rows), try tables.gather(&scope, &ids));
        }
        std.debug.print("PASS: 640 PLE rows (first, last, adjacent and middle) across all 128 shards exactly match independent MLX reads/dequantization\n", .{});
    }
    pub fn checkResident(io: std.Io, dir: []const u8) !void {
        try mx.init();
        defer mx.shutdown();
        {
            var tables = try Tables.init(io, dir);
            defer tables.deinit();
            try tables.makeResident();
            var kernels = mx.Kernels.init();
            defer kernels.deinit();
            for (0..128) |shard| {
                var s = mx.Scope{};
                defer s.deinit();
                const end = tables.starts[shard + 1] - tables.starts[shard];
                const rows = [_]i64{ 0, 1, @divTrunc(end, 2), end - 2, end - 1 };
                var ids: [5 * 16]i64 = undefined;
                for (&ids, 0..) |*id, i| id.* = tables.starts[shard] + rows[@divTrunc(i, 16)];
                const input = try s.cast(try s.data(&ids, &.{ 5, 16 }, mx.c.MLX_INT64), mx.c.MLX_UINT32);
                const actual = try tables.resident.?.gather(&kernels, &s, input);
                const expected = try s.reshape(try tables.gather(&s, &ids), &.{ 5, 2560 });
                try @import("sampling_checks.zig").equal(&s, actual, expected);
            }
            var peak: usize = 0;
            try mx.check(mx.c.mlx_get_peak_memory(&peak));
            const packed_bytes: usize = @intCast(tables.starts[128] * 100);
            if (peak > packed_bytes + 32 * 1024 * 1024) return error.ResidentLoadingPeakExceeded;
            std.debug.print("PASS: resident/bounded PLE at 640 shard boundary/interior rows; peak MLX bytes {d}, packed bytes {d}\n", .{ peak, packed_bytes });
        }
        try mx.check(mx.c.mlx_synchronize(mx.stream));
        var active: usize = 0;
        try mx.check(mx.c.mlx_get_active_memory(&active));
        try std.testing.expectEqual(@as(usize, 0), active);
    }
};
test "PLE shard lookup handles every edge and rejects out-of-table IDs" {
    var t = Tables{};
    for (&t.starts, 0..) |*start, i| start.* = @intCast(i * 97);
    for (0..128) |i| {
        const first = try t.locate(@intCast(i * 97));
        const last = try t.locate(@intCast((i + 1) * 97 - 1));
        try std.testing.expectEqual(i, first.shard);
        try std.testing.expectEqual(@as(usize, 0), first.row);
        try std.testing.expectEqual(i, last.shard);
        try std.testing.expectEqual(@as(usize, 96), last.row);
    }
    try std.testing.expectError(error.InvalidToken, t.locate(-1));
    try std.testing.expectError(error.InvalidToken, t.locate(t.starts[128]));
}
