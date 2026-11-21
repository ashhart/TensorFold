//! A Qwen3 dense checkpoint's names and host reads: its config, the reranker's `model.` scope, head rows as fp32.
const std = @import("std");
const core = @import("core");
const config = @import("config.zig");

const quant = core.quant;
const Checkpoint = core.checkpoint.Checkpoint;
const Allocator = std.mem.Allocator;

/// The reranker's save keeps the base model under `model.` (beside `lm_head`); the embedding's has no prefix.
pub fn scopeOf(ck: *const Checkpoint) []const u8 {
    return if (ck.has("model.embed_tokens.weight")) "model." else "";
}

/// The head rows' stored name: `lm_head.weight`, or the embedding's (prefixed as the checkpoint stores it) when tied.
pub fn headKey(tied: bool, scope: []const u8) []const u8 {
    if (!tied) return "lm_head.weight";
    return if (scope.len > 0) "model.embed_tokens.weight" else "embed_tokens.weight";
}

pub fn readSpec(gpa: Allocator, io: std.Io, dir: []const u8) !config.Spec {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(bytes);
    return config.parse(gpa, bytes);
}

pub fn refuse(key: []const u8, why: []const u8) error{UnexpectedTensor} {
    std.log.err("{s} {s}", .{ key, why });
    return error.UnexpectedTensor;
}

/// `name` (e.g. "layers.0.mlp.up_proj.weight") as this checkpoint stores it: under `model.` for the reranker's save.
pub fn scopedKey(gpa: Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]u8 {
    var ck = try Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    return std.mem.concat(gpa, u8, &.{ scopeOf(&ck), name });
}

/// The 1-D tensor `key` (named as stored) as fp32 host values.
pub fn readVector(gpa: Allocator, io: std.Io, dir: []const u8, key: []const u8) ![]f32 {
    var ck = try Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    const t = try ck.get(key);
    if (t.rank != 1) return refuse(key, "is not a vector");
    var scratch: std.heap.ArenaAllocator = .init(gpa);
    defer scratch.deinit();
    const wide = try quant.convert.float32(scratch.allocator(), t);
    const out = try gpa.alloc(f32, t.shape[0]);
    for (out, 0..) |*x, j| x.* = @bitCast(std.mem.readInt(u32, wide.bytes[4 * j ..][0..4], .little));
    return out;
}

/// Rows `rows` of the 2-D tensor `key` (named as stored, no prefix) as fp32 host values, row after row.
pub fn readRows(gpa: Allocator, io: std.Io, dir: []const u8, key: []const u8, rows: []const u32) ![]f32 {
    var ck = try Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    const t = try ck.get(key);
    if (t.rank != 2) return refuse(key, "is not a matrix");
    const width = t.shape[1];
    const out = try gpa.alloc(f32, rows.len * width);
    errdefer gpa.free(out);
    const size = t.dtype.size();
    for (rows, 0..) |r, i| {
        if (r >= t.shape[0]) {
            std.log.warn("{s} has no row {d} of {d}", .{ key, r, t.shape[0] });
            return error.UnexpectedTensor;
        }
        var shape = t.shape;
        shape[0] = width;
        const one: quant.Tensor = .{ .bytes = t.bytes[r * width * size ..][0 .. width * size], .dtype = t.dtype, .rank = 1, .shape = shape };
        var scratch: std.heap.ArenaAllocator = .init(gpa);
        defer scratch.deinit();
        const wide = try quant.convert.float32(scratch.allocator(), one);
        // little-endian fp32 words, read one by one: a mapped fp32 tensor need not be 4-byte aligned
        for (out[i * width ..][0..width], 0..) |*x, j| x.* = @bitCast(std.mem.readInt(u32, wide.bytes[4 * j ..][0..4], .little));
    }
    return out;
}

fn writeFixture(dir: std.Io.Dir, prefixed: bool) !void {
    const fixture = core.checkpoint_fixture;
    const a = std.testing.allocator;
    // lm_head: 3 rows of 2 bf16 values: row r is (r + 1, -(r + 1)) as 1.0, 2.0, 3.0 and their negatives
    const head = [_]u16{ 0x3f80, 0xbf80, 0x4000, 0xc000, 0x4040, 0xc040 };
    const embed = [_]u16{ 0, 0, 0, 0, 0, 0 };
    try fixture.write(a, std.testing.io, dir, "model.safetensors", &.{
        .{ .name = if (prefixed) "model.embed_tokens.weight" else "embed_tokens.weight", .dtype = "BF16", .shape = &.{ 3, 2 }, .bytes = std.mem.sliceAsBytes(&embed) },
        .{ .name = "lm_head.weight", .dtype = "BF16", .shape = &.{ 3, 2 }, .bytes = std.mem.sliceAsBytes(&head) },
    });
}

test "the reranker's model. prefix is found, the embedding's bare names are not, and lm_head rows read as fp32" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    for ([_]bool{ true, false }) |prefixed| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        try writeFixture(tmp.dir, prefixed);
        const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
        defer a.free(dir);
        var ck = try Checkpoint.openModel(a, io, dir);
        defer ck.close();
        try std.testing.expectEqualStrings(if (prefixed) "model." else "", scopeOf(&ck));
        const rows = try readRows(a, io, dir, "lm_head.weight", &.{ 2, 0 });
        defer a.free(rows);
        try std.testing.expectEqualSlices(f32, &.{ 3, -3, 1, -1 }, rows);
        try std.testing.expectError(error.UnexpectedTensor, readRows(a, io, dir, "lm_head.weight", &.{3}));
        // a tied model reads its head rows from the embedding, as stored
        const tied = try readRows(a, io, dir, headKey(true, scopeOf(&ck)), &.{1});
        defer a.free(tied);
        try std.testing.expectEqualSlices(f32, &.{ 0, 0 }, tied);
        try std.testing.expectEqualStrings("lm_head.weight", headKey(false, scopeOf(&ck)));
    }
}
