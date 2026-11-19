//! Nemotron-H's token table stored bf16: its rows kept as the file holds them, gathered with no arithmetic.
const std = @import("std");
const core = @import("core");
const config = @import("../config.zig");
const tensors = @import("tensors.zig");

const quant = core.quant;
const Tensor = core.checkpoint.Tensor;

/// The bf16 table `t` holds, its file bytes shared and not widened: [vocab, hidden], else UnexpectedTensor.
pub fn read(a: std.mem.Allocator, t: anytype, c: *const config.Config) !quant.Host {
    const h = try quant.readStored(a, t, tensors.embed);
    if (h != .dense) return error.UnexpectedTensor;
    const w: Tensor = h.dense.weight;
    if (w.dtype != .bf16 or w.rank != 2 or w.shape[0] != c.vocab or w.shape[1] != c.hidden) return error.UnexpectedTensor;
    return h;
}

const testing = std.testing;
const round = @import("round.zig");

const rows = 5;
const hidden = 8;

/// A small table of every kind of bf16 pattern: signed zeros, a subnormal, infinities, NaNs and a mixed hash.
fn pattern(out: []u16) void {
    const special = [_]u16{ 0x0000, 0x8000, 0x0001, 0x7F80, 0xFF80, 0x7FC1, 0xFFFF, 0x3F80 };
    for (out, 0..) |*v, i| v.* = if (i < special.len) special[i] else @truncate((i + 1) *% 0x9E37 ^ (i >> 3) *% 0x7F4B);
}

/// The Lightning config cut down to a `rows` x `hidden` table.
fn small() !config.Config {
    var why: config.Why = .{};
    var c = try config.parse(testing.allocator, @embedFile("../fixtures/config.json"), &why);
    c.vocab = rows;
    c.hidden = hidden;
    c.embed_bf16 = true;
    return c;
}

/// `model.safetensors` in `dir` holding the entry `name` with `dtype`, `shape` and `bytes`.
fn write(dir: std.Io.Dir, name: []const u8, dtype: []const u8, shape: []const usize, bytes: []const u8) !void {
    const entry: core.checkpoint_fixture.Entry = .{ .name = name, .dtype = dtype, .shape = shape, .bytes = bytes };
    try core.checkpoint_fixture.write(testing.allocator, testing.io, dir, "model.safetensors", &.{entry});
}

/// The checkpoint as the loader's table reads it: tensors by name, no width of its own.
const Table = struct {
    ck: *core.checkpoint.Checkpoint,

    pub fn has(t: *const Table, key: []const u8) bool {
        return t.ck.has(key);
    }

    pub fn get(t: *const Table, key: []const u8) !Tensor {
        return t.ck.get(key);
    }

    pub fn width(_: *const Table, _: []const u8) !quant.mlx.Width {
        return error.UnsupportedQuantization;
    }
};

fn opened(dir: std.Io.Dir) !core.checkpoint.Checkpoint {
    const path = try dir.realPathFileAlloc(testing.io, ".", testing.allocator);
    defer testing.allocator.free(path);
    var ck: core.checkpoint.Checkpoint = .{ .gpa = testing.allocator, .io = testing.io };
    errdefer ck.close();
    try ck.add(path, "model.safetensors");
    return ck;
}

test "the gathered rows are the file's bytes, first row and last, widened exactly; an id past the table is refused" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var values: [rows * hidden]u16 = undefined;
    pattern(&values);
    const file = std.mem.sliceAsBytes(&values);
    try write(tmp.dir, tensors.embed ++ ".weight", "BF16", &.{ rows, hidden }, file);
    var ck = try opened(tmp.dir);
    defer ck.close();
    const c = try small();
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const h = try read(arena.allocator(), &Table{ .ck = &ck }, &c);
    const table = h.dense.weight;
    try testing.expectEqual(core.checkpoint.DType.bf16, table.dtype);
    try testing.expectEqual(@as(usize, rows), h.rows());
    // the row the gather launch reads for `id`: `hidden` values at id * hidden, as stored
    for ([_]usize{ 0, 1, rows - 2, rows - 1 }) |id| {
        const row = table.bytes[id * hidden * 2 ..][0 .. hidden * 2];
        try testing.expectEqualSlices(u8, file[id * hidden * 2 ..][0 .. hidden * 2], row);
        for (0..hidden) |col| {
            const stored = std.mem.readInt(u16, row[col * 2 ..][0..2], .little);
            const widened = try quant.convert.float32(arena.allocator(), table);
            const bits = std.mem.readInt(u32, widened.bytes[(id * hidden + col) * 4 ..][0..4], .little);
            try testing.expectEqual(@as(u32, stored) << 16, bits);
        }
    }
    // no kernel bounds an id: a round refuses one at the table's row count before its plan is written
    var p: round.Plan = .{ .host = undefined, .dev = undefined, .rows = 4, .slots = 1, .pool = 0, .pages = 0, .vocab = h.rows() };
    const last: round.Window = .{ .s = undefined, .pos = 0, .tokens = &.{rows} };
    try testing.expectError(error.TokenOutsideVocabulary, p.send(&.{last}, 4096, undefined));
}

test "a table that is not bf16 [vocab, hidden] is refused" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try small();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const wide: [rows * hidden]f32 = @splat(0);
    try write(tmp.dir, tensors.embed ++ ".weight", "F32", &.{ rows, hidden }, std.mem.sliceAsBytes(&wide));
    var ck = try opened(tmp.dir);
    defer ck.close();
    try testing.expectError(error.UnexpectedTensor, read(a, &Table{ .ck = &ck }, &c));
    var short: [(rows - 1) * hidden]u16 = undefined;
    pattern(&short);
    try write(tmp.dir, tensors.embed ++ ".weight", "BF16", &.{ rows - 1, hidden }, std.mem.sliceAsBytes(&short));
    var again = try opened(tmp.dir);
    defer again.close();
    try testing.expectError(error.UnexpectedTensor, read(a, &Table{ .ck = &again }, &c));
}
