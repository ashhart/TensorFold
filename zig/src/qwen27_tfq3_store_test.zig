//! Real mmap/no-copy Store ownership checks with the recording Metal boundary.
const std = @import("std");
const mtl = @import("metal");
const tfq3 = @import("families/qwen27/tfq3.zig");
const check = tfq3.check;

const Spec = struct {
    name: []const u8,
    bits: u32 = 3,
    word_off: ?u64 = null,
    rows: u32 = 128,
    cols: u32 = 128,
};

fn writeTablesLen(a: std.mem.Allocator, dir: std.Io.Dir, len: usize) !void {
    const bytes = try a.alloc(u8, len);
    defer a.free(bytes);
    @memset(bytes, 0);
    std.mem.writeInt(u32, bytes[0..4], check.magic_v3_tables, .little);
    std.mem.writeInt(u32, bytes[4..8], check.format_version, .little);
    try dir.writeFile(std.testing.io, .{ .sub_path = check.tables_file, .data = bytes });
}

fn writeTables(a: std.mem.Allocator, dir: std.Io.Dir) !void {
    return writeTablesLen(a, dir, check.section_align);
}

fn shardBytes(a: std.mem.Allocator, specs: []const Spec) ![]u8 {
    var words: usize = 0;
    var halfs: usize = 0;
    for (specs) |spec| {
        words = @max(words, @as(usize, @intCast(spec.word_off orelse words)) + try check.matWords(spec.rows, spec.cols, spec.bits));
        halfs += try check.matScales(spec.rows, spec.cols);
    }
    const off_weights = 2 * check.section_align;
    const bytes = try a.alloc(u8, off_weights + try check.mappedLen(words * 4));
    @memset(bytes, 0);
    const h: check.Header = .{
        .magic = check.magic_nonmoe,
        .version = check.format_version,
        .num_tensors = @intCast(specs.len),
        .pad0 = 0,
        .off_meta = 64,
        .off_scales = check.section_align,
        .off_weights = off_weights,
        .total_scale_halfs = halfs,
        .total_weight_words = words,
        .pad1 = 0,
    };
    @memcpy(bytes[0..@sizeOf(check.Header)], std.mem.asBytes(&h));
    var next_words: usize = 0;
    var next_halfs: usize = 0;
    for (specs, 0..) |spec, i| {
        var m: check.TensorMeta = std.mem.zeroes(check.TensorMeta);
        @memcpy(m.name[0..spec.name.len], spec.name);
        m.R = spec.rows;
        m.C = spec.cols;
        m.b_bits = spec.bits;
        m.sc_off = @intCast(next_halfs);
        m.word_off = spec.word_off orelse next_words;
        m.word_count = try check.matWords(spec.rows, spec.cols, spec.bits);
        next_words = @intCast(m.word_off + m.word_count);
        next_halfs += try check.matScales(spec.rows, spec.cols);
        @memcpy(bytes[64 + i * @sizeOf(check.TensorMeta) ..][0..@sizeOf(check.TensorMeta)], std.mem.asBytes(&m));
    }
    return bytes;
}

fn writeShard(a: std.mem.Allocator, dir: std.Io.Dir, name: []const u8, specs: []const Spec) !void {
    const bytes = try shardBytes(a, specs);
    defer a.free(bytes);
    try dir.writeFile(std.testing.io, .{ .sub_path = name, .data = bytes });
}

fn root(a: std.mem.Allocator, tmp: *std.testing.TmpDir) ![:0]u8 {
    return tmp.dir.realPathFileAlloc(std.testing.io, ".", a);
}

fn expectReleased(r: *const mtl.Record) !void {
    try std.testing.expectEqual(@as(usize, 0), r.live_buffers);
    try std.testing.expectEqual(@as(usize, 0), r.live_bytes);
    try std.testing.expectEqual(r.buffers_created, r.buffers_released);
}

const one = "qwen27_tfq3-00001-of-00002.tffpw";
const two = "qwen27_tfq3-00002-of-00002.tffpw";
const gate = "model.layers.0.mlp.gate_proj";
const up = "model.layers.0.mlp.up_proj";

fn twoShardFiles(a: std.mem.Allocator, dir: std.Io.Dir, second: []const Spec) !void {
    try writeTables(a, dir);
    try writeShard(a, dir, one, &.{.{ .name = gate }});
    try writeShard(a, dir, two, second);
}

test "qwen27 tfq3 store: two real shards resolve, classify shape errors and release all views" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try twoShardFiles(a, tmp.dir, &.{.{ .name = up }});
    const dir = try root(a, &tmp);
    defer a.free(dir);
    var r: mtl.Record = .{};
    {
        var store = (try tfq3.Store.open(a, std.testing.io, r.device(), dir)).?;
        defer store.deinit();
        try std.testing.expectEqual(@as(usize, 0), (try store.find(gate, 128, 128)).?.part.shard);
        try std.testing.expectEqual(@as(usize, 1), (try store.find(up, 128, 128)).?.part.shard);
        try std.testing.expect((try store.find("model.layers.0.mlp.down_proj", 128, 128)) == null);
        try std.testing.expect((try store.lookupBits("model.layers.0.mlp.down_proj", 128, 128)) == null);
        try std.testing.expectError(error.Tfq3RecordShape, store.find(gate, 256, 128));
        try std.testing.expectError(error.Tfq3RecordShape, store.find(gate, 128, 256));
        try std.testing.expectError(error.Tfq3RecordShape, store.lookupBits(gate, 256, 128));
    }
    try std.testing.expectEqual(@as(usize, 5), r.no_copy_calls);
    try expectReleased(&r);
}

test "qwen27 tfq3 store: duplicate canonical records in one shard release exactly once" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeTables(a, tmp.dir);
    try writeShard(a, tmp.dir, check.single_file, &.{
        .{ .name = gate },
        .{ .name = "language_model." ++ gate },
    });
    const dir = try root(a, &tmp);
    defer a.free(dir);
    var r: mtl.Record = .{};
    try std.testing.expectError(error.Tfq3Overlap, tfq3.Store.open(a, std.testing.io, r.device(), dir));
    try expectReleased(&r);
}

test "qwen27 tfq3 store: duplicate in committed second shard releases both shards exactly once" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try twoShardFiles(a, tmp.dir, &.{.{ .name = "language_model." ++ gate }});
    const dir = try root(a, &tmp);
    defer a.free(dir);
    var r: mtl.Record = .{};
    try std.testing.expectError(error.Tfq3Overlap, tfq3.Store.open(a, std.testing.io, r.device(), dir));
    try std.testing.expectEqual(@as(usize, 5), r.buffers_created);
    try expectReleased(&r);
}

test "qwen27 tfq3 store: malformed second shard cleans the committed first before any second view" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try twoShardFiles(a, tmp.dir, &.{.{ .name = up, .bits = 4, .word_off = 1 }});
    const dir = try root(a, &tmp);
    defer a.free(dir);
    var r: mtl.Record = .{};
    try std.testing.expectError(error.Tfq3Misaligned, tfq3.Store.open(a, std.testing.io, r.device(), dir));
    try std.testing.expectEqual(@as(usize, 3), r.no_copy_calls);
    try expectReleased(&r);
}

test "qwen27 tfq3 store: failed second weights view releases uncommitted scales and committed shards" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try twoShardFiles(a, tmp.dir, &.{.{ .name = up }});
    const dir = try root(a, &tmp);
    defer a.free(dir);
    var r: mtl.Record = .{ .fail_no_copy = 5 };
    try std.testing.expectError(error.NoBuffer, tfq3.Store.open(a, std.testing.io, r.device(), dir));
    try std.testing.expectEqual(@as(usize, 5), r.no_copy_calls);
    try std.testing.expectEqual(@as(usize, 4), r.buffers_created);
    try expectReleased(&r);
}

test "qwen27 tfq3 store: rounded device limit refuses before the weights no-copy call" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeTables(a, tmp.dir);
    try writeShard(a, tmp.dir, check.single_file, &.{.{ .name = gate, .rows = 256, .cols = 256 }});
    const root_dir = try root(a, &tmp);
    defer a.free(root_dir);
    var r: mtl.Record = .{ .max_buffer = check.section_align };
    try std.testing.expectError(error.Tfq3BufferTooLarge, tfq3.Store.open(a, std.testing.io, r.device(), root_dir));
    try std.testing.expectEqual(@as(usize, 2), r.no_copy_calls);
    try expectReleased(&r);
}

test "qwen27 tfq3 store: rounded views beyond EOF copy exact payloads with zero padding" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeTablesLen(a, tmp.dir, check.tables_bytes);
    const padded = try shardBytes(a, &.{.{ .name = gate, .bits = 4 }});
    defer a.free(padded);
    const physical_len = 2 * check.section_align + try check.matWords(128, 128, 4) * 4;
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = check.single_file, .data = padded[0..physical_len] });
    const dir = try root(a, &tmp);
    defer a.free(dir);
    var r: mtl.Record = .{ .copy_allocator = a };
    {
        var store = (try tfq3.Store.open(a, std.testing.io, r.device(), dir)).?;
        defer store.deinit();
        try std.testing.expectEqual(@as(usize, 1), r.no_copy_calls);
        try std.testing.expectEqual(@as(usize, 3 * check.section_align), r.live_bytes);
        const tables = store.tables_buf.contents()[0..check.section_align];
        try std.testing.expectEqual(check.magic_v3_tables, std.mem.readInt(u32, tables[0..4], .little));
        try std.testing.expectEqual(check.format_version, std.mem.readInt(u32, tables[4..8], .little));
        for (tables[8..]) |byte| try std.testing.expectEqual(@as(u8, 0), byte);
        const found = (try store.find(gate, 128, 128)).?;
        const weights = found.tfq3.words.buffer.contents()[0..check.section_align];
        const payload_len = physical_len - 2 * check.section_align;
        try std.testing.expectEqualSlices(u8, padded[2 * check.section_align .. physical_len], weights[0..payload_len]);
        for (weights[payload_len..]) |byte| try std.testing.expectEqual(@as(u8, 0), byte);
    }
    try expectReleased(&r);
}

test "qwen27 tfq3 store: table tails never enter Metal buffers or residency bytes" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeTablesLen(a, tmp.dir, 2 * check.section_align + 4096);
    try writeShard(a, tmp.dir, check.single_file, &.{.{ .name = gate }});
    const dir = try root(a, &tmp);
    defer a.free(dir);
    var r: mtl.Record = .{};
    {
        var store = (try tfq3.Store.open(a, std.testing.io, r.device(), dir)).?;
        defer store.deinit();
        try std.testing.expectEqual(@as(usize, check.section_align), store.tables_buf.length());
        try std.testing.expectEqual(@as(usize, 3 * check.section_align), r.live_bytes);
    }
    try expectReleased(&r);
}

test "qwen27 tfq3 store: orphan tables are ignored and valid shards require tables" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = check.tables_file, .data = "bad orphan bytes" });
    const dir = try root(a, &tmp);
    defer a.free(dir);
    var r: mtl.Record = .{};
    try std.testing.expect((try tfq3.Store.open(a, std.testing.io, r.device(), dir)) == null);
    try std.testing.expectEqual(@as(usize, 0), r.no_copy_calls);
    try tmp.dir.deleteFile(std.testing.io, check.tables_file);
    try writeShard(a, tmp.dir, check.single_file, &.{.{ .name = gate }});
    try std.testing.expectError(error.UnsupportedQuantization, tfq3.Store.open(a, std.testing.io, r.device(), dir));
    try std.testing.expectEqual(@as(usize, 0), r.no_copy_calls);
}

test "qwen27 tfq3 store: malformed candidate name refuses in the real directory scan" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "qwen27_tfq3-backup.tffpw", .data = "anything" });
    const dir = try root(a, &tmp);
    defer a.free(dir);
    var r: mtl.Record = .{};
    try std.testing.expectError(error.Tfq3ShardName, tfq3.Store.open(a, std.testing.io, r.device(), dir));
    try std.testing.expectEqual(@as(usize, 0), r.no_copy_calls);
}

test "qwen27 tfq3 store: allocated joined paths have no 512 byte cap" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const segment = try a.alloc(u8, 184);
    defer a.free(segment);
    @memcpy(segment[0..4], "q27-");
    @memset(segment[4..], 'x');
    const nested = try std.fs.path.join(a, &.{ segment, segment, segment });
    defer a.free(nested);
    try tmp.dir.createDirPath(std.testing.io, nested);
    var sub = try tmp.dir.openDir(std.testing.io, nested, .{});
    defer sub.close(std.testing.io);
    try writeTables(a, sub);
    try writeShard(a, sub, check.single_file, &.{.{ .name = gate }});
    const base = try root(a, &tmp);
    defer a.free(base);
    const dir = try std.fs.path.join(a, &.{ base, nested });
    defer a.free(dir);
    var r: mtl.Record = .{};
    var store = (try tfq3.Store.open(a, std.testing.io, r.device(), dir)).?;
    store.deinit();
    try expectReleased(&r);
}

fn allocationTrial(a: std.mem.Allocator, dir: []const u8) !void {
    var r: mtl.Record = .{};
    var store = (try tfq3.Store.open(a, std.testing.io, r.device(), dir)).?;
    store.deinit();
    try expectReleased(&r);
}

test "qwen27 tfq3 store: every two-shard allocation failure preserves ownership" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try twoShardFiles(a, tmp.dir, &.{.{ .name = up }});
    const dir = try root(a, &tmp);
    defer a.free(dir);
    try std.testing.checkAllAllocationFailures(a, allocationTrial, .{dir});
}
