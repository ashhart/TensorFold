//! Darwin Engram reader: pread-thread issues byte-equal to the portable reader; a truncated shard errors.
const std = @import("std");
const t = std.testing;
const er = @import("engram_rows.zig");
const portable = @import("engram_rows_portable.zig");
const fx = @import("engram_fixture.zig");

/// The portable reader as the byte reference for the fixture shard.
fn portableFor(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !*portable.Rows {
    const root = try dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    return portable.Rows.openWithIo(a, io, root, 0, 2, &.{1}, 4096);
}

test "engram rows (darwin): overlapping issues on the reader threads, byte-equal to the portable reader" {
    const a = t.allocator;
    const io = t.io;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var lut_sha: [32]u8 = undefined;
    const fxp = try fx.parseFx(arena.allocator(), &lut_sha);
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try fx.writeShard(a, io, tmp.dir, fxp);
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    // 4 reader threads, over whatever openWithIo started (the default).
    const r = try er.Rows.openWithIo(a, io, root, 0, 2, &.{1}, 4096);
    r.stopPool();
    try r.startPool(4);
    defer r.close();
    const ref = try portableFor(a, io, tmp.dir);
    defer ref.close();
    var rng = std.Random.DefaultPrng.init(0x5eed);
    const rand = rng.random();
    // Many overlapping batches, each issued before the next and read out of order; every row byte-equal to portable.
    const sets = try a.alloc([]i64, 24);
    defer {
        for (sets) |q| a.free(q);
        a.free(sets);
    }
    for (sets) |*q| {
        q.* = try a.alloc(i64, 1 + rand.uintLessThan(usize, 200));
        for (q.*) |*x| x.* = @intCast(fxp.lo + rand.uintLessThan(u64, fxp.rows));
    }
    for (sets) |q| try r.issue(0, q);
    var k = sets.len;
    while (k > 0) {
        k -= 1;
        const q = sets[k];
        const got = try a.alloc(u16, q.len * 256);
        defer a.free(got);
        const want = try a.alloc(u16, q.len * 256);
        defer a.free(want);
        try ref.rows(0, q, 256, want);
        try r.rows(0, q, 256, got);
        try t.expectEqualSlices(u16, want, got);
    }
    // The fixture's own sets too, each byte-equal to its reference sha (duplicates included).
    for (fxp.sets) |s| {
        const got = try a.alloc(u16, s.idx.len * 256);
        defer a.free(got);
        try r.rows(0, s.idx, 256, got);
        try t.expectEqual(s.sha, fx.sha(got));
    }
}

test "engram rows (darwin): a truncated shard is an error, not stale rows" {
    const a = t.allocator;
    const io = t.io;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var lut_sha: [32]u8 = undefined;
    const fxp = try fx.parseFx(arena.allocator(), &lut_sha);
    // A shard whose header promises more rows than the file holds.
    const n: usize = @intCast(er.header_bytes + fxp.rows * er.row_bytes);
    const b = try a.alloc(u8, n);
    defer a.free(b);
    @memset(b, 0);
    for ([_]u64{ er.magic, 1, fxp.lo, fxp.lo + fxp.rows, fxp.total, er.row_bytes }, 0..) |v, i| std.mem.writeInt(u64, b[8 * i ..][0..8], v, .little);
    for (0..fxp.rows) |k2| {
        for (0..er.row_bytes) |j| b[er.header_bytes + k2 * er.row_bytes + j] = 0x77;
    }
    const file = try tmp.dir.createFile(io, "engram-l1-r0of2.bin", .{});
    defer file.close(io);
    try file.writePositionalAll(io, b, 0);
    try file.setLength(io, er.header_bytes + (fxp.rows - 64) * er.row_bytes); // short of the header's promise
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    // The header's size no longer matches the file: the reader must refuse, not serve truncated rows.
    try t.expectError(error.BadEngramShard, er.Rows.open(a, root, 0, 2, &.{1}, 4096));
}
