//! engram_pack.zig on a synthetic release table: the shard bytes, chunking, and the verifier catching a bad row.
const std = @import("std");
const t = std.testing;
const er = @import("engram_rows.zig");
const ep = @import("engram_pack.zig");
const pk = @import("pack.zig");

const rows = 300;

/// A release-style shard holding layer `layer`'s table with random codes and scales (and a bf16 tensor before it).
fn release(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, layer: u32, seed: u64) !void {
    var wn: [64]u8 = undefined;
    var sn: [64]u8 = undefined;
    var nn: [64]u8 = undefined;
    const specs = [_]pk.Spec{
        .{ .name = try std.fmt.bufPrint(&nn, "norm{d}.weight", .{layer}), .dtype = "BF16", .shape = &.{3}, .fill = 0 },
        .{ .name = try std.fmt.bufPrint(&wn, "layers.{d}.engram.embed.weight", .{layer}), .dtype = "F8_E4M3", .shape = &.{ rows, 256 }, .fill = 0 },
        .{ .name = try std.fmt.bufPrint(&sn, "layers.{d}.engram.embed.scale", .{layer}), .dtype = "F8_E8M0", .shape = &.{ rows, 8 }, .fill = 0 },
    };
    const img = try pk.image(a, &specs);
    defer a.free(img);
    const at: usize = 8 + @as(usize, @intCast(std.mem.readInt(u64, img[0..8], .little))) + 6;
    var prng = std.Random.DefaultPrng.init(seed);
    prng.random().bytes(img[at..]);
    var nb: [48]u8 = undefined;
    try dir.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&nb, "model-{d}.safetensors", .{layer}), .data = img });
}

test "engram pack: the shard is the header then each row's 256 codes and 8 scales, however the rows are chunked" {
    const a = t.allocator;
    const io = t.io;
    var src = t.tmpDir(.{});
    defer src.cleanup();
    var out = t.tmpDir(.{});
    defer out.cleanup();
    try release(a, io, src.dir, 1, 7);
    try release(a, io, src.dir, 14, 8);
    const sdir = try src.dir.realPathFileAlloc(io, ".", a);
    defer a.free(sdir);
    var p = try pk.Pack.open(a, io, sdir);
    defer p.deinit();
    const layers = try ep.layersOf(a, &p);
    defer a.free(layers);
    try t.expectEqualSlices(u32, &.{ 1, 14 }, layers);
    // 300 rows in chunks of 77 (a short last chunk) and in one chunk
    try t.expectEqual(@as(u64, rows), try ep.pack(a, io, &p, out.dir, 1, 77, null));
    try t.expectEqual(@as(u64, rows), try ep.pack(a, io, &p, out.dir, 14, 1 << 20, null));
    for (layers) |l| {
        var nb: [64]u8 = undefined;
        const bytes = try out.dir.readFileAlloc(io, try ep.shardName(&nb, l), a, .limited(1 << 24));
        defer a.free(bytes);
        const h = try er.Header.parse(bytes[0..48], bytes.len);
        try t.expectEqual(@as(u64, rows), h.hi);
        try t.expectEqual(@as(u64, l), h.layer);
        try t.expect(std.mem.allEqual(u8, bytes[48..er.header_bytes], 0));
        var w: [rows * 256]u8 = undefined;
        var s: [rows * 8]u8 = undefined;
        const tab = try ep.tableOf(&p, l);
        try p.read(io, tab.weight, .{ .offset = 0, .len = w.len }, &w);
        try p.read(io, tab.scale, .{ .offset = 0, .len = s.len }, &s);
        for (0..rows) |r| {
            const rec = bytes[er.header_bytes + r * er.row_bytes ..][0..er.row_bytes];
            try t.expectEqualSlices(u8, w[r * 256 ..][0..256], rec[0..256]);
            try t.expectEqualSlices(u8, s[r * 8 ..][0..8], rec[256..264]);
        }
    }
    try t.expectError(error.FileNotFound, out.dir.access(io, "engram-l1-r0of1.bin.part", .{}));
    const odir = try out.dir.realPathFileAlloc(io, ".", a);
    defer a.free(odir);
    const good = try ep.verify(a, io, &p, odir, layers, 64, 3);
    try t.expectEqual(@as(u64, 0), good.bad);
    try t.expectEqual(@as(u64, 2 * 66), good.rows);
    // every row of a table with one byte flipped in row 123's scales: exactly that row differs
    var f = try out.dir.openFile(io, "engram-l14-r0of1.bin", .{ .mode = .read_write });
    defer f.close(io);
    var b: [1]u8 = undefined;
    const at = er.header_bytes + 123 * er.row_bytes + 260;
    _ = try f.readPositionalAll(io, &b, at);
    b[0] +%= 1;
    try f.writePositionalAll(io, &b, at);
    const bad = try ep.verify(a, io, &p, odir, layers, rows, 3);
    try t.expectEqual(@as(u64, 2 * rows), bad.rows);
    try t.expectEqual(@as(u64, 1), bad.bad);
}
