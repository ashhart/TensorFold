//! engram_manifest.zig on synthetic releases: good passes; a flipped row, a moved tensor, no manifest refuse.
const std = @import("std");
const t = std.testing;
const er = @import("engram_rows.zig");
const ep = @import("engram_pack.zig");
const em = @import("engram_manifest.zig");
const pk = @import("pack.zig");

const rows = 300;

/// A release-style shard for `layer`: a bf16 norm of `norm` elements, then the table's random codes and scales.
fn release(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, layer: u32, seed: u64, norm: usize) !void {
    var wn: [64]u8 = undefined;
    var sn: [64]u8 = undefined;
    var nn: [64]u8 = undefined;
    const specs = [_]pk.Spec{
        .{ .name = try std.fmt.bufPrint(&nn, "norm{d}.weight", .{layer}), .dtype = "BF16", .shape = &.{norm}, .fill = 0 },
        .{ .name = try std.fmt.bufPrint(&wn, "layers.{d}.engram.embed.weight", .{layer}), .dtype = "F8_E4M3", .shape = &.{ rows, 256 }, .fill = 0 },
        .{ .name = try std.fmt.bufPrint(&sn, "layers.{d}.engram.embed.scale", .{layer}), .dtype = "F8_E8M0", .shape = &.{ rows, 8 }, .fill = 0 },
    };
    const img = try pk.image(a, &specs);
    defer a.free(img);
    const at: usize = 8 + @as(usize, @intCast(std.mem.readInt(u64, img[0..8], .little))) + 2 * norm;
    var prng = std.Random.DefaultPrng.init(seed);
    prng.random().bytes(img[at..]);
    var nb: [48]u8 = undefined;
    try dir.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&nb, "model-{d}.safetensors", .{layer}), .data = img });
}

/// The two-layer release in `src_dir`, packed to `out_dir`, with its manifest beside the tables.
fn packRelease(a: std.mem.Allocator, io: std.Io, src: []const u8, out: std.Io.Dir, seed: u64) !void {
    var p = try pk.Pack.open(a, io, src);
    defer p.deinit();
    const layers = try ep.layersOf(a, &p);
    defer a.free(layers);
    var plans: std.ArrayList(ep.Plan) = .empty;
    defer plans.deinit(a);
    for (layers) |l| {
        _ = try ep.pack(a, io, &p, out, l, 77, null);
        try plans.append(a, try ep.tableOf(&p, l));
    }
    const out_dir = try out.realPathFileAlloc(io, ".", a);
    defer a.free(out_dir);
    const m = try em.Manifest.build(a, io, out_dir, &p, plans.items, seed);
    defer m.deinit();
    try m.write(a, io, out_dir);
}

test "engram manifest: a good table passes, the parse holds its fields, and full verifies every byte" {
    const a = t.allocator;
    const io = t.io;
    var src = t.tmpDir(.{});
    defer src.cleanup();
    var out = t.tmpDir(.{});
    defer out.cleanup();
    try release(a, io, src.dir, 1, 7, 3);
    try release(a, io, src.dir, 14, 8, 3);
    const sdir = try src.dir.realPathFileAlloc(io, ".", a);
    defer a.free(sdir);
    try packRelease(a, io, sdir, out.dir, 11);
    const odir = try out.dir.realPathFileAlloc(io, ".", a);
    defer a.free(odir);
    const ck = try em.check(a, io, sdir, odir, false);
    try t.expect(ck.samples > 0);
    // the manifest's bytes parse back: geometry, sample, both tables with their hashes
    var mpath: [512]u8 = undefined;
    const mpath_s = try std.fmt.bufPrint(&mpath, "{s}/{s}", .{ odir, em.filename });
    const text = try std.Io.Dir.cwd().readFileAlloc(io, mpath_s, a, .limited(1 << 22));
    defer a.free(text);
    const m = try em.Manifest.parse(a, text);
    defer m.deinit();
    try t.expectEqual(@as(u64, 1), m.version);
    try t.expectEqual(@as(u64, er.row_bytes), m.rows);
    try t.expectEqual(@as(u32, 2), m.tables.len);
    try t.expectEqual(@as(usize, 4), m.sources.len);
    try t.expectEqual(@as(u32, 14), m.tables[1].layer);
    try t.expectEqual(@as(u64, rows), m.tables[1].hi);
    try t.expect(m.samples.len >= 49);
    const full = try em.check(a, io, sdir, odir, true);
    try t.expectEqual(ck.samples, full.samples);
    try t.expect(full.bytes > ck.bytes);
}

test "engram manifest: a flipped byte in a sampled row refuses the boot" {
    const a = t.allocator;
    const io = t.io;
    var src = t.tmpDir(.{});
    defer src.cleanup();
    var out = t.tmpDir(.{});
    defer out.cleanup();
    try release(a, io, src.dir, 1, 7, 3);
    try release(a, io, src.dir, 14, 8, 3);
    const sdir = try src.dir.realPathFileAlloc(io, ".", a);
    defer a.free(sdir);
    try packRelease(a, io, sdir, out.dir, 11);
    const odir = try out.dir.realPathFileAlloc(io, ".", a);
    defer a.free(odir);
    // a sampled row that is neither the first nor the last
    const mpath = try std.fs.path.join(a, &.{ odir, em.filename });
    defer a.free(mpath);
    const mtext = try std.Io.Dir.cwd().readFileAlloc(io, mpath, a, .limited(1 << 22));
    defer a.free(mtext);
    const m = try em.Manifest.parse(a, mtext);
    defer m.deinit();
    const x = m.samples[1];
    try t.expect(x != 0 and x != rows - 1);
    var f = try out.dir.openFile(io, "engram-l14-r0of1.bin", .{ .mode = .read_write });
    defer f.close(io);
    var b: [1]u8 = undefined;
    const at = er.header_bytes + x * er.row_bytes + 10;
    _ = try f.readPositionalAll(io, &b, at);
    b[0] +%= 1;
    try f.writePositionalAll(io, &b, at);
    try t.expectError(error.EngramSampleMismatch, em.check(a, io, sdir, odir, false));
}

test "engram manifest: shards that moved in the release refuse the boot" {
    const a = t.allocator;
    const io = t.io;
    var src = t.tmpDir(.{});
    defer src.cleanup();
    var out = t.tmpDir(.{});
    defer out.cleanup();
    try release(a, io, src.dir, 1, 7, 3);
    try release(a, io, src.dir, 14, 8, 3);
    const sdir = try src.dir.realPathFileAlloc(io, ".", a);
    defer a.free(sdir);
    try packRelease(a, io, sdir, out.dir, 11);
    const odir = try out.dir.realPathFileAlloc(io, ".", a);
    defer a.free(odir);
    // the same tensors in wider shards: the weight's offset moved, and its bytes are other bytes
    var other = t.tmpDir(.{});
    defer other.cleanup();
    try release(a, io, other.dir, 1, 7, 9);
    try release(a, io, other.dir, 14, 8, 9);
    const odir2 = try other.dir.realPathFileAlloc(io, ".", a);
    defer a.free(odir2);
    try t.expectError(error.EngramSourceMismatch, em.check(a, io, odir2, odir, false));
}

test "engram manifest: the manifest path rebuilds for existing tables and the boot passes" {
    const a = t.allocator;
    const io = t.io;
    var src = t.tmpDir(.{});
    defer src.cleanup();
    var out = t.tmpDir(.{});
    defer out.cleanup();
    try release(a, io, src.dir, 1, 7, 3);
    try release(a, io, src.dir, 14, 8, 3);
    const sdir = try src.dir.realPathFileAlloc(io, ".", a);
    defer a.free(sdir);
    try packRelease(a, io, sdir, out.dir, 11);
    const odir = try out.dir.realPathFileAlloc(io, ".", a);
    defer a.free(odir);
    // the packed tables already exist; drop the manifest the packer wrote and rebuild it from the shards
    try out.dir.deleteFile(io, em.filename);
    const mr = try em.manifestFor(a, io, sdir, odir, 11);
    try t.expect(mr.tables == 2);
    const ck = try em.check(a, io, sdir, odir, false);
    try t.expect(ck.samples > 0);
    // the rebuilt manifest parses back with both tables and the same seeded sample
    const mpath = try std.fs.path.join(a, &.{ odir, em.filename });
    defer a.free(mpath);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, mpath, a, .limited(1 << 22));
    defer a.free(text);
    const m = try em.Manifest.parse(a, text);
    defer m.deinit();
    try t.expectEqual(@as(u32, 2), m.tables.len);
    try t.expectEqual(@as(u32, 14), m.tables[1].layer);
    try t.expect(m.samples.len >= 49);
}

test "engram manifest: a flipped table byte makes the manifest path refuse and write nothing" {
    const a = t.allocator;
    const io = t.io;
    var src = t.tmpDir(.{});
    defer src.cleanup();
    var out = t.tmpDir(.{});
    defer out.cleanup();
    try release(a, io, src.dir, 1, 7, 3);
    try release(a, io, src.dir, 14, 8, 3);
    const sdir = try src.dir.realPathFileAlloc(io, ".", a);
    defer a.free(sdir);
    try packRelease(a, io, sdir, out.dir, 11);
    const odir = try out.dir.realPathFileAlloc(io, ".", a);
    defer a.free(odir);
    try out.dir.deleteFile(io, em.filename);
    // turn a byte in the last row, a sampled one by construction, then ask for the manifest back
    const x: u64 = rows - 1;
    var f = try out.dir.openFile(io, "engram-l14-r0of1.bin", .{ .mode = .read_write });
    defer f.close(io);
    var b: [1]u8 = undefined;
    const at = er.header_bytes + x * er.row_bytes + 10;
    _ = try f.readPositionalAll(io, &b, at);
    b[0] +%= 1;
    try f.writePositionalAll(io, &b, at);
    try t.expectError(error.EngramSampleMismatch, em.manifestFor(a, io, sdir, odir, 11));
    // the refusal wrote nothing: still no manifest, and no temp file left behind
    const mpath = try std.fs.path.join(a, &.{ odir, em.filename });
    defer a.free(mpath);
    try t.expectError(error.FileNotFound, std.Io.Dir.cwd().statFile(io, mpath, .{}));
    const tpath = try std.fs.path.join(a, &.{ odir, em.tmp_name });
    defer a.free(tpath);
    try t.expectError(error.FileNotFound, std.Io.Dir.cwd().statFile(io, tpath, .{}));
}

test "engram manifest: no manifest beside the tables refuses the boot" {
    const a = t.allocator;
    const io = t.io;
    var src = t.tmpDir(.{});
    defer src.cleanup();
    var out = t.tmpDir(.{});
    defer out.cleanup();
    try release(a, io, src.dir, 1, 7, 3);
    try release(a, io, src.dir, 14, 8, 3);
    const sdir = try src.dir.realPathFileAlloc(io, ".", a);
    defer a.free(sdir);
    try packRelease(a, io, sdir, out.dir, 11);
    const odir = try out.dir.realPathFileAlloc(io, ".", a);
    defer a.free(odir);
    try out.dir.deleteFile(io, em.filename);
    try t.expectError(error.EngramManifestMissing, em.check(a, io, sdir, odir, false));
}
