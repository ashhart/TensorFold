//! EXLR8 decoding against fixtures the reference loader (MIT) wrote outside the tree: layers 0, 3, 40, 78, vocab.

const std = @import("std");
const codec = @import("exlr8_codec.zig");
const ex = @import("exlr8_experts.zig");
const dense = @import("exlr8_dense.zig");
const mf = @import("exlr8_manifest.zig");
const release = @import("exlr8_release.zig");
const node = @import("exlr8_node.zig");
const pack = @import("exlr8_pack.zig");
const xxh = @import("xxhash64.zig");
const st = @import("safetensors.zig");
const Sha = std.crypto.hash.sha2.Sha256;

const io = std.testing.io;
const u = 0x1p-24;
/// The reference reconstructs in fp32 with two 128-term products and three roundings: its error against exact W.
const bound_factor = 2 * (128 * u / (1 - 128 * u)) + 5 * u;

const Tables = struct {
    codebook_codes_sha256: []const u8,
    codebook_spots: []const [2]u32,
    f16_to_e4m3_sha256: []const u8,
    f16_nan_count: usize,
    e4m3_to_f32_sha256: []const u8,
    e4m3_nan_codes: []const u8,
};
const Vector = struct { len: usize, seed: []const u8, xxh64: []const u8 };
const Unit = struct { layer: u16, expert: u16, proj: []const u8, half: u8, k: u8, checksum: []const u8, tiles_sha256: []const u8, w_samples: []const u32 };
const DenseUnit = struct {
    name: []const u8,
    record: [8]i64,
    checksums_match: bool,
    w_sha256: ?[]const u8 = null,
    w_samples: []const u32,
    w_samples_rows: usize,
    tiles_sha256: ?[]const u8 = null,
};
const ManifestId = struct { name: []const u8, set_id: []const u8, layers: usize, widths_sha256: []const u8 };
const Map = std.json.ArrayHashMap;
const Fixture = struct {
    tables: Tables,
    xxh64: []const Vector,
    experts: []const Unit,
    verified: Map(Map([2]usize)),
    dense: Map([]const DenseUnit),
    plain: Map(Map([]const u8)),
    manifests: Map(ManifestId),
};

fn load(arena: std.mem.Allocator) !Fixture {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, "zig/tests/fixtures/exlr8/oracle.json", arena, .limited(1 << 22));
    return std.json.parseFromSliceLeaky(Fixture, arena, text, .{ .ignore_unknown_fields = true });
}

fn hexOf(h: *Sha) [64]u8 {
    var d: [32]u8 = undefined;
    h.final(&d);
    return std.fmt.bytesToHex(d, .lower);
}
fn sha(bytes: []const u8) [64]u8 {
    var h = Sha.init(.{});
    h.update(bytes);
    return hexOf(&h);
}
fn position(j: usize, n: usize) usize {
    return @intCast((@as(u64, j) * 2654435761 + 12345) % n);
}

/// Our fp32 W must equal the reference's bits; the exact f64 W must sit within the reference's fp32 bound.
const Stats = struct {
    values: usize = 0,
    max_fraction: f64 = 0,

    fn add(s: *Stats, mine: f32, exact: f64, scale: f64, want_bits: u32) !void {
        if (@as(u32, @bitCast(mine)) != want_bits) {
            std.debug.print("W {x} want {x} (exact {d})\n", .{ @as(u32, @bitCast(mine)), want_bits, exact });
            return error.Exlr8WeightMismatch;
        }
        const err = @abs(@as(f64, mine) - exact);
        const bound = bound_factor * scale + 0x1p-149;
        if (!(err <= bound)) return error.OutsideExlr8Bound;
        s.values += 1;
        s.max_fraction = @max(s.max_fraction, err / bound);
    }
    fn print(s: Stats, what: []const u8) void {
        std.debug.print("{s}: {d} W values equal the reference's fp32 bits; worst {d:.4} of its fp32 bound from exact W\n", .{ what, s.values, s.max_fraction });
    }
};

/// The reference's W samples at their positions in a (rows, cols) W (or its transpose), block by block.
fn checkSamples(codes: []const u8, rows: usize, cols: usize, suh: []const f32, svh: []const f32, samples: []const u32, transposed: bool, stats: *Stats) !void {
    const gpa = std.testing.allocator;
    const exact = try gpa.create([128 * 128]f64);
    defer gpa.destroy(exact);
    const mine = try gpa.create([128 * 128]f32);
    defer gpa.destroy(mine);
    var cached: ?[2]usize = null;
    var abs: f64 = 0;
    for (samples, 0..) |want, j| {
        const p = position(j, rows * cols);
        const r, const c = if (transposed) .{ p % rows, p / rows } else .{ p / cols, p % cols };
        const key = [2]usize{ r / 128, c / 128 };
        if (cached == null or !std.mem.eql(usize, &cached.?, &key)) {
            codec.reconstructBlock(codes, cols, key[0], key[1], suh, svh, exact);
            codec.reconstructBlockF32(codes, cols, key[0], key[1], suh, svh, mine);
            abs = 0;
            for (0..128) |i| for (0..128) |q| {
                abs += @abs(codec.e4m3Value(codes[(key[0] * 128 + i) * cols + key[1] * 128 + q]));
            };
            cached = key;
        }
        const at = (r % 128) * 128 + c % 128;
        try stats.add(mine[at], exact[at], @abs(@as(f64, suh[r]) * svh[c]) / 128 * abs, want);
    }
}

test "the codebook, E4M3 rounding and decoding equal the reference's tables for every input" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const fx = try load(arena.allocator());
    const book = try arena.allocator().create(codec.Codebook);
    book.init();
    try std.testing.expectEqualStrings(fx.tables.codebook_codes_sha256, &sha(&book.codes));
    for (fx.tables.codebook_spots) |s| try std.testing.expectEqual(s[1], book.codes[s[0]]);
    var h = Sha.init(.{});
    var nans: usize = 0;
    for (0..65536) |bits| {
        const v: f16 = @bitCast(@as(u16, @intCast(bits)));
        if (std.math.isNan(v)) {
            nans += 1;
        } else h.update(&.{codec.e4m3FromF32(v)});
    }
    try std.testing.expectEqualStrings(fx.tables.f16_to_e4m3_sha256, &hexOf(&h));
    try std.testing.expectEqual(fx.tables.f16_nan_count, nans);
    h = Sha.init(.{});
    var nan_codes: std.ArrayList(u8) = .empty;
    for (0..256) |c| {
        const v = codec.e4m3Value(@intCast(c));
        if (std.math.isNan(v)) try nan_codes.append(arena.allocator(), @intCast(c)) else h.update(std.mem.asBytes(&v));
    }
    try std.testing.expectEqualStrings(fx.tables.e4m3_to_f32_sha256, &hexOf(&h));
    try std.testing.expectEqualSlices(u8, fx.tables.e4m3_nan_codes, nan_codes.items);
}

test "XXH64 equals the reference package's vectors" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const fx = try load(arena.allocator());
    var data: [10007]u8 = undefined;
    for (&data, 0..) |*b, i| b.* = @truncate(i *% 31 +% 7);
    for (fx.xxh64) |v| {
        const seed = try std.fmt.parseInt(u64, v.seed, 16);
        try std.testing.expectEqual(try std.fmt.parseInt(u64, v.xxh64, 16), xxh.hash(seed, data[0..v.len]));
    }
}

/// The layers a fixture names, each opened once from the release folder.
const Layers = struct {
    open: [79]?ex.Layer = @splat(null),

    fn get(l: *Layers, root: []const u8, n: u16) !*ex.Layer {
        if (l.open[n] == null) l.open[n] = try ex.Layer.open(std.testing.allocator, io, root, n);
        return &l.open[n].?;
    }
    fn close(l: *Layers) void {
        for (&l.open) |*o| if (o.*) |*layer| layer.close(io);
    }
};

test "expert halves of layers 3, 40 and 78: stored checksums, the reference's tiles and W bits" {
    const root = std.testing.environ.getPosix("TF_EXLR8_DIR") orelse return error.SkipZigTest;
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const fx = try load(arena.allocator());
    var layers: Layers = .{};
    defer layers.close();
    const book = try arena.allocator().create(codec.Codebook);
    book.init();
    const codes = try arena.allocator().alloc(u8, ex.hidden * 128);
    var suh: [ex.hidden]f32 = undefined;
    var svh: [ex.hidden]f32 = undefined;
    var stats: Stats = .{};
    for (fx.experts) |unit| {
        const layer = try layers.get(root, unit.layer);
        const p = std.meta.stringToEnum(ex.Proj, unit.proj) orelse return error.BadFixture;
        const h = try layer.half(unit.expert, p, unit.half, unit.k);
        try std.testing.expectEqual(try std.fmt.parseInt(u64, unit.checksum, 16), h.checksum);
        try std.testing.expect(h.verify());
        try codec.decodeRun(book, h.k, h.order(), h.trellis, h.rows(), h.cols(), codes);
        try std.testing.expectEqualStrings(unit.tiles_sha256, &sha(codes));
        h.suh(suh[0..h.rows()]);
        h.svh(svh[0..h.cols()]);
        try checkSamples(codes, h.rows(), h.cols(), suh[0..h.rows()], svh[0..h.cols()], unit.w_samples, false, &stats);
        if (std.testing.environ.getPosix("TF_EXLR8_ORACLE_FULL")) |full| try wholeHalf(full, unit, codes, h, suh[0..h.rows()], svh[0..h.cols()]);
    }
    stats.print("expert halves");
}

/// Every value of a half against the reference's own dump of it (written beside the fixture, outside the tree).
fn wholeHalf(dir: []const u8, unit: Unit, codes: []const u8, h: ex.Half, suh: []const f32, svh: []const f32) !void {
    const gpa = std.testing.allocator;
    var number: [8]u8 = undefined;
    var name: [64]u8 = undefined;
    const e: []const u8 = if (unit.expert == ex.shared) "shared" else try std.fmt.bufPrint(&number, "{d}", .{unit.expert});
    const tag = try std.fmt.bufPrint(&name, "L{d}-{s}-{s}-{d}-k{d}", .{ unit.layer, e, unit.proj, unit.half, unit.k });
    const tiles_path = try std.fmt.allocPrint(gpa, "{s}/tiles-{s}.u8", .{ dir, tag });
    defer gpa.free(tiles_path);
    const want_codes = std.Io.Dir.cwd().readFileAlloc(io, tiles_path, gpa, .limited(1 << 22)) catch return;
    defer gpa.free(want_codes);
    try std.testing.expectEqualSlices(u8, want_codes, codes);
    const w_path = try std.fmt.allocPrint(gpa, "{s}/w-{s}.f32", .{ dir, tag });
    defer gpa.free(w_path);
    const want_w = try std.Io.Dir.cwd().readFileAlloc(io, w_path, gpa, .limited(1 << 24));
    defer gpa.free(want_w);
    const exact = try gpa.alloc(f64, codes.len);
    defer gpa.free(exact);
    const mine = try gpa.alloc(f32, codes.len);
    defer gpa.free(mine);
    try codec.reconstruct(codes, h.rows(), h.cols(), suh, svh, exact);
    try codec.reconstructF32(codes, h.rows(), h.cols(), suh, svh, mine);
    var abs: [48]f64 = undefined;
    codec.blockAbs(codes, h.rows(), h.cols(), &abs);
    var stats: Stats = .{};
    for (mine, exact, 0..) |m, x, i| {
        const r = i / h.cols();
        const c = i % h.cols();
        const scale = @abs(@as(f64, suh[r]) * svh[c]) / 128 * abs[(r / 128) * (h.cols() / 128) + c / 128];
        try stats.add(m, x, scale, std.mem.readInt(u32, want_w[4 * i ..][0..4], .little));
    }
    stats.print(tag);
}

test "every half of every stored width verifies, as the reference's --verify counts them" {
    const root = std.testing.environ.getPosix("TF_EXLR8_DIR") orelse return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const fx = try load(arena.allocator());
    var layers: Layers = .{};
    defer layers.close();
    var it = fx.verified.map.iterator();
    while (it.next()) |kv| {
        const layer = try layers.get(root, try std.fmt.parseInt(u16, kv.key_ptr.*, 10));
        var widths = kv.value_ptr.map.iterator();
        while (widths.next()) |w| {
            const shared = std.mem.startsWith(u8, w.key_ptr.*, "shared.");
            const k = w.key_ptr.*[w.key_ptr.len - 1] - '0';
            var passed: usize = 0;
            var total: usize = 0;
            for (0..if (shared) 1 else ex.routed) |e| for ([_]ex.Proj{ .gate, .up, .down }) |p| for (0..ex.halves) |h| {
                const half = try layer.half(if (shared) ex.shared else @intCast(e), p, h, k);
                passed += @intFromBool(half.verify());
                total += 1;
            };
            try std.testing.expectEqual(w.value_ptr.*, [2]usize{ passed, total });
            try std.testing.expectEqual(passed, total);
        }
    }
}

test "dense and vocabulary files: records, checksums, every format bit for bit, plain tensors as shipped" {
    const root = std.testing.environ.getPosix("TF_EXLR8_DIR") orelse return error.SkipZigTest;
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const fx = try load(arena.allocator());
    const book = try arena.allocator().create(codec.Codebook);
    book.init();
    var stats: Stats = .{};
    var it = fx.dense.map.iterator();
    while (it.next()) |kv| {
        const path = try std.fmt.allocPrint(gpa, "{s}/dense/{s}.safetensors", .{ root, kv.key_ptr.* });
        defer gpa.free(path);
        var f = try st.File.open(gpa, io, path);
        defer f.close(io);
        const names = try dense.names(gpa, &f);
        defer gpa.free(names);
        try std.testing.expectEqual(kv.value_ptr.len, names.len);
        for (kv.value_ptr.*, names) |unit, name| {
            try std.testing.expectEqualStrings(unit.name, name);
            const m = try dense.Matrix.get(&f, unit.name);
            const r = m.record;
            for ([_]i64{ @backingInt(r.format), @intCast(r.out), @intCast(r.in), r.k, @backingInt(r.side), @intCast(r.channels), r.rotation, r.solve }, unit.record) |got, want|
                try std.testing.expectEqual(want, got);
            try std.testing.expectEqual(unit.checksums_match, try m.verify());
            if (r.format != .trellis) {
                try denseBits(m, unit);
                continue;
            }
            const codes = try gpa.alloc(u8, r.in * r.out);
            defer gpa.free(codes);
            try m.trellisCodes(book, codes);
            try std.testing.expectEqualStrings(unit.tiles_sha256.?, &sha(codes));
            const suh = try gpa.alloc(f32, r.in);
            defer gpa.free(suh);
            const svh = try gpa.alloc(f32, r.out);
            defer gpa.free(svh);
            m.suh(suh);
            m.svh(svh);
            try checkSamples(codes, r.in, r.out, suh, svh, unit.w_samples, true, &stats);
        }
        const plain = try dense.plainNames(gpa, &f);
        defer gpa.free(plain);
        const want = fx.plain.map.get(kv.key_ptr.*) orelse return error.BadFixture;
        try std.testing.expectEqual(want.map.count(), plain.len);
        for (plain) |n| try std.testing.expectEqualStrings(want.map.get(n) orelse return error.UnexpectedPlainTensor, &sha(f.get(n).?.bytes));
    }
    stats.print("dense trellis");
}

/// A non-trellis matrix's whole fp32 weight against the reference's digest, a chunk of rows at a time.
fn denseBits(m: dense.Matrix, unit: DenseUnit) !void {
    const gpa = std.testing.allocator;
    const r = m.record;
    const chunk: usize = @min(r.out, 8192);
    const rows = try gpa.alloc(f32, chunk * r.in);
    defer gpa.free(rows);
    var h = Sha.init(.{});
    var row: usize = 0;
    while (row < r.out) : (row += chunk) {
        const n = @min(chunk, r.out - row);
        try m.rows(row, n, rows[0 .. n * r.in]);
        for (unit.w_samples, 0..) |want, j| {
            const p = position(j, unit.w_samples_rows * r.in);
            if (p >= row * r.in and p < (row + n) * r.in) try std.testing.expectEqual(want, @as(u32, @bitCast(rows[p - row * r.in])));
        }
        h.update(std.mem.sliceAsBytes(rows[0 .. n * r.in]));
    }
    try std.testing.expectEqualStrings(unit.w_sha256.?, &hexOf(&h));
}

test "the eight manifests parse to the reference's widths, and every width they ask of a present layer is stored" {
    const root = std.testing.environ.getPosix("TF_EXLR8_DIR") orelse return error.SkipZigTest;
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const fx = try load(arena.allocator());
    var layers: Layers = .{};
    defer layers.close();
    var it = fx.manifests.map.iterator();
    while (it.next()) |kv| {
        const path = try std.fmt.allocPrint(gpa, "{s}/manifests/exlr8-manifest-{s}.json", .{ root, kv.key_ptr.* });
        defer gpa.free(path);
        var m = try mf.Manifest.load(gpa, io, path);
        defer m.deinit();
        try std.testing.expectEqualStrings(kv.value_ptr.name, m.name);
        try std.testing.expectEqualStrings(kv.value_ptr.set_id, m.set_id);
        var h = Sha.init(.{});
        var count: usize = 0;
        for (m.layers) |maybe| if (maybe) |w| {
            count += 1;
            for (w.gate) |g| h.update(&g);
            for (w.up) |g| h.update(&g);
            h.update(&w.down);
            h.update(&w.shared_gate);
            h.update(&w.shared_up);
            h.update(&.{w.shared_down});
        };
        try std.testing.expectEqual(kv.value_ptr.layers, count);
        try std.testing.expectEqualStrings(kv.value_ptr.widths_sha256, &hexOf(&h));
        var vit = fx.verified.map.iterator();
        while (vit.next()) |v| {
            const n = try std.fmt.parseInt(u16, v.key_ptr.*, 10);
            const layer = try layers.get(root, n);
            for (0..mf.experts + 1) |e| for (0..mf.halves) |hh| inline for (.{ .gate, .up, .down }) |p| {
                try std.testing.expect(layer.stored(@intCast(e), m.width(n, e, p, hh).?));
            };
        }
    }
}

test "every present release file matches its published sha256 and size" {
    const root = std.testing.environ.getPosix("TF_EXLR8_DIR") orelse return error.SkipZigTest;
    const gpa = std.testing.allocator;
    var checked: usize = 0;
    for ([_][]const u8{ "experts", "dense" }) |sub| {
        const path = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ root, sub });
        defer gpa.free(path);
        var dir = try std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true });
        defer dir.close(io);
        var walk = dir.iterate();
        while (try walk.next(io)) |entry| {
            if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
            var buf: [256]u8 = undefined;
            const rel = try std.fmt.bufPrint(&buf, "{s}/{s}", .{ sub, entry.name });
            try std.testing.expect(try release.checkFile(gpa, io, root, rel));
            checked += 1;
        }
    }
    std.debug.print("{d} files match their published sha256\n", .{checked});
    try std.testing.expect(checked > 0);
}

test {
    _ = codec;
    _ = ex;
    _ = dense;
    _ = mf;
    _ = release;
    _ = node;
    _ = pack;
    _ = xxh;
}
