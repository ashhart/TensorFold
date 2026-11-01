//! engram-manifest.json beside the packed Engram tables: the release's tensors, the shard hashes, a seeded row sample.
const std = @import("std");
const er = @import("engram_rows.zig");
const ep = @import("engram_pack.zig");
const Pack = @import("pack.zig").Pack;
const Info = @import("pack.zig").Info;
const DType = @import("core").safetensors.DType;
const Plan = @import("engram_tables.zig").Plan;

const Io = std.Io;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Allocator = std.mem.Allocator;

pub const filename = "engram-manifest.json";
pub const tmp_name = ".engram-manifest.json.tmp";
const format_name = "dsv41-engram-manifest";

/// One source tensor of a release's Engram table: where its bytes live, as the packer read them.
pub const Source = struct {
    name: []const u8,
    file: []const u8,
    dtype: []const u8,
    shape: []const usize,
    offset: u64,
    length: u64,
};

/// One packed row shard: its row span, on-disk size and SHA-256.
pub const Table = struct {
    layer: u32,
    file: []const u8,
    lo: u64,
    hi: u64,
    total: u64,
    size: u64,
    sha: [32]u8,
};

pub const Manifest = struct {
    root: Allocator,
    arena: *std.heap.ArenaAllocator,
    /// 1: the first layout; a newer packer writes a higher version, and this reader refuses it.
    version: u64 = 1,
    rows: u64 = er.row_bytes,
    values: u64 = er.values,
    seed: u64 = 1,
    /// The seeded row indices, first and last included, within every table's 0 .. rows.
    samples: []const u64 = &.{},
    sources: []const Source = &.{},
    tables: []const Table = &.{},

    /// The packer's manifest for `plans`, in ascending layer order; the shards in `tables_dir` must already be written.
    pub fn build(a: Allocator, io: Io, tables_dir: []const u8, p: *const Pack, plans: []const Plan, seed: u64) !Manifest {
        const arena_p = try a.create(std.heap.ArenaAllocator);
        errdefer a.destroy(arena_p);
        arena_p.* = std.heap.ArenaAllocator.init(a);
        const al = arena_p.allocator();
        // the indices must fit every table, so draw them under the smallest table's row count
        var span: u64 = 0;
        for (plans) |t| span = if (span == 0 or t.rows < span) t.rows else span;
        var m = Manifest{ .root = a, .arena = arena_p, .seed = seed };
        if (span > 0) m.samples = try ep.samples(al, span, 48, seed);
        var srcs: std.ArrayList(Source) = .empty;
        var tabs: std.ArrayList(Table) = .empty;
        for (plans) |t| {
            for ([_]usize{ 0, 1 }) |k| {
                const i: Info = if (k == 0) t.weight else t.scale;
                var nb: [96]u8 = undefined;
                const name = try std.fmt.bufPrint(&nb, "layers.{d}.engram.embed.{s}", .{ t.layer, if (k == 0) "weight" else "scale" });
                try srcs.append(al, Source{
                    .name = try al.dupe(u8, name),
                    .file = try al.dupe(u8, p.files.items[i.file]),
                    .dtype = try al.dupe(u8, dtypeName(i.dtype)),
                    .shape = try al.dupe(usize, i.dims()),
                    .offset = i.start,
                    .length = i.nbytes,
                });
            }
            var tnb: [64]u8 = undefined;
            const tname = try ep.shardName(&tnb, t.layer);
            var spb: [512]u8 = undefined;
            const spath = try std.fmt.bufPrint(&spb, "{s}/{s}", .{ tables_dir, tname });
            try tabs.append(al, Table{
                .layer = t.layer,
                .file = try al.dupe(u8, tname),
                .lo = 0,
                .hi = t.rows,
                .total = t.rows,
                .size = try fileLength(io, spath),
                .sha = try fileSha256(io, spath),
            });
        }
        m.sources = srcs.items;
        m.tables = tabs.items;
        return m;
    }

    /// The manifest's own bytes: format marker, sample, sources, tables with hashes.
    pub fn render(m: *const Manifest) ![]u8 {
        var out: std.Io.Writer.Allocating = .init(m.arena.allocator());
        defer out.deinit();
        const w = &out.writer;
        try w.print("{{\"format\":\"{s}\",\"version\":{d},\"row_bytes\":{d},\"values\":{d},\"seed\":{d},\"samples\":[", .{ format_name, m.version, m.rows, m.values, m.seed });
        for (m.samples, 0..) |x, i| try w.print("{s}{d}", .{ if (i == 0) "" else ",", x });
        try w.print("],\"sources\":[", .{});
        for (m.sources, 0..) |s, i| {
            if (i != 0) try w.writeByte(',');
            try w.print("{{\"tensor\":\"{s}\",\"file\":\"{s}\",\"dtype\":\"{s}\",\"shape\":[", .{ s.name, s.file, s.dtype });
            for (s.shape, 0..) |d, j| try w.print("{s}{d}", .{ if (j == 0) "" else ",", d });
            try w.print("],\"offset\":{d},\"length\":{d}}}", .{ s.offset, s.length });
        }
        try w.print("],\"tables\":[", .{});
        for (m.tables, 0..) |t, i| {
            if (i != 0) try w.writeByte(',');
            const hex = std.fmt.bytesToHex(t.sha, .lower);
            try w.print("{{\"layer\":{d},\"file\":\"{s}\",\"lo\":{d},\"hi\":{d},\"total\":{d},\"size\":{d},\"sha256\":\"{s}\"}}", .{ t.layer, t.file, t.lo, t.hi, t.total, t.size, &hex });
        }
        try w.writeAll("]}");
        return out.written();
    }

    /// The manifest's bytes beside `dir`'s tables: format marker, sample, sources, tables with hashes.
    pub fn write(m: *const Manifest, a: Allocator, io: Io, dir: []const u8) !void {
        const data = try render(m);
        const path = try std.fs.path.join(a, &.{ dir, filename });
        defer a.free(path);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = data });
    }

    /// Parses a manifest's bytes; the caller deinit's the result.
    pub fn parse(a: Allocator, text: []const u8) !Manifest {
        const arena_p = try a.create(std.heap.ArenaAllocator);
        errdefer a.destroy(arena_p);
        arena_p.* = std.heap.ArenaAllocator.init(a);
        const al = arena_p.allocator();
        var m = Manifest{ .root = a, .arena = arena_p };
        errdefer {
            m.arena.deinit();
            m.root.destroy(m.arena);
        }
        const v = std.json.parseFromSliceLeaky(std.json.Value, al, text, .{}) catch return error.EngramManifestBad;
        if (v != .object) return error.EngramManifestBad;
        if (!std.mem.eql(u8, str(v.object.get("format")) orelse "", format_name)) return error.EngramManifestBad;
        m.version = num(v.object.get("version")) orelse return error.EngramManifestBad;
        m.rows = num(v.object.get("row_bytes")) orelse return error.EngramManifestBad;
        m.values = num(v.object.get("values")) orelse return error.EngramManifestBad;
        m.seed = num(v.object.get("seed")) orelse return error.EngramManifestBad;
        const samples = (v.object.get("samples") orelse return error.EngramManifestBad);
        if (samples != .array) return error.EngramManifestBad;
        const sam = try al.alloc(u64, samples.array.items.len);
        for (samples.array.items, 0..) |x, i| sam[i] = num(x) orelse return error.EngramManifestBad;
        m.samples = sam;
        const srcs = (v.object.get("sources") orelse return error.EngramManifestBad);
        if (srcs != .array) return error.EngramManifestBad;
        const src = try al.alloc(Source, srcs.array.items.len);
        for (srcs.array.items, 0..) |x, i| src[i] = try parseSource(al, x);
        m.sources = src;
        const tabs = (v.object.get("tables") orelse return error.EngramManifestBad);
        if (tabs != .array) return error.EngramManifestBad;
        const tab = try al.alloc(Table, tabs.array.items.len);
        for (tabs.array.items, 0..) |x, i| tab[i] = try parseTable(al, x);
        m.tables = tab;
        return m;
    }

    pub fn deinit(m: *const Manifest) void {
        m.arena.deinit();
        m.root.destroy(m.arena);
    }
};

fn parseSource(al: Allocator, v: std.json.Value) !Source {
    if (v != .object) return error.EngramManifestBad;
    const o = v.object;
    const name = str(o.get("tensor")) orelse return error.EngramManifestBad;
    const file = str(o.get("file")) orelse return error.EngramManifestBad;
    const dtype = str(o.get("dtype")) orelse return error.EngramManifestBad;
    const shape_v = (o.get("shape") orelse return error.EngramManifestBad);
    if (shape_v != .array) return error.EngramManifestBad;
    const shape = try al.alloc(usize, shape_v.array.items.len);
    for (shape_v.array.items, 0..) |x, i| shape[i] = num(x) orelse return error.EngramManifestBad;
    return Source{
        .name = try al.dupe(u8, name),
        .file = try al.dupe(u8, file),
        .dtype = try al.dupe(u8, dtype),
        .shape = shape,
        .offset = num(o.get("offset")) orelse return error.EngramManifestBad,
        .length = num(o.get("length")) orelse return error.EngramManifestBad,
    };
}

fn parseTable(al: Allocator, v: std.json.Value) !Table {
    if (v != .object) return error.EngramManifestBad;
    const o = v.object;
    const layer = num(o.get("layer")) orelse return error.EngramManifestBad;
    if (layer > std.math.maxInt(u32)) return error.EngramManifestBad;
    const file = str(o.get("file")) orelse return error.EngramManifestBad;
    const hex = str(o.get("sha256")) orelse return error.EngramManifestBad;
    var sha: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&sha, hex) catch return error.EngramManifestBad;
    return Table{
        .layer = @intCast(layer),
        .file = try al.dupe(u8, file),
        .lo = num(o.get("lo")) orelse return error.EngramManifestBad,
        .hi = num(o.get("hi")) orelse return error.EngramManifestBad,
        .total = num(o.get("total")) orelse return error.EngramManifestBad,
        .size = num(o.get("size")) orelse return error.EngramManifestBad,
        .sha = sha,
    };
}

fn str(v: ?std.json.Value) ?[]const u8 {
    return if (v) |x| switch (x) {
        .string => x.string,
        else => null,
    } else null;
}

fn num(v: ?std.json.Value) ?u64 {
    return switch (v orelse return null) {
        .integer => |i| std.math.cast(u64, i),
        else => null,
    };
}

pub fn dtypeName(d: DType) []const u8 {
    return switch (d) {
        .bool => "BOOL",
        .u8 => "U8",
        .i8 => "I8",
        .u16 => "U16",
        .i16 => "I16",
        .f16 => "F16",
        .bf16 => "BF16",
        .u32 => "U32",
        .i32 => "I32",
        .f32 => "F32",
        .u64 => "U64",
        .i64 => "I64",
        .f64 => "F64",
        .f8_e4m3 => "F8_E4M3",
        .f8_e5m2 => "F8_E5M2",
        .f8_e8m0 => "F8_E8M0",
    };
}

pub const Checked = struct { samples: u64, bytes: u64 };

/// The boot's check: the release tensors still match, the shards hold, and the sampled rows read byte-identical.
pub fn check(a: Allocator, io: Io, release_dir: []const u8, tables_dir: []const u8, full: bool) !Checked {
    const mpath = try std.fs.path.join(a, &.{ tables_dir, filename });
    defer a.free(mpath);
    const text = std.Io.Dir.cwd().readFileAlloc(io, mpath, a, .limited(1 << 22)) catch |e| {
        if (e == error.FileNotFound) return error.EngramManifestMissing;
        return error.EngramManifestBad;
    };
    defer a.free(text);
    return checkText(a, io, release_dir, tables_dir, text, full);
}

pub const ManifestResult = struct { tables: u64, samples: u64 };

/// The manifest for tables that already exist: shard headers, release tensors, a seeded sample; the rows proved first.
pub fn manifestFor(a: Allocator, io: Io, release_dir: []const u8, tables_dir: []const u8, seed: u64) !ManifestResult {
    var dir = std.Io.Dir.cwd().openDir(io, tables_dir, .{ .iterate = true }) catch return error.EngramManifestMissing;
    defer dir.close(io);
    var names: std.ArrayList([]const u8) = .empty;
    defer {
        for (names.items) |s| a.free(s);
        names.deinit(a);
    }
    var it = dir.iterate();
    while (try it.next(io)) |e| {
        if (std.mem.startsWith(u8, e.name, "engram-l") and std.mem.endsWith(u8, e.name, ".bin")) try names.append(a, try a.dupe(u8, e.name));
    }
    if (names.items.len == 0) return error.EngramManifestBad;
    std.mem.sort([]const u8, names.items, {}, lessName);
    var pack = try Pack.open(a, io, release_dir);
    defer pack.deinit();
    const arena_p = try a.create(std.heap.ArenaAllocator);
    arena_p.* = std.heap.ArenaAllocator.init(a);
    const al = arena_p.allocator();
    var m = Manifest{ .root = a, .arena = arena_p, .seed = seed };
    defer m.deinit();
    var srcs: std.ArrayList(Source) = .empty;
    var tabs: std.ArrayList(Table) = .empty;
    var span: u64 = 0;
    for (names.items) |n| {
        const tpath = try std.fs.path.join(a, &.{ tables_dir, n });
        defer a.free(tpath);
        const f = std.Io.Dir.cwd().openFile(io, tpath, .{}) catch return error.EngramShardCorrupt;
        defer f.close(io);
        const size = try f.length(io);
        var hb: [48]u8 = undefined;
        if (try f.readPositionalAll(io, &hb, 0) != 48) return error.EngramShardCorrupt;
        const h = er.Header.parse(&hb, size) catch return error.EngramShardCorrupt;
        const p = Plan.fromPack(&pack, @intCast(h.layer), h.total) catch return error.EngramSourceMismatch;
        span = if (span == 0 or h.total < span) h.total else span;
        var nb: [96]u8 = undefined;
        for ([_]usize{ 0, 1 }) |k| {
            const i: Info = if (k == 0) p.weight else p.scale;
            const name = try std.fmt.bufPrint(&nb, "layers.{d}.engram.embed.{s}", .{ h.layer, if (k == 0) "weight" else "scale" });
            try srcs.append(al, Source{
                .name = try al.dupe(u8, name),
                .file = try al.dupe(u8, pack.files.items[i.file]),
                .dtype = try al.dupe(u8, dtypeName(i.dtype)),
                .shape = try al.dupe(usize, i.dims()),
                .offset = i.start,
                .length = i.nbytes,
            });
        }
        var tb: [64]u8 = undefined;
        const tname = try ep.shardName(&tb, @intCast(h.layer));
        if (!std.mem.eql(u8, n, tname)) return error.EngramShardCorrupt;
        try tabs.append(al, Table{
            .layer = @intCast(h.layer),
            .file = try al.dupe(u8, tname),
            .lo = h.lo,
            .hi = h.hi,
            .total = h.total,
            .size = size,
            .sha = try fileSha256(io, tpath),
        });
    }
    m.samples = try ep.samples(al, span, 48, seed);
    m.sources = srcs.items;
    m.tables = tabs.items;
    const text = try m.render();
    const got = try checkText(a, io, release_dir, tables_dir, text, false);
    const res = ManifestResult{ .tables = m.tables.len, .samples = got.samples };
    const tpath = try std.fs.path.join(a, &.{ tables_dir, tmp_name });
    defer a.free(tpath);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = tpath, .data = text });
    try std.Io.Dir.rename(dir, tmp_name, dir, filename, io);
    return res;
}

/// One manifest's bytes against a release: the tensors still match, the shards hold, the sampled rows byte-identical.
fn checkText(a: Allocator, io: Io, release_dir: []const u8, tables_dir: []const u8, text: []const u8, full: bool) !Checked {
    const m = try Manifest.parse(a, text);
    defer m.deinit();
    if (m.version != 1) return error.EngramManifestBad;
    if (m.rows != er.row_bytes or m.values != er.values) return error.EngramManifestBad;
    if (m.tables.len == 0 or m.sources.len != m.tables.len * 2 or m.samples.len == 0) return error.EngramManifestBad;
    var pack = try Pack.open(a, io, release_dir);
    defer pack.deinit();
    var got: Checked = .{ .samples = 0, .bytes = 0 };
    for (m.tables, 0..) |t, ti| {
        const w = m.sources[ti * 2];
        const s = m.sources[ti * 2 + 1];
        try sourceMatches(a, io, &pack, w);
        try sourceMatches(a, io, &pack, s);
        const tpath = try std.fs.path.join(a, &.{ tables_dir, t.file });
        defer a.free(tpath);
        const f = std.Io.Dir.cwd().openFile(io, tpath, .{}) catch return error.EngramShardCorrupt;
        defer f.close(io);
        const size = try f.length(io);
        if (size != t.size) return error.EngramShardCorrupt;
        var hb: [48]u8 = undefined;
        if (try f.readPositionalAll(io, &hb, 0) != 48) return error.EngramShardCorrupt;
        const h = er.Header.parse(&hb, size) catch return error.EngramShardCorrupt;
        if (h.layer != t.layer or h.lo != t.lo or h.hi != t.hi or h.total != t.total) return error.EngramShardCorrupt;
        const wi = pack.get(w.name).?;
        const si = pack.get(s.name).?;
        for (m.samples) |x| {
            if (x < h.lo or x >= h.hi) continue;
            var row: [er.row_bytes]u8 = undefined;
            if (try f.readPositionalAll(io, &row, er.header_bytes + x * er.row_bytes) != er.row_bytes) return error.EngramShardCorrupt;
            var wrow: [er.values]u8 = undefined;
            var srow: [8]u8 = undefined;
            try pack.read(io, wi, .{ .offset = x * er.values, .len = er.values }, &wrow);
            try pack.read(io, si, .{ .offset = x * 8, .len = 8 }, &srow);
            if (!std.mem.eql(u8, row[0..er.values], &wrow) or !std.mem.eql(u8, row[er.values..], &srow)) return error.EngramSampleMismatch;
            got.samples += 1;
            got.bytes += er.row_bytes;
        }
        if (full) {
            const whole = try shaOf(f, io);
            if (!std.mem.eql(u8, &whole, &t.sha)) return error.EngramShardCorrupt;
            got.bytes += t.size;
        }
    }
    return got;
}

/// One recorded source tensor against its release pack: file, dtype, shape, offset and length still match.
fn sourceMatches(a: Allocator, io: Io, p: *const Pack, s: Source) !void {
    const i = p.get(s.name) orelse return error.EngramSourceMismatch;
    if (!std.mem.eql(u8, dtypeName(i.dtype), s.dtype) or !std.mem.eql(usize, i.dims(), s.shape) or !std.mem.eql(u8, p.files.items[i.file], s.file) or i.start != s.offset or i.nbytes != s.length) return error.EngramSourceMismatch;
    const spath = try std.fs.path.join(a, &.{ p.dir, s.file });
    defer a.free(spath);
    const f = std.Io.Dir.cwd().openFile(io, spath, .{}) catch return error.EngramSourceMismatch;
    defer f.close(io);
    const len = try f.length(io);
    if (len < i.start + i.nbytes) return error.EngramSourceMismatch;
}

fn lessName(_: void, x: []const u8, y: []const u8) bool {
    return std.mem.order(u8, x, y) == .lt;
}

fn fileLength(io: Io, path: []const u8) !u64 {
    const f = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    return f.length(io);
}

fn fileSha256(io: Io, path: []const u8) ![32]u8 {
    const f = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    return shaOf(f, io);
}

fn shaOf(f: Io.File, io: Io) ![32]u8 {
    var h = Sha256.init(.{});
    const buf = std.heap.page_allocator.alloc(u8, 1 << 20) catch @panic("the Engram manifest check cannot get a 1 MiB read buffer");
    defer std.heap.page_allocator.free(buf);
    var pos: u64 = 0;
    while (true) {
        const n = try f.readPositionalAll(io, buf, pos);
        if (n == 0) break;
        h.update(buf[0..n]);
        pos += n;
    }
    return h.finalResult();
}
