//! Lane products at any MLX width on the recorded lane kernel's text: one launch per bit-width run, mapped columns.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const replay = @import("replay.zig");
const Run = replay.Run;
const Buf = replay.Buf;
const pack_lane = @import("pack_lane.zig");
const variants = @import("variants.zig");
const rows_w = @import("rows_w.zig");

/// One run's product: n columns over k inputs at one width, and where its columns land in the fused output.
pub const Product = struct { wq: Buf, sbt: Buf, n: usize, k: usize, bits: u8, group: u16, map: ?Buf = null, tiled: bool = true };

/// A lane projection read at its own widths: its runs and the fused output's width.
pub const Wide = struct { runs: []const Product, n: usize };

/// A pack lane: the recorded 6/32 `.wq` and `.sbt`, or its runs at other widths (`w`); never both.
pub const Lane = struct {
    six: ?Six = null,
    w: ?*const Wide = null,
    rows: ?*const rows_w.Dense = null, // the same projection in MLX layout on the row kernels (rows_load.zig)

    pub const Six = struct { wq: Buf, sbt: Buf };

    /// The buffers a 6/32 kernel reads; a lane at other widths has none, so no 6-bit kernel reads one of its runs.
    pub fn recorded(l: Lane) error{LaneNot6Bit32}!Six {
        return l.six orelse error.LaneNot6Bit32;
    }
};

/// One compiled variant: shape, width, row tiles, output stride, column map, words tiled (else MLX's, decode.py nt 0).
pub const Spec = struct { n: usize, k: usize, bits: u8, group: u16, tmr: usize, ld: usize, mapped: bool, tiled: bool = true };

/// Compiled variants by their spec.
pub const Cache = std.AutoHashMapUnmanaged(Spec, mtl.Pipeline);

/// K slices for an n by k product as lane_qmm.split_k: fixed by the shape, never by the row count.
pub fn splitK(n: usize, k: usize) usize {
    const tiles = (n + 31) / 32;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < 1024 and (k / 64) / (sk * 2) >= 8) sk *= 2;
    return sk;
}

/// The recorded lane kernel's text for this GPU: the -lanes twin without tensor units.
pub fn template(tensor_units: bool) ![]const u8 {
    const name = if (tensor_units) "lane_qmm_bytes_grouped_68a0697c6f38c26d.metal" else "lane_qmm_bytes_grouped_68a0697c6f38c26d-lanes.metal";
    for (ks.flashnext_gen.sources) |s| if (std.mem.eql(u8, s.name, name)) return s.text;
    return error.NoSource;
}

fn swap(a: std.mem.Allocator, text: []const u8, from: []const u8, to: []const u8) ![]const u8 {
    if (std.mem.count(u8, text, from) != 1) return error.LanePatch;
    return std.mem.replaceOwned(u8, a, text, from, to);
}

/// The lane kernel at `s` (any width staged as bytes, sums over GS, stores at stride `ld`): the recorded arithmetic.
pub fn source(a: std.mem.Allocator, text_in: []const u8, s: Spec) ![]const u8 {
    var text = text_in;
    for ([_][]const u8{ "TMR", "N", "K", "SK", "BITS", "GS", "TILED" }, [_]usize{ s.tmr, s.n, s.k, splitK(s.n, s.k), s.bits, s.group, @intFromBool(s.tiled) }) |name, v|
        text = try variants.constant(a, text, name, v);
    text = try swap(a, text, "static_assert(BITS == 5 || BITS == 6 || BITS == 8, \"bytes for 5-, 6- and 8-bit weights\");", "static_assert(BITS >= 2 && BITS <= 8, \"codes of 2 to 8 bits staged as bytes\");");
    text = try swap(a, text, "for (int i = 0; i < 32; i++) acc += float(X[m * K + g * 32 + i]);", "for (int i = 0; i < FZ_GS; i++) acc += float(X[m * K + g * FZ_GS + i]);");
    const lanes = std.mem.indexOf(u8, text, "Y[m * N + nn + (j & 1) + 8 * (j >> 1)]") != null;
    text = if (lanes)
        try swap(a, text, "Y[m * N + nn + (j & 1) + 8 * (j >> 1)]", "Y[m * FZ_LD + FZ_COL(nn + (j & 1) + 8 * (j >> 1))]")
    else
        try swap(a, text, "Y[m * N + nn + j]", "Y[m * FZ_LD + FZ_COL(nn + j)]");
    if (s.mapped) text = try swap(a, text, "  device bfloat16_t* Y [[buffer(5)]],\n", "  device bfloat16_t* Y [[buffer(5)]],\n  const device uint32_t* FZ_MAP [[buffer(6)]],\n");
    const col = if (s.mapped) "(FZ_MAP[c])" else "(c)";
    return std.fmt.allocPrint(a, "#define FZ_GS {d}\n#define FZ_LD {d}\n#define FZ_COL(c) {s}\n{s}", .{ s.group, s.ld, col, text });
}

/// The kernel function a source defines.
pub fn functionName(a: std.mem.Allocator, text: []const u8) ![:0]const u8 {
    const at = (std.mem.indexOf(u8, text, "[[kernel]] void ") orelse return error.LanePatch) + "[[kernel]] void ".len;
    const end = at + (std.mem.indexOfScalar(u8, text[at..], '(') orelse return error.LanePatch);
    return std.fmt.allocPrintSentinel(a, "{s}", .{text[at..end]}, 0);
}

/// The pipeline for `s` on `device`, compiled once and kept in `cache`.
pub fn pipeline(a: std.mem.Allocator, device: mtl.Device, cache: *Cache, s: Spec) !mtl.Pipeline {
    if (s.n == 0 or s.n % 4 != 0 or s.tiled != (s.n % 32 == 0)) return error.LaneShape;
    if (cache.get(s)) |p| return p;
    const text = try source(a, try template(device.tensorUnits()), s);
    const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const pipe = try mtl.Pipeline.init(device, lib, try functionName(a, text), false);
    errdefer pipe.deinit();
    try cache.put(a, s, pipe);
    return pipe;
}

/// Rows tile 16 at a time up to 32 (lane_qmm's ROW_BLOCK): a row's sums never depend on its window.
pub fn rowBlock(rows: usize) usize {
    return @min((rows + 15) / 16 * 16, 32);
}

/// y[rows, w.n] = x[rows, k] W^T, every run at its own width; `mdims` holds rows and rows padded to 16.
pub fn encode(enc: mtl.ComputeEncoder, a: std.mem.Allocator, device: mtl.Device, cache: *Cache, w: *const Wide, x: Buf, mdims: Buf, rows: usize, y: Buf) !void {
    const block = rowBlock(rows);
    const mp = (rows + 15) / 16 * 16;
    for (w.runs) |p| {
        const s: Spec = .{ .n = p.n, .k = p.k, .bits = p.bits, .group = p.group, .tmr = block / 16, .ld = w.n, .mapped = p.map != null, .tiled = p.tiled };
        enc.setPipeline(try pipeline(a, device, cache, s));
        for ([_]Buf{ x, x, p.wq, p.sbt, mdims, y }, 0..) |b, j| enc.setBuffer(b.b, b.off, j);
        if (p.map) |m| enc.setBuffer(m.b, m.off, 6);
        const sk = splitK(p.n, p.k);
        enc.dispatchThreads(mtl.Size.of((p.n + 31) / 32 * 32 * sk, (mp + block - 1) / block, 1), mtl.Size.of(32 * sk, 1, 1));
    }
}

/// `w` on the run's encoder at `rows` rows, unless the role is knocked out.
pub fn launch(r: *Run, w: *const Wide, role: []const u8, x: Buf, mdims: Buf, rows: usize, y: Buf) !void {
    if (r.skip & Run.class(role) != 0) return;
    try encode(r.enc, r.arena, r.device, &r.wide_pipes, w, x, mdims, rows, y);
    if (!r.serial) r.enc.barrier();
}

/// Prefetches a lane's weights: the recorded pair, or every run's.
pub fn touch(r: *Run, l: Lane) void {
    if (l.six) |s| for ([_]Buf{ s.wq, s.sbt }) |b| r.touch(b);
    if (l.w) |w| for (w.runs) |p| for ([_]Buf{ p.wq, p.sbt }) |b| r.touch(b);
}

/// Run columns to fused columns from a `.members` table of (run, row in run, row in fused output) per member.
pub fn columnMaps(a: std.mem.Allocator, members: []const u32, run_rows: []const usize) ![][]u32 {
    const maps = try a.alloc([]u32, run_rows.len);
    for (maps, run_rows) |*m, n| {
        m.* = try a.alloc(u32, n);
        @memset(m.*, std.math.maxInt(u32));
    }
    const count = members.len / 3;
    for (0..count) |i| {
        const run = members[i * 3];
        if (run >= run_rows.len) return error.BadLaneMembers;
        const start = members[i * 3 + 1];
        var end: usize = run_rows[run]; // a member runs to the next start in its run
        for (0..count) |j| if (members[j * 3] == run and members[j * 3 + 1] > start) {
            end = @min(end, members[j * 3 + 1]);
        };
        if (start >= end) return error.BadLaneMembers;
        for (start..end) |c| maps[run][c] = members[i * 3 + 2] + @as(u32, @intCast(c - start));
    }
    var total: usize = 0;
    for (run_rows) |n| total += n;
    const seen = try a.alloc(bool, total); // every fused column written exactly once
    @memset(seen, false);
    for (maps) |m| for (m) |c| {
        if (c >= total or seen[c]) return error.BadLaneMembers;
        seen[c] = true;
    };
    return maps;
}

/// A pack lane's widths: null for a 6/32 lane (no `.runs` table), else its runs, each `.rN` when there are several.
pub fn of(r: *Run, stem: []const u8) !?*const Wide {
    const a = r.arena;
    const runs_name = try std.fmt.allocPrint(a, "{s}.runs", .{stem});
    if (!r.index.contains(runs_name)) return null;
    const t = try variants.table(r, runs_name);
    const n_runs = t.len / 3;
    if (n_runs == 0 or t.len % 3 != 0) return error.BadLaneRuns;
    const run_rows = try a.alloc(usize, n_runs);
    for (run_rows, 0..) |*n, i| n.* = t[i * 3 + 2];
    var maps: ?[][]u32 = null;
    if (n_runs > 1) {
        const members_name = try std.fmt.allocPrint(a, "{s}.members", .{stem});
        maps = try columnMaps(a, try variants.table(r, members_name), run_rows);
    }
    const runs = try a.alloc(Product, n_runs);
    var total: usize = 0;
    for (runs, 0..) |*p, i| {
        const base = if (n_runs == 1) stem else try std.fmt.allocPrint(a, "{s}.r{d}", .{ stem, i });
        const wq_name = try std.fmt.allocPrint(a, "{s}.wq", .{base});
        const bits: u8 = @intCast(t[i * 3]);
        const n = run_rows[i];
        if (n == 0 or bits == 0) return error.BadLaneRuns;
        const k = (try r.entry(wq_name)).len * 8 / (n * bits);
        p.* = .{ .wq = try r.load(wq_name), .sbt = try r.loadf("{s}.sbt", .{base}), .n = n, .k = k, .bits = bits, .group = @intCast(t[i * 3 + 1]), .tiled = pack_lane.tiled(n) };
        if (maps) |m| {
            const b = try r.buffer(n * 4);
            @memcpy(b.slice(u32, n), m[i]);
            p.map = .{ .b = b };
        }
        total += n;
    }
    const w = try a.create(Wide);
    w.* = .{ .runs = runs, .n = total };
    return w;
}

/// A pack lane: its runs when not 6/32 (`of`), else the recorded `.wq` and `.sbt`.
pub fn laneOf(r: *Run, comptime fmt: []const u8, args: anytype) !Lane {
    var name: [96]u8 = undefined;
    const stem = try std.fmt.bufPrint(&name, fmt, args);
    var full: [128]u8 = undefined;
    if (try of(r, stem)) |w| return .{ .w = w };
    return .{ .six = .{ .wq = try r.load(try std.fmt.bufPrint(&full, "{s}.wq", .{stem})), .sbt = try r.load(try std.fmt.bufPrint(&full, "{s}.sbt", .{stem})) } };
}

test {
    _ = @import("lane_w_test.zig");
}
