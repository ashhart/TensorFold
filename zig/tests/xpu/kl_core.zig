//! Model-independent KL-vs-base-logits harness: base-file reader (tools/klref.py), scoring, running stats, csv resume.

const std = @import("std");
const tfix = @import("fx.zig");

extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn pread(fd: c_int, buf: [*]u8, n: usize, off: i64) isize;
extern "c" fn read(fd: c_int, buf: [*]u8, n: usize) isize;
extern "c" fn write(fd: c_int, buf: [*]const u8, n: usize) isize;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn mkdir(path: [*:0]const u8, mode: c_uint) c_int;

pub fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

pub fn preadAll(fd: c_int, buf: []u8, off: u64) !void {
    var done: usize = 0;
    while (done < buf.len) {
        const n = pread(fd, buf.ptr + done, buf.len - done, @intCast(off + done));
        if (n <= 0) return error.ShortRead;
        done += @intCast(n);
    }
}

pub fn appendFile(path: [*:0]const u8, bytes: []const u8) !void {
    const fd = open(path, 0x441, @as(c_int, 0o644)); // O_WRONLY | O_CREAT | O_APPEND
    if (fd < 0) return error.OpenFailed;
    defer _ = close(fd);
    if (write(fd, bytes.ptr, bytes.len) != @as(isize, @intCast(bytes.len))) return error.WriteFailed;
}

/// Whole file as bytes, empty when missing.
pub fn slurp(gpa: std.mem.Allocator, path: [*:0]const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    const fd = open(path, 0);
    if (fd < 0) return out.toOwnedSlice(gpa);
    defer _ = close(fd);
    var buf: [1 << 16]u8 = undefined;
    while (true) {
        const n = read(fd, &buf, buf.len);
        if (n <= 0) break;
        try out.appendSlice(gpa, buf[0..@intCast(n)]);
    }
    return out.toOwnedSlice(gpa);
}

pub fn makeDir(gpa: std.mem.Allocator, path: []const u8) !void {
    _ = mkdir(try std.fmt.allocPrintSentinel(gpa, "{s}", .{path}, 0), 0o755);
}

/// "A:B" (B exclusive) and plain indices, comma separated.
pub fn parseChunks(gpa: std.mem.Allocator, s: []const u8, n_chunk: u32) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    var it = std.mem.tokenizeScalar(u8, s, ',');
    while (it.next()) |t| {
        if (std.mem.indexOfScalar(u8, t, ':')) |c| {
            const a = try std.fmt.parseInt(u32, t[0..c], 10);
            const b = try std.fmt.parseInt(u32, t[c + 1 ..], 10);
            var k = a;
            while (k < @min(b, n_chunk)) : (k += 1) try out.append(gpa, k);
        } else {
            const k = try std.fmt.parseInt(u32, t, 10);
            if (k < n_chunk) try out.append(gpa, k);
        }
    }
    return out.toOwnedSlice(gpa);
}

/// An opened base-logits file: header fields and the stored token ids of every chunk.
pub const Ref = struct {
    fd: c_int,
    n_ctx: u32,
    n_vocab: u32,
    n_chunk: u32,
    first: u32,
    rows: u32, // scored rows a chunk: positions first .. n_ctx-2
    nv: usize, // u16 a stored row
    data_off: u64,
    tokens: []i32,

    pub fn load(gpa: std.mem.Allocator, path: [*:0]const u8) !Ref {
        const fd = open(path, 0);
        if (fd < 0) return error.OpenFailed;
        var hdr: [20]u8 = undefined;
        try preadAll(fd, &hdr, 0);
        if (!std.mem.eql(u8, hdr[0..8], "_logits_")) return error.NotLogitsFile;
        const n_ctx = std.mem.readInt(u32, hdr[8..12], .little);
        const n_vocab = std.mem.readInt(u32, hdr[12..16], .little);
        const n_chunk = std.mem.readInt(u32, hdr[16..20], .little);
        const tokens = try gpa.alloc(i32, @as(usize, n_ctx) * n_chunk);
        try preadAll(fd, std.mem.sliceAsBytes(tokens), 20);
        return .{
            .fd = fd,
            .n_ctx = n_ctx,
            .n_vocab = n_vocab,
            .n_chunk = n_chunk,
            .first = n_ctx / 2,
            .rows = n_ctx - 1 - n_ctx / 2,
            .nv = 2 * ((n_vocab + 1) / 2) + 4,
            .data_off = 20 + 4 * @as(u64, n_ctx) * n_chunk,
            .tokens = tokens,
        };
    }

    pub fn chunkTokens(r: Ref, c: u32) []const i32 {
        return r.tokens[@as(usize, c) * r.n_ctx ..][0..r.n_ctx];
    }

    /// Stored row `row` (position first + row) of chunk `c` into `buf` (nv u16).
    pub fn readRow(r: Ref, c: u32, row: u32, buf: []u16) !void {
        try preadAll(r.fd, std.mem.sliceAsBytes(buf[0..r.nv]), r.data_off + (@as(u64, c) * r.rows + row) * r.nv * 2);
    }
};

pub const Row = struct { kld: f64, same: bool, lp_ours: f64, lp_ref: f64 };

/// The stored row as f32 log-probs (scale * q + min), as scoreRow reads it.
pub fn dequantRow(ref_row: []const u16, out: []f32) void {
    const scale: f32 = @bitCast(@as(u32, ref_row[0]) | @as(u32, ref_row[1]) << 16);
    const min_lp: f32 = @bitCast(@as(u32, ref_row[2]) | @as(u32, ref_row[3]) << 16);
    for (out, 0..) |*o, k| o.* = scale * @as(f32, @floatFromInt(ref_row[4 + k])) + min_lp;
}

/// Scores one position like llama-perplexity: fp64 ours, f32 dequantized base, KL over base > -16; lpb is scratch.
pub fn scoreRow(logits: []const f32, ref_row: []const u16, lpb: []f32, target: usize) Row {
    const scale: f32 = @bitCast(@as(u32, ref_row[0]) | @as(u32, ref_row[1]) << 16);
    const min_lp: f32 = @bitCast(@as(u32, ref_row[2]) | @as(u32, ref_row[3]) << 16);
    var mx = logits[0];
    var imax: usize = 0;
    for (logits[1..], 1..) |v, k| if (v > mx) {
        mx = v;
        imax = k;
    };
    var se: f64 = 0;
    for (logits) |v| se += @exp(@as(f64, v - mx));
    const lse: f64 = @as(f64, mx) + @log(se);
    var bmax: f32 = -std.math.inf(f32);
    var ibase: usize = 0;
    var kld: f64 = 0;
    for (lpb, 0..) |*o, k| {
        const lb: f32 = scale * @as(f32, @floatFromInt(ref_row[4 + k])) + min_lp;
        o.* = lb;
        if (k == 0 or lb > bmax) {
            bmax = lb;
            ibase = k;
        }
        if (lb > -16.0) kld += @exp(@as(f64, lb)) * (@as(f64, lb) - (@as(f64, logits[k]) - lse));
    }
    return .{ .kld = kld, .same = imax == ibase, .lp_ours = @as(f64, logits[target]) - lse, .lp_ref = lpb[target] };
}

/// Running sums as in llama.cpp's kl_divergence_result.
pub const Acc = struct {
    nll: f64 = 0,
    nll2: f64 = 0,
    nllb: f64 = 0,
    nllb2: f64 = 0,
    nll_nllb: f64 = 0,
    kld: f64 = 0,
    kld2: f64 = 0,
    same: u64 = 0,
    n: u64 = 0,

    pub fn add(a: *Acc, x: Row) void {
        const nll = -x.lp_ours;
        const nllb = -x.lp_ref;
        a.nll += nll;
        a.nll2 += nll * nll;
        a.nllb += nllb;
        a.nllb2 += nllb * nllb;
        a.nll_nllb += nll * nllb;
        a.kld += x.kld;
        a.kld2 += x.kld * x.kld;
        a.same += @intFromBool(x.same);
        a.n += 1;
    }
};

const MU = struct { m: f64, u: f64 };

fn meanUnc(sum: f64, sum2: f64, n: u64) MU {
    if (n < 1) return .{ .m = 0, .u = 0 };
    const nf: f64 = @floatFromInt(n);
    const f = sum / nf;
    const df = sum2 / nf - f * f;
    return .{ .m = f, .u = if (df > 0 and n > 10) @sqrt(df / (nf - 1)) else 0 };
}

pub fn printRunning(chunk: u32, a: Acc) void {
    const nf: f64 = @floatFromInt(a.n);
    const ppl = meanUnc(a.nll, a.nll2, a.n);
    const base = meanUnc(a.nllb, a.nllb2, a.n);
    const cov = (a.nll_nllb / nf - (a.nll / nf) * (a.nllb / nf)) / (nf - 1);
    const lr = ppl.m - base.m;
    const lru = @sqrt(@max(ppl.u * ppl.u + base.u * base.u - 2 * cov, 0));
    const k = meanUnc(a.kld, a.kld2, a.n);
    const pt = @as(f64, @floatFromInt(a.same)) / nf;
    std.debug.print("{d:>4}  PPL {d:>9.4}  ln(PPL(Q)/PPL(base)) {d:>9.5} +- {d:.5}  KLD {d:>9.5} +- {d:.5}  same top {d:>7.3} +- {d:.3} %\n", .{
        chunk, @exp(ppl.m), lr, lru, k.m, k.u, 100 * pt, 100 * @sqrt(pt * (1 - pt) / (nf - 1)),
    });
}

/// Per-chunk row counts already in the csv (resume); writes the header when the file is new.
pub fn resumeCounts(gpa: std.mem.Allocator, csv_path: [*:0]const u8, n_chunk: u32) ![]u32 {
    const done = try gpa.alloc(u32, n_chunk);
    @memset(done, 0);
    const old = try slurp(gpa, csv_path);
    var lines = std.mem.tokenizeScalar(u8, old, '\n');
    while (lines.next()) |ln| {
        const c = std.mem.indexOfScalar(u8, ln, ',') orelse continue;
        const k = std.fmt.parseInt(u32, ln[0..c], 10) catch continue;
        if (k < n_chunk) done[k] += 1;
    }
    if (old.len == 0) try appendFile(csv_path, "chunk,pos,kld,same_top,lp_ours,lp_ref\n");
    return done;
}

/// One csv line of a scored position.
pub fn csvLine(buf: []u8, chunk: u32, pos: u32, x: Row) ![]const u8 {
    return std.fmt.bufPrint(buf, "{d},{d},{e:.9},{d},{d:.6},{d:.6}\n", .{ chunk, pos, x.kld, @intFromBool(x.same), x.lp_ours, x.lp_ref });
}
