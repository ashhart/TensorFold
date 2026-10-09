//! KL divergence vs a llama.cpp base-logits file. usage: xpu-qwen_kl-test CKPT REF.logits TAG [--chunks A:B] [opts]

const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");
const ld = @import("xpu").loader;
const cfgm = @import("qwen_xpu").config;
const model = @import("qwen_xpu").model;
const qg = @import("qwen_xpu").gguf;
const qw = @import("qwen_xpu").win;
const stop = @import("xpu").stop;

extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn pread(fd: c_int, buf: [*]u8, n: usize, off: i64) isize;
extern "c" fn read(fd: c_int, buf: [*]u8, n: usize) isize;
extern "c" fn write(fd: c_int, buf: [*]const u8, n: usize) isize;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn mkdir(path: [*:0]const u8, mode: c_uint) c_int;

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn preadAll(fd: c_int, buf: []u8, off: u64) !void {
    var done: usize = 0;
    while (done < buf.len) {
        const n = pread(fd, buf.ptr + done, buf.len - done, @intCast(off + done));
        if (n <= 0) return error.ShortRead;
        done += @intCast(n);
    }
}

fn appendFile(path: [*:0]const u8, bytes: []const u8) !void {
    const fd = open(path, 0x441, @as(c_int, 0o644)); // O_WRONLY | O_CREAT | O_APPEND
    if (fd < 0) return error.OpenFailed;
    defer _ = close(fd);
    if (write(fd, bytes.ptr, bytes.len) != @as(isize, @intCast(bytes.len))) return error.WriteFailed;
}

/// Whole file as bytes, empty when missing.
fn slurp(gpa: std.mem.Allocator, path: [*:0]const u8) ![]u8 {
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

/// "A:B" (B exclusive) and plain indices, comma separated.
fn parseChunks(gpa: std.mem.Allocator, s: []const u8, n_chunk: u32) ![]u32 {
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

/// Running sums as in llama.cpp's kl_divergence_result.
const Acc = struct {
    nll: f64 = 0,
    nll2: f64 = 0,
    nllb: f64 = 0,
    nllb2: f64 = 0,
    nll_nllb: f64 = 0,
    kld: f64 = 0,
    kld2: f64 = 0,
    same: u64 = 0,
    n: u64 = 0,

    fn add(a: *Acc, kld: f64, same: bool, nll: f64, nllb: f64) void {
        a.nll += nll;
        a.nll2 += nll * nll;
        a.nllb += nllb;
        a.nllb2 += nllb * nllb;
        a.nll_nllb += nll * nllb;
        a.kld += kld;
        a.kld2 += kld * kld;
        a.same += @intFromBool(same);
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

fn printRunning(chunk: u32, a: Acc) void {
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

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main(init: std.process.Init) !u8 {
    stop.install();
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 4) {
        std.debug.print("usage: xpu-qwen_kl-test CHECKPOINT REF.logits TAG [--chunks A:B,N,...] [--bf16-logits] [--out DIR]\n", .{});
        return 2;
    }
    const dir = args[1];
    const ref_path = args[2];
    const tag = args[3];
    var chunk_spec: []const u8 = "0:1";
    var out_dir: []const u8 = "out";
    var bf16_logits = false;
    var prefill_rows: u32 = 0;
    var i: usize = 4;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--chunks")) {
            i += 1;
            chunk_spec = args[i];
        } else if (std.mem.eql(u8, args[i], "--out")) {
            i += 1;
            out_dir = args[i];
        } else if (std.mem.eql(u8, args[i], "--prefill")) {
            i += 1;
            prefill_rows = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--bf16-logits")) bf16_logits = true else return error.BadArgument;
    }

    // base-logits header: "_logits_", u32 n_ctx, i32 n_vocab, i32 n_chunk, i32 tokens[n_ctx * n_chunk]
    const rfd = open(ref_path, 0);
    if (rfd < 0) return error.OpenFailed;
    var hdr: [20]u8 = undefined;
    try preadAll(rfd, &hdr, 0);
    if (!std.mem.eql(u8, hdr[0..8], "_logits_")) return error.NotLogitsFile;
    const n_ctx = std.mem.readInt(u32, hdr[8..12], .little);
    const n_vocab = std.mem.readInt(u32, hdr[12..16], .little);
    const n_chunk = std.mem.readInt(u32, hdr[16..20], .little);
    const first = n_ctx / 2;
    const rows = n_ctx - 1 - first;
    const nv: usize = 2 * ((n_vocab + 1) / 2) + 4;
    const tokens = try gpa.alloc(i32, @as(usize, n_ctx) * n_chunk);
    try preadAll(rfd, std.mem.sliceAsBytes(tokens), 20);
    const data_off: u64 = 20 + 4 * @as(u64, n_ctx) * n_chunk;
    std.debug.print("ref: n_ctx {d} n_vocab {d} n_chunk {d}, {d} scored rows per chunk\n", .{ n_ctx, n_vocab, n_chunk, rows });

    const chunks = try parseChunks(gpa, chunk_spec, n_chunk);
    _ = mkdir(try std.fmt.allocPrintSentinel(gpa, "{s}", .{out_dir}, 0), 0o755);
    const csv_path = try std.fmt.allocPrintSentinel(gpa, "{s}/kl_{s}.csv", .{ out_dir, tag }, 0);
    const time_path = try std.fmt.allocPrintSentinel(gpa, "{s}/kl_{s}.time", .{ out_dir, tag }, 0);

    // resume: chunks that already have all their rows in the csv are skipped
    const done_rows = try gpa.alloc(u32, n_chunk);
    @memset(done_rows, 0);
    const old = try slurp(gpa, csv_path);
    var lines = std.mem.tokenizeScalar(u8, old, '\n');
    while (lines.next()) |ln| {
        const c = std.mem.indexOfScalar(u8, ln, ',') orelse continue;
        const k = std.fmt.parseInt(u32, ln[0..c], 10) catch continue;
        if (k < n_chunk) done_rows[k] += 1;
    }
    if (old.len == 0) try appendFile(csv_path, "chunk,pos,kld,same_top,lp_ours,lp_ref\n");

    const is_gguf = std.mem.endsWith(u8, dir, ".gguf");
    var r = try rt.open();
    defer r.deinit();
    var l = if (is_gguf) try ld.Loader.initBare(gpa, &r) else try ld.Loader.init(gpa, &r, dir);
    const cfg: std.json.Parsed(cfgm.Config) = if (is_gguf) blk: {
        const meta = try qg.readHeader(gpa, &l, dir);
        break :blk .{ .arena = undefined, .value = try qg.config(gpa, &l, meta) };
    } else try cfgm.parse(gpa, try ld.readFile(gpa, dir, "config.json"));
    if (cfg.value.text_config.vocab_size != n_vocab) return error.VocabMismatch;
    if (prefill_rows > 16) qw.default_rows = prefill_rows;
    var m = try model.Model.load(gpa, &r, &l, cfg.value, n_ctx);
    // a big window (GGUF / EXL3 prefill GEMMs) scores all rows from one pass; otherwise 16-row windows
    const big = prefill_rows >= rows and (m.ops.ex != null or m.ops.gg != null);
    const batch = try gpa.alloc(f32, 16 * @as(usize, n_vocab));
    const tok32 = try gpa.alloc(u32, n_ctx);
    m.bf16_logits = bf16_logits;
    const zeros = try gpa.alloc(u8, model.state_bytes);
    @memset(zeros, 0);

    const logits = try gpa.alloc(f32, n_vocab);
    const ref_row = try gpa.alloc(u16, nv);
    const lpb = try gpa.alloc(f32, n_vocab);
    var csv: std.ArrayList(u8) = .empty;
    var acc: Acc = .{};
    var tot_steps: u64 = 0;
    var tot_gpu_ns: u64 = 0;
    var line: [160]u8 = undefined;

    for (chunks) |c| {
        if (stop.requested()) return error.Interrupted; // /tmp/arc_stop or a signal: finished chunks are already in the csv
        if (done_rows[c] >= rows) continue;
        const toks = tokens[@as(usize, c) * n_ctx ..][0..n_ctx];
        csv.clearRetainingCapacity();
        try m.reset(zeros);
        var gpu_ns: u64 = 0;
        var t: u32 = 0;
        var lg: []f32 = logits;
        var batch_t0: u32 = 0;
        var batch_n: u32 = 0;
        var big_fed = false;
        for (toks, 0..) |tk, k| tok32[k] = @intCast(tk);
        while (t + 1 < n_ctx) : (t += 1) { // the last position of a chunk is never scored, so it is never fed
            if (prefill_rows > 0) {
                const t0 = nowNs();
                if (t == 0) { // the unscored prefix in windows
                    var k: u32 = 0;
                    while (k < first) {
                        const n = @min(prefill_rows, first - k);
                        try m.forwardRows(tok32[k .. k + n], 0, true);
                        k += n;
                    }
                    t = first;
                }
                if (t >= batch_t0 + batch_n or batch_n == 0) { // next batch of up to 16 scored rows
                    const nb = @min(16, n_ctx - 1 - t);
                    if (big) {
                        if (!big_fed) {
                            try m.forwardRows(tok32[first .. n_ctx - 1], qw.keep_hidden, true);
                            big_fed = true;
                        }
                        try m.headBatch(t - first, nb);
                    } else try m.forwardRows(tok32[t .. t + nb], nb, true);
                    try m.fetchRowLogits(batch);
                    batch_t0 = t;
                    batch_n = nb;
                }
                lg = batch[@as(usize, t - batch_t0) * n_vocab ..][0..n_vocab];
                gpu_ns += nowNs() - t0;
            } else {
                const need = t >= first;
                const t0 = nowNs();
                try m.forward(@intCast(toks[t]), need);
                if (need) try m.fetchLogits(logits) else try r.sync();
                gpu_ns += nowNs() - t0;
                if (!need) continue;
            }

            const row = t - first;
            const target: usize = @intCast(toks[t + 1]);
            try preadAll(rfd, std.mem.sliceAsBytes(ref_row), data_off + (@as(u64, c) * rows + row) * nv * 2);
            const scale: f32 = @bitCast(@as(u32, ref_row[0]) | @as(u32, ref_row[1]) << 16);
            const min_lp: f32 = @bitCast(@as(u32, ref_row[2]) | @as(u32, ref_row[3]) << 16);

            // ours: fp64 log-softmax and argmax (first maximum)
            var mx = lg[0];
            var imax: usize = 0;
            for (lg[1..], 1..) |v, k| if (v > mx) {
                mx = v;
                imax = k;
            };
            var se: f64 = 0;
            for (lg) |v| se += @exp(@as(f64, v - mx));
            const lse: f64 = @as(f64, mx) + @log(se);
            // base: log-probs dequantized like llama.cpp (f32 scale * q + min), argmax, KL over entries above -16
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
                if (lb > -16.0) kld += @exp(@as(f64, lb)) * (@as(f64, lb) - (@as(f64, lg[k]) - lse));
            }
            const lp_ours = @as(f64, lg[target]) - lse;
            const lp_ref: f64 = lpb[target];
            const same = imax == ibase;
            acc.add(kld, same, -lp_ours, -lp_ref);
            const s = try std.fmt.bufPrint(&line, "{d},{d},{e:.9},{d},{d:.6},{d:.6}\n", .{ c, t, kld, @intFromBool(same), lp_ours, lp_ref });
            try csv.appendSlice(gpa, s);
        }
        try appendFile(csv_path, csv.items);
        const ts = try std.fmt.bufPrint(&line, "{d},{d},{d:.4}\n", .{ c, n_ctx - 1, @as(f64, @floatFromInt(gpu_ns)) / 1e9 });
        try appendFile(time_path, ts);
        tot_steps += n_ctx - 1;
        tot_gpu_ns += gpu_ns;
        printRunning(c, acc);
    }
    const secs = @as(f64, @floatFromInt(tot_gpu_ns)) / 1e9;
    std.debug.print("done: {d} tokens in {d:.1} s of device time = {d:.2} tokens/s ({s}, logits downloaded at scored rows), device memory {d:.2} GB\n", .{
        tot_steps, secs, @as(f64, @floatFromInt(tot_steps)) / @max(secs, 1e-9), if (prefill_rows > 0) "prefill windows" else "decode path", @as(f64, @floatFromInt(l.total)) / 1e9,
    });
    return 0;
}
