//! KL divergence of Nemotron-H vs a llama.cpp base-logits file, scored like llama-perplexity; csv per chunk.

const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");
const ld = @import("xpu").loader;
const cfgm = @import("nemotron_xpu").cfg;
const model = @import("nemotron_xpu").model;
const nw = @import("nemotron_xpu").win;
const core = @import("kl_core.zig");
const stop = @import("xpu").stop;

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main(init: std.process.Init) !u8 {
    stop.install();
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 4) {
        std.debug.print("usage: xpu-nem_kl-test CHECKPOINT_DIR REF.logits TAG [--chunks A:B,N,...] [--rows N] [--prefill R] [--bos ID] [--legacy] [--check] [--bf16-logits] [--out DIR]\n", .{});
        return 2;
    }
    const dir = args[1];
    const tag = args[3];
    var chunk_spec: []const u8 = "0:1";
    var out_dir: []const u8 = "out";
    var bf16_logits = false;
    var score_rows: u32 = 1;
    var pre_rows: u32 = 16;
    var bos: ?u32 = null;
    var legacy = false;
    var check = false;
    var i: usize = 4;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--chunks")) {
            i += 1;
            chunk_spec = args[i];
        } else if (std.mem.eql(u8, args[i], "--out")) {
            i += 1;
            out_dir = args[i];
        } else if (std.mem.eql(u8, args[i], "--rows")) {
            i += 1;
            score_rows = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--prefill")) {
            i += 1;
            pre_rows = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--bos")) {
            i += 1;
            bos = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--legacy")) {
            legacy = true;
        } else if (std.mem.eql(u8, args[i], "--check")) {
            check = true;
        } else if (std.mem.eql(u8, args[i], "--bf16-logits")) bf16_logits = true else return error.BadArgument;
    }
    if (score_rows == 0 or score_rows > 16 or pre_rows == 0) return error.BadArgument;

    const ref = try core.Ref.load(gpa, try std.fmt.allocPrintSentinel(gpa, "{s}", .{args[2]}, 0));
    std.debug.print("ref: n_ctx {d} n_vocab {d} n_chunk {d}, {d} scored rows per chunk\n", .{ ref.n_ctx, ref.n_vocab, ref.n_chunk, ref.rows });
    const chunks = try core.parseChunks(gpa, chunk_spec, ref.n_chunk);
    try core.makeDir(gpa, out_dir);
    const csv_path = try std.fmt.allocPrintSentinel(gpa, "{s}/kl_{s}.csv", .{ out_dir, tag }, 0);
    const time_path = try std.fmt.allocPrintSentinel(gpa, "{s}/kl_{s}.time", .{ out_dir, tag }, 0);
    if (check) return selfCheck(gpa, ref, chunks);
    const done_rows = try core.resumeCounts(gpa, csv_path, ref.n_chunk);

    const cfg = try cfgm.parse(gpa, try ld.readFile(gpa, dir, "config.json"));
    if (cfg.value.vocab_size != ref.n_vocab) return error.VocabMismatch;
    var r = try rt.open();
    defer r.deinit();
    var l = try ld.Loader.init(gpa, &r, dir);
    if (pre_rows > 16) nw.default_rows = pre_rows;
    var m = try model.Model.load(gpa, &r, &l, cfg.value, ref.n_ctx);
    m.bf16_logits = bf16_logits;
    var w = try nw.Win.init(gpa, &m);

    const vocab: usize = ref.n_vocab;
    const logits = try gpa.alloc(f32, vocab * score_rows);
    const ref_row = try gpa.alloc(u16, ref.nv);
    const lpb = try gpa.alloc(f32, vocab);
    const toks = try gpa.alloc(u32, ref.n_ctx);
    var csv: std.ArrayList(u8) = .empty;
    var acc: core.Acc = .{};
    var tot_steps: u64 = 0;
    var tot_gpu_ns: u64 = 0;
    var line: [160]u8 = undefined;

    for (chunks) |c| {
        if (done_rows[c] >= ref.rows) continue;
        for (ref.chunkTokens(c), toks) |s, *d| d.* = @intCast(s);
        if (bos) |b| toks[0] = b;
        csv.clearRetainingCapacity();
        try w.reset(&m);
        const t0 = core.nowNs();
        var t: u32 = 0;
        if (legacy) {
            while (t + 1 < ref.n_ctx) : (t += 1) {
                try nw.stopCheck(&r);
                const need = t >= ref.first;
                try m.forward(toks[t], need);
                if (!need) {
                    try r.sync(); // Model.forward uploads from one reused host token slot: the copy must finish before the next token overwrites it
                    continue;
                }
                try m.fetchLogits(logits[0..vocab]);
                const x = core.scoreRow(logits[0..vocab], try readRef(ref, c, t, ref_row), lpb, @intCast(ref.tokens[@as(usize, c) * ref.n_ctx + t + 1]));
                acc.add(x);
                try csv.appendSlice(gpa, try core.csvLine(&line, c, t, x));
            }
        }
        while (!legacy and t < ref.first) { // positions before the scored half: states only
            try nw.stopCheck(&r);
            const n = @min(pre_rows, ref.first - t);
            try w.forward(&m, toks[t .. t + n], true, 0);
            t += n;
        }
        while (!legacy and t + 1 < ref.n_ctx) { // the last position of a chunk is never scored, so it is never fed
            try nw.stopCheck(&r);
            const n = @min(score_rows, ref.n_ctx - 1 - t);
            try w.forward(&m, toks[t .. t + n], true, n);
            try w.fetchRowLogits(&m, logits);
            for (0..n) |k| {
                const pos = t + @as(u32, @intCast(k));
                try ref.readRow(c, pos - ref.first, ref_row);
                const x = core.scoreRow(logits[k * vocab ..][0..vocab], ref_row, lpb, @intCast(ref.tokens[@as(usize, c) * ref.n_ctx + pos + 1]));
                acc.add(x);
                try csv.appendSlice(gpa, try core.csvLine(&line, c, pos, x));
            }
            t += n;
        }
        try r.sync();
        const gpu_ns = core.nowNs() - t0;
        try core.appendFile(csv_path, csv.items);
        const ts = try std.fmt.bufPrint(&line, "{d},{d},{d:.4}\n", .{ c, ref.n_ctx - 1, @as(f64, @floatFromInt(gpu_ns)) / 1e9 });
        try core.appendFile(time_path, ts);
        tot_steps += ref.n_ctx - 1;
        tot_gpu_ns += gpu_ns;
        core.printRunning(c, acc);
    }
    const secs = @as(f64, @floatFromInt(tot_gpu_ns)) / 1e9;
    std.debug.print("done: {d} tokens in {d:.1} s of device time = {d:.2} tokens/s (windows of {d} scored rows), device memory {d:.2} GB\n", .{
        tot_steps, secs, @as(f64, @floatFromInt(tot_steps)) / @max(secs, 1e-9), score_rows, @as(f64, @floatFromInt(l.total)) / 1e9,
    });
    return 0;
}

fn readRef(ref: core.Ref, c: u32, pos: u32, buf: []u16) ![]const u16 {
    try ref.readRow(c, pos - ref.first, buf);
    return buf;
}

/// CPU-only check: each stored row dequantized and scored against itself gives KLD ~ 0, same top and target log-probs.
fn selfCheck(gpa: std.mem.Allocator, ref: core.Ref, chunks: []const u32) !u8 {
    const vocab: usize = ref.n_vocab;
    const row = try gpa.alloc(u16, ref.nv);
    const lp = try gpa.alloc(f32, vocab);
    const scratch = try gpa.alloc(f32, vocab);
    var worst_kld: f64 = 0;
    var worst_lp: f64 = 0;
    var bad_top: u32 = 0;
    var n: u32 = 0;
    for (chunks[0..@min(chunks.len, 3)]) |c| {
        var k: u32 = 0;
        while (k < ref.rows) : (k += 17) {
            try ref.readRow(c, k, row);
            core.dequantRow(row, lp);
            const target: usize = @intCast(ref.tokens[@as(usize, c) * ref.n_ctx + ref.first + k + 1]);
            const x = core.scoreRow(lp, row, scratch, target);
            worst_kld = @max(worst_kld, @abs(x.kld));
            worst_lp = @max(worst_lp, @abs(x.lp_ours - x.lp_ref));
            bad_top += @intFromBool(!x.same);
            n += 1;
        }
    }
    std.debug.print("self-check: {d} rows, max |kld| {e:.2}, max |lp_ours - lp_ref| {d:.4} (lse of the clamped tail), rows with different top-1: {d}\n", .{ n, worst_kld, worst_lp, bad_top });
    return 0;
}
