//! exl3-kv full: all 48 layers of the EXL3 pack through the family's own pass code, one layer resident at a time
//! (layer streaming: every routed expert of that layer resident, then freed), and the logits of the last rows through
//! the mixer and the EXL3 lm_head, once per KV variant. The variants differ only in the attention caches: after each
//! attention block the rows it wrote are replaced by what an int8 / int4 cache would hold (fn_kv_roundtrip), so every
//! later row and layer reads the quantised keys and values through upstream's kernels.
//! Usage: tf-cuda-test exl3-kv full DIR PROBES CHUNK VARIANTS FILE[,FILE...]
//!   PROBES  the prompt's last tokens fed one row at a time (decode rows); their logits are compared
//!   CHUNK   prompt chunk rows (<= 256 keeps one attention block a chunk)
//!   VARIANTS comma list of bf16 | int8r | int8 | int4r | int4 (the first is the reference; bf16 recommended)
//! Layers 0 and 1 (GDN, the PLE) run once through Pass.chunk (they have no attention cache); 2..47 step block by block.
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("check.zig");
const fx = @import("flashnext_exl3_tests.zig");
const rig = @import("flashnext_rig.zig");

const Kv = flash.exl3.kv;
const X = flash.exl3;
const wt = X.prompt_wt;
const pl = X.prompt_load;
const batch = X.batch;
const bf = rig.bf;

const dims = wt.dims;
const wide = wt.wide;
const streams = wt.streams;
const vocab = wt.vocab;

const Side = struct { bits: u32 = 16, rot: bool = false };
/// kv8: the attention caches really are int8 (cuda_exl3_kv.zig attnBlock and its kernels), not an emulation.
const Variant = struct { name: []const u8, k: Side = .{}, v: Side = .{}, chunk: usize = 0, kv8: bool = false, rot: bool = false };

/// bf16 | int8 | int8r | int4 | int4r (K and V alike) | kbf16v8 | k8v4r (K and V apart), each with an optional @CHUNK
/// (the prompt fed in chunks of CHUNK rows instead of the default: with bf16 it is the run's numerical noise floor).
fn parseVariant(a: std.mem.Allocator, t: []const u8) !Variant {
    var v: Variant = .{ .name = try a.dupe(u8, t) };
    var base = t;
    if (std.mem.indexOfScalar(u8, t, '@')) |at| {
        v.chunk = try std.fmt.parseInt(usize, t[at + 1 ..], 10);
        base = t[0..at];
    }
    const both = struct {
        fn of(bits: u32, rot: bool) [2]Side {
            return .{ .{ .bits = bits, .rot = rot }, .{ .bits = bits, .rot = rot } };
        }
    };
    if (std.mem.eql(u8, base, "kv8") or std.mem.eql(u8, base, "kv8r")) {
        v.kv8 = true;
        v.rot = base.len == 4;
        return v;
    }
    const kv: [2]Side = if (std.mem.eql(u8, base, "bf16")) both.of(16, false) else if (std.mem.eql(u8, base, "int8")) both.of(8, false) else if (std.mem.eql(u8, base, "int8r")) both.of(8, true) else if (std.mem.eql(u8, base, "int4")) both.of(4, false) else if (std.mem.eql(u8, base, "int12")) both.of(12, false) else if (std.mem.eql(u8, base, "int10")) both.of(10, false) else if (std.mem.eql(u8, base, "int6")) both.of(6, false) else if (std.mem.eql(u8, base, "int4r")) both.of(4, true) else if (std.mem.eql(u8, base, "kbf16v8")) .{ .{}, .{ .bits = 8 } } else if (std.mem.eql(u8, base, "k8v4r")) .{ .{ .bits = 8 }, .{ .bits = 4, .rot = true } } else return error.UnknownVariant;
    v.k = kv[0];
    v.v = kv[1];
    return v;
}

/// One variant's sequence state on the device, laid out as Engine.newSeq makes it.
const VSeq = struct {
    conv: [wt.linear_n]cuda.DeviceBuffer = undefined,
    state: [wt.linear_n]cuda.DeviceBuffer = undefined,
    kc: [wt.attn_n]cuda.DeviceBuffer = undefined,
    vc: [wt.attn_n]cuda.DeviceBuffer = undefined,
    ikc: [wt.attn_n]cuda.DeviceBuffer = undefined,
    pooled: [wt.attn_n]cuda.DeviceBuffer = undefined,
    tail: cuda.DeviceBuffer = undefined,
    history: [2]i64 = .{ 0, 0 },
    cap: usize = 0,
    /// The int8 caches of a kv8 variant: codes and scales for K and V, each attention layer.
    c8: [wt.attn_n]Kv.Kv8.Cache = undefined,
    c8_bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,

    fn zeroed(d: *const cuda.Driver, s: cuda.Stream, n: usize) !cuda.DeviceBuffer {
        const b = try cuda.DeviceBuffer.alloc(d, n);
        try b.fill8(0, s.handle);
        return b;
    }

    fn init(v: *VSeq, gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, cap: usize, eos: i64, kv8: bool) !void {
        const gdn = flash.gdn;
        if (kv8) {
            const cb = Kv.Kv8.cacheBytes(cap, wt.kv_heads);
            for (&v.c8) |*c| {
                var parts: [4]u64 = undefined;
                for (&parts, 0..) |*pp, i| {
                    const b = try zeroed(d, s, if (i % 2 == 0) cb.codes else cb.scales);
                    try v.c8_bufs.append(gpa, b);
                    pp.* = b.ptr;
                }
                c.* = .{ .k8 = parts[0], .ks = parts[1], .v8 = parts[2], .vs = parts[3] };
            }
        }
        v.cap = cap;
        v.history = .{ eos, eos };
        for (&v.conv, &v.state) |*c, *x| {
            c.* = try zeroed(d, s, (gdn.taps - 1) * gdn.conv_dim * 2);
            x.* = try zeroed(d, s, wt.state_n * 4);
        }
        for (&v.kc, &v.vc, &v.ikc, &v.pooled) |*kc, *vc, *ikc, *pooled| {
            kc.* = try zeroed(d, s, cap * wt.kv_heads * wt.head_dim * 2);
            vc.* = try zeroed(d, s, cap * wt.kv_heads * wt.head_dim * 2);
            ikc.* = try zeroed(d, s, cap * wt.index_dim * 2);
            pooled.* = try zeroed(d, s, flash.prompt.pooledBytes(cap));
        }
        v.tail = try zeroed(d, s, wt.ple_tail_n * wide * 2);
        try s.synchronize();
    }

    fn deinit(v: *VSeq, gpa: std.mem.Allocator) void {
        for (v.c8_bufs.items) |*b| b.free();
        v.c8_bufs.deinit(gpa);
        for (&v.conv, &v.state) |*c, *x| {
            c.free();
            x.free();
        }
        for (&v.kc, &v.vc, &v.ikc, &v.pooled) |*a, *b, *c, *e| {
            a.free();
            b.free();
            c.free();
            e.free();
        }
        v.tail.free();
    }

    fn view(v: *VSeq, scores: u64) batch.SeqState {
        return .{ .conv = &v.conv, .state = &v.state, .kc = &v.kc, .vc = &v.vc, .ikc = &v.ikc, .pooled = &v.pooled, .tail = v.tail, .history = &v.history, .cap = v.cap, .scores = scores };
    }
};

/// One decoder layer (>= 2: no PLE) for `n` rows at `pos0` from the streams in `h` (host, bf16), in place.
fn stepLayer(gpu: check.Gpu, p: *fx.Pass, k8: *const Kv.Kv8, v: Variant, vq: *const VSeq, lw: *const pl.Layer, l: usize, ids: []const u32, pos0: usize, h: []u16) !void {
    const s = p.s;
    const n = ids.len;
    p.rows = n;
    p.pos0 = pos0;
    p.pending = null;
    try p.stage(ids);
    try fx.uploadTo(gpu, p.stream, s.h, std.mem.sliceAsBytes(h));
    try p.writeback(0, n);
    try p.readout(lw.attn_hc, 0, n);
    if (lw.linear) {
        try p.gdnBlock(lw, wt.linearIndex(l), .{});
    } else {
        const ai = wt.attnIndex(l);
        if (v.kv8) return stepRest(gpu, p, lw, l, n, h, Kv.attnBlock(p, k8, lw, ai, vq.c8[ai], v.rot));
        try p.attnBlock(lw, ai);
        const row = wt.kv_heads * wt.head_dim * 2;
        if (v.k.bits != 16) try k8.roundTrip(p.stream, p.seq.kc[ai].ptr + pos0 * row, n * wt.kv_heads, v.k.rot, v.k.bits);
        if (v.v.bits != 16) try k8.roundTrip(p.stream, p.seq.vc[ai].ptr + pos0 * row, n * wt.kv_heads, v.v.rot, v.v.bits);
    }
    try p.tri.hcWriteback(s.h, s.h, s.pss, s.part, "*fp32", s.inj, s.dummy, "*bf16", s.dummy, "*bf16", n * dims, n, dims, streams, 4, 1, 1, wt.out_sk);
    try p.readout(lw.mlp_hc, 0, n);
    try p.moeBlock(lw);
    p.pending = l;
    try p.writeback(0, n);
    const out = try fx.download(u16, gpu, p.stream, s.h, n * wide);
    defer gpu.gpa.free(out);
    @memcpy(h, out);
}

/// The rest of stepLayer after an attention block that returned `mixer` (the int8 block).
fn stepRest(gpu: check.Gpu, p: *fx.Pass, lw: *const pl.Layer, l: usize, n: usize, h: []u16, mixer: anyerror!void) !void {
    try mixer;
    const s = p.s;
    try p.tri.hcWriteback(s.h, s.h, s.pss, s.part, "*fp32", s.inj, s.dummy, "*bf16", s.dummy, "*bf16", n * dims, n, dims, streams, 4, 1, 1, wt.out_sk);
    try p.readout(lw.mlp_hc, 0, n);
    try p.moeBlock(lw);
    p.pending = l;
    try p.writeback(0, n);
    const out = try fx.download(u16, gpu, p.stream, s.h, n * wide);
    defer gpu.gpa.free(out);
    @memcpy(h, out);
}

/// The chunks a prompt of `total` tokens is fed in: prefill chunks of `chunk` rows, then `probes` single rows.
fn chunksOf(a: std.mem.Allocator, total: usize, probes: usize, chunk: usize) ![][2]usize {
    var list: std.ArrayList([2]usize) = .empty;
    var at: usize = 0;
    while (at < total - probes) {
        const n = @min(chunk, total - probes - at);
        try list.append(a, .{ at, n });
        at += n;
    }
    while (at < total) : (at += 1) try list.append(a, .{ at, 1 });
    return list.toOwnedSlice(a);
}

fn logSoftmax(row: []const u16, out: []f64) void {
    var best = -std.math.inf(f64);
    for (row) |x| best = @max(best, bf(x));
    var sum: f64 = 0;
    for (row) |x| sum += @exp(bf(x) - best);
    const lse = best + @log(sum);
    for (row, out) |x, *o| o.* = bf(x) - lse;
}

fn top5(row: []const u16) [5]u32 {
    var best: [5]u32 = @splat(0);
    var val: [5]f64 = @splat(-std.math.inf(f64));
    for (row, 0..) |x, i| {
        const v = bf(x);
        if (v <= val[4]) continue;
        var j: usize = 4;
        while (j > 0 and v > val[j - 1]) : (j -= 1) {
            val[j] = val[j - 1];
            best[j] = best[j - 1];
        }
        val[j] = v;
        best[j] = @intCast(i);
    }
    return best;
}

pub fn run(gpu: check.Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 5) return error.MissingArgument;
    const dir = rest[0];
    const probes = try std.fmt.parseInt(usize, rest[1], 10);
    const chunk = try std.fmt.parseInt(usize, rest[2], 10);
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var variants: std.ArrayList(Variant) = .empty;
    {
        var it = std.mem.splitScalar(u8, rest[3], ',');
        while (it.next()) |t| try variants.append(a, try parseVariant(a, t));
    }
    var ids: std.ArrayList(u32) = .empty;
    {
        var it = std.mem.splitScalar(u8, rest[4], ',');
        while (it.next()) |path| {
            const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, a, .limited(1 << 26));
            var toks = std.mem.tokenizeAny(u8, text, ", \n\r\t");
            while (toks.next()) |t| try ids.append(a, try std.fmt.parseInt(u32, t, 10));
        }
    }
    const total = ids.items.len;
    if (probes == 0 or probes >= total) return error.TooManyProbes;
    const cap = total + 16;
    const nv = variants.items.len;
    // logits from row STATS_FROM (6th argument; default: every row of a short prompt, else the decode rows) to the end
    const stats_from: usize = if (rest.len > 5) try std.fmt.parseInt(usize, rest[5], 10) else if (total <= 64) 0 else total - probes;
    const block: usize = 64;
    std.debug.print("== exl3-kv full: {d} tokens ({d} decode rows), chunk {d}, 48 layers streamed, variants:", .{ total, probes, chunk });
    for (variants.items) |v| std.debug.print(" {s};", .{v.name});
    std.debug.print("\n", .{});
    const t_all = check.now(gpu.io);

    var r: fx.Rig = undefined;
    try r.init(gpu, dir, @max(chunk, block), 16);
    defer r.deinit();
    const d: *cuda.Driver = @constCast(gpu.d);
    var k8 = try Kv.Kv8.load(gpu.d);
    defer k8.deinit();
    const eos = try flash.ple.eosOf(gpu.gpa, gpu.io, dir);
    const vs = try a.alloc(VSeq, nv);
    for (vs) |*v| v.* = .{};
    var made: usize = 0;
    defer for (vs[0..made]) |*v| v.deinit(gpu.gpa);
    for (vs, variants.items) |*v, vr| {
        try v.init(gpu.gpa, gpu.d, r.stream, cap, eos, vr.kv8);
        made += 1;
    }
    var max_chunk = chunk;
    for (variants.items) |v| max_chunk = @max(max_chunk, v.chunk);
    if (max_chunk > chunk) return error.VariantChunkAboveScratch; // a variant's chunk may only be smaller
    var scores = try cuda.DeviceBuffer.alloc(gpu.d, batch.scoreBytes(cap, chunk));
    defer scores.free();
    const plan = try chunksOf(a, total, probes, chunk);
    const plans = try a.alloc([][2]usize, nv);
    for (variants.items, plans) |v, *pv| pv.* = if (v.chunk == 0) plan else try chunksOf(a, total, probes, v.chunk);
    const hs = try a.alloc([]u16, nv);
    for (hs) |*h| h.* = try a.alloc(u16, total * wide);

    // layers 0 and 1 (embedding, GDN, the PLE) through Pass.chunk, the streams after layer 1 kept: the same for every variant
    {
        var table = try flash.embed.Table.uploadDense(gpu.d, try r.store.tensor(flash.embed.weight_name));
        defer table.deinit();
        var ngram = try flash.ple.Ngram.init(gpu.gpa, gpu.io, dir);
        defer ngram.deinit();
        var first: [2]pl.Layer = undefined;
        first[0] = try pl.loadLayerRank(gpu.gpa, d, &r.stream, &r.store, r.pack_fn, 0, 1, 0, null);
        defer first[0].free(gpu.gpa);
        first[1] = try pl.loadLayerRank(gpu.gpa, d, &r.stream, &r.store, r.pack_fn, 1, 1, 0, null);
        defer first[1].free(gpu.gpa);
        r.store.release();
        var hbuf = try cuda.DeviceBuffer.alloc(gpu.d, chunk * wide * 2);
        defer hbuf.free();
        var p = r.pass(&first);
        p.seq = vs[0].view(scores.ptr);
        p.m.table_w = table.w.ptr;
        p.m.table_dense = table.dense;
        p.m.ngram = &ngram;
        p.hooks = .{ .logits = .none, .streams = hbuf.ptr };
        for (plan) |c| {
            try p.chunk(ids.items[c[0]..][0..c[1]], c[0], null);
            const got = try fx.download(u16, gpu, r.stream, hbuf.ptr, c[1] * wide);
            defer gpu.gpa.free(got);
            @memcpy(hs[0][c[0] * wide ..][0 .. c[1] * wide], got);
        }
        for (hs[1..]) |h| @memcpy(h, hs[0]);
        std.debug.print("  layers 0-1 (embed, GDN, PLE) for {d} rows in {d:.1} s\n", .{ total, @as(f64, @floatFromInt(check.now(gpu.io) - t_all)) / 1e9 });
    }

    // layers 2..47, one resident at a time, every variant through it
    var p = r.pass(&.{});
    p.hooks = .{ .logits = .none };
    var load_ns: i96 = 0;
    var run_ns: i96 = 0;
    for (2..wt.layers_n) |l| {
        var t0 = check.now(gpu.io);
        var lw = try pl.loadLayerRank(gpu.gpa, d, &r.stream, &r.store, r.pack_fn, l, 1, 0, null);
        defer lw.free(gpu.gpa);
        r.store.release();
        load_ns += check.now(gpu.io) - t0;
        t0 = check.now(gpu.io);
        for (variants.items, vs, hs, plans) |v, *vq, h, pv| {
            p.seq = vq.view(scores.ptr);
            for (pv) |c| try stepLayer(gpu, &p, &k8, v, vq, &lw, l, ids.items[c[0]..][0..c[1]], c[0], h[c[0] * wide ..][0 .. c[1] * wide]);
        }
        run_ns += check.now(gpu.io) - t0;
        if (l % 8 == 7) std.debug.print("  layer {d}/47: loads {d:.1} s, passes {d:.1} s so far\n", .{ l, @as(f64, @floatFromInt(load_ns)) / 1e9, @as(f64, @floatFromInt(run_ns)) / 1e9 });
    }

    // the head: mixer read-out and the EXL3 lm_head over the kept rows, a variant at a time
    var mixer = try wt.loadHc(gpu.gpa, d, &r.store, "language_model.model.hyper_connection_mixer", false);
    defer mixer.free();
    var head = try X.projection_load.load(a, d, &r.store, "language_model.lm_head", vocab, dims);
    defer head.free();
    r.store.release();
    var logits_d = try cuda.DeviceBuffer.alloc(gpu.d, block * vocab * 2);
    defer logits_d.free();
    const lg = try a.alloc([]u16, nv);
    for (lg) |*x| x.* = try a.alloc(u16, block * vocab);
    const Acc = struct { rows: usize = 0, correct: usize = 0, agree: usize = 0, agree5: usize = 0, kl_sum: f64 = 0, kl_max: f64 = 0, flips: [4]usize = @splat(0), max_d: f64 = 0 };
    const acc = try a.alloc(Acc, nv);
    for (acc) |*x| x.* = .{};
    const lp_ref = try a.alloc(f64, vocab);
    const lp = try a.alloc(f64, vocab);
    const margins = [_]f64{ 0.5, 1, 2, 4 };
    var b0 = stats_from;
    while (b0 < total) : (b0 += block) {
        const n = @min(block, total - b0);
        for (hs, 0..) |h, vi| {
            p.rows = n;
            p.pending = null;
            try fx.uploadTo(gpu, p.stream, p.s.h, std.mem.sliceAsBytes(h[b0 * wide ..][0 .. n * wide]));
            try p.writeback(0, n);
            try p.readout(mixer, 0, n);
            try p.lane(p.s.mixed, p.s.xsm, head, logits_d.ptr, p.s.dummy, n, 1);
            const got = try fx.download(u16, gpu, r.stream, logits_d.ptr, n * vocab);
            defer gpu.gpa.free(got);
            @memcpy(lg[vi][0 .. n * vocab], got);
        }
        for (0..n) |i| {
            const pos = b0 + i;
            const next: ?u32 = if (pos + 1 < total) ids.items[pos + 1] else null;
            const ra = lg[0][i * vocab ..][0..vocab];
            const ta = top5(ra);
            if (total <= 64) std.debug.print("  pos {d} -> {?d} | {any}\n", .{ pos, next, ta });
            logSoftmax(ra, lp_ref);
            for (acc, lg, variants.items) |*ac, l, v| {
                const rb = l[i * vocab ..][0..vocab];
                const tb = top5(rb);
                ac.rows += 1;
                if (next) |nx| ac.correct += @intFromBool(tb[0] == nx);
                if (ta[0] == tb[0]) ac.agree += 1 else {
                    const m = bf(ra[ta[0]]) - bf(ra[tb[0]]);
                    for (margins, 0..) |mm, mi| ac.flips[mi] += @intFromBool(m > mm);
                    if (m > 1) std.debug.print("  FLIP {s} pos {d} (next {?d}): ref top-3 {any} {d:.3} {d:.3} {d:.3}; var top-3 {any} {d:.3} {d:.3} {d:.3}\n", .{ v.name, pos, next, ta[0..3], bf(ra[ta[0]]), bf(ra[ta[1]]), bf(ra[ta[2]]), tb[0..3], bf(rb[tb[0]]), bf(rb[tb[1]]), bf(rb[tb[2]]) });
                }
                for (tb) |x| ac.agree5 += @intFromBool(x == ta[0]);
                for (ra, rb) |x, y| ac.max_d = @max(ac.max_d, @abs(bf(x) - bf(y)));
                logSoftmax(rb, lp);
                var kl: f64 = 0;
                for (lp_ref, lp) |x, y| kl += @exp(x) * (x - y);
                ac.kl_sum += kl;
                ac.kl_max = @max(ac.kl_max, kl);
            }
        }
    }
    // the prompt's last rows (a needle prompt's answer): every variant's top-1 against the next token
    {
        const tail = @min(6, total - stats_from);
        const t0r = total - tail;
        // recompute the logits of the tail block (the loop above kept only the last block)
        for (hs, 0..) |h, vi| {
            p.rows = tail;
            p.pending = null;
            try fx.uploadTo(gpu, p.stream, p.s.h, std.mem.sliceAsBytes(h[t0r * wide ..][0 .. tail * wide]));
            try p.writeback(0, tail);
            try p.readout(mixer, 0, tail);
            try p.lane(p.s.mixed, p.s.xsm, head, logits_d.ptr, p.s.dummy, tail, 1);
            const got = try fx.download(u16, gpu, r.stream, logits_d.ptr, tail * vocab);
            defer gpu.gpa.free(got);
            @memcpy(lg[vi][0 .. tail * vocab], got);
        }
        for (0..tail) |i| {
            const pos = t0r + i;
            std.debug.print("  tail pos {d} next {?d}:", .{ pos, if (pos + 1 < total) @as(?u32, ids.items[pos + 1]) else null });
            for (variants.items, lg) |v, l| std.debug.print(" {s}={d}", .{ v.name, top5(l[i * vocab ..][0..vocab])[0] });
            std.debug.print("\n", .{});
        }
    }
    const rows_n = total - stats_from;
    for (acc, variants.items) |ac, v| {
        std.debug.print("RESULT {s:<12} rows {d}..{d}: teacher-forced top-1 = next token {d}/{d}; vs {s}: top-1 equal {d}/{d}, its top-1 in our top-5 {d}/{d}, flips with reference margin > 0.5/1/2/4: {d}/{d}/{d}/{d}; KL mean {e:.3} max {e:.3}; max |logit diff| {d:.3}\n", .{ v.name, stats_from, total - 1, ac.correct, rows_n - 1, variants.items[0].name, ac.agree, rows_n, ac.agree5, rows_n, ac.flips[0], ac.flips[1], ac.flips[2], ac.flips[3], ac.kl_sum / @as(f64, @floatFromInt(rows_n)), ac.kl_max, ac.max_d });
    }
    std.debug.print("  time: layer loads {d:.1} s, layer passes {d:.1} s, total {d:.1} s\n", .{ @as(f64, @floatFromInt(load_ns)) / 1e9, @as(f64, @floatFromInt(run_ns)) / 1e9, @as(f64, @floatFromInt(check.now(gpu.io) - t_all)) / 1e9 });
    check.pass("exl3-kv full: 48 layers streamed over {d} tokens, {d} variants; numbers above", .{ total, nv });
}
