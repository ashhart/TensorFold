//! exl3-kv real: the first LAYERS decoder layers of the EXL3 pack through the family's own pass (Pass.chunk: embed).
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("../../check.zig");
const fx = @import("../model/flashnext_exl3_tests.zig");
const kt = @import("exl3_kv_tests.zig");
const rig = @import("../../flashnext_rig.zig");

const Kv = flash.exl3.kv;
const X = flash.exl3;
const wt = X.prompt_wt;
const pl = X.prompt_load;
const bf = rig.bf;
const bf16 = rig.bf16;

const q_heads = 24;
const kv_heads = 2;
const head_dim = 256;
const idw = wt.index_idw;
const top = wt.index_top;
const ratio = wt.index_ratio;
const nch = wt.nch;
const group = 32;
const ng = head_dim / group;

fn readIds(gpa: std.mem.Allocator, io: std.Io, paths: []const u8, out: *std.ArrayList(u32)) !void {
    var it = std.mem.splitScalar(u8, paths, ',');
    while (it.next()) |p| {
        const text = try std.Io.Dir.cwd().readFileAlloc(io, p, gpa, .limited(1 << 26));
        defer gpa.free(text);
        var toks = std.mem.tokenizeAny(u8, text, ", \n\r\t");
        while (toks.next()) |t| try out.append(gpa, try std.fmt.parseInt(u32, t, 10));
    }
}

/// A row's selected block set (sparse rows) as sorted block ids; dense rows: null.
fn blocksOf(ids: []const i32, nk: i32, spr: i32, out: []u32) ?[]u32 {
    if (spr == 0) return null;
    _ = nk;
    for (0..top) |i| out[i] = @intCast(@divTrunc(ids[i * ratio], @as(i32, ratio)));
    return out[0..top];
}

fn overlap(a: []const u32, b: []const u32) usize {
    var i: usize = 0;
    var j: usize = 0;
    var n: usize = 0;
    while (i < a.len and j < b.len) {
        if (a[i] == b[j]) {
            n += 1;
            i += 1;
            j += 1;
        } else if (a[i] < b[j]) i += 1 else j += 1;
    }
    return n;
}

const Stats = struct {
    rows: usize = 0,
    sum1: f64 = 0,
    min1: f64 = 1,
    sum4: f64 = 0,
    rows4: usize = 0,
    sum16: f64 = 0,
    rows16: usize = 0,
    hist: [11]usize = @splat(0),
    blocks: usize = 0,
    distinct: usize = 0,
    max_distinct: usize = 0,
};

fn has(set: []const u32, x: u32) bool {
    var lo: usize = 0;
    var hi: usize = set.len;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        if (set[mid] == x) return true;
        if (set[mid] < x) lo = mid + 1 else hi = mid;
    }
    return false;
}

/// Each sparse row against the previous row's set, and against the union of the last 4 and 16 rows' sets.
const Window = struct {
    sets: std.ArrayList([]u32) = .empty,

    fn add(w: *Window, a: std.mem.Allocator, set: []const u32, st: *Stats) !void {
        const n = w.sets.items.len;
        if (n > 0) {
            const o1 = @as(f64, @floatFromInt(overlap(set, w.sets.items[n - 1]))) / top;
            st.sum1 += o1;
            st.min1 = @min(st.min1, o1);
            st.hist[@min(10, @as(usize, @intFromFloat(o1 * 10)))] += 1;
            for ([_]usize{ 4, 16 }) |k| {
                if (n < k) continue;
                var hit: usize = 0;
                for (set) |blk| {
                    for (w.sets.items[n - k ..]) |prev| {
                        if (has(prev, blk)) {
                            hit += 1;
                            break;
                        }
                    }
                }
                const f = @as(f64, @floatFromInt(hit)) / top;
                if (k == 4) {
                    st.sum4 += f;
                    st.rows4 += 1;
                } else {
                    st.sum16 += f;
                    st.rows16 += 1;
                }
            }
            st.rows += 1;
        }
        try w.sets.append(a, try a.dupe(u32, set));
    }
};

const Probe = struct { pos: usize, q: []u16, ao: []u16, ids: []i32, nk: i32, spr: i32 };

fn errRow(got: []const f64, want: []const f64, e: *kt.Err) void {
    for (got, want) |x, w| {
        e.max_abs = @max(e.max_abs, @abs(x - w));
        e.peak = @max(e.peak, @abs(w));
        e.ss += (x - w) * (x - w);
        e.ss_ref += w * w;
    }
}

pub fn run(gpu: check.Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 5) return error.MissingArgument;
    const dir = rest[0];
    const n_layers = try std.fmt.parseInt(usize, rest[1], 10);
    const probes = try std.fmt.parseInt(usize, rest[2], 10);
    var chunk_list: std.ArrayList(usize) = .empty;
    defer chunk_list.deinit(gpu.gpa);
    {
        var it = std.mem.splitScalar(u8, rest[3], ',');
        while (it.next()) |t| try chunk_list.append(gpu.gpa, try std.fmt.parseInt(usize, t, 10));
    }
    if (n_layers == 0 or n_layers > wt.layers_n or wt.linearLayer(n_layers - 1)) return error.LastLayerNotAttention;
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var ids: std.ArrayList(u32) = .empty;
    try readIds(a, gpu.io, rest[4], &ids);
    const total = ids.items.len;
    if (probes >= total) return error.TooManyProbes;
    const n_prefill = total - probes;
    var max_chunk: usize = 0;
    for (chunk_list.items) |c| max_chunk = @max(max_chunk, c);
    const cap = total + 16;
    std.debug.print("== exl3-kv real: {d} prompt tokens ({d} prefilled, {d} probe rows), layers 0..{d}, chunks {any}\n", .{ total, n_prefill, probes, n_layers - 1, chunk_list.items });

    var r: fx.Rig = undefined;
    try r.init(gpu, dir, max_chunk, cap);
    defer r.deinit();
    const d: *cuda.Driver = @constCast(gpu.d);
    var t0 = check.now(gpu.io);
    var table = try flash.embed.Table.uploadDense(gpu.d, try r.store.tensor(flash.embed.weight_name));
    defer table.deinit();
    var ngram = try flash.ple.Ngram.init(gpu.gpa, gpu.io, dir);
    defer ngram.deinit();
    const eos = try flash.ple.eosOf(gpu.gpa, gpu.io, dir);
    const layers = try a.alloc(pl.Layer, n_layers);
    var loaded: usize = 0;
    defer for (layers[0..loaded]) |*lw| lw.free(gpu.gpa);
    for (0..n_layers) |l| {
        layers[l] = try pl.loadLayerRank(gpu.gpa, d, &r.stream, &r.store, r.pack_fn, l, 1, 0, null);
        loaded += 1;
        if (l % 16 == 15) r.store.release();
    }
    r.store.release();
    var free_b: usize = 0;
    var total_b: usize = 0;
    try gpu.d.check(gpu.d.api.cuMemGetInfo_v2(&free_b, &total_b), "cuMemGetInfo");
    std.debug.print("  loaded table, n-gram reader and {d} layers (every routed expert resident) in {d:.1} s; device free {d:.2} of {d:.2} GiB\n", .{ n_layers, @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e9, @as(f64, @floatFromInt(free_b)) / Kv.gib, @as(f64, @floatFromInt(total_b)) / Kv.gib });

    var p = r.pass(layers);
    p.m.table_w = table.w.ptr;
    p.m.table_dense = table.dense;
    p.m.ngram = &ngram;
    p.hooks.logits = .none;
    const ai = wt.attnIndex(n_layers - 1);
    const s = p.s;

    // -- prompt passes: the first chunk size drives the analysis.
    var st_prefill: Stats = .{};
    var win: Window = .{};
    var blk_buf: [top]u32 = undefined;
    var times: std.ArrayList(f64) = .empty;
    for (chunk_list.items, 0..) |chunk, ci| {
        try r.reset();
        r.history = .{ eos, eos };
        try r.stream.synchronize();
        var at: usize = 0;
        t0 = check.now(gpu.io);
        var analysis_ns: i96 = 0;
        while (at < n_prefill) {
            const n = @min(chunk, n_prefill - at);
            try p.chunk(ids.items[at..][0..n], at, null);
            try r.stream.synchronize();
            if (ci == 0 and n <= X.batch.att_rows) {
                // n <= att_rows: the selection buffers hold every row of this chunk (the last attention layer's)
                const ta = check.now(gpu.io);
                const qids = try fx.download(i32, gpu, r.stream, s.qids, n * idw);
                defer gpu.gpa.free(qids);
                const qnk = try fx.download(i32, gpu, r.stream, s.qnk, n);
                defer gpu.gpa.free(qnk);
                const qspr = try fx.download(i32, gpu, r.stream, s.qspr, n);
                defer gpu.gpa.free(qspr);
                var seen = try std.DynamicBitSetUnmanaged.initEmpty(a, cap);
                defer seen.deinit(a);
                var any_sparse = false;
                for (0..n) |row| {
                    const nkr: usize = @intCast(qnk[row]);
                    if (qspr[row] != 0) {
                        any_sparse = true;
                        for (qids[row * idw ..][0..nkr]) |k| seen.set(@intCast(k));
                        if (blocksOf(qids[row * idw ..][0..idw], qnk[row], qspr[row], &blk_buf)) |set| try win.add(a, set, &st_prefill);
                    } else for (0..nkr) |k| seen.set(k);
                }
                if (any_sparse) {
                    st_prefill.blocks += 1;
                    st_prefill.distinct += seen.count();
                    st_prefill.max_distinct = @max(st_prefill.max_distinct, seen.count());
                }
                analysis_ns += check.now(gpu.io) - ta;
            }
            at += n;
        }
        const ms = @as(f64, @floatFromInt(check.now(gpu.io) - t0 - analysis_ns)) / 1e6;
        try times.append(a, ms);
        std.debug.print("RESULT prompt {d} tokens in chunks of {d} through layers 0..{d}: {d:.0} ms ({d:.3} ms a token)\n", .{ n_prefill, chunk, n_layers - 1, ms, ms / @as(f64, @floatFromInt(n_prefill)) });
    }
    if (st_prefill.rows > 0) {
        const rows_f: f64 = @floatFromInt(st_prefill.rows);
        std.debug.print("RESULT layer {d} selection, {d} consecutive sparse prompt rows: overlap with the previous row mean {d:.3} min {d:.3}; covered by the last 4 rows' union {d:.3}, last 16 rows' {d:.3}\n", .{ n_layers - 1, st_prefill.rows, st_prefill.sum1 / rows_f, st_prefill.min1, st_prefill.sum4 / @as(f64, @floatFromInt(@max(1, st_prefill.rows4))), st_prefill.sum16 / @as(f64, @floatFromInt(@max(1, st_prefill.rows16))) });
        std.debug.print("  overlap histogram (tenths): {any}\n", .{st_prefill.hist});
        std.debug.print("RESULT layer {d}: a sparse 256-row prompt block reads {d:.0} distinct keys on average, {d} at most ({d} blocks)\n", .{ n_layers - 1, @as(f64, @floatFromInt(st_prefill.distinct)) / @as(f64, @floatFromInt(st_prefill.blocks)), st_prefill.max_distinct, st_prefill.blocks });
    }

    // -- probe rows: one token a pass (a decode row each), the pass's own q, selection.
    const probe = try a.alloc(Probe, probes);
    var st_decode: Stats = .{};
    for (0..probes) |i| {
        const pos = n_prefill + i;
        try p.chunk(ids.items[pos..][0..1], pos, null);
        try r.stream.synchronize();
        const q = try fx.download(u16, gpu, r.stream, s.q, q_heads * head_dim);
        const ao = try fx.download(u16, gpu, r.stream, s.ao, q_heads * head_dim);
        const qids = try fx.download(i32, gpu, r.stream, s.qids, idw);
        const qnk = try fx.download(i32, gpu, r.stream, s.qnk, 1);
        const qspr = try fx.download(i32, gpu, r.stream, s.qspr, 1);
        probe[i] = .{ .pos = pos, .q = try a.dupe(u16, q), .ao = try a.dupe(u16, ao), .ids = try a.dupe(i32, qids), .nk = qnk[0], .spr = qspr[0] };
        if (blocksOf(qids, qnk[0], qspr[0], &blk_buf)) |set| try win.add(a, set, &st_decode);
        gpu.gpa.free(q);
        gpu.gpa.free(ao);
        gpu.gpa.free(qids);
        gpu.gpa.free(qnk);
        gpu.gpa.free(qspr);
    }
    if (st_decode.rows > 0) {
        const rows_f: f64 = @floatFromInt(st_decode.rows);
        std.debug.print("RESULT layer {d} selection, {d} decode rows: overlap with the previous row mean {d:.3} min {d:.3}; last 4 rows' union {d:.3}, last 16 {d:.3}\n", .{ n_layers - 1, st_decode.rows, st_decode.sum1 / rows_f, st_decode.min1, st_decode.sum4 / @as(f64, @floatFromInt(@max(1, st_decode.rows4))), st_decode.sum16 / @as(f64, @floatFromInt(@max(1, st_decode.rows16))) });
    }

    // -- the cache variants at each probe row ------------------------------------------------------------------------
    var k8 = try Kv.Kv8.load(gpu.d);
    defer k8.deinit();
    var g = try flash.Glue.load(gpu.d);
    defer g.deinit();
    const nrow = cap * kv_heads;
    const kc = try fx.download(u16, gpu, r.stream, r.kc[ai].ptr, nrow * head_dim);
    defer gpu.gpa.free(kc);
    const vc = try fx.download(u16, gpu, r.stream, r.vc[ai].ptr, nrow * head_dim);
    defer gpu.gpa.free(vc);
    const used = total * kv_heads; // rows past the prompt are zero
    // the host's codes for every variant, and their float64 values in the model's domain
    const Variant = struct { name: []const u8, rot: bool, int4: bool };
    const variants = [_]Variant{ .{ .name = "int8 plain", .rot = false, .int4 = false }, .{ .name = "int8 rotated", .rot = true, .int4 = false }, .{ .name = "int4 plain", .rot = false, .int4 = true }, .{ .name = "int4 rotated", .rot = true, .int4 = true } };
    const kf0 = try a.alloc(f64, nrow * head_dim);
    const vf0 = try a.alloc(f64, nrow * head_dim);
    for (kf0, kc) |*o, x| o.* = bf(x);
    for (vf0, vc) |*o, x| o.* = bf(x);
    var vals: [variants.len][2][]f64 = undefined;
    for (variants, 0..) |v, vi| for (0..2) |kvi| {
        const src = if (kvi == 0) kc else vc;
        const out = try a.alloc(f64, nrow * head_dim);
        @memset(out, 0);
        for (0..used) |row| {
            for (0..ng) |gi| {
                var x: [group]f32 = undefined;
                for (0..group) |j| x[j] = @floatCast(bf(src[row * head_dim + gi * group + j]));
                if (v.rot) kt.wht32(&x);
                var y: [group]f32 = undefined;
                if (v.int4) kt.roundTrip4(&x, &y) else {
                    var codes: [group]i8 = undefined;
                    const sc = kt.quant8(&x, &codes);
                    for (0..group) |j| y[j] = kt.dequant8(codes[j], sc);
                }
                for (0..group) |j| {
                    var acc: f64 = 0;
                    if (v.rot) {
                        for (0..group) |m| acc += (if (@popCount(j & m) % 2 == 0) @as(f64, 1) else -1) * y[m];
                        acc /= @sqrt(32.0);
                    } else acc = y[j];
                    out[row * head_dim + gi * group + j] = acc;
                }
            }
        }
        vals[vi][kvi] = out;
    };
    // the device's int8 caches (rotated and plain), packed from the pass's own bf16 cache
    var codes: [2]Kv.Kv8.Cache = undefined;
    var cbufs: [8]cuda.DeviceBuffer = undefined;
    const cb = Kv.Kv8.cacheBytes(cap, kv_heads);
    for (0..8) |i| cbufs[i] = try cuda.DeviceBuffer.alloc(gpu.d, if (i % 2 == 0) cb.codes else cb.scales);
    defer for (&cbufs) |*x| x.free();
    for (0..2) |ri| {
        codes[ri] = .{ .k8 = cbufs[ri * 4].ptr, .ks = cbufs[ri * 4 + 1].ptr, .v8 = cbufs[ri * 4 + 2].ptr, .vs = cbufs[ri * 4 + 3].ptr };
        try k8.packRows(r.stream, r.kc[ai].ptr, codes[ri].k8, codes[ri].ks, nrow, ri == 1);
        try k8.packRows(r.stream, r.vc[ai].ptr, codes[ri].v8, codes[ri].vs, nrow, ri == 1);
    }
    var qb = try cuda.DeviceBuffer.alloc(gpu.d, q_heads * head_dim * 2);
    defer qb.free();
    var ib = try cuda.DeviceBuffer.alloc(gpu.d, idw * 4 + 32);
    defer ib.free();
    var posb = try cuda.DeviceBuffer.alloc(gpu.d, 16);
    defer posb.free();
    var outb = try cuda.DeviceBuffer.alloc(gpu.d, q_heads * head_dim * 2);
    defer outb.free();
    const want = try a.alloc(f64, q_heads * head_dim);
    const got = try a.alloc(f64, q_heads * head_dim);
    const sbuf = try a.alloc(f64, cap + idw);
    const kbuf = try a.alloc(i32, cap);
    var e_up: kt.Err = .{};
    var e_dev: [2]kt.Err = .{ .{}, .{} };
    var e_host: [variants.len]kt.Err = @splat(.{});
    var e_own: [2]kt.Err = .{ .{}, .{} };
    var replay_equal: usize = 0;
    var flips_up: usize = 0;
    for (probe) |pr| {
        const n: usize = @intCast(pr.nk);
        const keys: []const i32 = if (pr.spr != 0) pr.ids[0..n] else blk: {
            for (0..n) |i| kbuf[i] = @intCast(i);
            break :blk kbuf[0..n];
        };
        for (0..q_heads) |h| oracleRow(pr.q[h * head_dim ..][0..head_dim], kf0, vf0, keys, h, want[h * head_dim ..][0..head_dim], sbuf);
        // the pass's own output (upstream's kernels inside Pass.attnRows)
        for (got, pr.ao) |*o, x| o.* = bf(x);
        errRow(got, want, &e_up);
        // upstream's kernels re-run here on the same buffers: the same bits (the harness reproduces the pass)
        try posb.upload(0, std.mem.sliceAsBytes(&[_]i32{ @intCast(pr.pos), 0, 0, 0 }));
        try ib.upload(0, std.mem.sliceAsBytes(pr.ids));
        try ib.upload(idw * 4, std.mem.sliceAsBytes(&[_]i32{ pr.nk, pr.spr }));
        try qb.upload(0, std.mem.sliceAsBytes(pr.q));
        const chunks = if (pr.spr != 0) nch else (pr.pos + 1 + 511) / 512;
        try g.attnChunks(r.stream, qb.ptr, r.kc[ai].ptr, r.vc[ai].ptr, posb.ptr, s.po, s.pm, s.pl, ib.ptr, ib.ptr + idw * 4, ib.ptr + idw * 4 + 4, 1, kv_heads, chunks, true, .{});
        try g.attnMerge(r.stream, s.po, s.pm, s.pl, posb.ptr, outb.ptr, ib.ptr + idw * 4, ib.ptr + idw * 4 + 4, 1, kv_heads, true);
        const again = try fx.download(u16, gpu, r.stream, outb.ptr, q_heads * head_dim);
        replay_equal += @intFromBool(std.mem.eql(u16, again, pr.ao));
        gpu.gpa.free(again);
        // int8 on the device: plain, and rotated (the query rotated from the pass's bf16 query, then rounded)
        for (0..2) |ri| {
            const rot = ri == 1;
            const qr = try a.alloc(u16, q_heads * head_dim);
            for (0..q_heads * ng) |gi| {
                var x: [group]f32 = undefined;
                for (0..group) |j| x[j] = @floatCast(bf(pr.q[gi * group + j]));
                if (rot) kt.wht32(&x);
                for (0..group) |j| qr[gi * group + j] = bf16(x[j]);
            }
            try qb.upload(0, std.mem.sliceAsBytes(qr));
            try k8.attnChunks(r.stream, qb.ptr, codes[ri], posb.ptr, s.po, s.pm, s.pl, ib.ptr, ib.ptr + idw * 4, ib.ptr + idw * 4 + 4, 1, q_heads, kv_heads, chunks, true);
            try k8.attnMerge(r.stream, s.po, s.pm, s.pl, posb.ptr, outb.ptr, ib.ptr + idw * 4, ib.ptr + idw * 4 + 4, 1, q_heads, kv_heads, true, rot);
            const o8 = try fx.download(u16, gpu, r.stream, outb.ptr, q_heads * head_dim);
            defer gpu.gpa.free(o8);
            for (got, o8) |*o, x| o.* = bf(x);
            errRow(got, want, &e_dev[ri]);
            // against the float64 of its own cache (the kernel's arithmetic alone)
            const own = try a.alloc(f64, q_heads * head_dim);
            const vi: usize = if (rot) 1 else 0;
            for (0..q_heads) |h| oracleRow(pr.q[h * head_dim ..][0..head_dim], vals[vi][0], vals[vi][1], keys, h, own[h * head_dim ..][0..head_dim], sbuf);
            errRow(got, own, &e_own[ri]);
        }
        // every variant's cache in float64 (no kernel): the cache's own error
        for (variants, 0..) |_, vi| {
            for (0..q_heads) |h| oracleRow(pr.q[h * head_dim ..][0..head_dim], vals[vi][0], vals[vi][1], keys, h, got[h * head_dim ..][0..head_dim], sbuf);
            errRow(got, want, &e_host[vi]);
        }
    }
    _ = &flips_up;
    std.debug.print("RESULT layer {d} attention at {d} probe rows ({d}..{d}), against float64 of the pass's bf16 cache (max | rms, of the peak):\n", .{ n_layers - 1, probes, n_prefill, total - 1 });
    std.debug.print("  upstream bf16 (the pass's own output)         {e:.3} | {e:.3}   (re-run here bit-equal on {d}/{d} rows)\n", .{ kt.rel(e_up), kt.rms(e_up), replay_equal, probes });
    std.debug.print("  int8 plain   device kernel                    {e:.3} | {e:.3}   (kernel vs its own cache {e:.3})\n", .{ kt.rel(e_dev[0]), kt.rms(e_dev[0]), kt.rel(e_own[0]) });
    std.debug.print("  int8 rotated device kernel                    {e:.3} | {e:.3}   (kernel vs its own cache {e:.3})\n", .{ kt.rel(e_dev[1]), kt.rms(e_dev[1]), kt.rel(e_own[1]) });
    for (variants, 0..) |v, vi| std.debug.print("  {s:<13} cache alone (float64)            {e:.3} | {e:.3}\n", .{ v.name, kt.rel(e_host[vi]), kt.rms(e_host[vi]) });
    try check.expect(replay_equal == probes, "exl3-kv real: upstream's kernels re-run on the pass's buffers differ on {d} rows", .{probes - replay_equal});
    check.pass("exl3-kv real: layers 0..{d} over {d} tokens; numbers above", .{ n_layers - 1, total });
}

fn oracleRow(q: []const u16, kf: []const f64, vf: []const f64, keys: []const i32, h: usize, out: []f64, sbuf: []f64) void {
    const kvh = h / (q_heads / kv_heads);
    var best = -std.math.inf(f64);
    for (keys, 0..) |key, j| {
        const k: usize = @intCast(key);
        var dot: f64 = 0;
        for (0..head_dim) |dd| dot += bf(q[dd]) * kf[(k * kv_heads + kvh) * head_dim + dd];
        sbuf[j] = dot * 0.0625;
        best = @max(best, sbuf[j]);
    }
    var l: f64 = 0;
    for (sbuf[0..keys.len]) |*v| {
        v.* = @exp(v.* - best);
        l += v.*;
    }
    for (0..head_dim) |dd| {
        var o: f64 = 0;
        for (keys, 0..) |key, j| o += sbuf[j] * vf[(@as(usize, @intCast(key)) * kv_heads + kvh) * head_dim + dd];
        out[dd] = o / l;
    }
}
