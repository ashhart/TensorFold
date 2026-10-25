//! MiniMax M3.1's device parity: tf_m3_scores / tf_m3_topk / tf_m3_decode / tf_m3_decode_combine
//! and the family's tf_m3_swigluoai against the CPU references on synthetic tensors — the indexer's
//! sparse.zig contracts (fp32 block-max-pool scores, the boost override, ties at the smaller id,
//! -1 right-pad) and serve.zig's block-sparse fp32 attention; the NB <= 1024 / nsplit <= 8 bound
//! guards ride the same run. Prints max|delta| per case so receipts can quote them verbatim.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const m3 = @import("minimax_m3_vl_cuda");

const indexer = m3.indexer; // sparse.zig: the CPU reference the kernels are graded against
const sp = m3.sparse_cuda; // the launch wrappers
const glue = m3.serve_cuda; // the family's own glue (the swigluoai rows)

const d = sp.d; // 128: the index head's width
const block = sp.block; // 128 keys a scored block
const topk = sp.topk; // 16: the fixed selection width
const page = sp.page; // 256: the fp16 cache's page

fn f16r(x: f32) f32 {
    return @floatCast(@as(f16, @floatCast(x)));
}

/// The run's shared state: buffers collected for one freeing sweep at the end.
const Rig = struct {
    gpu: check.Gpu,
    a: std.mem.Allocator,
    s: cuda.Stream,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,

    fn dev(r: *Rig, bytes: []const u8) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(r.gpu.d, bytes);
        try r.bufs.append(r.a, b);
        return b.ptr;
    }

    fn uninit(r: *Rig, bytes: usize) !u64 {
        const b = try cuda.DeviceBuffer.alloc(r.gpu.d, bytes);
        try r.bufs.append(r.a, b);
        try b.fill8(0xff, r.s.handle); // an entry the kernel misses reads NaN, never a stale zero
        return b.ptr;
    }

    fn back(r: *Rig, ptr: u64, len: usize) ![]u8 {
        const o = try r.a.alloc(u8, len);
        try r.s.synchronize();
        const b: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = ptr, .len = len };
        try b.download(0, o);
        return o;
    }

    /// fp16 values a scorer or cache reads: the storage width of every indexed row.
    fn h16(r: *Rig, n: usize, scale: f32) ![]f16 {
        const v = try r.a.alloc(f16, n);
        var prng = std.Random.DefaultPrng.init(0xd57c_a7e1);
        const rng = prng.random();
        for (v) |*x| x.* = @floatCast((rng.float(f32) * 2 - 1) * scale);
        return v;
    }
};

fn backF16(bytes: []u8) []f16 {
    return @as([*]f16, @ptrCast(@alignCast(bytes.ptr)))[0 .. bytes.len / 2];
}

fn backF32(bytes: []u8) []f32 {
    return @as([*]f32, @ptrCast(@alignCast(bytes.ptr)))[0 .. bytes.len / 4];
}

/// max |got - want| plus class agreement (+inf stays +inf, -inf stays -inf, NaN stays NaN).
fn delta(got: []const f32, want: []const f32) struct { d: f64, classes: bool } {
    var worst: f64 = 0;
    var classes = true;
    for (got, want) |g, w| {
        if (std.math.isPositiveInf(g) != std.math.isPositiveInf(w) or
            std.math.isNegativeInf(g) != std.math.isNegativeInf(w) or
            (g != g) != (w != w)) classes = false;
        if (std.math.isFinite(g) and std.math.isFinite(w)) {
            const e = @abs(@as(f64, g) - @as(f64, w));
            if (e > worst) worst = e;
        }
    }
    return .{ .d = worst, .classes = classes };
}

pub fn run(gpu: check.Gpu) !void {
    const mod = try cuda.Module.load(gpu.d, cuda.kernels.minimax_m3);
    const ks = try sp.Kernels.load(mod);
    const gk = try glue.Kernels.load(mod);
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var rig = Rig{ .gpu = gpu, .a = gpu.gpa, .s = s };

    try scoresParity(&rig, ks);
    try topkParity(&rig, ks, 300);
    try marksParity(&rig, ks); // NB = 1024 inclusive: the wide-chunk ceiling runs, 1025 refuses
    try decodeParity(&rig, ks, 4);
    try decodeParity(&rig, ks, 1);
    try zeroRowParity(&rig, ks);
    try engineGeometryParity(&rig, ks); // the engine's exact failing invocation: the stage-6b regression
    try engineSelectionParity(&rig, ks); // the engine's +inf-boosted scores -> real topk -> decode: the stage-6c regression
    try swigluoaiParity(&rig, gk);
    for (rig.bufs.items) |*b| b.free();
    rig.bufs.deinit(gpu.gpa);
}

/// tf_m3_scores against sparse.scores on fp16 storage rows: fp32 scoring, the right-open causal
/// mask, -inf fully-masked blocks, the own-block +inf override, and the NaN class contract.
fn scoresParity(r: *Rig, ks: sp.Kernels) !void {
    const a = r.a;
    const h: usize = 4; // the pack's 4 index heads
    const sq: usize = 5;
    const scale_of_q: f32 = 0.4;
    const cases = [_]usize{ 1, 129, 300 }; // a partial block, one over, several
    for (cases) |sk| {
        const nb = sp.nbOf(sk);
        const q = try r.h16(h * sq * d, scale_of_q);
        const k = try r.h16(sk * d, scale_of_q);
        var qpos = try a.alloc(i64, sq);
        var kpos = try a.alloc(i64, sk);
        // query rows sit at positions 0, 3, 6, ... so every block's mask edge is exercised
        for (0..sq) |ri| qpos[ri] = @intCast(3 * ri);
        for (0..sk) |ki| kpos[ki] = @intCast(ki); // content keys, none padded
        // the CPU reference over the SAME fp16-rounded storage values, f32 inside
        const qf = try a.alloc(f32, q.len);
        for (q, 0..) |v, i| qf[i] = @floatCast(v);
        const kf = try a.alloc(f32, k.len);
        for (k, 0..) |v, i| kf[i] = @floatCast(v);
        const ref = try indexer.scores(a, qf, kf, qpos, kpos, h, sq, sk, d, block);
        // the boost override the kernel applies at the write (local 1, init 0): the row's own block
        for (0..h) |hi| for (0..sq) |ri| {
            const b = @divTrunc(qpos[ri], @as(i64, @intCast(block)));
            if (b >= 0 and b < @as(i64, @intCast(nb))) ref[(hi * sq + ri) * nb + @as(usize, @intCast(b))] = std.math.inf(f32);
        };
        const qu = try r.dev(std.mem.sliceAsBytes(q));
        const ku = try r.dev(std.mem.sliceAsBytes(k));
        const qpu = try r.dev(std.mem.sliceAsBytes(qpos));
        const kpu = try r.dev(std.mem.sliceAsBytes(kpos));
        const ou = try r.uninit(h * sq * nb * 4);
        try sp.scores(ks, r.s, qu, ku, qpu, kpu, ou, sq, nb, sk, h, 1, 1, 0);
        const got_b = try r.back(ou, h * sq * nb * 4);
        const got = backF32(got_b);
        const res = delta(got, ref);
        try check.expect(res.classes, "scores: a +/-inf/NaN class disagrees at sk={d}", .{sk});
        const tol = 1e-4 + 1e-6 * @as(f64, @floatFromInt(sk * d));
        try check.expect(res.d <= tol, "scores sk={d}: max|delta| {e} over tolerance {e}", .{ sk, res.d, tol });
        // the override itself, read back: every row's own block sits at +inf on the device
        for (0..h) |hi| for (0..sq) |ri| {
            const b: usize = @intCast(@divTrunc(qpos[ri], @as(i64, @intCast(block))));
            try check.expect(std.math.isPositiveInf(got[(hi * sq + ri) * nb + b]), "scores sk={d}: row {d} head {d} own block {d} not boosted", .{ sk, ri, hi, b });
        };
        check.pass("scores sk={d} nb={d}: max|delta| {e} (classes hold, override at +inf)", .{ sk, nb, res.d });
    }
    // The NaN rule at a NON-own block: one NaN key element turns the block's dot (and so amax)
    // NaN — the kernel's stated amax class (sparse.zig's CPU rows have no NaN producer; this pins
    // the device class). The row's own block stays +inf: the boost lands after the reduction and
    // replaces any pooled value there, NaN included.
    {
        const sk: usize = 300; // 3 blocks; the row sits in block 1, block 0 stays unboosted
        const nb = sp.nbOf(sk);
        const q = try a.alloc(f16, h * 1 * d);
        const k = try a.alloc(f16, sk * d);
        for (q) |*x| x.* = @floatCast(0.1);
        for (k) |*x| x.* = @floatCast(0.1);
        k[7] = std.math.nan(f16); // block 0's keys, valid under the causal mask (kpos 0 <= 200)
        const qpos = [_]i64{200};
        const kpos = try a.alloc(i64, sk);
        for (0..sk) |ki| kpos[ki] = @intCast(ki);
        const qu = try r.dev(std.mem.sliceAsBytes(q));
        const ku = try r.dev(std.mem.sliceAsBytes(k));
        const qpu = try r.dev(std.mem.sliceAsBytes(&qpos));
        const kpu = try r.dev(std.mem.sliceAsBytes(kpos));
        const ou = try r.uninit(h * nb * 4);
        try sp.scores(ks, r.s, qu, ku, qpu, kpu, ou, 1, nb, sk, h, 1, 1, 0);
        const got_b = try r.back(ou, h * nb * 4);
        const got = backF32(got_b);
        for (0..h) |hi| {
            try check.expect(got[hi * nb] != got[hi * nb], "scores: head {d} block 0 did not carry the NaN amax", .{hi}); // unboosted: NaN wins
            try check.expect(std.math.isPositiveInf(got[hi * nb + 1]), "scores: head {d} own block is not +inf over the NaN", .{hi}); // the boost lands after reduction
        }
        check.pass("scores NaN rule: the unboosted block's amax carries NaN, the own block's boost still wins", .{});
    }
}

/// tf_m3_topk against sparse.topk: synthetic score rows with exact ties, boost overrides already
/// in the table, descending picks, ties at the smaller id, fewer-valid rows padded -1.
fn topkParity(r: *Rig, ks: sp.Kernels, sk: usize) !void {
    const a = r.a;
    const h: usize = 4;
    const sq: usize = 12;
    const nb = sp.nbOf(sk);
    // scores with quantized values so exact ties are common, and -inf pads behind the causal edge
    var prng = std.Random.DefaultPrng.init(0x5c0a_e123);
    const rng = prng.random();
    const s = try a.alloc(f32, h * sq * nb);
    const qpos = try a.alloc(i64, sq);
    for (0..sq) |ri| {
        qpos[ri] = @intCast(if (ri == 0) 0 else ri * 31 % (sk - 1));
        for (0..h) |hi| for (0..nb) |b| {
            const row = (hi * sq + ri) * nb;
            const causal = @as(i64, @intCast((b + 1) * block)) <= qpos[ri];
            s[row + b] = if (causal) std.math.round(rng.float(f32) * 8) else -std.math.inf(f32);
        };
        // the override: the row's own block, as sparse.topk would scatter it
        const b = @divTrunc(qpos[ri], @as(i64, @intCast(block)));
        if (b >= 0 and b < @as(i64, @intCast(nb))) {
            for (0..h) |hi| s[(hi * sq + ri) * nb + @as(usize, @intCast(b))] = std.math.inf(f32);
        }
    }
    const ref = try indexer.topk(a, s, h, sq, nb, topk, block, 1, 0, qpos);
    const su = try r.dev(std.mem.sliceAsBytes(s));
    const ou = try r.uninit(h * sq * topk * 4);
    try sp.topk16(ks, r.s, su, ou, nb, h * sq);
    const got_b = try r.back(ou, h * sq * topk * 4);
    const got = backI32(got_b);
    try check.expect(std.mem.eql(i32, got, ref), "topk: pick ids disagree", .{});
    // two rows' trace quoted for the receipts
    check.pass("topk rows={d} nb={d}: every pick equal (order, ties, -1 pads)", .{ h * sq, nb });
}

fn backI32(bytes: []u8) []i32 {
    return @as([*]i32, @ptrCast(@alignCast(bytes.ptr)))[0 .. bytes.len / 4];
}

/// The mark bitmap's NB <= 1024 bound at its ceiling: a 131,072-key table (NB exactly 1024)
/// scores and selects; 1025 blocks is refused by the wrapper before any launch.
fn marksParity(r: *Rig, ks: sp.Kernels) !void {
    const a = r.a;
    const h: usize = 4;
    const sq: usize = 2;
    const sk: usize = block * sp.max_nb; // 131,072
    const nb = sp.nbOf(sk);
    try check.expect(nb == sp.max_nb, "the wide table's block count is the mark bitmap's ceiling", .{});
    const q = try r.h16(h * sq * d, 0.4);
    const k = try r.h16(sk * d, 0.4);
    var qpos = [_]i64{ @as(i64, @intCast(sk - 1)), @as(i64, @intCast(sk / 2)) };
    var kpos = try a.alloc(i64, sk);
    for (0..sk) |ki| kpos[ki] = @intCast(ki);
    const qf = try a.alloc(f32, q.len);
    for (q, 0..) |v, i| qf[i] = @floatCast(v);
    const kf = try a.alloc(f32, k.len);
    for (k, 0..) |v, i| kf[i] = @floatCast(v);
    const ref = try indexer.scores(a, qf, kf, &qpos, kpos, h, sq, sk, d, block);
    for (0..h) |hi| for (0..sq) |ri| {
        const b: usize = @intCast(@divTrunc(qpos[ri], @as(i64, @intCast(block))));
        ref[(hi * sq + ri) * nb + b] = std.math.inf(f32);
    };
    const qu = try r.dev(std.mem.sliceAsBytes(q));
    const ku = try r.dev(std.mem.sliceAsBytes(k));
    const kpu = try r.dev(std.mem.sliceAsBytes(kpos));
    const qpu = try r.dev(std.mem.sliceAsBytes(&qpos));
    const ou = try r.uninit(h * sq * nb * 4);
    try sp.scores(ks, r.s, qu, ku, qpu, kpu, ou, sq, nb, sk, h, 1, 1, 0);
    const got_b = try r.back(ou, h * sq * nb * 4);
    const got = backF32(got_b);
    const res = delta(got, ref);
    try check.expect(res.classes, "scores nb=1024: a +/-inf class disagrees", .{});
    const tol = 1e-4 + 1e-6 * @as(f64, @floatFromInt(sk * d));
    try check.expect(res.d <= tol, "scores nb=1024: max|delta| {e} over {e}", .{ res.d, tol });
    const su = try r.dev(std.mem.sliceAsBytes(ref)); // the same table back through the selection
    const pu = try r.uninit(h * sq * topk * 4);
    try sp.topk16(ks, r.s, su, pu, nb, h * sq);
    const p_b = try r.back(pu, h * sq * topk * 4);
    const picks = backI32(p_b);
    const want = try indexer.topk(a, ref, h, sq, nb, topk, block, 1, 0, &qpos);
    try check.expect(std.mem.eql(i32, picks, want), "topk nb=1024: pick ids disagree", .{});
    check.pass("marks nb=1024 ceiling (sk={d}): scores max|delta| {e}, picks equal; nb=1025 refused below", .{ sk, res.d });
    const refused: anyerror = blk: {
        sp.topk16(ks, r.s, 0, 0, sp.max_nb + 1, 1) catch |e| break :blk e;
        break :blk error.MissingError;
    };
    try check.expect(refused == error.ScoreColumns, "an nb of {d} was not refused ({s})", .{ sp.max_nb + 1, @errorName(refused) });
}

/// tf_m3_decode + tf_m3_decode_combine against serve.zig's block-sparse attention over the same
/// fp16 cache rows and selection lists — the gather, the causal seqlen mask, the fp32 online
/// softmax and split merge, fp16 out. nsplit runs 4 (the served envelope) and 1.
fn decodeParity(r: *Rig, ks: sp.Kernels, nsplit: u32) !void {
    const a = r.a;
    const hq: usize = 8;
    const hkv: usize = 4; // index heads == kv heads; GQA group of 2
    const sk: usize = 1200; // 10 blocks, so the 16-wide lists carry -1 pads
    const seqlen: i64 = 1100; // block 9 sits past the logical end: masked, never read
    const pages = sk / page + 1;
    const pages_of_seq = (sk + page - 1) / page;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));
    var prng = std.Random.DefaultPrng.init(0xdec0_de55);
    const rng = prng.random();
    const kt = try r.h16(pages * page * hkv * d, 0.6);
    const vt = try r.h16(pages * page * hkv * d, 0.6);
    const q = try r.h16(hq * d, 0.6);
    // per kv head: a shuffled set of the 9 in-range blocks, then -1 pads
    const sel = try a.alloc(i32, hkv * topk);
    for (0..hkv) |hv| {
        const list = sel[hv * topk ..][0..topk];
        for (0..topk) |i| list[i] = -1;
        var perm = [_]i32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        for (1..10) |i| { // a shuffle of the ids; the kernel never trusts the order
            const j = rng.intRangeAtMost(usize, 0, i);
            const t = perm[i];
            perm[i] = perm[j];
            perm[j] = t;
        }
        const count: usize = if (hv == 0) 9 else 10;
        for (0..count) |i| list[i] = perm[i];
    }
    // the reference: serve.zig's math at f32, from the same fp16-rounded storage rows
    const kf = try a.alloc(f32, kt.len);
    for (kt, 0..) |v, i| kf[i] = @floatCast(v);
    const vf = try a.alloc(f32, vt.len);
    for (vt, 0..) |v, i| vf[i] = @floatCast(v);
    const qf = try a.alloc(f32, q.len);
    for (q, 0..) |v, i| qf[i] = @floatCast(v);
    const table = try a.alloc(i32, pages_of_seq);
    const seqlens = [_]i64{seqlen};
    for (0..pages_of_seq) |p| table[p] = @intCast(p); // identity page table, 2 sequences' pages fresh
    const ref = try a.alloc(f32, hq * d);
    for (0..hq) |i| {
        const hv = i / (hq / hkv);
        const qr = qf[i * d ..][0..d];
        var mx: f32 = -std.math.inf(f32);
        var scores = try a.alloc(f32, topk * block);
        defer a.free(scores);
        var n: usize = 0;
        for (0..topk) |sl| {
            const bid = sel[hv * topk + sl];
            if (bid < 0) break;
            var j: usize = 0;
            while (j < block) : (j += 1) {
                const p: i64 = @as(i64, bid) * @as(i64, @intCast(block)) + @as(i64, @intCast(j));
                if (p >= seqlens[0]) continue;
                const pg: usize = @intCast(@divTrunc(p, @as(i64, @intCast(page))));
                const off: usize = @intCast(@mod(p, @as(i64, @intCast(page))));
                const krow = kf[((pg * page + off) * hkv + hv) * d ..][0..d];
                var dot: f32 = 0;
                for (0..d) |c| dot += qr[c] * krow[c];
                scores[n] = dot * scale;
                if (scores[n] > mx) mx = scores[n];
                n += 1;
            }
        }
        var se: f32 = 0;
        for (0..n) |mi| {
            scores[mi] = @exp(scores[mi] - mx);
            se += scores[mi];
        }
        const out = ref[i * d ..][0..d];
        @memset(out, 0);
        var at: usize = 0;
        for (0..topk) |sl| {
            const bid = sel[hv * topk + sl];
            if (bid < 0) break;
            var j: usize = 0;
            while (j < block) : (j += 1) {
                const p: i64 = @as(i64, bid) * @as(i64, @intCast(block)) + @as(i64, @intCast(j));
                if (p >= seqlens[0]) continue;
                const pg: usize = @intCast(@divTrunc(p, @as(i64, @intCast(page))));
                const off: usize = @intCast(@mod(p, @as(i64, @intCast(page))));
                const vrow = vf[((pg * page + off) * hkv + hv) * d ..][0..d];
                const w = scores[at]; at += 1;
                for (0..d) |c| out[c] += w / se * vrow[c];
            }
        }
    }
    // the device run
    const npp: usize = pages_of_seq;
    const ktu = try r.dev(std.mem.sliceAsBytes(kt));
    const vtu = try r.dev(std.mem.sliceAsBytes(vt));
    const qcu = try r.dev(std.mem.sliceAsBytes(q));
    const tbu = try r.dev(std.mem.sliceAsBytes(table));
    const slu = try r.dev(std.mem.sliceAsBytes(sel));
    const seqs = try r.dev(std.mem.sliceAsBytes(&seqlens));
    const pu = try r.uninit(sp.scratchBytes(hq, 1, nsplit));
    const ou = try r.uninit(hq * d * 2);
    try sp.decode(ks, r.s, qcu, ktu, vtu, tbu, slu, seqs, pu, hq, hkv, npp, scale, 1, nsplit);
    try sp.combine(ks, r.s, pu, ou, hq, 1, nsplit);
    const got_b = try r.back(ou, hq * d * 2);
    const got = backF16(got_b);
    var worst: f64 = 0;
    for (0..hq) |i| for (0..d) |c| {
        const g: f64 = @floatCast(got[i * d + c]);
        const w0: f64 = @floatCast(f16r(ref[i * d + c])); // the reference's fp16 store too
        const e = @abs(g - w0);
        if (e > worst) worst = e;
    };
    try check.expect(worst <= 5e-3, "decode nsplit={d}: max|delta| {e} over 5e-3", .{ nsplit, worst });
    check.pass("decode+combine nsplit={d}: attention max|delta| {e} vs the fp16-rounded reference", .{ nsplit, worst });
}

/// An all-(-1) selection is the zero-valid-key row: the combine returns zeros, never a NaN —
/// byte for byte, including the two heads a normal list keeps finite.
fn zeroRowParity(r: *Rig, ks: sp.Kernels) !void {
    const a = r.a;
    const hq: usize = 2;
    const hkv: usize = 2;
    const scale: f32 = 1.0 / @sqrt(128.0);
    const kt = try r.h16(2 * page * hkv * d, 0.6);
    const q = try r.h16(hq * d, 0.6);
    const sel = try a.alloc(i32, hkv * topk);
    @memset(sel, -1);
    const table = try a.alloc(i32, 1);
    table[0] = 0;
    const seqlens = [_]i64{0};
    const ktu = try r.dev(std.mem.sliceAsBytes(kt));
    const vtu = ktu; // never read with l = 0, aliased harmlessly
    const qcu = try r.dev(std.mem.sliceAsBytes(q));
    const tbu = try r.dev(std.mem.sliceAsBytes(table));
    const slu = try r.dev(std.mem.sliceAsBytes(sel));
    const seqs = try r.dev(std.mem.sliceAsBytes(&seqlens));
    const pu = try r.uninit(sp.scratchBytes(hq, 1, 4));
    const ou = try r.uninit(hq * d * 2);
    try sp.decode(ks, r.s, qcu, ktu, vtu, tbu, slu, seqs, pu, hq, hkv, 1, scale, 1, 4);
    try sp.combine(ks, r.s, pu, ou, hq, 1, 4);
    const got_b = try r.back(ou, hq * d * 2);
    for (got_b) |byte| try check.expect(byte == 0, "zero-valid-key row returned non-zero bytes", .{});
    check.pass("zero-valid-key row: all-zero fp16 bytes, never a NaN", .{});
}

/// The ENGINE's exact failing-invocation geometry (graph.zig layerRow's sp.decode/sp.combine), the
/// stage-6b regression for the max|a|=947 hunt: the fixture's dims (Hq 64, Hkv 4, D 128), the
/// layer-strided [nl*keys_cap][Hkv][D] cache whose L0 base sits at +0 (so the kernel's
/// [(P*PAGE)+off][Hkv][D] row addressing lines up), the 16-entry identity table, sel
/// (B, Hkv, 16) = [0, -1 x 15], seqlens [1] (ONE valid key), sp.scratchBytes(Hq, 1, 4), nsplit 4,
/// and the magnitudes the smoke's engine measured (max|q| 7.78 fp16, key row 0 to 6.07, v row 0
/// between -497.75 and 631.5). With one valid key the softmax weight is 1: every combine row must
/// equal its kv head's v row bit for bit, split 0's scratch must carry m = the q·k score with
/// l = 1 and acc = the v row, and the empty splits must hold m = -inf, acc = 0.
fn engineGeometryParity(r: *Rig, ks: sp.Kernels) !void {
    const a = r.a;
    const hq: usize = 64; // the fixture's attn_heads
    const hkv: usize = 4; // the fixture's kv_heads; GQA group of 16
    const kvr: usize = hkv * d; // 512
    const rows: usize = 16 * page; // keys_cap 4096: (P=16, PAGE=256, Hkv, D) at the L0 base
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));
    var prng = std.Random.DefaultPrng.init(0x6b5e_947a);
    const rng = prng.random();
    // q: noise at the engine's scale with the smoke's fp16 max in row 0
    const q = try a.alloc(f16, hq * d);
    for (q, 0..) |*x, i| {
        const s: f32 = if (i == 0) 7.7773 else (rng.float(f32) * 2 - 1) * 4.0;
        x.* = @floatCast(s);
    }
    // the cache: key row 0 at the smoke's max 6.07, v row 0 within [min -497.75, max 631.5],
    // the rest noise on the same scales (rows >= 1 are masked; they may be anything)
    const kt = try a.alloc(f16, rows * kvr);
    const vt = try a.alloc(f16, rows * kvr);
    for (kt) |*x| x.* = @floatCast((rng.float(f32) * 2 - 1) * 3.0);
    for (vt) |*x| x.* = @floatCast((rng.float(f32) * 2 - 1) * 60.0);
    for (0..d) |i| {
        kt[0 * kvr + i] = @floatCast((rng.float(f32) * 2 - 1) * 6.0664);
        const vv: f32 = @round((rng.float(f32) * 2 - 1) * 497.75);
        vt[0 * kvr + i] = @floatCast(if (i == 0) 631.5 else vv);
    }
    const sel = try a.alloc(i32, hkv * topk);
    for (0..hkv) |hv| {
        const list = sel[hv * topk ..][0..topk];
        list[0] = 0;
        for (1..topk) |i| list[i] = -1;
    }
    const table = try a.alloc(i32, 16);
    for (0..16) |p| table[p] = @intCast(p); // the engine's identity upload
    const seqlens = [_]i64{1};
    // the reference: single valid key, weight 1 — out is the fp16 v row; split 0's m is the score
    const kf = try a.alloc(f32, kt.len);
    for (kt, 0..) |v, i| kf[i] = @floatCast(v);
    const vf = try a.alloc(f32, vt.len);
    for (vt, 0..) |v, i| vf[i] = @floatCast(v);
    const qf = try a.alloc(f32, q.len);
    for (q, 0..) |v, i| qf[i] = @floatCast(v);
    const want = try a.alloc(f16, hq * d);
    const m0 = try a.alloc(f32, hq); // split 0's m per head, from the same fp16 rows
    for (0..hq) |i| {
        const hv = i / (hq / hkv);
        for (0..d) |c| want[i * d + c] = @floatCast(vf[0 * kvr + hv * d + c]);
        var dot: f32 = 0;
        const k0 = kf[0 * kvr + hv * d ..][0..d];
        for (0..d) |c| dot += qf[i * d + c] * k0[c];
        m0[i] = dot * scale;
    }
    // the device run: the engine's layout and args, verbatim
    const ktu = try r.dev(std.mem.sliceAsBytes(kt));
    const vtu = try r.dev(std.mem.sliceAsBytes(vt));
    const qcu = try r.dev(std.mem.sliceAsBytes(q));
    const tbu = try r.dev(std.mem.sliceAsBytes(table));
    const slu = try r.dev(std.mem.sliceAsBytes(sel));
    const seqs = try r.dev(std.mem.sliceAsBytes(&seqlens));
    const pu = try r.uninit(sp.scratchBytes(hq, 1, 4));
    const ou = try r.uninit(hq * d * 2);
    try sp.decode(ks, r.s, qcu, ktu, vtu, tbu, slu, seqs, pu, hq, hkv, 16, scale, 1, 4);
    try sp.combine(ks, r.s, pu, ou, hq, 1, 4);
    const got_b = try r.back(ou, hq * d * 2);
    const got = backF16(got_b);
    const pb = try r.back(pu, sp.scratchBytes(hq, 1, 4));
    const parts = backF32(pb);
    const sestr: usize = d + 2; // one (split, head)'s scratch: the 128 acc lanes then m, l
    var worst: usize = 0; // mismatching fp16 bits
    for (0..hq * d) |i| {
        if (got[i] != want[i]) worst += 1;
    }
    try check.expect(worst == 0, "engine geometry: {d} of {d} combine bits differ from the v rows", .{ worst, hq * d });
    var bad: usize = 0;
    for (0..hq) |hh| {
        const row = hh * 4 * sestr; // (B, Hq, nsplit, D+2), B = 1
        var m0e: f64 = 0;
        for (0..d) |c| {
            const want0: f64 = @floatCast(want[hh * d + c]); // acc = p * v with p = 1
            m0e = @max(m0e, @abs(@as(f64, parts[row + c]) - want0));
        }
        try check.expect(m0e == 0, "head {d} split 0 acc != v ({e})", .{ hh, m0e });
        try check.expect(@abs(@as(f64, parts[row + sestr - 1]) - 1.0) < 1e-7, "head {d} l != 1", .{hh});
        const md = @abs(@as(f64, parts[row + sestr - 2]) - @as(f64, m0[hh]));
        try check.expect(md <= 2e-5, "head {d} m {e} off by {e}", .{ hh, m0[hh], md });
        for (1..4) |spp| {
            const srow = row + spp * sestr;
            try check.expect(std.math.isNegativeInf(parts[srow + sestr - 2]), "head {d} split {d} m not -inf", .{ hh, spp });
            try check.expect(parts[srow + sestr - 1] == 0, "head {d} split {d} l not 0", .{ hh, spp });
            for (srow..srow + d) |c| if (parts[c] != 0) {
                bad += 1;
                break;
            };
        }
    }
    try check.expect(bad == 0, "{d} empty-split heads carry non-zero acc", .{bad});
    for (got) |g| try check.expect(!std.math.isNan(g), "combine wrote a NaN at the engine geometry", .{});
    check.pass("engine geometry (Hq 64, sel [0,-1x15], seqlens 1, layer-strided L0, nsplit 4): combine == v rows bit-for-bit, scratch m/l/acc exact, empty splits clean", .{});
}

/// The engine's selection path end to end — the stage-6c regression for the delegated open corner:
/// tf_m3_scores at the engine's NB=1 shape producing the own-block +inf boost, tf_m3_topk at the
/// engine's consumer rows (i_heads == kv_heads is the config's admission assert), that +inf-boosted
/// score list then driving decode+combine exactly as the engine hands the device buffer over. The
/// boost cannot make topk pathological: the own block wins pick 0, its per-pick mark invalidates
/// it, and the exhausted picks right-pad -1 — the single valid key then returns each head's v row.
fn engineSelectionParity(r: *Rig, ks: sp.Kernels) !void {
    const a = r.a;
    const hq: usize = 64; // the fixture's attn_heads
    const hkv: usize = 4; // kv_heads; the indexer's i_heads == kv_heads (the admission assert)
    const ih: usize = 4; // the indexer's heads: the scores table's rows
    const kvr: usize = hkv * d;
    const rows: usize = 16 * page; // keys_cap 4096 at the L0 base: (P=16, PAGE=256, Hkv, D)
    const sk: usize = 1;
    const nb: usize = sp.nbOf(sk); // 1: the query's own block carries the +inf boost
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));
    var prng = std.Random.DefaultPrng.init(0x7e57_ca7e);
    const rng = prng.random();
    // the indexer's side: fp16 q rows (the smoke's max) and the one fp16 plane row pos 0 appends
    const iq = try a.alloc(f16, ih * d);
    for (iq, 0..) |*x, i| {
        const s: f32 = if (i == 0) 2.5867 else 2.6;
        x.* = @floatCast((rng.float(f32) * 2 - 1) * s);
    }
    const plane_row = try a.alloc(f16, sk * d);
    for (plane_row) |*x| x.* = @floatCast((rng.float(f32) * 2 - 1) * 1.2);
    const qpos = [_]i64{0};
    const kpos = try a.alloc(i64, rows); // the engine's identity 0..cap-1 upload
    for (kpos, 0..) |*x, i| x.* = @intCast(i);
    // the attention side at the engine's shapes (the same scales engineGeometryParity grades)
    const q = try a.alloc(f16, hq * d);
    for (q) |*x| x.* = @floatCast((rng.float(f32) * 2 - 1) * 4.0);
    const kt = try a.alloc(f16, rows * kvr);
    const vt = try a.alloc(f16, rows * kvr);
    for (kt) |*x| x.* = @floatCast((rng.float(f32) * 2 - 1) * 3.0);
    for (vt) |*x| x.* = @floatCast((rng.float(f32) * 2 - 1) * 60.0);
    for (0..d) |i| vt[0 * kvr + i] = @floatCast((rng.float(f32) * 2 - 1) * 60.0);
    const table = try a.alloc(i32, 16);
    for (0..16) |p| table[p] = @intCast(p); // the engine's identity page table
    const seqlens = [_]i64{1};
    // the reference: the produced list's only valid key is block 0's p=0, weight 1 — each head's
    // combine row is its kv head's v row (GQA hv = hq/16), the same fp16 rows the cache stores
    const want = try a.alloc(f16, hq * d);
    for (0..hq) |i| {
        const hv = i / (hq / hkv);
        for (0..d) |c| want[i * d + c] = vt[0 * kvr + hv * d + c];
    }
    // scores -> topk -> decode -> combine, the device chain in the engine's order and shapes
    const iqu = try r.dev(std.mem.sliceAsBytes(iq));
    const plu = try r.dev(std.mem.sliceAsBytes(plane_row));
    const qpu = try r.dev(std.mem.sliceAsBytes(&qpos));
    const kpu = try r.dev(std.mem.sliceAsBytes(kpos));
    const scu = try r.uninit(ih * nb * 4);
    try sp.scores(ks, r.s, iqu, plu, qpu, kpu, scu, 1, nb, sk, ih, 1, 1, 0);
    const sc = backF32(try r.back(scu, ih * nb * 4));
    for (sc) |s| try check.expect(std.math.isPositiveInf(s), "the pos-0 own block must score +inf, got {d}", .{s});
    const selu = try r.uninit(hkv * topk * 4);
    try sp.topk16(ks, r.s, scu, selu, nb, hkv); // the engine's topk rows = kv_heads over the same table
    const sel = backI32(try r.back(selu, hkv * topk * 4));
    for (0..hkv) |hv| for (0..topk) |ti| {
        const want_v: i32 = if (ti == 0) 0 else -1;
        try check.expect(sel[hv * topk + ti] == want_v, "sel[{d}][{d}] = {d}, want {d}: the +inf boost distorted the picks", .{ hv, ti, sel[hv * topk + ti], want_v });
    };
    const ktu = try r.dev(std.mem.sliceAsBytes(kt));
    const vtu = try r.dev(std.mem.sliceAsBytes(vt));
    const qcu = try r.dev(std.mem.sliceAsBytes(q));
    const tbu = try r.dev(std.mem.sliceAsBytes(table));
    const seqs = try r.dev(std.mem.sliceAsBytes(&seqlens));
    const pu = try r.uninit(sp.scratchBytes(hq, 1, 4)); // the engine's scratch sizing
    const ou = try r.uninit(hq * d * 2);
    try sp.decode(ks, r.s, qcu, ktu, vtu, tbu, selu, seqs, pu, hq, hkv, 16, scale, 1, 4);
    try sp.combine(ks, r.s, pu, ou, hq, 1, 4);
    const got = backF16(try r.back(ou, hq * d * 2));
    var worst: usize = 0;
    for (0..hq * d) |i| {
        if (got[i] != want[i]) worst += 1;
    }
    try check.expect(worst == 0, "engine selection: {d} of {d} combine bits differ from the v rows", .{ worst, hq * d });
    check.pass("engine selection (scores +inf boost -> topk rows=kv_heads -> decode/combine nsplit 4, NB=1 sk=1): sel == [0,-1x15] per kv head, combine == v rows bit-for-bit", .{});
}

/// tf_m3_swigluoai against serve.zig's swigluoai at the pack's alpha 1.702 / limit 7: fp16 rows,
/// clamps above-only (g) and both-ways (u), the fast-exp sigmoid's class allowed its few ulps.
fn swigluoaiParity(r: *Rig, ks: glue.Kernels) !void {
    const a = r.a;
    const rows: usize = 17;
    const n: usize = 1536; // not a 512 multiple boundary case: a partial block of the 4-elem lanes
    var prng = std.Random.DefaultPrng.init(0x1a2b_3c4d);
    const rng = prng.random();
    const g = try a.alloc(f16, rows * n);
    const u = try a.alloc(f16, rows * n);
    for (g) |*x| {
        const v = if (rng.boolean()) rng.float(f32) * 20 - 4 else rng.float(f32) - 0.5; // heavy at the clamp edge
        x.* = @floatCast(v);
    }
    for (u) |*x| {
        const v = if (rng.boolean()) rng.float(f32) * 30 - 15 else rng.float(f32) * 2 - 1;
        x.* = @floatCast(v);
    }
    const want = try a.alloc(f32, rows * n);
    defer a.free(want);
    for (0..rows * n) |i| want[i] = m3.serve.swigluoai(@floatCast(g[i]), @floatCast(u[i]), 1.702, 7.0);
    const gu = try r.dev(std.mem.sliceAsBytes(g));
    const uu = try r.dev(std.mem.sliceAsBytes(u));
    const ou = try r.uninit(rows * n * 2);
    try glue.swigluoai(ks, r.s, gu, uu, ou, rows, n, 7.0);
    const got_b = try r.back(ou, rows * n * 2);
    const got = backF16(got_b);
    var worst: f64 = 0;
    for (0..rows * n) |i| {
        const e = @abs(@as(f64, @floatCast(got[i])) - @as(f64, @floatCast(f16r(want[i]))));
        if (e > worst) worst = e;
    }
    try check.expect(worst <= 5e-3, "swigluoai: max|delta| {e} over 5e-3", .{worst});
    check.pass("swigluoai rows={d} n={d}: max|delta| {e} vs serve.swigluoai at the fp16 store", .{ rows, n, worst });
}