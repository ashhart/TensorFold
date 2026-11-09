//! Ripple's DFlash2 drafter behind lanes.drafter, cut over the ranks as the target: taps, a 16-row block, the selector.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const fwd = @import("tp2_forward.zig");
const ld = @import("tp2_draft_load.zig");

const dr = lanes.drafter;
const Stream = lanes.Stream;
const Cost = lanes.config.Cost;

/// cuBLASLt's workspace for the drafter's matmuls.
const work_bytes = 32 << 20;
/// Streams one hold runs together (a round's others go in further holds), and the widest block.
const hold_streams = 4;
const max_block = 16;

/// A stream's context: its layers' key and value rings over positions [begin, end), and the chain it holds.
pub const Ctx = struct { ring: cuda.DeviceBuffer, id: i32 = 0, begin: u64 = 0, end: u64 = 0, held: [max_block]u32 = undefined, held_n: usize = 0 };

const Bufs = struct { taps: u64, th: u64, thn: u64, ka: u64, va: u64, h: u64, h2: u64, x: u64, xin: u64, dyn: u64, q: u64, kr: u64, vr: u64, qb: u64, kb: u64, vb: u64, att: u64, o: u64, gate: u64, up: u64, act: u64, fin: u64, fin16: u64, logits: u64, hp: u64, tk: u64, pm: u64, pl: u64, po: u64, ids: u64, segs: u64 };

pub const Draft = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    m: *fwd.Model,
    c: ld.Config, // the rank's widths (tp2_draft_load.zig's share)
    rank: usize,
    world: usize,
    w: ld.Weights,
    k: ld.Kernels,
    dense: cuda.dense_rows.DenseRows,
    lt: ?cuda.cublaslt.Library = null, // bf16 matmuls by cuBLASLt (null: dense_rows)
    plans: std.AutoHashMapUnmanaged([3]u64, cuda.cublaslt.Linear) = .empty,
    work: u64 = 0,
    x3: cuda.exl3.Dense, // the head's own EXL3 scratch: the target's captured graphs hold theirs
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    b: Bufs = undefined,
    streams: std.AutoHashMapUnmanaged(*Stream, *Ctx) = .empty,
    pool: std.ArrayList(*Ctx) = .empty,
    by_id: std.AutoHashMapUnmanaged(i32, *Ctx) = .empty, // rank 1: rank 0's contexts by id
    next_id: i32 = 1,
    tkh: []u32 = &.{}, // the ranks' top-k slices: values then ids, a rank's after another's
    hp: []f32 = &.{},
    step_ms: f64 = 1.0,
    costs: [fwd.tree_rows]Cost = undefined,
    n_costs: usize = 0,

    /// The drafter in `dir` over the target `m`, this rank's share; rank 0 first has rank 1 open its own.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, m: *fwd.Model, dir: []const u8) !*Draft {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
        defer gpa.free(bytes);
        const full = try ld.Config.parse(gpa, bytes, m.c.hidden, m.c.vocab, m.c.layers);
        const world = m.o.place.world;
        const c = try full.share(world);
        if (m.rank() == 0) {
            var h: fwd.Header = .{ .op = @backingInt(fwd.Op.draft_open) };
            try m.plan(&h);
        }
        const d = try gpa.create(Draft);
        errdefer gpa.destroy(d);
        d.* = .{ .gpa = gpa, .io = io, .m = m, .c = c, .rank = m.rank(), .world = world, .w = undefined, .k = undefined, .dense = undefined, .x3 = undefined };
        d.k = try ld.Kernels.load(m.d);
        errdefer d.k.unload();
        d.dense = try cuda.dense_rows.DenseRows.load(m.d);
        errdefer d.dense.unload();
        d.x3 = try cuda.exl3.Dense.load(m.d);
        errdefer d.x3.unload();
        d.lt = cuda.cublaslt.Library.open() catch |e| blk: {
            std.log.warn("ricursive: drafter without cuBLASLt ({s}): dense rows", .{@errorName(e)});
            break :blk null;
        };
        d.w = try ld.Weights.load(gpa, io, m.d, dir, full, .{ .rank = d.rank, .world = world });
        errdefer d.w.deinit();
        errdefer for (d.bufs.items) |*x| x.free();
        try d.alloc();
        d.work = try d.buf(work_bytes);
        d.tkh = try gpa.alloc(u32, world * d.tkWords(c.block * hold_streams));
        d.hp = try gpa.alloc(f32, d.c.block * hold_streams * c.rank);
        std.log.info("ricursive: DFlash2 drafter, rank {d} of {d}: {d} layers, block {d}, conv {d}, top-{d}, {d:.2} GiB of BF16 weights", .{ d.rank, world, c.layers, c.block, c.kernel, c.topk, @as(f64, @floatFromInt(d.w.bytes)) / (1 << 30) });
        return d;
    }

    fn buf(d: *Draft, n: usize) !u64 {
        try d.bufs.ensureUnusedCapacity(d.gpa, 1);
        const b = try cuda.DeviceBuffer.alloc(d.m.d, @max(n, 256));
        d.bufs.appendAssumeCapacity(b);
        return b.ptr;
    }

    fn alloc(d: *Draft) !void {
        const c = d.c;
        const R = fwd.max_rows;
        const B = c.block * hold_streams;
        const h = c.hidden;
        const ch = c.chunks();
        d.b = .{ .taps = try d.buf(R * c.n_taps * h * 2 / d.world), .th = try d.buf(R * h * 4), .thn = try d.buf(R * h * 2), .ka = try d.buf(R * c.kvWidth() * 4), .va = try d.buf(R * c.kvWidth() * 4), .h = try d.buf(B * h * 4), .h2 = try d.buf(B * h * 4), .x = try d.buf(B * h * 2), .xin = try d.buf(B * h * 2), .dyn = try d.buf(B * c.dynamic() * 4), .q = try d.buf(B * c.qWidth() * 4), .kr = try d.buf(B * c.kvWidth() * 4), .vr = try d.buf(B * c.kvWidth() * 4), .qb = try d.buf(B * c.qWidth() * 2), .kb = try d.buf(B * c.kvWidth() * 2), .vb = try d.buf(B * c.kvWidth() * 2), .att = try d.buf(B * c.qWidth() * 2), .o = try d.buf(B * h * 4), .gate = try d.buf(B * c.intermediate * 4), .up = try d.buf(B * c.intermediate * 4), .act = try d.buf(B * c.intermediate * 2), .fin = try d.buf(B * h * 2), .fin16 = try d.buf(B * h * 2), .logits = try d.buf(B * d.m.vr * 4), .hp = try d.buf(B * c.rank * 4), .tk = try d.buf(d.world * d.tkWords(B) * 4), .pm = try d.buf(B * c.heads * ch * 4), .pl = try d.buf(B * c.heads * ch * 4), .po = try d.buf(B * c.heads * ch * 128 * 4), .ids = try d.buf(B * 4), .segs = try d.buf(hold_streams * @sizeOf(ld.Seg)) };
    }

    /// A rank's top-k words for `rows` rows: values then ids over its vocabulary share's slices.
    fn tkWords(d: *const Draft, rows: usize) usize {
        return 2 * rows * ld.slices(d.m.vr) * d.c.topk;
    }

    pub fn deinit(d: *Draft) void {
        d.m.stream.synchronize() catch {};
        var it = d.streams.valueIterator();
        while (it.next()) |x| d.drop(x.*);
        d.streams.deinit(d.gpa);
        for (d.pool.items) |x| d.drop(x);
        d.pool.deinit(d.gpa);
        for (d.bufs.items) |*x| x.free();
        d.bufs.deinit(d.gpa);
        var pit = d.plans.valueIterator();
        while (pit.next()) |l| l.deinit();
        d.plans.deinit(d.gpa);
        if (d.lt) |*l| l.close();
        d.x3.unload();
        var bit = d.by_id.valueIterator();
        while (bit.next()) |x| d.drop(x.*);
        d.by_id.deinit(d.gpa);
        d.gpa.free(d.tkh);
        d.gpa.free(d.hp);
        d.w.deinit();
        d.dense.unload();
        d.k.unload();
        d.gpa.destroy(d);
    }

    fn drop(d: *Draft, x: *Ctx) void {
        x.ring.free();
        d.gpa.destroy(x);
    }

    fn create(d: *Draft) !*Ctx {
        const x = try d.gpa.create(Ctx);
        errdefer d.gpa.destroy(x);
        x.* = .{ .ring = try cuda.DeviceBuffer.alloc(d.m.d, d.c.ringBytes()), .id = d.next_id };
        d.next_id += 1;
        try x.ring.fill8(0, null);
        return x;
    }

    /// Rank 1: rank 0's context `id` (made on first use).
    fn ctxOf(d: *Draft, id: i32) !*Ctx {
        if (d.by_id.get(id)) |x| return x;
        const x = try d.create();
        errdefer d.drop(x);
        x.id = id;
        try d.by_id.put(d.gpa, id, x);
        return x;
    }

    /// `n` contexts ready for the streams the server admits.
    pub fn reserve(d: *Draft, n: usize) !void {
        while (d.pool.items.len < n) try d.pool.append(d.gpa, try d.create());
    }

    /// x [rows, k] bf16 times w [n, k] bf16 into out [rows, n] fp32.
    fn mm(d: *Draft, x: u64, w: u64, out: u64, rows: usize, n: usize, k: usize) !void {
        const lt = if (d.lt) |*l| l else return d.dense.rows(d.m.stream, x, k, w, out, n, rows, n, k);
        const got = try d.plans.getOrPut(d.gpa, .{ rows, n, k });
        if (!got.found_existing) got.value_ptr.* = cuda.cublaslt.Linear.init(lt, rows, n, k, .f32, work_bytes) catch |e| {
            d.plans.removeByPtr(got.key_ptr);
            return e;
        };
        try got.value_ptr.run(x, w, out, d.work, work_bytes, d.m.stream.handle);
    }

    fn ringK(d: *const Draft, x: *Ctx, layer: usize) u64 {
        return x.ring.ptr + layer * 2 * d.c.ringRows() * d.c.kvWidth() * 2;
    }

    /// out = x through a row-cut matrix: this rank's partial, then the ranks' partials added in rank order.
    fn mmSum(d: *Draft, x: u64, w: u64, out: u64, rows: usize, n: usize, k: usize) !void {
        try d.mm(x, w, d.m.b.parts + d.rank * rows * n * 4, rows, n, k);
        try d.m.sumRanks(out, rows * n * 4);
    }

    /// Rank 0: target rows [start, start + n) of tap slot `slot` into the context (past the widest window dropped).
    fn absorb(d: *Draft, x: *Ctx, slot: usize, start_: u64, n_: usize) !void {
        const most = d.c.ringRows() - d.c.block;
        var start = start_;
        var n = n_;
        if (n == 0) return;
        if (start < x.end) x.end = start;
        if (start > x.end or x.begin > start) x.begin = start; // a gap (a resumed prompt): the context starts here
        if (n > most) {
            start += n - most;
            n = most;
            x.begin = start;
        }
        while (n > 0) {
            const rows = @min(n, fwd.max_rows);
            if (start + rows > x.begin + most) x.begin = start + rows - most;
            var h: fwd.Header = .{ .op = @backingInt(fwd.Op.draft_absorb), .slot = x.id, .a = @intCast(slot), .p0 = @intCast(start), .n = @intCast(rows), .b = @intCast(x.begin) };
            try d.m.plan(&h);
            try d.absorbRows(x, slot, start, rows);
            start += rows;
            n -= rows;
        }
    }

    /// Every rank: tap rows [start, start + rows) of `slot` through fc and each layer's keys and values into x's rings.
    fn absorbRows(d: *Draft, x: *Ctx, slot: usize, start: u64, rows: usize) !void {
        const c = d.c;
        const s = d.m.stream;
        const stride = c.n_taps * c.hidden;
        const kc = stride / d.world;
        try d.k.gather(s, d.m.tap_ring + slot * fwd.tap_rows * stride * 2, fwd.tap_rows, start, stride, d.rank * kc, kc, d.b.taps, rows);
        try d.mmSum(d.b.taps, d.w.fc, d.b.th, rows, c.hidden, kc);
        try d.k.rms(s, d.b.th, d.w.hidden_norm, d.b.thn, c.hidden, c.eps, 0, 0, rows);
        for (d.w.layers[0..c.layers], 0..) |l, i| {
            try d.mm(d.b.thn, l.k, d.b.ka, rows, c.kvWidth(), c.hidden);
            try d.mm(d.b.thn, l.v, d.b.va, rows, c.kvWidth(), c.hidden);
            const rk = d.ringK(x, i);
            try d.k.kvPut(s, d.b.ka, d.b.va, l.k_norm, rk, rk + c.ringRows() * c.kvWidth() * 2, c, start, c.rope[i], rows);
        }
        x.end = start + rows;
    }

    /// Rank 0: one block a stream (pending then masks) on every rank; each stream's chain into its context.
    fn hold(d: *Draft, items: []const dr.Hold, ctxs: []const *Ctx) !void {
        var h: fwd.Header = .{ .op = @backingInt(fwd.Op.draft_hold), .ns = @intCast(items.len) };
        for (items, ctxs, 0..) |it, x, i| {
            const p = it.position - 1; // the pending token's position: the context holds every row before it
            if (x.end > p) x.end = p;
            if (x.begin > x.end) x.begin = x.end;
            h.streams[i] = .{ x.id, @intCast(it.pending), @intCast(p), @intCast(x.begin) };
        }
        try d.m.plan(&h);
        try d.holdRun(&h, ctxs);
        const c = d.c;
        const rows = items.len * (c.block - 1);
        try d.m.stream.synchronize();
        const tw = d.tkWords(rows);
        try (cuda.DeviceBuffer{ .d = d.m.d, .ptr = d.b.tk, .len = d.world * tw * 4 }).download(0, std.mem.sliceAsBytes(d.tkh[0 .. d.world * tw]));
        try (cuda.DeviceBuffer{ .d = d.m.d, .ptr = d.b.hp, .len = rows * c.rank * 4 }).download(0, std.mem.sliceAsBytes(d.hp[0 .. rows * c.rank]));
        for (items, ctxs, 0..) |it, x, i| {
            d.chain(rows, i * (c.block - 1), it.pending, x.held[0 .. c.block - 1]);
            x.held_n = c.block - 1;
        }
    }

    /// Every rank: the plan's blocks through its share; the head's top-k slices gathered (rank 0: the selector too).
    fn holdRun(d: *Draft, hw: *const fwd.Header, ctxs: []const *Ctx) !void {
        const c = d.c;
        const m = d.m;
        const s = m.stream;
        const b = &d.b;
        const ns: usize = @intCast(hw.ns);
        const B = ns * c.block;
        var ids: [hold_streams * max_block]i32 = undefined;
        var segs: [hold_streams]ld.Seg = undefined;
        for (0..ns) |i| {
            const e = hw.streams[i];
            ids[i * c.block] = e[1];
            @memset(ids[i * c.block + 1 ..][0 .. c.block - 1], @intCast(c.mask));
            segs[i] = .{ .ring = ctxs[i].ring.ptr, .begin = e[3], .position = e[2] };
        }
        try s.synchronize();
        try m.up(b.ids, std.mem.sliceAsBytes(ids[0..B]));
        try m.up(b.segs, std.mem.sliceAsBytes(segs[0..ns]));
        const h = c.hidden;
        try d.k.embed(s, b.ids, m.w.embed, b.h, h, B);
        for (d.w.layers[0..c.layers], 0..) |l, i| {
            try d.k.rms(s, b.h, l.input_norm, b.x, h, c.eps, 0, 0, B);
            try d.mm(b.x, l.attn_conv, b.dyn, B, c.dynamic(), h);
            try d.k.convIn(s, b.x, b.dyn, l.attn_base, b.xin, c, B);
            try d.mm(b.xin, l.q, b.q, B, c.qWidth(), h);
            try d.mm(b.xin, l.k, b.kr, B, c.kvWidth(), h);
            try d.mm(b.xin, l.v, b.vr, B, c.kvWidth(), h);
            try d.k.heads(s, b.q, c.qWidth(), l.q_norm, b.qb, c.heads, b.segs, c, c.rope[i], B);
            try d.k.heads(s, b.kr, c.kvWidth(), l.k_norm, b.kb, c.kv_heads, b.segs, c, c.rope[i], B);
            try d.k.f2b(s, b.vr, b.vb, B * c.kvWidth());
            try d.k.attention(s, b.qb, b.kb, b.vb, b.segs, c, i, .{ b.pm, b.pl, b.po }, b.att, B);
            try d.mmSum(b.att, l.o, b.o, B, h, c.qWidth());
            try d.k.convOut(s, b.o, b.dyn, l.attn_base, b.h, b.h2, c, B);
            try d.k.rms(s, b.h2, l.post_norm, b.x, h, c.eps, 0, 0, B);
            try d.mm(b.x, l.mlp_conv, b.dyn, B, c.dynamic(), h);
            try d.k.convIn(s, b.x, b.dyn, l.mlp_base, b.xin, c, B);
            try d.mm(b.xin, l.gate, b.gate, B, c.intermediate, h);
            try d.mm(b.xin, l.up, b.up, B, c.intermediate, h);
            try d.k.silu(s, b.gate, b.up, b.act, B * c.intermediate);
            try d.mmSum(b.act, l.down, b.o, B, h, c.intermediate);
            try d.k.convOut(s, b.o, b.dyn, l.mlp_base, b.h2, b.h, c, B);
        }
        const rows = ns * (c.block - 1); // each block's mask rows: its drafts
        try d.k.rms(s, b.h, d.w.final_norm, b.fin16, h, c.eps, 1, c.block, rows);
        try d.x3.prompt(s, m.w.head, b.fin16, .f16, rows, b.logits, .f32); // this rank's vocabulary share
        const tw = d.tkWords(rows);
        const mine = b.tk + d.rank * tw * 4;
        try d.k.topk(s, b.logits, m.vr, c.topk, mine, mine + tw * 2, rows);
        try m.group.allGather(s, mine, b.tk, tw * 4);
        if (d.rank == 0) {
            try d.k.rms(s, b.h, d.w.final_norm, b.fin, h, c.eps, 0, c.block, rows);
            try d.mm(b.fin, d.w.selector, b.hp, rows, c.rank, h);
        }
    }

    fn bf(v: u16) f32 {
        return @bitCast(@as(u32, v) << 16);
    }

    /// Rows row0.. of the last hold (`rows` in all): each row's best candidate after the one before (the selector).
    fn chain(d: *const Draft, rows: usize, row0: usize, pending: u32, out: []u32) void {
        const c = d.c;
        const sr = ld.slices(d.m.vr) * c.topk; // a rank's candidates a row
        const tw = d.tkWords(rows);
        var pred = pending;
        for (out, 0..) |*o, r| {
            var vals: [2 * ld.slices_most * ld.most_topk]f32 = undefined;
            var toks: [2 * ld.slices_most * ld.most_topk]i32 = undefined;
            const per = d.world * sr;
            for (0..d.world) |q| { // rank q's slices: its values, then its ids (offset to the whole vocabulary)
                const base = q * tw + (row0 + r) * sr;
                for (0..sr) |j| {
                    vals[q * sr + j] = @bitCast(d.tkh[base + j]);
                    const t: i32 = @bitCast(d.tkh[base + tw / 2 + j]);
                    toks[q * sr + j] = if (t < 0) -1 else t + @as(i32, @intCast(q * d.m.vr));
                }
            }
            const hp = d.hp[(row0 + r) * c.rank ..][0..c.rank];
            var order: [2 * ld.slices_most * ld.most_topk]u16 = undefined;
            for (order[0..per], 0..) |*e, i| e.* = @intCast(i);
            const By = struct {
                v: []const f32,
                t: []const i32,
                fn lt(x: @This(), a: u16, b: u16) bool {
                    if (x.t[a] < 0 or x.t[b] < 0) return x.t[b] < 0 and x.t[a] >= 0;
                    return x.v[a] > x.v[b] or (x.v[a] == x.v[b] and x.t[a] < x.t[b]);
                }
            };
            std.sort.pdq(u16, order[0..per], By{ .v = vals[0..per], .t = toks[0..per] }, By.lt);
            var best_score = -std.math.inf(f32);
            var best: u32 = pending;
            for (order[0..c.topk]) |e| {
                if (toks[e] < 0) break;
                const cand: usize = @intCast(toks[e]);
                var edge: f32 = 0;
                for (0..c.rank) |k| edge += bf(d.w.predecessor[pred * c.rank + k]) * hp[k] * bf(d.w.successor[cand * c.rank + k]);
                const score = vals[e] + edge;
                if (score > best_score) {
                    best_score = score;
                    best = @intCast(cand);
                }
            }
            o.* = best;
            pred = best;
        }
    }

    /// The target's window ms by rows (one stream, slot 0 before any request) and a hold's, for the round loop.
    pub fn time(d: *Draft) !void {
        var toks: [fwd.tree_rows]u32 = undefined;
        for (&toks, 0..) |*t, i| t.* = @intCast(1000 + i);
        for (1..fwd.tree_rows + 1) |w| {
            var best = std.math.inf(f64);
            for (0..3) |_| {
                const t0 = std.Io.Clock.awake.now(d.io);
                try d.m.decode(&.{.{ .slot = 0, .p0 = 0, .tokens = toks[0..w] }});
                try d.m.stream.synchronize();
                best = @min(best, ms(d.io, t0));
            }
            d.costs[w - 1] = .{ .width = @intCast(w), .ms = best };
        }
        d.n_costs = fwd.tree_rows;
        const x = try d.create();
        var best = std.math.inf(f64);
        for (0..4) |_| {
            x.* = .{ .ring = x.ring, .id = x.id, .end = 1 };
            const t0 = std.Io.Clock.awake.now(d.io);
            try d.hold(&.{.{ .stream = undefined, .pending = 1000, .position = 2, .depth = @intCast(d.c.block - 1) }}, &.{x});
            best = @min(best, ms(d.io, t0));
        }
        try d.pool.append(d.gpa, x);
        d.step_ms = best / @as(f64, @floatFromInt(d.c.block - 1));
        std.log.info("ricursive: target windows 1 / 2 / 16 rows {d:.1} / {d:.1} / {d:.1} ms, a draft block {d:.1} ms", .{ d.costs[0].ms, d.costs[1].ms, d.costs[15].ms, best });
    }

    fn ms(io: std.Io, t0: std.Io.Timestamp) f64 {
        return @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6;
    }

    // the Drafter contract

    pub fn drafter(d: *Draft) dr.Drafter {
        return .{ .kind = "DFlash2", .ptr = d, .vtable = &.{ .taps = tapsFn, .facts = factsFn, .open = openFn, .absorb = absorbFn, .hold = holdFn, .held = heldFn, .release = releaseFn } };
    }

    fn cast(p: *anyopaque) *Draft {
        return @ptrCast(@alignCast(p));
    }

    fn tapsFn(p: *anyopaque) []const u32 {
        const d = cast(p);
        return d.c.taps[0..d.c.n_taps];
    }

    fn factsFn(p: *anyopaque) dr.Facts {
        const d = cast(p);
        return .{ .depth = @intCast(d.c.block - 1), .step_ms = d.step_ms, .prior = &prior, .plain_guard = true, .batched = true, .stream_bytes = d.c.ringBytes(), .window_costs = d.costs[0..d.n_costs], .lane_costs = d.costs[0..d.n_costs] };
    }

    fn openFn(p: *anyopaque, s: *Stream) anyerror!void {
        const d = cast(p);
        if (d.streams.contains(s)) return error.StreamOpen;
        const x = d.pool.pop() orelse try d.create();
        errdefer d.pool.append(d.gpa, x) catch d.drop(x);
        x.* = .{ .ring = x.ring };
        try d.streams.put(d.gpa, s, x);
    }

    fn absorbFn(p: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const d = cast(p);
        const most = d.c.ringRows() - d.c.block;
        for (items) |it| {
            const x = d.streams.get(it.stream) orelse return error.UnknownStream;
            if (it.features.space != .device or it.features.dtype != .bf16) return error.UnsupportedFeatures;
            if (it.rows != null) return error.ChainsOnly;
            var start = it.features.offset;
            var n: u64 = it.features.rows;
            if (it.prompt) { // a prompt's rows past the drafter's widest window are never read: skip them
                const plen: u64 = it.stream.prompt_len;
                const from = if (plen > most + 1) plen - 1 - most else 0;
                if (start + n <= from) {
                    x.begin = start + n;
                    x.end = start + n;
                    continue;
                }
                if (start < from) {
                    n -= from - start;
                    start = from;
                }
            }
            const row = fwd.tap_rows * d.c.n_taps * d.c.hidden * 2;
            try d.absorb(x, @intCast((it.features.buffer - d.m.tap_ring) / row), start, @intCast(n));
        }
    }

    fn holdFn(p: *anyopaque, items: []const dr.Hold) anyerror!void {
        const d = cast(p);
        var at: usize = 0;
        while (at < items.len) {
            const n = @min(hold_streams, items.len - at);
            var ctxs: [hold_streams]*Ctx = undefined;
            for (items[at..][0..n], ctxs[0..n]) |it, *x| x.* = d.streams.get(it.stream) orelse return error.UnknownStream;
            try d.hold(items[at..][0..n], ctxs[0..n]);
            at += n;
        }
    }

    fn heldFn(p: *anyopaque, streams: []const *Stream, out: []const []u32) anyerror!void {
        const d = cast(p);
        for (streams, out) |s, o| {
            const x = d.streams.get(s) orelse return error.UnknownStream;
            if (o.len > x.held_n) return error.TooFewDrafts;
            @memcpy(o, x.held[0..o.len]);
        }
    }

    fn releaseFn(p: *anyopaque, s: *Stream) void {
        const d = cast(p);
        if (d.streams.fetchRemove(s)) |kv| d.pool.append(d.gpa, kv.value) catch d.drop(kv.value);
    }
};

/// Rank 1's side of rank 0's drafter plan words (Model.draft_hook): opens its share, absorbs and holds by context id.
pub fn follow(m: *fwd.Model, h: *const fwd.Header) anyerror!void {
    if (h.op == @backingInt(fwd.Op.draft_open)) {
        const dir = m.o.drafter orelse return error.RicursiveNoDrafterDir;
        if (m.draft_ptr != null) return error.RicursiveDrafterOpen;
        m.draft_ptr = try Draft.init(m.gpa, m.io, m, dir);
        return;
    }
    const d: *Draft = @ptrCast(@alignCast(m.draft_ptr orelse return error.RicursiveNoDrafter));
    if (h.op == @backingInt(fwd.Op.draft_absorb)) {
        const x = try d.ctxOf(h.slot);
        x.begin = @intCast(h.b);
        return d.absorbRows(x, @intCast(h.a), @intCast(h.p0), @intCast(h.n));
    }
    if (h.ns <= 0 or h.ns > hold_streams) return error.RicursiveRows;
    var ctxs: [hold_streams]*Ctx = undefined;
    for (ctxs[0..@intCast(h.ns)], 0..) |*x, i| x.* = try d.ctxOf(h.streams[i][0]);
    try d.holdRun(h, ctxs[0..@intCast(h.ns)]);
}

/// Acceptance by depth until a stream has its own (a prior: the round loop replaces it with what it measures).
const prior = [_]f64{ 0.8, 0.7, 0.62, 0.56, 0.5, 0.46, 0.42, 0.39, 0.36, 0.34, 0.32, 0.3, 0.28, 0.27, 0.26 };

test "the drafter's paths compile on the host" {
    _ = &Draft.init;
    _ = &Draft.drafter;
    _ = &Draft.time;
    _ = &Draft.reserve;
    _ = &Draft.deinit;
    _ = &@import("tp2_native.zig").openDrafter;
}
