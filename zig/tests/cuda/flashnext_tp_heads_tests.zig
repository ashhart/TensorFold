//! tf-cuda-test flashnext-tp-heads MODEL: each rank's cut in-projections and attention heads, the whole run's bytes.
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("check.zig");
const rig = @import("flashnext_rig.zig");

const cuts = flash.projection_load;
const hc = flash.heads_cut;
const Heads = flash.Heads;
const Kernels = flash.prompt.batch.Kernels;
/// The family's checkpoint reader (cuda_prompt_wt.zig Store).
const Store = @FieldType(flash.prompt.Engine, "store");
const Weight = cuda.qmmf.Weight;

const D = 2560; // dims
const most = 1536; // the default chunk
/// Decode rows, a stage of 16, the prompt tile's first row count and the default chunk.
const counts = [_]usize{ 1, 4, 16, 64, most };
const head_dim = 256;
const index_heads = 4;
const index_dim = 128;
const eps: f32 = 1e-6;
const cap = 4096; // the attention cache's positions
const idw = 2052; // a sparse row's key list (cuda_prompt_wt.zig index_idw)
const dense_keys = 2051; // keys a row reads densely at most (index_top + 1 blocks of 4, less one)
const nch = 5; // a sparse row's 512-key chunks
const att_rows = flash.batch_state.att_rows;
const fused_rows = flash.batch_state.fused_rows;

const Ctx = struct {
    r: *rig.Rig,
    k: *const Kernels,
    g: *const flash.Glue,
    store: *Store,
    part: cuda.DeviceBuffer,
    checks: usize = 0,

    /// A packed projection's weight, its buffer freed with the rig's.
    fn keep(c: *Ctx, p: flash.projection.Projection) !Weight {
        try c.r.bufs.append(c.r.gpu.gpa, p.w);
        return p.native.?;
    }

    /// One lane call as the pass makes it (bf16 rows), or its fp32 rows, into a buffer of 0xff bytes.
    fn lane(c: *Ctx, comptime T: type, x: u64, m: usize, w: Weight) ![]T {
        const dst = try c.r.out(m * w.n * @sizeOf(T));
        try c.k.native_lane.matmulOut(c.r.s, x, w.k, m, w, dst, if (T == f32) .f32 else .bf16, c.part.ptr);
        return c.r.back(T, dst, m * w.n);
    }

    fn same(c: *Ctx, comptime fmt: []const u8, args: anytype, got: []const u8, want: []const u8) !void {
        try check.sameBytes(try std.fmt.allocPrint(c.r.a, fmt, args), got, want);
        c.checks += 1;
    }
};

/// Rank `mine`'s kept columns of whole rows [m, whole.n]: each part's ranges in order, as the cut's rows hold them.
fn kept(comptime T: type, a: std.mem.Allocator, rows: []const T, m: usize, whole: hc.Stitch, mine: hc.Stitch) ![]T {
    const out = try a.alloc(T, m * mine.n);
    for (0..m) |row| {
        var at: usize = 0;
        var base: usize = 0;
        for (0..4) |p| {
            for (mine.ranges[p].items()) |c| {
                @memcpy(out[row * mine.n + at ..][0..c.width()], rows[row * whole.n + base + c.from ..][0..c.width()]);
                at += c.width();
            }
            base += whole.ranges[p].rows();
        }
    }
    return out;
}

/// Activations: BF16 of a unit normal.
fn inputs(r: *rig.Rig, len: usize) ![]u16 {
    const v = try r.a.alloc(u16, len);
    for (v) |*h| h.* = rig.bf16(r.rng.floatNorm(f32));
    return v;
}

/// One class's in-projection: each rank's pinned cut against the whole one's kept rows; returns the whole's bf16 rows.
fn projection(c: *Ctx, comptime what: []const u8, prefix: []const u8, parts: []const []const u8, whole: hc.Stitch, mine: [2]hc.Stitch, x: u64) ![]const u16 {
    const r = c.r;
    const all = try c.keep(try cuts.loadStitched(r.gpu.gpa, @constCast(r.gpu.d), c.store, prefix, parts, whole.n, D));
    var ranks: [2]Weight = undefined;
    var loose: [2]Weight = undefined;
    for (mine, &ranks, &loose) |s, *w, *l| {
        w.* = try c.keep(try cuts.loadStitchedCut(r.gpu.gpa, r.gpu.d, c.store, prefix, parts, &s.ranges, s.n, D, s.sk));
        try check.expect(w.sk == whole.sk and w.n == s.n, what ++ ": the cut keeps the whole shape's {d} K slices", .{whole.sk});
        // the witness: the same rows packed without the pin, so its K slices follow its own shape
        const bytes = try cuts.stitchCut(r.a, c.store, prefix, parts, &s.ranges, s.n, D);
        l.* = try c.keep(try flash.projection.dense(r.gpu.gpa, r.gpu.d, .{ .dtype = .bf16, .rank = 2, .shape = .{ s.n, D, 1, 1, 1 }, .bytes = bytes }));
        try check.expect(cuda.qmmf.planFor(most, l.*, false).sk != cuda.qmmf.planFor(most, w.*, false).sk, what ++ ": the unpinned cut slices K its own way", .{});
    }
    var rows: []const u16 = &.{};
    for (counts) |m| {
        const want16 = try c.lane(u16, x, m, all);
        const want32 = try c.lane(f32, x, m, all);
        if (m == most) rows = want16;
        for (mine, ranks, loose, 0..) |s, w, l, rank| {
            const k16 = try kept(u16, r.a, want16, m, whole, s);
            const k32 = try kept(f32, r.a, want32, m, whole, s);
            try c.same(what ++ " rank {d}, {d} rows: bf16 rows", .{ rank, m }, std.mem.sliceAsBytes(try c.lane(u16, x, m, w)), std.mem.sliceAsBytes(k16));
            try c.same(what ++ " rank {d}, {d} rows: fp32 rows", .{ rank, m }, std.mem.sliceAsBytes(try c.lane(f32, x, m, w)), std.mem.sliceAsBytes(k32));
            if (m != most) continue;
            const off = try c.lane(f32, x, m, l);
            try check.expect(!std.mem.eql(f32, off, k32), what ++ " rank {d}: the unpinned witness equals the whole rows, so the test cannot fail", .{rank});
        }
    }
    check.pass("flashnext-tp-heads {s}: each rank's {d} rows equal the whole {d}-row projection's kept rows byte for byte " ++
        "(bf16 and fp32) at 1 to 1,536 rows; the unpinned cut differs", .{ what, mine[0].n, whole.n });
    return rows;
}

/// RoPE's inverse frequencies as the pass uploads them (base 10,000,000 over 32 pairs).
fn inverse(r: *rig.Rig) ![]f32 {
    const inv = try r.a.alloc(f32, 32);
    for (inv, 0..) |*o, i| o.* = @floatCast(std.math.pow(f64, 10_000_000.0, -@as(f64, @floatFromInt(i)) / 32));
    return inv;
}

/// Head `kv` of every position of a whole cache [cap, 2, 256]: a rank's own cache [cap, 1, 256].
fn headOf(a: std.mem.Allocator, whole: []const u16, kv: usize) ![]u16 {
    const out = try a.alloc(u16, cap * head_dim);
    for (0..cap) |p| @memcpy(out[p * head_dim ..][0..head_dim], whole[(p * 2 + kv) * head_dim ..][0..head_dim]);
    return out;
}

/// Heads [from, from + n) of each of `m` rows of `per`-head rows `width` wide.
fn heads(a: std.mem.Allocator, comptime T: type, rows: []const T, m: usize, per: usize, from: usize, n: usize, width: usize) ![]T {
    const out = try a.alloc(T, m * n * width);
    for (0..m) |row| @memcpy(out[row * n * width ..][0 .. n * width], rows[(row * per + from) * width ..][0 .. n * width]);
    return out;
}

/// The layer's norm weights and RoPE table on the device, one cache's initial bytes, and the rows' key lists.
const Fixed = struct { qw: u64, kw: u64, iw: u64, inv: u64, kc: []const u16, vc: []const u16, ikc: []const u16, ids: u64, nk: u64, spr: u64 };

/// What one run leaves: queries, caches, indexer queries, merged and fused rows, gated rows and their sums.
const Run = struct { q: []u16, kc: []u16, vc: []u16, iq: []u16, ikc: []u16, merged: []u16, fused: []u16, gated: []u16, sums: []f32 };

/// The pass's attention for `m` rows at the end of the cache with heads `h`: prep, chunks then merge, fused, gate.
fn attend(c: *Ctx, f: Fixed, h: Heads, pa: []const u16, m: usize, kc0: []const u16, vc0: []const u16) !Run {
    const r = c.r;
    const g = c.g;
    const s = r.s;
    const pos0 = cap - m;
    const qr = h.q * head_dim * 2;
    // a 16-byte position record a block of att_rows rows, as the pass stages them
    const blocks = (m + att_rows - 1) / att_rows;
    const recs = try r.a.alloc(i32, 4 * blocks);
    @memset(recs, 0);
    for (0..blocks) |b| recs[4 * b] = @intCast(pos0 + b * att_rows);
    const pos = try r.dev(std.mem.sliceAsBytes(recs));
    const dpa = try r.dev(std.mem.sliceAsBytes(pa));
    const q = try r.out(m * qr);
    const iq = try r.out(m * index_heads * index_dim * 2);
    const kc = try r.dev(std.mem.sliceAsBytes(kc0));
    const vc = try r.dev(std.mem.sliceAsBytes(vc0));
    const ikc = try r.dev(std.mem.sliceAsBytes(f.ikc));
    try g.attnPrep(s, dpa, pos, f.qw, f.kw, f.iw, f.inv, q, kc, vc, iq, ikc, eps, m, h, .{});
    const po = try r.out(att_rows * nch * h.q * head_dim * 4);
    const pm = try r.out(att_rows * nch * h.q * 4);
    const pl = try r.out(att_rows * nch * h.q * 4);
    const merged = try r.out(m * qr);
    var r0: usize = 0;
    while (r0 < m) : (r0 += att_rows) {
        const n = @min(att_rows, m - r0);
        const at = pos + (r0 / att_rows) * 16;
        const chunks = (@min(pos0 + r0 + n, dense_keys) + 511) / 512;
        try g.attnChunks(s, q + r0 * qr, kc, vc, at, po, pm, pl, f.ids + r0 * idw * 4, f.nk + r0 * 4, f.spr + r0 * 4, n, h, chunks, true, .{});
        try g.attnMerge(s, po, pm, pl, at, merged + r0 * qr, f.nk + r0 * 4, f.spr + r0 * 4, n, h, true);
    }
    const fused = try r.out(m * qr);
    var g0: usize = 0;
    while (g0 < m) : (g0 += fused_rows) {
        const n = @min(fused_rows, m - g0);
        try g.attnFused(s, q + g0 * qr, kc, vc, pos + (g0 / att_rows) * 16, fused + g0 * qr, f.ids + g0 * idw * 4, f.nk + g0 * 4, f.spr + g0 * 4, n, h, .{});
    }
    const gated = try r.out(m * qr);
    const sums = try r.out(m * h.q * head_dim / 32 * 4);
    try g.attnGate(s, merged, dpa, gated, sums, m, h.projN(), h.q, head_dim);
    return .{
        .q = try r.back(u16, q, m * h.q * head_dim),
        .kc = try r.back(u16, kc, cap * h.kv * head_dim),
        .vc = try r.back(u16, vc, cap * h.kv * head_dim),
        .iq = try r.back(u16, iq, m * index_heads * index_dim),
        .ikc = try r.back(u16, ikc, cap * index_dim),
        .merged = try r.back(u16, merged, m * h.q * head_dim),
        .fused = try r.back(u16, fused, m * h.q * head_dim),
        .gated = try r.back(u16, gated, m * h.q * head_dim),
        .sums = try r.back(f32, sums, m * h.q * head_dim / 32),
    };
}

/// fn_attn_prep as Glue.attnPrep launches it, but with the row width `pw` given (the witness's whole layout's).
fn prepAt(c: *Ctx, f: Fixed, h: Heads, pw: usize, pa: u64, pos: u64, q: u64, kc: u64, vc: u64, iq: u64, ikc: u64, m: usize) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ pa, pos, f.qw, f.kw, f.iw, f.inv, q, kc, vc, iq, ikc }) |v| a.add(v);
    a.add(eps);
    for ([_]usize{ pw, h.q, h.kv, index_heads, index_dim, 32 }) |v| a.add(@as(c_int, @intCast(v)));
    try cuda.launch.launch(c.g.f.prep, .{ .grid = .{ .x = @intCast(m), .y = h.q + h.kv + index_heads + 1 }, .block = .{ .x = head_dim } }, c.r.s, &a);
}

/// Attention at 12 queries and one KV head a rank against the same heads of the whole layer's run.
fn attention(c: *Ctx, rows: []const u16, whole: hc.Stitch, mine: [2]hc.Stitch) !void {
    const r = c.r;
    var buf: [120]u8 = undefined;
    const root = "language_model.model.layers.3.self_attn";
    var norms: [3]u64 = undefined;
    for ([_][]const u8{ ".q_norm.weight", ".k_norm.weight", ".indexer.q_layernorm.weight" }, &norms) |name, *d| {
        const w = try c.store.normScale(r.a, try c.store.tensor(try std.fmt.bufPrint(&buf, "{s}{s}", .{ root, name })));
        d.* = try r.dev(std.mem.sliceAsBytes(w));
    }
    // every row's key list as the selection leaves it past 2,048 keys: one list both ranks read (the indexer is whole)
    const ids = try r.a.alloc(i32, most * idw);
    const nk = try r.a.alloc(i32, most);
    const spr = try r.a.alloc(i32, most);
    const kc = r.bfs(cap * 2 * head_dim, 1);
    const vc = r.bfs(cap * 2 * head_dim, 1);
    const f0: Fixed = .{ .qw = norms[0], .kw = norms[1], .iw = norms[2], .inv = try r.dev(std.mem.sliceAsBytes(try inverse(r))), .kc = kc, .vc = vc, .ikc = r.bfs(cap * index_dim, 1), .ids = 0, .nk = 0, .spr = 0 };
    const full = Heads.full;
    for ([_]usize{ 1, most }) |m| {
        const pos0 = cap - m;
        for (0..m) |row| {
            nk[row] = @intCast(idw - 1 - row % 3);
            spr[row] = 1;
            for (0..idw) |i| ids[row * idw + i] = @intCast((i * 1237 + row * 31) % (pos0 + row + 1));
        }
        var f = f0;
        f.ids = try r.dev(std.mem.sliceAsBytes(ids[0 .. m * idw]));
        f.nk = try r.dev(std.mem.sliceAsBytes(nk[0..m]));
        f.spr = try r.dev(std.mem.sliceAsBytes(spr[0..m]));
        const pa = rows[0 .. m * whole.n];
        const want = try attend(c, f, full, pa, m, kc, vc);
        for (mine, 0..) |s, rank| {
            const h = Heads.half;
            const got = try attend(c, f, h, try kept(u16, r.a, pa, m, whole, s), m, try headOf(r.a, kc, rank), try headOf(r.a, vc, rank));
            const fmt = "flashnext-tp-heads attention rank {d}, {d} rows: {s}";
            const theirs = [_][]const u16{ try heads(r.a, u16, want.q, m, full.q, 12 * rank, 12, head_dim), try heads(r.a, u16, want.merged, m, full.q, 12 * rank, 12, head_dim), try heads(r.a, u16, want.fused, m, full.q, 12 * rank, 12, head_dim), try heads(r.a, u16, want.gated, m, full.q, 12 * rank, 12, head_dim) };
            for ([_][]const u16{ got.q, got.merged, got.fused, got.gated }, theirs, [_][]const u8{ "queries", "merged rows", "fused rows", "gated rows" }) |mine_rows, w, what|
                try c.same(fmt, .{ rank, m, what }, std.mem.sliceAsBytes(mine_rows), std.mem.sliceAsBytes(w));
            try c.same(fmt, .{ rank, m, "gate sums" }, std.mem.sliceAsBytes(got.sums), std.mem.sliceAsBytes(try heads(r.a, f32, want.sums, m, full.q, 12 * rank, 12, head_dim / 32)));
            try c.same(fmt, .{ rank, m, "cache keys" }, std.mem.sliceAsBytes(got.kc), std.mem.sliceAsBytes(try headOf(r.a, want.kc, rank)));
            try c.same(fmt, .{ rank, m, "cache values" }, std.mem.sliceAsBytes(got.vc), std.mem.sliceAsBytes(try headOf(r.a, want.vc, rank)));
            try c.same(fmt, .{ rank, m, "indexer queries" }, std.mem.sliceAsBytes(got.iq), std.mem.sliceAsBytes(want.iq));
            try c.same(fmt, .{ rank, m, "indexer keys" }, std.mem.sliceAsBytes(got.ikc), std.mem.sliceAsBytes(want.ikc));
        }
        if (m == 1) continue;
        try witness(c, f, try kept(u16, r.a, pa[0 .. 16 * whole.n], 16, whole, mine[1]), whole.n, kc, vc);
    }
    check.pass("flashnext-tp-heads attention: queries, cache rows, merged, fused and gated rows of 12 queries and one KV head " ++
        "a rank equal the whole run's same heads byte for byte at 1 and 1,536 rows over a 4,096-token cache; the whole PW differs", .{});
}

/// Prep and gate at 16 rows with the whole layout's PW over a rank's rows in a buffer sized for it: they must differ.
fn witness(c: *Ctx, f: Fixed, local: []const u16, whole_pw: usize, kc: []const u16, vc: []const u16) !void {
    const r = c.r;
    const h = Heads.half;
    const m = 16;
    const right = try attend(c, f, h, local, m, try headOf(r.a, kc, 1), try headOf(r.a, vc, 1));
    const padded = try r.a.alloc(u16, m * whole_pw);
    @memset(padded, 0);
    @memcpy(padded[0..local.len], local);
    const pa = try r.dev(std.mem.sliceAsBytes(padded));
    var recs = [_]i32{ cap - m, 0, 0, 0 };
    const pos = try r.dev(std.mem.sliceAsBytes(&recs));
    const q = try r.out(m * h.q * head_dim * 2);
    const iq = try r.out(m * index_heads * index_dim * 2);
    const kd = try r.dev(std.mem.sliceAsBytes(try headOf(r.a, kc, 1)));
    const vd = try r.dev(std.mem.sliceAsBytes(try headOf(r.a, vc, 1)));
    try prepAt(c, f, h, whole_pw, pa, pos, q, kd, vd, iq, try r.dev(std.mem.sliceAsBytes(f.ikc)), m);
    const gated = try r.out(m * h.q * head_dim * 2);
    const sums = try r.out(m * h.q * head_dim / 32 * 4);
    try c.g.attnGate(r.s, try r.dev(std.mem.sliceAsBytes(right.merged)), pa, gated, sums, m, whole_pw, h.q, head_dim);
    try r.s.synchronize();
    const wq = try r.back(u16, q, m * h.q * head_dim);
    const wg = try r.back(u16, gated, m * h.q * head_dim);
    try check.expect(!std.mem.eql(u16, wq, right.q), "flashnext-tp-heads witness: prep at the whole PW gives the rank's queries", .{});
    try check.expect(!std.mem.eql(u16, wg, right.gated), "flashnext-tp-heads witness: the gate at the whole PW gives the rank's rows", .{});
    // row 0 starts where the rank's own row 0 does, so only the rows after it move
    try check.expect(std.mem.eql(u16, wq[0 .. h.q * head_dim], right.q[0 .. h.q * head_dim]), "flashnext-tp-heads witness: row 0's queries", .{});
}

pub fn run(gpu: check.Gpu, dir: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var rng = std.Random.DefaultPrng.init(0x7f17_4ead);
    var r = try rig.Rig.init(gpu, a, rng.random());
    defer r.deinit();
    var k = try Kernels.load(gpu.d);
    defer k.deinit();
    var g = try flash.Glue.load(gpu.d);
    defer g.deinit();
    // the unpinned witness's two K slices of up to 8,240 fp32 columns a row
    var part = try cuda.DeviceBuffer.alloc(gpu.d, 2 * most * 8240 * 4);
    defer part.free();
    var store = try Store.open(gpu.gpa, gpu.io, dir);
    defer store.deinit();
    try store.detectNorms();
    var c: Ctx = .{ .r = &r, .k = &k, .g = &g, .store = &store, .part = part };
    const x = try r.dev(std.mem.sliceAsBytes(try inputs(&r, most * D)));
    const half = Heads.half;
    _ = try projection(&c, "layer 0 DeltaNet in-projection", "language_model.model.layers.0.linear_attn", &hc.delta_parts, try hc.delta(.full, 0), .{ try hc.delta(half, 0), try hc.delta(half, 1) }, x);
    const whole = try hc.attention(.full, 0);
    const mine = [2]hc.Stitch{ try hc.attention(half, 0), try hc.attention(half, 1) };
    const rows = try projection(&c, "layer 3 attention in-projection", "language_model.model.layers.3.self_attn", &hc.attention_parts, whole, mine, x);
    try attention(&c, rows, whole, mine);
    try outputs(&c, "language_model.model.layers.0.linear_attn.out_proj", D, 6144);
    try outputs(&c, "language_model.model.layers.3.self_attn.o_proj", D, 6144);
    try outputs(&c, "language_model.model.layers.0.mlp.shared_expert.gate_proj", 640, D);
    try outputs(&c, "language_model.model.layers.0.mlp.shared_expert.up_proj", 640, D);
    try outputs(&c, "language_model.model.layers.0.mlp.shared_expert.down_proj", D, 640);
    check.pass("flashnext-tp-heads: {d} byte-for-byte checks of each rank's heads against the whole layer's", .{c.checks});
}

/// Output rows keep the whole projection's input and K slices, including the single-row and prompt paths.
fn outputs(c: *Ctx, prefix: []const u8, outputs_n: usize, width: usize) !void {
    const r = c.r;
    const all = try c.keep(try cuts.load(r.gpu.gpa, @constCast(r.gpu.d), c.store, prefix, outputs_n, width));
    var ranks: [2]Weight = undefined;
    for (&ranks, 0..) |*w, rank| {
        const cut = try flash.modelopt_experts.Cut.init(outputs_n, 2, rank);
        w.* = try c.keep(try cuts.loadCut(r.gpu.gpa, r.gpu.d, c.store, prefix, outputs_n, width, cut, .rows));
        try check.expect(w.sk == cuda.qmmf.splitK(outputs_n, width), "output row cut pins the full K split", .{});
    }
    const values = try inputs(r, most * width);
    for ([_]bool{ false, true }) |nonfinite| {
        if (nonfinite) values[0..3].* = .{ 0x7fc0, 0x7f80, 0xff80 };
        const x = try r.dev(std.mem.sliceAsBytes(values));
        for (counts) |m| {
            const want = try c.lane(f32, x, m, all);
            const got = try r.a.alloc(f32, m * outputs_n);
            for (0..5) |_| {
                for (ranks, 0..) |w, rank| {
                    const mine = try c.lane(f32, x, m, w);
                    for (0..m) |row| @memcpy(got[row * outputs_n + rank * (outputs_n / 2) ..][0 .. outputs_n / 2], mine[row * (outputs_n / 2) ..][0 .. outputs_n / 2]);
                }
                try c.same("{s} output columns, {d} rows, nonfinite={any}", .{ prefix, m, nonfinite }, std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(want));
            }
        }
    }
    check.pass("{s}: each rank's full-input output columns equal one GPU's FP32 bytes, five repeats", .{prefix});
}
