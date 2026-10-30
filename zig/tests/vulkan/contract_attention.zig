// Attention contract, group `attention`: the stage3_shaders pair rope_bf16 (fast_rope.comp, -DUSE_BF16 -DASH_QWEN35) and sdpa_rows_bf16 (sdpa_decode_native.comp, -DBF16_IO -DSDPA_DIM=256 -DSDPA_ROWS) against stage3_fixture.zig's f64 references. Host-checkable; no device code here.
//  rope: stage3_fixture's Rope gate verbatim (f64 ropeRef, budget(2, 6, 1), bit-exact passthrough tail). The fixture owns it; nothing here rederives it.
//  sdpa: the fixture's Attend f64 reference (f64 dots, max, softmax, output over exactly each row's visible keys) generalized to the served shapes -- the fixture's fixed [8]f64 score buffer caps it at 8 visible keys. The f64 operation order is identical, element for element.
//  The bound is derived from the shader text, not asserted. Reading the bf16 arm of sdpa_decode_native.comp, per (head, row) 1024-thread workgroup: - q*scale: alpha = 1/16 = 2^-4, a power of two, so the multiply is exact. - score: ONE f32 accumulator over head_dim = 256 ascending-d FMAs (16-wide tiles with exact-zero tail padding, drained by *1.0f, which is exact): depth 256 at the absolute-terms scale. - max: comparisons and max() only -- zero roundings (NaN propagates). - denominator: NOT a plain f32 sum over the keys. 256 lanes each add ceil(n/256) exp terms sequentially (thread, thread+256, ...), then the 8-level binary tree (stride 128 -> 1) adds the lane partials: depth ceil(n/256) + 8. At n = 7168 that is 36, not 7168; claiming the plain-sum depth would overstate the denominator's bound ~200x. - normalizer 1.0/sum: one division. probs: exp recomputed from the same stored f32 scores (same inputs, same bits as the denominator pass) times the normalizer: one multiply. - output: ONE f32 accumulator over the n visible keys ascending (16-wide tiles, exact-zero tail padding, *1.0f drain): depth n. - one RNE bf16 store (cast.comp's pair). Served shapes (attention.zig, the engine's): decode rows = 1, prefill rows <= 32, visible keys 1..max_visible_keys = 7168 = the shader's kStreamKeys shared-memory bound. Cases sit on the 16-key tile edges, the 256-lane scan edges and the 1024-thread key-stride edges.
const std = @import("std");
const contract = @import("contract");
const c = contract.spec;
pub const fx = @import("stage3_fixture.zig");
const sources = @import("attention_sources");

pub const Report = fx.Report;

// The reduction geometry the depth derivation names, pinned to the shader source by the tests below.
pub const lanes: u64 = 256; // softmax threads: s_tree's width
pub const tree_levels: u64 = 8; // log2(lanes) binary-tree levels
pub const dot_terms: u64 = fx.head_dim; // 256 ascending-d score FMAs
pub const key_tile: u64 = 16; // 16-key output tiles / load batches
pub const key_stride: u64 = 1024; // threads striding the score scan
pub const max_visible_keys: u64 = 7168; // kStreamKeys: the shared-memory stream

// The shader's denominator depth for n visible keys: one sequential chain per lane plus the binary tree. This is the honest depth -- not n.
pub fn softmaxDepth(n: u64) u64 {
    return (n + lanes - 1) / lanes + tree_levels;
}

/// The output accumulation's depth: sequential over the n visible keys.
pub fn outputDepth(n: u64) u64 {
    return n;
}

fn rel(depth: f64) f64 {
    const u: f64 = 0x1p-24;
    return depth * u / (1 - depth * u);
}

/// The reference walk's magnitudes for one case, at the shader's own scales.
pub const Mags = struct {
    /// max over keys of sum_d |alpha*q_d*k_d|: the score dot's term scale.
    terms: f64 = 0,
    /// max |score|: exp's argument is |s - max| <= 2*score.
    score: f64 = 0,
    /// max |value|: probs sum to 1, so the output term scale is this.
    value: f64 = 0,
};

// The stated bound for one output element, from the derivation above. Each stage carries its own depth at its own magnitude scale, exactly the house budget() form; two exp ops absorb the f32 exp's ulps. The final term is the one RNE bf16 store, with the fixture's 2^-8 store constant.
pub fn sdpaBound(n: u64, m: Mags, want: f64) f64 {
    const u: f64 = 0x1p-24;
    const dot = rel(dot_terms) * m.terms; // score dot depth 256
    const arg = dot + u * 2 * m.score; // s - max: subtraction at |arg| <= 2S
    const e_rel = arg + 2 * u; // exp's own f32 error, two ops
    const den_rel = rel(@floatFromInt(softmaxDepth(n))) + e_rel; // chain + tree, NOT n
    const prob_rel = e_rel + den_rel + u; // p = e * (1/D): one multiply
    const work = prob_rel * m.value + rel(@floatFromInt(outputDepth(n))) * m.value;
    return work + 0x1p-8 * (@abs(want) + work) + 0x1p-133;
}

pub const SdpaCase = struct { name: []const u8, rows: u64, prefix: u64 };

// The cache exactly spans the visible keys: attendPlan requires prefix + rows <= capacity and refuses totals past max_visible_keys.
pub fn capacityOf(case: SdpaCase) u64 {
    return case.prefix + case.rows;
}

/// One row's visible key count: the causal bound, prefix + r + 1.
pub fn visibleKeys(case: SdpaCase, r: u64) u64 {
    return case.prefix + r + 1;
}

// The served shapes the engine dispatches (decode rows 1, prefill rows <= 32, visible keys 1..7168), with every reduction edge: - n = 1: the single-key transport corner (softmax is exactly 1.0). - n = 15/16/17: the 16-key tile edge (full tile, one-past tail). - n = 255/256/257: the 256-lane scan edge (chain depth 1 -> 2). - n = 1023/1024/1025: the 1024-thread key-stride edge. - n = 7167/7168: the kStreamKeys shared-memory bound. - rows 17/32 crossing the 1024 edge and filling the 7168 context.
pub const served_cases = [_]SdpaCase{
    .{ .name = "decode n=1", .rows = 1, .prefix = 0 },
    .{ .name = "decode n=16 tile", .rows = 1, .prefix = 15 },
    .{ .name = "decode n=17 tile+1", .rows = 1, .prefix = 16 },
    .{ .name = "decode n=255 scan", .rows = 1, .prefix = 254 },
    .{ .name = "decode n=256 scan", .rows = 1, .prefix = 255 },
    .{ .name = "decode n=257 scan+1", .rows = 1, .prefix = 256 },
    .{ .name = "decode n=1024 stride", .rows = 1, .prefix = 1023 },
    .{ .name = "decode n=1025 stride+1", .rows = 1, .prefix = 1024 },
    .{ .name = "decode n=7167 stream-1", .rows = 1, .prefix = 7166 },
    .{ .name = "decode n=7168 stream", .rows = 1, .prefix = 7167 },
    .{ .name = "prefill rows32 n1..32", .rows = 32, .prefix = 0 },
    .{ .name = "prefill rows17 across 1024", .rows = 17, .prefix = 1007 },
    .{ .name = "prefill rows32 to 7168", .rows = 32, .prefix = 7136 },
};

// The fixture's Attend f64 reference over the served shapes: per query row r and head h, f64 dots (ascending d), f64 max, f64 softmax and f64 output over exactly the n = prefix + r + 1 visible keys, on the cache layout the engine writes: (kv_head, capacity, head_dim). Collects the magnitudes the bound needs. Op order matches fx.Attend.init exactly.
pub fn attendRef(
    a: std.mem.Allocator,
    q_bits: []const u16,
    keys: []const u16,
    values: []const u16,
    case: SdpaCase,
    capacity: u64,
    mags: *Mags,
    want: []f64,
    visible: []usize,
) !void {
    const row_elems: usize = @intCast(fx.heads * fx.head_dim);
    const cap: usize = @intCast(capacity);
    const kh_span: usize = @intCast(fx.heads / fx.kv_heads); // repeats: 8/2
    // Per-(row, head) score scratch; capacity <= 7168 (attendPlan's bound).
    const scores = try a.alloc(f64, cap);
    defer a.free(scores);
    for (0..@intCast(case.rows)) |r| {
        const n: usize = @intCast(visibleKeys(case, @intCast(r)));
        visible[r] = n;
        for (0..@intCast(fx.heads)) |h| {
            const kh = h / kh_span;
            const qrow = q_bits[r * row_elems + h * @as(usize, @intCast(fx.head_dim)) ..][0..@intCast(fx.head_dim)];
            var m: f64 = -std.math.floatMax(f64);
            for (0..n) |j| {
                const kbase = (kh * cap + j) * @as(usize, @intCast(fx.head_dim));
                var dot: f64 = 0;
                var terms: f64 = 0;
                for (0..@intCast(fx.head_dim)) |d| {
                    const qv = c.fromBf16(qrow[d]);
                    const kv = c.fromBf16(keys[kbase + d]);
                    dot += qv * kv;
                    terms += @abs(qv) * @abs(kv);
                }
                scores[j] = dot * fx.scale;
                mags.terms = @max(mags.terms, terms * fx.scale);
                m = @max(m, scores[j]);
                mags.score = @max(mags.score, @abs(scores[j]));
            }
            var denom: f64 = 0;
            for (scores[0..n]) |s| denom += @exp(s - m);
            for (0..@intCast(fx.head_dim)) |d| {
                var acc: f64 = 0;
                for (0..n) |j| {
                    const vbase = (kh * cap + j) * @as(usize, @intCast(fx.head_dim));
                    acc += (@exp(scores[j] - m) / denom) * c.fromBf16(values[vbase + d]);
                }
                want[r * row_elems + h * @as(usize, @intCast(fx.head_dim)) + d] = acc;
            }
        }
    }
}

// One causal SDPA gate over a served shape: bf16 q/k/v shells, the f64 reference from the very bits the GPU will read, and the derived bound. A row whose visible count is 1 has softmax exactly 1.0, so its output is the V row's bits unchanged; check() enforces that exactly.
pub const Sdpa = struct {
    a: std.mem.Allocator,
    case: SdpaCase,
    capacity: u64,
    q_bits: []u16,
    k_bits: []u16,
    v_bits: []u16,
    want: []f64,
    visible: []usize,
    mags: Mags,

    pub fn init(a: std.mem.Allocator, case: SdpaCase, seed: u64) !Sdpa {
        const capacity = capacityOf(case);
        const q_elems: usize = @intCast(case.rows * fx.heads * fx.head_dim);
        const cache_elems: usize = @intCast(fx.kv_heads * capacity * fx.head_dim);
        const q_bits = try a.alloc(u16, q_elems);
        errdefer a.free(q_bits);
        const k_bits = try a.alloc(u16, cache_elems);
        errdefer a.free(k_bits);
        const v_bits = try a.alloc(u16, cache_elems);
        errdefer a.free(v_bits);
        const want = try a.alloc(f64, q_elems);
        errdefer a.free(want);
        const visible = try a.alloc(usize, @intCast(case.rows));
        errdefer a.free(visible);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (q_bits) |*x| x.* = fx.sampleDyadic(rng, -5, 3);
        for (k_bits) |*x| x.* = fx.sampleDyadic(rng, -5, 3);
        for (v_bits) |*x| x.* = fx.sampleDyadic(rng, -5, 3);
        var f: Sdpa = .{
            .a = a,
            .case = case,
            .capacity = capacity,
            .q_bits = q_bits,
            .k_bits = k_bits,
            .v_bits = v_bits,
            .want = want,
            .visible = visible,
            .mags = .{},
        };
        for (f.v_bits) |bits| f.mags.value = @max(f.mags.value, @abs(c.fromBf16(bits)));
        try attendRef(a, f.q_bits, f.k_bits, f.v_bits, case, capacity, &f.mags, f.want, f.visible);
        return f;
    }

    pub fn deinit(f: Sdpa) void {
        f.a.free(f.q_bits);
        f.a.free(f.k_bits);
        f.a.free(f.v_bits);
        f.a.free(f.want);
        f.a.free(f.visible);
    }

    // Every output element against its row's derived bound; single-key rows must carry the V bits exactly. Every finite reference demands a finite output.
    pub fn check(f: Sdpa, bits: []const u16) !Report {
        var r: Report = .{};
        const row_elems: usize = @intCast(fx.heads * fx.head_dim);
        const kh_span: usize = @intCast(fx.heads / fx.kv_heads);
        for (0..@intCast(f.case.rows)) |row| {
            const n = f.visible[row];
            for (0..@intCast(fx.heads)) |h| {
                for (0..@intCast(fx.head_dim)) |d| {
                    const i = row * row_elems + h * @as(usize, @intCast(fx.head_dim)) + d;
                    const got = c.fromBf16(bits[i]);
                    if (!std.math.isFinite(got)) return error.NonFiniteOutput;
                    const w = f.want[i];
                    const bound = sdpaBound(n, f.mags, w);
                    const dist = @abs(got - w);
                    if (dist > bound) return error.OverBound;
                    const ratio = dist / bound;
                    if (ratio > r.worst) r = .{ .worst = ratio, .where = i };
                    if (n == 1) {
                        const vb = f.v_bits[(h / kh_span) * @as(usize, @intCast(f.capacity)) * @as(usize, @intCast(fx.head_dim)) + d];
                        if (bits[i] != vb) return error.SingleKeyNotTransport;
                    }
                }
            }
        }
        return r;
    }
};

const testing = std.testing;

test "softmax depth is the shader's chain plus tree, never the plain key sum" {
    // One strided pass per lane + 8 tree levels at and under one full scan.
    try testing.expectEqual(@as(u64, 9), softmaxDepth(1));
    try testing.expectEqual(@as(u64, 9), softmaxDepth(255));
    try testing.expectEqual(@as(u64, 9), softmaxDepth(256));
    try testing.expectEqual(@as(u64, 10), softmaxDepth(257));
    try testing.expectEqual(@as(u64, 12), softmaxDepth(1024));
    try testing.expectEqual(@as(u64, 13), softmaxDepth(1025));
    try testing.expectEqual(@as(u64, 36), softmaxDepth(7168));
    // The point of the derivation: 36, not 7168.
    try testing.expect(softmaxDepth(max_visible_keys) < max_visible_keys / 100);
}

test "sdpaBound covers a planted wrong finite and grows with the honest depths" {
    const m: Mags = .{ .terms = 128, .score = 64, .value = 8 };
    const b1 = sdpaBound(1, m, 1);
    try testing.expect(b1 > 0);
    try testing.expect(sdpaBound(257, m, 1) > sdpaBound(256, m, 1)); // chain edge
    try testing.expect(sdpaBound(7168, m, 1) > sdpaBound(1025, m, 1)); // output depth
    try testing.expect(sdpaBound(7168, m, 1) < 1 << 20); // 2^31 plants are refused
    try testing.expect(sdpaBound(1, .{}, 0) > 0); // finite on zeros
}

test "generalized reference equals the fixture's Attend element for element" {
    const a = testing.allocator;
    // Same shape family the fixture's [8]f64 score buffer allows.
    const case = SdpaCase{ .name = "fixture parity", .rows = 2, .prefix = 1 };
    const capacity = capacityOf(case);
    const row_elems: usize = @intCast(fx.heads * fx.head_dim);
    const cache_elems: usize = @intCast(fx.kv_heads * capacity * fx.head_dim);
    var prng = std.Random.DefaultPrng.init(0xc0ffee);
    const rng = prng.random();
    const q = try a.alloc(u16, case.rows * row_elems);
    defer a.free(q);
    const k = try a.alloc(u16, cache_elems);
    defer a.free(k);
    const v = try a.alloc(u16, cache_elems);
    defer a.free(v);
    for (q) |*x| x.* = fx.sampleDyadic(rng, -5, 3);
    for (k) |*x| x.* = fx.sampleDyadic(rng, -5, 3);
    for (v) |*x| x.* = fx.sampleDyadic(rng, -5, 3);
    const fixture = try fx.Attend.init(a, .{ .name = case.name, .rows = case.rows, .prefix = case.prefix, .capacity = capacity }, q, k, v, 0);
    defer fixture.deinit();
    const want = try a.alloc(f64, case.rows * row_elems);
    defer a.free(want);
    const visible = try a.alloc(usize, @intCast(case.rows));
    defer a.free(visible);
    var mags: Mags = .{};
    try attendRef(a, q, k, v, case, capacity, &mags, want, visible);
    try testing.expectEqualSlices(f64, fixture.want, want);
    for (0..case.rows) |r| try testing.expectEqual(@as(usize, @intCast(visibleKeys(case, @intCast(r)))), visible[r]);
}

test "served cases stay inside the engine's shapes and hit every reduction edge" {
    for (served_cases) |case| {
        try testing.expect(case.rows == 1 or case.rows <= 32); // decode rows 1, prefill <= 32
        const capacity = capacityOf(case);
        try testing.expect(capacity <= max_visible_keys); // kStreamKeys
        try testing.expectEqual(visibleKeys(case, case.rows - 1), capacity); // exact fit
        try testing.expect(visibleKeys(case, 0) >= 1);
    }
    // The named edges all appear as some decode case's single-row count.
    const decode_edges = [_]u64{ 1, 16, 17, 256, 257, 1024, 1025, 7168 };
    for (decode_edges) |n| {
        var found = false;
        for (served_cases) |case| {
            if (case.rows == 1 and visibleKeys(case, 0) == n) found = true;
        }
        try testing.expect(found); // every claimed edge is actually served
    }
    // Multi-row prefill crosses the 1024 stride and fills 7168.
    try testing.expect(served_cases[11].prefix < key_stride and capacityOf(served_cases[11]) >= key_stride);
    try testing.expectEqual(max_visible_keys, capacityOf(served_cases[served_cases.len - 1]));
}

test "sdpa checkers accept the honest bf16 output and refuse planted ones" {
    const a = testing.allocator;
    const case = SdpaCase{ .name = "plants", .rows = 2, .prefix = 0 }; // row 0 sees exactly 1 key
    const f = try Sdpa.init(a, case, 0x5d1);
    defer f.deinit();
    const good = try a.alloc(u16, f.want.len);
    defer a.free(good);
    for (f.want, 0..) |w, i| good[i] = c.toBf16(@floatCast(w));
    const rep = try f.check(good); // the honest rounded output passes
    try testing.expect(rep.worst <= 1);
    // single-key row: bit-exact V transport, and tampering with it is caught
    try testing.expectEqual(f.v_bits[0], good[0]); // row 0, head 0, d 0: n=1
    good[0] ^= 1;
    try testing.expectError(error.SingleKeyNotTransport, f.check(good));
    good[0] = f.v_bits[0];
    good[1] = 0x4f00; // 2^31: an absolutely wrong finite past any bound
    try testing.expectError(error.OverBound, f.check(good));
    good[1] = 0x7fc0;
    try testing.expectError(error.NonFiniteOutput, f.check(good));
    // an all-poison output (dead kernel) is refused
    const dead = try a.alloc(u16, f.want.len);
    defer a.free(dead);
    @memset(dead, fx.poison16);
    try testing.expectError(error.NonFiniteOutput, f.check(dead));
}

test "depth derivation is pinned to the shader source this contract reads" {
    const src = sources.sdpa_decode_native;
    // 256-lane scan and the 8-level binary tree: the denominator's depth.
    try testing.expect(std.mem.indexOf(u8, src, "if (thread < 256u)") != null);
    try testing.expect(std.mem.indexOf(u8, src, "index = thread; index < key_end; index += 256u") != null);
    try testing.expect(std.mem.indexOf(u8, src, "for (uint stride = 128u; stride > 0u; stride >>= 1u)") != null);
    // The output accumulation's depth: sequential over the visible keys.
    try testing.expect(std.mem.indexOf(u8, src, "key + 16u <= key_end; key += 16u") != null);
    // The causal multi-row bound the served shapes ride.
    try testing.expect(std.mem.indexOf(u8, src, "params.matrix_k - params.dims + row + 1u") != null);
    // The shared-memory bound that caps the served context.
    try testing.expect(std.mem.indexOf(u8, src, "kStreamKeys = 7168u") != null);
}
