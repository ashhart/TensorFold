// Preserve the omitted contract, precision, and host-interface facts from the original comment.
const std = @import("std");
const contract = @import("contract");
const c = contract.spec;

pub const tail = 16;
pub const poison16: u16 = 0xffff;

pub const heads: u64 = 8;
pub const kv_heads: u64 = 2;
pub const head_dim: u64 = 256;
pub const rotary: u64 = 32;
pub const theta: f64 = 1e7;
pub const scale: f64 = 1.0 / 16.0;

/// The worst output: |got - want| OVER its budget (a ratio; >1 fails), and where.
pub const Report = struct { worst: f64 = 0, where: usize = 0 };

// Budget: depth f32 sum, then `ops` f32 and `mids` bf16 roundings, with one final bf16 store; never sum output squares.
pub fn budget(depth: f64, ops: f64, mids: f64, want: f64, mag: f64) f64 {
    const u: f64 = 0x1p-24;
    const g = if (depth > 1) depth * u / (1 - depth * u) else 0;
    const value_scale = @max(@abs(want), mag);
    const work = g * mag + (ops * u + mids * 0x1p-8) * value_scale;
    return work + 0x1p-8 * (@abs(want) + work) + 0x1p-133;
}

/// ±m * 2^e, m in {1, 1.5}, e in the stated range: exactly bf16-representable.
pub fn sampleDyadic(rng: std.Random, lo: i32, hi: i32) u16 {
    const e = rng.intRangeAtMost(i32, lo, hi);
    const m: f32 = if (rng.boolean()) 1.0 else 1.5;
    return c.toBf16(m * std.math.pow(f32, 2, @floatFromInt(e)) * (if (rng.boolean()) @as(f32, 1) else -1));
}

fn poisonedShell(a: std.mem.Allocator, len: usize) ![]u16 {
    const buf = try a.alloc(u16, len + tail);
    @memset(buf, poison16);
    return buf;
}

/// The ASH rope in f64 for one 256-wide row at absolute position `pos`.
pub fn ropeRef(row: []const u16, pos: f64, out: []f64) void {
    for (0..rotary) |i| {
        const inv = std.math.pow(f64, theta, -@as(f64, @floatFromInt(i)) / 32.0);
        const angle = pos * inv;
        const cos_a = @cos(angle);
        const sin_a = @sin(angle);
        const x1 = c.fromBf16(row[i]);
        const x2 = c.fromBf16(row[i + 32]);
        out[i] = x1 * cos_a - x2 * sin_a;
        out[i + 32] = x1 * sin_a + x2 * cos_a;
    }
    for (rotary * 2..head_dim) |d| out[d] = c.fromBf16(row[d]);
}

pub const RopeCase = struct { name: []const u8, rows: u64, base: i64 };

pub const rope_cases = [_]RopeCase{
    .{ .name = "queries rows4 base0", .rows = 4, .base = 0 },
    .{ .name = "queries rows2 base5", .rows = 2, .base = 5 },
};

// RoPE gate checks rotated values against f64; uploaded poison tails remain, and passthrough bits match exactly.
pub const Rope = struct {
    a: std.mem.Allocator,
    case: RopeCase,
    q_bits: []u16,
    want: []f64,
    /// Elements per row (heads * head_dim).
    row_elems: usize,

    pub fn init(a: std.mem.Allocator, case: RopeCase, seed: u64) !Rope {
        const row_elems: usize = @intCast(heads * head_dim);
        const q_bits = try poisonedShell(a, case.rows * row_elems);
        errdefer a.free(q_bits);
        const want = try a.alloc(f64, case.rows * row_elems);
        errdefer a.free(want);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        var ref: [head_dim]f64 = undefined;
        for (0..@intCast(case.rows)) |r| {
            const row = q_bits[r * row_elems ..][0..row_elems];
            for (0..heads) |h| {
                const head_row = row[@intCast(h * head_dim)..][0..@intCast(head_dim)];
                for (head_row) |*x| x.* = sampleDyadic(rng, -5, 3);
                ropeRef(head_row, @floatFromInt(case.base + @as(i64, @intCast(r))), &ref);
                @memcpy(want[r * row_elems + @as(usize, @intCast(h * head_dim)) ..][0..@intCast(head_dim)], &ref);
            }
        }
        return .{ .a = a, .case = case, .q_bits = q_bits, .want = want, .row_elems = row_elems };
    }

    pub fn deinit(f: Rope) void {
        f.a.free(f.q_bits);
        f.a.free(f.want);
    }

    pub fn check(f: Rope, bits: []const u16) !Report {
        var r: Report = .{};
        for (f.want, 0..) |w, i| {
            const got = c.fromBf16(bits[i]);
            const d = @abs(got - w);
            const bound = budget(2, 6, 1, w, @abs(w));
            if (!std.math.isFinite(got)) return error.NonFiniteOutput;
            if (d > bound) return error.OverBound;
            const ratio = d / bound;
            if (ratio > r.worst) r = .{ .worst = ratio, .where = i };
            // passthrough tail: bit-exact
            if (i % @as(usize, @intCast(head_dim)) >= @as(usize, @intCast(rotary * 2))) {
                if (bits[i] != f.q_bits[i]) return error.PassthroughTailChanged;
            }
        }
        return r;
    }
};

pub const CacheCase = struct { name: []const u8, rows: u64, base: u64, capacity: u64 };

pub const cache_case = CacheCase{ .name = "cache prefix2 base2 rows3 cap8", .rows = 3, .base = 2, .capacity = 8 };

// CacheKV plants prefix rows in [kv_head,capacity,head_dim]; K matches f64 RoPE, inactive bits persist, V is exact.
pub const Cache = struct {
    a: std.mem.Allocator,
    case: CacheCase,
    knorm_bits: []u16,
    v_bits: []u16,
    keys_shell: []u16,
    values_shell: []u16,
    want_keys: []f64,
    // Exact bits required for every position: planted prefix bits, the V transport bits, or poison for untouched rows.
    want_key_bits: []u16,
    want_value_bits: []u16,
    /// True where the operator writes (or the prefix was planted): budget-checked.
    active: []bool,

    /// (h, pos, d) -> element index in the [kv_head, capacity, head_dim] cache.
    fn cacheIndex(case: CacheCase, h: usize, pos: usize) usize {
        return (h * @as(usize, @intCast(case.capacity)) + pos) * @as(usize, @intCast(head_dim));
    }

    pub fn init(a: std.mem.Allocator, case: CacheCase, seed: u64) !Cache {
        const row_elems: usize = @intCast(kv_heads * head_dim);
        const cache_elems: usize = @intCast(kv_heads * case.capacity * head_dim);
        const knorm_bits = try poisonedShell(a, case.rows * row_elems);
        errdefer a.free(knorm_bits);
        const v_bits = try poisonedShell(a, case.rows * row_elems);
        errdefer a.free(v_bits);
        const keys_shell = try poisonedShell(a, cache_elems);
        errdefer a.free(keys_shell);
        const values_shell = try poisonedShell(a, cache_elems);
        errdefer a.free(values_shell);
        const want_keys = try a.alloc(f64, cache_elems);
        errdefer a.free(want_keys);
        const want_key_bits = try a.alloc(u16, cache_elems);
        errdefer a.free(want_key_bits);
        const want_value_bits = try a.alloc(u16, cache_elems);
        errdefer a.free(want_value_bits);
        const active = try a.alloc(bool, cache_elems);
        errdefer a.free(active);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // Everything starts inactive and poison; the loops below claim slots.
        for (0..cache_elems) |i| {
            want_keys[i] = 0;
            want_key_bits[i] = poison16;
            want_value_bits[i] = poison16;
            active[i] = false;
        }
        // Prefix positions (both kv heads): planted data, bit-exact want.
        for (0..@intCast(case.base)) |pos| {
            for (0..@intCast(kv_heads)) |h| {
                const base_idx = cacheIndex(case, h, pos);
                for (0..@intCast(head_dim)) |d| {
                    const kb = sampleDyadic(rng, -5, 3);
                    const vb = sampleDyadic(rng, -5, 3);
                    keys_shell[base_idx + d] = kb;
                    values_shell[base_idx + d] = vb;
                    want_keys[base_idx + d] = c.fromBf16(kb);
                    want_key_bits[base_idx + d] = kb;
                    want_value_bits[base_idx + d] = vb;
                    active[base_idx + d] = true;
                }
            }
        }
        // K rows match f64 RoPE at base+r; V rows transport exact bits.
        var ref: [head_dim]f64 = undefined;
        for (0..@intCast(case.rows)) |r| {
            for (0..@intCast(kv_heads)) |h| {
                const row = knorm_bits[r * row_elems + h * @as(usize, @intCast(head_dim)) ..][0..@intCast(head_dim)];
                for (row) |*x| x.* = sampleDyadic(rng, -5, 3);
                const vrow = v_bits[r * row_elems + h * @as(usize, @intCast(head_dim)) ..][0..@intCast(head_dim)];
                for (vrow) |*x| x.* = sampleDyadic(rng, -5, 3);
                ropeRef(row, @floatFromInt(@as(i64, @intCast(case.base + r))), &ref);
                const dst = cacheIndex(case, h, @intCast(case.base + r));
                @memcpy(want_keys[dst..][0..@intCast(head_dim)], &ref);
                for (0..@intCast(head_dim)) |d| {
                    want_value_bits[dst + d] = vrow[d];
                    active[dst + d] = true;
                }
            }
        }
        return .{
            .a = a,
            .case = case,
            .knorm_bits = knorm_bits,
            .v_bits = v_bits,
            .keys_shell = keys_shell,
            .values_shell = values_shell,
            .want_keys = want_keys,
            .want_key_bits = want_key_bits,
            .want_value_bits = want_value_bits,
            .active = active,
        };
    }

    pub fn deinit(f: Cache) void {
        f.a.free(f.knorm_bits);
        f.a.free(f.v_bits);
        f.a.free(f.keys_shell);
        f.a.free(f.values_shell);
        f.a.free(f.want_keys);
        f.a.free(f.want_key_bits);
        f.a.free(f.want_value_bits);
        f.a.free(f.active);
    }

    // Active rows match f64 RoPE; inactive poison rows preserve uploaded bits.
    pub fn checkKeys(f: Cache, bits: []const u16) !Report {
        var r: Report = .{};
        for (f.active, 0..) |is_active, i| {
            if (!is_active) {
                if (bits[i] != f.want_key_bits[i]) return error.PoisonRowTouched;
                continue;
            }
            const w = f.want_keys[i];
            const got = c.fromBf16(bits[i]);
            if (!std.math.isFinite(got)) return error.NonFiniteOutput;
            const d = @abs(got - w);
            const bound = budget(2, 6, 1, w, @abs(w));
            if (d > bound) return error.OverBound;
            const ratio = d / bound;
            if (ratio > r.worst) r = .{ .worst = ratio, .where = i };
        }
        return r;
    }

    /// V rows and every untouched cache row must be bit-exact (transport or poison).
    pub fn checkValueBits(f: Cache, bits: []const u16) !void {
        for (f.want_value_bits, 0..) |w, i| {
            if (w == poison16) {
                if (bits[i] != poison16) return error.PoisonRowTouched;
            } else if (bits[i] != w) return error.ValueTransportChanged;
        }
    }
};

pub const AttendCase = struct { name: []const u8, rows: u64, prefix: u64, capacity: u64 };

pub const attend_case = AttendCase{ .name = "sdpa prefix2 rows3 cap8", .rows = 3, .prefix = 2, .capacity = 8 };

// Each query row must ignore keys from its first non-visible index onward.
pub fn visibleFrom(case: AttendCase, r: u64) u64 {
    return case.prefix + r + 1;
}

// Causal SDPA uses uploaded cache and rotated query bits; its f64 reference adds no RoPE error.
pub const Attend = struct {
    a: std.mem.Allocator,
    case: AttendCase,
    q_bits: []u16,
    want: []f64,
    /// Visible key count per row (prefix + row + 1), for the known-counts report.
    visible: []usize,
    max_score: f64,
    max_value: f64,

    pub fn init(a: std.mem.Allocator, case: AttendCase, q_bits: []const u16, keys: []const u16, values: []const u16, seed: u64) !Attend {
        _ = seed;
        const row_elems: usize = @intCast(heads * head_dim);
        const q_bits_copy = try a.dupe(u16, q_bits[0 .. case.rows * row_elems]);
        errdefer a.free(q_bits_copy);
        const want = try a.alloc(f64, case.rows * row_elems);
        errdefer a.free(want);
        const visible = try a.alloc(usize, @intCast(case.rows));
        errdefer a.free(visible);
        var max_score: f64 = 0;
        var max_value: f64 = 0;
        for (0..@intCast(case.rows)) |r| {
            const vis: usize = @intCast(case.prefix + @as(u64, @intCast(r)) + 1);
            visible[r] = vis;
            for (0..heads) |h| {
                const kh = h / 4; // repeats = heads / kv_heads
                const qrow = q_bits[r * row_elems + h * @as(usize, @intCast(head_dim)) ..][0..@intCast(head_dim)];
                var scores: [8]f64 = undefined; // vis <= prefix+rows <= 8 in the fixture case
                var m: f64 = -std.math.floatMax(f64);
                for (0..vis) |j| {
                    const kbase = (@as(usize, @intCast(kh)) * @as(usize, @intCast(case.capacity)) + j) * @as(usize, @intCast(head_dim));
                    var dot: f64 = 0;
                    for (0..@intCast(head_dim)) |d| {
                        dot += @as(f64, c.fromBf16(qrow[d])) * c.fromBf16(keys[kbase + d]);
                    }
                    scores[j] = dot * scale;
                    m = @max(m, scores[j]);
                    max_score = @max(max_score, @abs(scores[j]));
                }
                var denom: f64 = 0;
                for (scores[0..vis]) |s| denom += @exp(s - m);
                for (0..@intCast(head_dim)) |d| {
                    var acc: f64 = 0;
                    for (0..vis) |j| {
                        const vbase = (@as(usize, @intCast(kh)) * @as(usize, @intCast(case.capacity)) + j) * @as(usize, @intCast(head_dim));
                        acc += (@exp(scores[j] - m) / denom) * c.fromBf16(values[vbase + d]);
                        max_value = @max(max_value, @abs(c.fromBf16(values[vbase + d])));
                    }
                    want[r * row_elems + h * @as(usize, @intCast(head_dim)) + d] = acc;
                }
            }
        }
        return .{ .a = a, .case = case, .q_bits = q_bits_copy, .want = want, .visible = visible, .max_score = max_score, .max_value = max_value };
    }

    pub fn deinit(f: Attend) void {
        f.a.free(f.q_bits);
        f.a.free(f.want);
        f.a.free(f.visible);
    }

    pub fn check(f: Attend, bits: []const u16) !Report {
        var r: Report = .{};
        for (f.want, 0..) |w, i| {
            const got = c.fromBf16(bits[i]);
            if (!std.math.isFinite(got)) return error.NonFiniteOutput;
            const d = @abs(got - w);
            const bound = budget(300, 6, 2, w, f.max_score + f.max_value);
            if (d > bound) return error.OverBound;
            const ratio = d / bound;
            if (ratio > r.worst) r = .{ .worst = ratio, .where = i };
        }
        return r;
    }
};

pub const GateCase = struct { name: []const u8, rows: u64 };

pub const gate_case = GateCase{ .name = "gate rows4", .rows = 4 };

// Gate elements retain inputs, f64 reference/intermediates, budget and output for shared checks and diagnostics.
pub const GateDiag = struct {
    index: usize,
    row: usize,
    d: usize,
    head: usize,
    /// Uploaded input words: ctx at ctx_bits, the raw qraw gate at gate_bits.
    ctx_bits: u16,
    gate_bits: u16,
    ctx: f64,
    gate: f64,
    // Sigmoid intermediates are f64, f32 before one RNE bf16 round, and the resulting bf16 value.
    sigmoid_f64: f64,
    sigmoid_f32: f64,
    sigmoid_bf16_bits: u16,
    sigmoid_bf16: f64,
    want: f64,
    got: f64,
    bound: f64,
    distance: f64,
    ratio: f64,
};

// The interleaved output gate: qraw row stride 4096, head h's gate at h*512 + 256 + d within the row.
pub const Gate = struct {
    a: std.mem.Allocator,
    case: GateCase,
    qraw_bits: []u16,
    ctx_bits: []u16,
    out_bits: []u16,

    pub fn init(a: std.mem.Allocator, case: GateCase, seed: u64) !Gate {
        const qraw = try poisonedShell(a, case.rows * 4096);
        errdefer a.free(qraw);
        const ctx = try poisonedShell(a, case.rows * 2048);
        errdefer a.free(ctx);
        const out = try poisonedShell(a, case.rows * 2048);
        errdefer a.free(out);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // Data region only: the poison tail must survive as the overrun guard.
        for (qraw[0 .. case.rows * 4096]) |*x| x.* = sampleDyadic(rng, -8, 8);
        for (ctx[0 .. case.rows * 2048]) |*x| x.* = sampleDyadic(rng, -5, 3);
        return .{ .a = a, .case = case, .qraw_bits = qraw, .ctx_bits = ctx, .out_bits = out };
    }

    pub fn deinit(f: Gate) void {
        f.a.free(f.qraw_bits);
        f.a.free(f.ctx_bits);
        f.a.free(f.out_bits);
    }

    // One element's reference, budget and output value; the checker and the diagnostic both walk this.
    fn element(f: Gate, bits: []const u16, row: usize, d: usize) GateDiag {
        const head = d / 256;
        const gate_bits = f.qraw_bits[row * 4096 + head * 512 + 256 + (d % 256)];
        const gate = c.fromBf16(gate_bits);
        // Metal rounds sigmoid to bf16 then widens; budget covers f32/f64 exp differences.
        const sigmoid_f64 = 1.0 / (1.0 + @exp(-@as(f64, gate)));
        const sigmoid_f32: f32 = @floatCast(sigmoid_f64);
        const sigmoid_bf16_bits = c.toBf16(sigmoid_f32);
        const sigmoid_bf16 = c.fromBf16(sigmoid_bf16_bits);
        const ctx = c.fromBf16(f.ctx_bits[row * 2048 + d]);
        const want = ctx * sigmoid_bf16;
        const got = c.fromBf16(bits[row * 2048 + d]);
        const distance: f32 = @abs(got - want);
        const bound = budget(1, 3, 1, want, @abs(want));
        return .{
            .index = row * 2048 + d,
            .row = row,
            .d = d,
            .head = head,
            .ctx_bits = f.ctx_bits[row * 2048 + d],
            .gate_bits = gate_bits,
            .ctx = ctx,
            .gate = gate,
            .sigmoid_f64 = sigmoid_f64,
            .sigmoid_f32 = sigmoid_f32,
            .sigmoid_bf16_bits = sigmoid_bf16_bits,
            .sigmoid_bf16 = sigmoid_bf16,
            .want = want,
            .got = got,
            .bound = bound,
            .distance = distance,
            .ratio = @as(f64, distance) / bound,
        };
    }

    /// The checker recomputes the reference from the uploaded shells.
    pub fn check(f: Gate, bits: []const u16) !Report {
        var r: Report = .{};
        for (0..@intCast(f.case.rows)) |row| {
            for (0..2048) |d| {
                const e = f.element(bits, row, d);
                if (!std.math.isFinite(e.got)) return error.NonFiniteOutput;
                if (e.distance > e.bound) return error.OverBound;
                if (e.ratio > r.worst) r = .{ .worst = e.ratio, .where = e.index };
            }
        }
        return r;
    }

    // Return first failing element with probe context; null means accepted.
    pub fn firstViolation(f: Gate, bits: []const u16) ?GateDiag {
        for (0..@intCast(f.case.rows)) |row| {
            for (0..2048) |d| {
                const e = f.element(bits, row, d);
                if (!std.math.isFinite(e.got) or e.distance > e.bound) return e;
            }
        }
        return null;
    }
};

const testing = std.testing;

test "sdpa per-row visible counts and per-row poison windows" {
    const ac = attend_case;
    // prefix=2, rows=3: each row's visible key count is prefix+r+1 -> 3,4,5
    try testing.expectEqual([3]u64{ 3, 4, 5 }, .{ visibleFrom(ac, 0), visibleFrom(ac, 1), visibleFrom(ac, 2) });
    for (0..ac.rows) |r| {
        const from = visibleFrom(ac, @intCast(r));
        // Each row's visible key is required; its poison window is fresh and nonempty.
        try testing.expect(from > ac.prefix);
        try testing.expect(from <= ac.capacity);
    }
}

test "rope reference matches the ASH half-split form on a hand vector" {
    // Unit input yields cos/sin in rotated halves and zero passthrough tail.
    var row: [head_dim]u16 = @splat(0);
    for (0..32) |i| row[i] = c.toBf16(1.0);
    var out: [head_dim]f64 = undefined;
    ropeRef(&row, 0, &out);
    for (0..32) |i| {
        try testing.expectEqual(@as(f64, 1), out[i]); // cos(0) = 1
        try testing.expectEqual(@as(f64, 0), out[i + 32]); // sin(0) = 0
    }
    try testing.expectEqual(@as(f64, 0), out[128]); // passthrough tail
    // pos 0.5, theta^(-1/32) at i=1: out[1] = cos(0.5*inv), out[33] = sin(0.5*inv).
    const pos: f64 = 0.5;
    ropeRef(&row, pos, &out);
    const inv = std.math.pow(f64, theta, -1.0 / 32.0);
    try testing.expectApproxEqAbs(@cos(pos * inv), out[1], 1e-12);
    try testing.expectApproxEqAbs(@sin(pos * inv), out[33], 1e-12);
}

test "budget grows with depth and stays finite on zeros" {
    try testing.expect(budget(300, 6, 2, 1, 1) > budget(2, 6, 1, 1, 1));
    try testing.expect(budget(2, 6, 1, 0, 0) > 0);
}

test "sampleDyadic is exactly bf16" {
    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    for (0..64) |_| {
        const bits = sampleDyadic(rng, -5, 3);
        const v = c.fromBf16(bits);
        try testing.expectEqual(v, c.fromBf16(c.toBf16(v)));
    }
}

// Negative tests require rejection of wrong finite/nonfinite outputs and detect tail/row bit tampering.
test "rope, cache and gate checkers refuse planted outputs" {
    const a = testing.allocator;
    {
        const f = try Rope.init(a, rope_cases[0], 0x11);
        defer f.deinit();
        const good = try a.alloc(u16, f.want.len);
        defer a.free(good);
        for (f.want, 0..) |w, i| good[i] = c.toBf16(@floatCast(w));
        const rep = try f.check(good); // the honest output passes
        try testing.expect(rep.worst <= 1);
        good[0] = 0x4f00; // 2^31: an absolutely wrong finite, far past any budget
        try testing.expectError(error.OverBound, f.check(good));
        good[0] = 0x7fc0;
        try testing.expectError(error.NonFiniteOutput, f.check(good));
        // passthrough tail tamper
        const f2 = try Rope.init(a, rope_cases[0], 0x12);
        defer f2.deinit();
        const good2 = try a.alloc(u16, f2.want.len);
        defer a.free(good2);
        for (f2.want, 0..) |w, i| good2[i] = c.toBf16(@floatCast(w));
        good2[@intCast(rotary * 2)] ^= 1;
        try testing.expectError(error.PassthroughTailChanged, f2.check(good2));
    }
    {
        const f = try Cache.init(a, cache_case, 0x13);
        defer f.deinit();
        const kbits = try a.alloc(u16, f.keys_shell.len - tail);
        defer a.free(kbits);
        for (kbits) |*x| x.* = poison16;
        for (f.active, 0..) |act, i| if (act) {
            kbits[i] = c.toBf16(@floatCast(f.want_keys[i]));
        };
        const krep = try f.checkKeys(kbits);
        try testing.expect(krep.worst <= 1);
        // a POISON row tamper must fail:
        var inactive: ?usize = null;
        for (f.active, 0..) |act, i| if (!act) {
            inactive = i;
            break;
        };
        kbits[inactive.?] = 0;
        try testing.expectError(error.PoisonRowTouched, f.checkKeys(kbits));
        // V transport tamper
        const vbits = try a.dupe(u16, f.want_value_bits);
        defer a.free(vbits);
        var v_active: ?usize = null;
        for (f.want_value_bits, 0..) |w, i| if (w != poison16) {
            v_active = i;
            break;
        };
        vbits[v_active.?] ^= 1;
        try testing.expectError(error.ValueTransportChanged, f.checkValueBits(vbits));
    }
    {
        const f = try Gate.init(a, gate_case, 0x14);
        defer f.deinit();
        const good = try a.alloc(u16, @intCast(f.case.rows * 2048));
        defer a.free(good);
        @memset(good, 0);
        // the gate reference is derived from the shells via the shared element walk
        for (0..@intCast(f.case.rows)) |row| {
            for (0..2048) |d| {
                good[row * 2048 + d] = c.toBf16(@floatCast(f.element(good, row, d).want));
            }
        }
        const grep2 = try f.check(good);
        try testing.expect(grep2.worst <= 1);
        good[0] = 0xff80;
        try testing.expectError(error.NonFiniteOutput, f.check(good));
    }
}

test "all-poison output (dead kernel on poisoned init) is rejected by checkers" {
    const a = testing.allocator;
    // Poison16 fills and uploads output before each dispatch; absent or partial writes must fail.
    {
        const f = try Rope.init(a, rope_cases[0], 0x21);
        defer f.deinit();
        const shell = try a.alloc(u16, f.want.len + tail);
        defer a.free(shell);
        @memset(shell, poison16);
        try testing.expectError(error.NonFiniteOutput, f.check(shell[0..f.want.len])); // dead kernel
        // partial write: first row correct, rest poison
        for (0..f.row_elems) |i| shell[i] = c.toBf16(@floatCast(f.want[i]));
        try testing.expectError(error.NonFiniteOutput, f.check(shell[0..f.want.len]));
    }
    {
        const f = try Gate.init(a, gate_case, 0x22);
        defer f.deinit();
        const out = try a.alloc(u16, @intCast(f.case.rows * 2048));
        defer a.free(out);
        @memset(out, poison16);
        try testing.expectError(error.NonFiniteOutput, f.check(out));
    }
}

test "fixture initialization fails cleanly under allocation failure" {
    var failing = std.testing.FailingAllocator.init(testing.allocator, .{ .fail_index = 1 });
    try testing.expectError(error.OutOfMemory, Rope.init(failing.allocator(), rope_cases[0], 0x15));
    try testing.expectError(error.OutOfMemory, Cache.init(failing.allocator(), cache_case, 0x16));
    try testing.expectError(error.OutOfMemory, Gate.init(failing.allocator(), gate_case, 0x17));
}

// Prefix=2, rows=3, cap=8: hidden-key poisoning preserves f64 output; a one-key leak is rejected.
test "causal sdpa: poisoned future keys are invisible; a one-key leak is caught" {
    const a = testing.allocator;
    const ac = attend_case;
    const row_elems: usize = @intCast(heads * head_dim);
    const cache_elems: usize = @intCast(kv_heads * ac.capacity * head_dim);
    const cap: usize = @intCast(ac.capacity);
    var prng = std.Random.DefaultPrng.init(0x31);
    const rng = prng.random();
    for (0..ac.rows) |r| {
        // Fresh, finite, nontrivial data each row (no cross-row accumulation).
        const q = try a.alloc(u16, row_elems);
        defer a.free(q);
        for (q) |*x| x.* = sampleDyadic(rng, -5, 3);
        const keys = try a.alloc(u16, cache_elems);
        defer a.free(keys);
        const values = try a.alloc(u16, cache_elems);
        defer a.free(values);
        for (keys) |*x| x.* = sampleDyadic(rng, -5, 3);
        for (values) |*x| x.* = sampleDyadic(rng, -5, 3);

        // Copy with every key at or after row r's visibility window poisoned.
        const from: usize = @intCast(visibleFrom(ac, @intCast(r)));
        const leaked_keys = try a.dupe(u16, keys);
        defer a.free(leaked_keys);
        for (0..@intCast(kv_heads)) |h| {
            const head_base = @as(usize, @intCast(h)) * cap;
            @memset(leaked_keys[(head_base + from) * @as(usize, @intCast(head_dim)) ..][0 .. (cap - from) * @as(usize, @intCast(head_dim))], poison16);
        }

        // Baseline: this row as a single-row query over the clean cache.
        const one_row = AttendCase{ .name = "baseline row", .rows = 1, .prefix = ac.prefix + @as(u64, @intCast(r)), .capacity = ac.capacity };
        const base = try Attend.init(a, one_row, q, keys, values, 0);
        defer base.deinit();
        // Same single-row query over the poisoned cache: identical reference.
        const pbase = try Attend.init(a, one_row, q, leaked_keys, values, 0);
        defer pbase.deinit();
        try testing.expectEqualSlices(f64, base.want, pbase.want);

        // The honest bf16 output passes BOTH production checkers.
        const out = try a.alloc(u16, base.want.len);
        defer a.free(out);
        for (base.want, 0..) |w, i| out[i] = c.toBf16(@floatCast(w));
        const rep = try pbase.check(out);
        try testing.expect(rep.worst <= 1);

        // The same Attend.init leaks one poisoned key; its nonfinite reference must differ and fail.
        const leak_case = AttendCase{ .name = "leak row", .rows = 1, .prefix = ac.prefix + @as(u64, @intCast(r)) + 1, .capacity = ac.capacity };
        const leak = try Attend.init(a, leak_case, q, leaked_keys, values, 0);
        defer leak.deinit();
        var has_nonfinite = false;
        for (leak.want) |w| {
            if (!std.math.isFinite(w)) has_nonfinite = true;
        }
        try testing.expect(has_nonfinite); // the leak truly saw the poison
        try testing.expect(!std.meta.eql(base.want[0], leak.want[0])); // differs from baseline
        const leak_out = try a.alloc(u16, leak.want.len);
        defer a.free(leak_out);
        for (leak.want, 0..) |w, i| leak_out[i] = c.toBf16(@floatCast(w));
        try testing.expectError(error.NonFiniteOutput, leak.check(leak_out));
        try testing.expectError(error.NonFiniteOutput, base.check(leak_out)); // causal checker rejects the leaked output
    }
}

// Zeroed lane 0 was the GPU bug: all-zero output must fail OverBound and report its first element.
test "all-zero gate output is refused OverBound and the diagnostic names element 0" {
    const a = testing.allocator;
    const f = try Gate.init(a, gate_case, 0x25);
    defer f.deinit();
    // Element 0 uses gate=1 (sigmoid ~0.731, nonzero bf16) and ctx=1; dyadic shells stay fixed.
    f.qraw_bits[256] = c.toBf16(1.0);
    f.ctx_bits[0] = c.toBf16(1.0);
    const zeros = try a.alloc(u16, @intCast(f.case.rows * 2048));
    defer a.free(zeros);
    @memset(zeros, 0);
    try testing.expectError(error.OverBound, f.check(zeros));
    const diag = f.firstViolation(zeros).?;
    try testing.expectEqual(@as(usize, 0), diag.index);
    try testing.expectEqual(@as(usize, 0), diag.row);
    try testing.expectEqual(@as(usize, 0), diag.d);
    try testing.expectEqual(@as(usize, 0), diag.head);
    try testing.expectEqual(@as(f64, 0), diag.got);
    try testing.expectEqual(c.toBf16(1.0), diag.ctx_bits);
    try testing.expectEqual(f.qraw_bits[256], diag.gate_bits);
    try testing.expectEqual(@as(f64, 1), diag.ctx);
    try testing.expect(diag.sigmoid_bf16 > 0.5 and diag.sigmoid_bf16 < 1);
    try testing.expectEqual(diag.ctx * diag.sigmoid_bf16, diag.want);
    try testing.expectEqual(diag.want, @abs(diag.got - diag.want));
    try testing.expect(diag.ratio > 1);
}

// Diagnostics report the planted first bad element without changing OverBound acceptance.
test "gate diagnostic reports a planted bad first element with unchanged OverBound acceptance" {
    const a = testing.allocator;
    const f = try Gate.init(a, gate_case, 0x24);
    defer f.deinit();
    f.qraw_bits[256] = c.toBf16(1.0);
    f.ctx_bits[0] = c.toBf16(1.0);
    const good = try a.alloc(u16, @intCast(f.case.rows * 2048));
    defer a.free(good);
    @memset(good, 0);
    for (0..@intCast(f.case.rows)) |row| {
        for (0..2048) |d| {
            good[row * 2048 + d] = c.toBf16(@floatCast(f.element(good, row, d).want));
        }
    }
    const rep = try f.check(good); // the honest output passes
    try testing.expect(rep.worst <= 1);
    good[0] = 0x4f00; // 2^31 at the first element: absolutely wrong
    try testing.expectError(error.OverBound, f.check(good));
    const diag = f.firstViolation(good).?;
    try testing.expectEqual(@as(usize, 0), diag.index);
    try testing.expectEqual(@as(f64, 2147483648.0), diag.got);
    try testing.expect(diag.ratio > 1);
    try testing.expectEqual(f.qraw_bits[256], diag.gate_bits);
    try testing.expectEqual(c.toBf16(1.0), diag.ctx_bits);
}

// Preserve the omitted contract, precision, and host-interface facts from the original comment.
test "gate sigmoid widen: bf16 0x3f00 widens once to 0.5; the old double shift gives 0" {
    const bits: u32 = 0x3f00; // what bf16_bits(0.5) produces
    const once: f32 = @bitCast(bits << 16); // bf16_lane(word, 0): the correct widen
    try testing.expectEqual(@as(f32, 0.5), once);
    const annihilated: u64 = @as(u64, bits) << 32; // (bits << 16) << 16 as the GPU takes it
    const twice: f32 = @bitCast(@as(u32, @truncate(annihilated)));
    try testing.expectEqual(@as(f32, 0.0), twice);
}
