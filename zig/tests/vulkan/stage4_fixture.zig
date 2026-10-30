// Stage 4 fixture: the GatedDeltaNet closure, 16 heads, head_dim 128, 6144 conv channels, 4 taps, vs f64 host refs.
// depth/ops/mids budgets, mag: split max|vals|^2, gate |g|, recur max state, norm max|x|^2*len; chunk=4/8 one ref
const std = @import("std");
const contract = @import("contract");
pub const c = contract.spec;
const gdn_mod = @import("vulkan").gdn;

pub const tail = 16;
pub const poison16: u16 = 0xffff;

pub const Model = gdn_mod.Model;
pub const tokens: u64 = 8;

/// `worst` is |got - want| OVER its budget (a ratio; >1 fails), with `where`.
pub const Report = struct { worst: f64 = 0, where: usize = 0 };

// First failing element, probe print. `want` = HOST Ash-chain ref; sp/exp never stored; negative plants pass null.
pub const Output = enum { soa, g, beta, y, state, snapshot };
pub const Violation = struct {
    index: usize,
    output: Output = .soa,
    want: f64 = 0,
    got: f64 = 0,
    bound: f64 = 0,
    ratio: f64 = 0,
    // The want value's f32 bits where the reference is an f32 scalar (g): shows f32 class (subnormal/underflow) only.
    want_f32_bits: u32 = 0,
};

fn violated(v: ?*Violation, output: Output, index: usize, want: f64, got: f64, bound: f64) void {
    if (v) |d| d.* = .{
        .output = output,
        .index = index,
        .want = want,
        .got = got,
        .bound = bound,
        .ratio = if (bound > 0) @abs(got - want) / bound else std.math.inf(f64),
        .want_f32_bits = @bitCast(@as(f32, @floatCast(want))),
    };
}

// budget: depth f32 sum@mag, ops f32 + mids bf16 rounds, 1 store; not sum-of-squares; never enlarge after GPU runs
pub fn budget(depth: f64, ops: f64, mids: f64, want: f64, mag: f64) f64 {
    const u: f64 = 0x1p-24;
    const g = if (depth > 1) depth * u / (1 - depth * u) else 0;
    const value_scale = @max(@abs(want), mag);
    const work = g * mag + (ops * u + mids * 0x1p-8) * value_scale;
    return work + 0x1p-8 * (@abs(want) + work) + 0x1p-133;
}

pub fn sampleDyadic(rng: std.Random, lo: i32, hi: i32) u16 {
    const e = rng.intRangeAtMost(i32, lo, hi);
    const m: f32 = if (rng.boolean()) 1.0 else 1.5;
    return c.toBf16(m * std.math.pow(f32, 2, @floatFromInt(e)) * (if (rng.boolean()) @as(f32, 1) else -1));
}

pub fn sampleDyadicF32(rng: std.Random, lo: i32, hi: i32) f32 {
    const e = rng.intRangeAtMost(i32, lo, hi);
    const m: f32 = if (rng.boolean()) 1.0 else 1.5;
    return m * std.math.pow(f32, 2, @floatFromInt(e)) * (if (rng.boolean()) @as(f32, 1) else -1);
}

fn poisonedShell(a: std.mem.Allocator, len: usize) ![]u16 {
    const buf = try a.alloc(u16, len + tail);
    @memset(buf, poison16);
    return buf;
}

fn f32Of(pairs: []const u16, word: usize) f64 {
    return @as(f32, @bitCast(@as(u32, pairs[2 * word]) | (@as(u32, pairs[2 * word + 1]) << 16)));
}

// f64 chain shared by split lane-0 and decode raw-gates; omits bf16 rounds (~20x); kept for amplification test
pub fn gateRef(a_in: f64, b_in: f64, a_log: f64, dt: f64) gdn_mod.GateF64 {
    return gdn_mod.gateReferenceF64(a_in, b_in, a_log, dt);
}

// ASH: s=BF16(a+dt); sp=BF16(max(s,0)+log(1+exp(-|s|))); g f32; beta=BF16(sigmoid(b)); residual = exp/log ulps
pub fn gateRefAsh(a_in: u16, b_in: u16, a_log: f64, dt_in: u16) gdn_mod.GateF64 {
    const a = c.fromBf16(a_in);
    const dt = c.fromBf16(dt_in);
    const s = c.fromBf16(c.toBf16(a + dt));
    const sp_raw: f32 = @max(s, 0.0) + @log(@as(f32, 1.0) + @exp(-@abs(s)));
    const sp = c.fromBf16(c.toBf16(sp_raw));
    // the shader's exp reads the f32 A_log: cast BEFORE the exp
    const ea: f32 = @exp(@as(f32, @floatCast(a_log)));
    const g: f32 = @exp(-ea * sp);
    const beta_raw: f32 = 1.0 / (1.0 + @exp(-c.fromBf16(b_in)));
    return .{ .g = @floatCast(g), .beta = @floatCast(c.fromBf16(c.toBf16(beta_raw))) };
}

// ---- Split ----

// Gate split at T=tokens: conv_out/a/b/dt/a_log uploaded; check reads downloaded soa (bf16), g (f32), beta (bf16) bits.
pub const Split = struct {
    a: std.mem.Allocator,
    conv_out_bits: []u16,
    a_bits: []u16,
    b_bits: []u16,
    a_log_bits: []u16, // f32 bits as u16 pairs, 16 words
    dt_bits: []u16,

    pub fn init(a: std.mem.Allocator, seed: u64) !Split {
        const conv_len: usize = @intCast(tokens * Model.conv_channels);
        const gate_len: usize = @intCast(tokens * Model.heads);
        const conv_out_bits = try poisonedShell(a, conv_len);
        errdefer a.free(conv_out_bits);
        const a_bits = try poisonedShell(a, gate_len);
        errdefer a.free(a_bits);
        const b_bits = try poisonedShell(a, gate_len);
        errdefer a.free(b_bits);
        const dt_bits = try poisonedShell(a, @intCast(Model.heads));
        errdefer a.free(dt_bits);
        const a_log_bits = try poisonedShell(a, @intCast(2 * Model.heads));
        errdefer a.free(a_log_bits);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (conv_out_bits[0..conv_len]) |*x| x.* = sampleDyadic(rng, -4, 2);
        for (a_bits[0..gate_len]) |*x| x.* = sampleDyadic(rng, -3, 2);
        for (b_bits[0..gate_len]) |*x| x.* = sampleDyadic(rng, -3, 2);
        for (dt_bits[0..@intCast(Model.heads)]) |*x| x.* = sampleDyadic(rng, -4, 1);
        for (0..@intCast(Model.heads)) |h| {
            const bits: u32 = @bitCast(sampleDyadicF32(rng, -4, 1));
            a_log_bits[2 * h] = @truncate(bits);
            a_log_bits[2 * h + 1] = @truncate(bits >> 16);
        }
        return .{ .a = a, .conv_out_bits = conv_out_bits, .a_bits = a_bits, .b_bits = b_bits, .a_log_bits = a_log_bits, .dt_bits = dt_bits };
    }

    pub fn deinit(f: Split) void {
        f.a.free(f.conv_out_bits);
        f.a.free(f.a_bits);
        f.a.free(f.b_bits);
        f.a.free(f.dt_bits);
        f.a.free(f.a_log_bits);
    }

    pub fn check(f: Split, soa_bits: []const u16, g_pairs: []const u16, beta_bits: []const u16, violation: ?*Violation) !Report {
        var r: Report = .{};
        const C: usize = @intCast(Model.conv_channels);
        const section: usize = @intCast(tokens * Model.linear_dim);
        for (0..@intCast(tokens)) |t| {
            for (0..@intCast(3 * Model.heads)) |head| {
                const hv: usize = @intCast(head % 16);
                var vals: [128]f64 = undefined;
                var ss: f64 = 0;
                var vmax: f64 = 0;
                for (0..128) |i| {
                    const cv = c.fromBf16(f.conv_out_bits[t * C + head * 128 + i]);
                    const sig = 1.0 / (1.0 + @exp(-cv));
                    vals[i] = cv * sig;
                    ss += vals[i] * vals[i];
                    vmax = @max(vmax, @abs(vals[i]));
                }
                const is_q = head < 16;
                const is_k = head >= 16 and head < 32;
                const inv: f64 = 1.0 / @sqrt(ss / 128.0 + 1e-6 / 128.0);
                const scale_f: f64 = if (is_q) 1.0 / 128.0 else @sqrt(1.0 / 128.0);
                const dst_base = (head / 16) * section + (t * 16 + hv) * 128;
                for (0..128) |i| {
                    const want = if (is_q or is_k) scale_f * (vals[i] * inv) else vals[i];
                    const got = c.fromBf16(soa_bits[dst_base + i]);
                    if (!std.math.isFinite(got)) {
                        violated(violation, .soa, dst_base + i, want, got, 0);
                        return error.NonFiniteOutput;
                    }
                    const d = @abs(got - want);
                    const bound = budget(40, 5, 3, want, vmax * vmax);
                    if (d > bound) {
                        violated(violation, .soa, dst_base + i, want, got, bound);
                        return error.OverBound;
                    }
                    const ratio = d / bound;
                    if (ratio > r.worst) r = .{ .worst = ratio, .where = dst_base + i };
                }
            }
            for (0..@intCast(Model.heads)) |hv| {
                // HOST Ash chain: the exact bf16 rounds the shader does
                const ref = gateRefAsh(
                    f.a_bits[t * 16 + hv],
                    f.b_bits[t * 16 + hv],
                    f32Of(f.a_log_bits, hv),
                    f.dt_bits[hv],
                );
                const g_idx = (t * 16 + hv) * 2;
                const g_got = f32Of(g_pairs, t * 16 + hv);
                if (!std.math.isFinite(g_got)) {
                    violated(violation, .g, g_idx / 2, ref.g, g_got, 0);
                    return error.NonFiniteOutput;
                }
                const gd = @abs(g_got - ref.g);
                const gbound = budget(1, 4, 2, ref.g, @abs(ref.g));
                if (gd > gbound) {
                    violated(violation, .g, g_idx / 2, ref.g, g_got, gbound);
                    return error.GateOverBound;
                }
                const gratio = gd / gbound;
                if (gratio > r.worst) r = .{ .worst = gratio, .where = g_idx };
                const beta_want = ref.beta;
                const beta_got = c.fromBf16(beta_bits[t * 16 + hv]);
                if (!std.math.isFinite(beta_got)) {
                    violated(violation, .beta, g_idx / 2, beta_want, beta_got, 0);
                    return error.NonFiniteOutput;
                }
                const bd = @abs(beta_got - beta_want);
                const bbound = budget(1, 3, 1, beta_want, @abs(beta_want));
                if (bd > bbound) {
                    violated(violation, .beta, g_idx / 2, beta_want, beta_got, bbound);
                    return error.GateOverBound;
                }
                const bratio = bd / bbound;
                if (bratio > r.worst) r = .{ .worst = bratio, .where = g_idx + 1 };
            }
        }
        return r;
    }
};

// ---- SplitOne: the gate split at exactly ONE token (the convDecodeStep tail) ----

// Split T=1: soa[3*2048], g/beta[16], conv_out[6144]; ACTUAL activated conv bits in `conv_out_bits`[6144] before check.
pub const SplitOne = struct {
    a: std.mem.Allocator,
    conv_out_bits: []u16,
    a_bits: []u16,
    b_bits: []u16,
    a_log_bits: []u16, // f32 pairs
    dt_bits: []u16,

    pub fn init(a: std.mem.Allocator, seed: u64) !SplitOne {
        const C: usize = @intCast(Model.conv_channels);
        const conv_out_bits = try poisonedShell(a, C);
        errdefer a.free(conv_out_bits);
        const a_bits = try poisonedShell(a, @intCast(Model.heads));
        errdefer a.free(a_bits);
        const b_bits = try poisonedShell(a, @intCast(Model.heads));
        errdefer a.free(b_bits);
        const dt_bits = try poisonedShell(a, @intCast(Model.heads));
        errdefer a.free(dt_bits);
        const a_log_bits = try poisonedShell(a, @intCast(2 * Model.heads));
        errdefer a.free(a_log_bits);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (conv_out_bits[0..C]) |*x| x.* = sampleDyadic(rng, -4, 2);
        for (a_bits[0..@intCast(Model.heads)]) |*x| x.* = sampleDyadic(rng, -3, 2);
        for (b_bits[0..@intCast(Model.heads)]) |*x| x.* = sampleDyadic(rng, -3, 2);
        for (dt_bits[0..@intCast(Model.heads)]) |*x| x.* = sampleDyadic(rng, -4, 1);
        for (0..@intCast(Model.heads)) |h| {
            const bits: u32 = @bitCast(sampleDyadicF32(rng, -4, 1));
            a_log_bits[2 * h] = @truncate(bits);
            a_log_bits[2 * h + 1] = @truncate(bits >> 16);
        }
        return .{ .a = a, .conv_out_bits = conv_out_bits, .a_bits = a_bits, .b_bits = b_bits, .a_log_bits = a_log_bits, .dt_bits = dt_bits };
    }

    pub fn deinit(f: SplitOne) void {
        f.a.free(f.conv_out_bits);
        f.a.free(f.a_bits);
        f.a.free(f.b_bits);
        f.a.free(f.dt_bits);
        f.a.free(f.a_log_bits);
    }

    // The gate section alone (g/beta are conv_out-independent), shared by check() and the host replay test.
    pub fn checkGates(f: SplitOne, g_pairs: []const u16, beta_bits: []const u16, violation: ?*Violation) !Report {
        var r: Report = .{};
        for (0..@intCast(Model.heads)) |hv| {
            // HOST Ash chain: the exact bf16 rounds the shader does
            const ref = gateRefAsh(
                f.a_bits[hv],
                f.b_bits[hv],
                f32Of(f.a_log_bits, hv),
                f.dt_bits[hv],
            );
            const g_got = f32Of(g_pairs, hv);
            if (!std.math.isFinite(g_got)) {
                violated(violation, .g, hv, ref.g, g_got, 0);
                return error.NonFiniteOutput;
            }
            const gd = @abs(g_got - ref.g);
            const gbound = budget(1, 4, 2, ref.g, @abs(ref.g));
            if (gd > gbound) {
                violated(violation, .g, hv, ref.g, g_got, gbound);
                return error.GateOverBound;
            }
            const gratio = gd / gbound;
            if (gratio > r.worst) r = .{ .worst = gratio, .where = hv };
            const beta_got = c.fromBf16(beta_bits[hv]);
            if (!std.math.isFinite(beta_got)) {
                violated(violation, .beta, hv, ref.beta, beta_got, 0);
                return error.NonFiniteOutput;
            }
            const bd = @abs(beta_got - ref.beta);
            const bbound = budget(1, 3, 1, ref.beta, @abs(ref.beta));
            if (bd > bbound) {
                violated(violation, .beta, hv, ref.beta, beta_got, bbound);
                return error.GateOverBound;
            }
            const bratio = bd / bbound;
            if (bratio > r.worst) r = .{ .worst = bratio, .where = 16 + hv };
        }
        return r;
    }

    pub fn check(f: SplitOne, soa_bits: []const u16, g_pairs: []const u16, beta_bits: []const u16, violation: ?*Violation) !Report {
        var r: Report = .{};
        const section: usize = @intCast(Model.linear_dim);
        for (0..@intCast(3 * Model.heads)) |head| {
            const hv: usize = @intCast(head % 16);
            var vals: [128]f64 = undefined;
            var ss: f64 = 0;
            var vmax: f64 = 0;
            for (0..128) |i| {
                const cv = c.fromBf16(f.conv_out_bits[head * 128 + i]);
                const sig = 1.0 / (1.0 + @exp(-cv));
                vals[i] = cv * sig;
                ss += vals[i] * vals[i];
                vmax = @max(vmax, @abs(vals[i]));
            }
            const is_q = head < 16;
            const is_k = head >= 16 and head < 32;
            const inv: f64 = 1.0 / @sqrt(ss / 128.0 + 1e-6 / 128.0);
            const scale_f: f64 = if (is_q) 1.0 / 128.0 else @sqrt(1.0 / 128.0);
            const dst_base = (head / 16) * section + hv * 128;
            for (0..128) |i| {
                const want = if (is_q or is_k) scale_f * (vals[i] * inv) else vals[i];
                const got = c.fromBf16(soa_bits[dst_base + i]);
                if (!std.math.isFinite(got)) {
                    violated(violation, .soa, dst_base + i, want, got, 0);
                    return error.NonFiniteOutput;
                }
                const d = @abs(got - want);
                const bound = budget(40, 5, 3, want, vmax * vmax);
                if (d > bound) {
                    violated(violation, .soa, dst_base + i, want, got, bound);
                    return error.OverBound;
                }
                const ratio = d / bound;
                if (ratio > r.worst) r = .{ .worst = ratio, .where = dst_base + i };
            }
        }
        const g = try f.checkGates(g_pairs, beta_bits, violation);
        if (g.worst > r.worst) r = g;
        return r;
    }
};

// ---- Conv prefill ----

// Prefill conv: fixture states the assembled input (state rows then qkv), weight, and checked output.
pub const Conv = struct {
    a: std.mem.Allocator,
    state_bits: []u16, // [3, C]
    qkv_bits: []u16, // [T, C]
    weight_bits: []u16, // [C, 4]
    conv_input_want: []u16, // [T+3, C]: what copy_u16 must assemble

    pub fn init(a: std.mem.Allocator, seed: u64) !Conv {
        const C: usize = @intCast(Model.conv_channels);
        const state = try poisonedShell(a, 3 * C);
        errdefer a.free(state);
        const qkv = try poisonedShell(a, @intCast(tokens * C));
        errdefer a.free(qkv);
        const weight = try poisonedShell(a, 4 * C);
        errdefer a.free(weight);
        const conv_input_want = try a.alloc(u16, @intCast((tokens + 3) * C));
        errdefer a.free(conv_input_want);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (state[0 .. 3 * C]) |*x| x.* = sampleDyadic(rng, -4, 2);
        for (qkv[0..@intCast(tokens * C)]) |*x| x.* = sampleDyadic(rng, -4, 2);
        for (weight[0 .. 4 * C]) |*x| x.* = sampleDyadic(rng, -3, 1);
        @memcpy(conv_input_want[0 .. 3 * C], state[0 .. 3 * C]);
        @memcpy(conv_input_want[3 * C ..], qkv[0..@intCast(tokens * C)]);
        return .{ .a = a, .state_bits = state, .qkv_bits = qkv, .weight_bits = weight, .conv_input_want = conv_input_want };
    }

    pub fn deinit(f: Conv) void {
        f.a.free(f.state_bits);
        f.a.free(f.qkv_bits);
        f.a.free(f.weight_bits);
        f.a.free(f.conv_input_want);
    }

    pub fn checkInput(f: Conv, got: []const u16) !void {
        if (std.mem.indexOfDiff(u16, f.conv_input_want, got[0..f.conv_input_want.len]) != null) return error.ConvInputMismatch;
    }

    pub fn checkOutput(f: Conv, bits: []const u16) !Report {
        var r: Report = .{};
        const C: usize = @intCast(Model.conv_channels);
        for (0..@intCast(tokens)) |t| {
            for (0..C) |ch| {
                var acc: f64 = 0;
                for (0..4) |k| {
                    acc += c.fromBf16(f.conv_input_want[(t + k) * C + ch]) * c.fromBf16(f.weight_bits[ch * 4 + k]);
                }
                const got = c.fromBf16(bits[t * C + ch]);
                if (!std.math.isFinite(got)) return error.NonFiniteOutput;
                const d = @abs(got - acc);
                const bound = budget(4, 0, 0, acc, @abs(acc));
                if (d > bound) return error.OverBound;
                const ratio = d / bound;
                if (ratio > r.worst) r = .{ .worst = ratio, .where = t * C + ch };
            }
        }
        return r;
    }
};

// ---- Prefill recurrence and decode: one serial f64 walker ----

pub const RecurWalk = struct {
    // state[head][row][dk]; y[token][head*128+row]; want_snap = FULL state after snap_after-1 (scan's snapshot slot)
    pub fn run(
        soa_bits: []const u16,
        g_pairs: []const u16, // f32 as u16 pairs [T*16*2]
        beta_bits: []const u16,
        h0_bits: []const u16, // f32 pairs [262144*2]
        want_y: []f64,
        want_state: []f64, // [262144]
        max_state: *f64,
        snap_after: ?u64,
        want_snap: ?[]f64,
    ) void {
        const dk: usize = @intCast(Model.head_dim);
        const heads: usize = @intCast(Model.heads);
        const state_elems: usize = @intCast(Model.state_elements);
        const section: usize = @intCast(tokens * Model.linear_dim);
        var st: [16 * 128 * 128]f64 = @splat(0);
        for (0..heads) |h| {
            for (0..dk) |row| {
                for (0..dk) |i| {
                    st[h * dk * dk + row * dk + i] = f32Of(h0_bits, h * dk * dk + row * dk + i);
                }
            }
        }
        for (0..@intCast(tokens)) |t| {
            for (0..heads) |h| {
                const g: f64 = f32Of(g_pairs, t * 16 + h);
                const beta: f64 = c.fromBf16(beta_bits[t * 16 + h]);
                const qsec = 0 * section + (t * 16 + h) * dk;
                const ksec = 1 * section + (t * 16 + h) * dk;
                const vsec = 2 * section + (t * 16 + h) * dk;
                for (0..dk) |row| {
                    var kv: f64 = 0;
                    for (0..dk) |i| {
                        const s = st[h * dk * dk + row * dk + i];
                        max_state.* = @max(max_state.*, @abs(s));
                        kv += (s * g) * c.fromBf16(soa_bits[ksec + i]);
                    }
                    const delta = (c.fromBf16(soa_bits[vsec + row]) - kv) * beta;
                    var o: f64 = 0;
                    for (0..dk) |i| {
                        const idx = h * dk * dk + row * dk + i;
                        const ns = (st[idx] * g) + delta * c.fromBf16(soa_bits[ksec + i]);
                        st[idx] = ns;
                        o += ns * c.fromBf16(soa_bits[qsec + i]);
                    }
                    want_y[(t * 16 + h) * dk + row] = o;
                }
            }
            // the interior chunk boundary's snapshot slot
            if (snap_after) |after| if (want_snap) |ws| {
                if (@as(u64, t) + 1 == after) @memcpy(ws[0..state_elems], st[0..state_elems]);
            };
        }
        @memcpy(want_state[0..state_elems], st[0..state_elems]);
    }
};

// Prefill recurrence gate: reference inputs are the ACTUAL soa/g/beta bits the GPU produced (downloaded first).
pub const Recur = struct {
    a: std.mem.Allocator,
    h0_bits: []u16, // f32 pairs
    want_y: []f64,
    want_state: []f64,
    max_state: f64,
    // Chunk boundary reference snapshot: non-null iff chunked (0 < chunk < tokens, chunks >= 2; boundary at `chunk`).
    want_snap: ?[]f64,

    pub fn init(a: std.mem.Allocator, soa_bits: []const u16, g_pairs: []const u16, beta_bits: []const u16, seed: u64, chunk: u64) !Recur {
        const state_elems: usize = @intCast(Model.state_elements);
        const y_elems: usize = @intCast(tokens * Model.linear_dim);
        const h0_bits = try poisonedShell(a, 2 * state_elems);
        errdefer a.free(h0_bits);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (0..state_elems) |i| {
            const bits: u32 = @bitCast(sampleDyadicF32(rng, -6, -2)); // small state
            h0_bits[2 * i] = @truncate(bits);
            h0_bits[2 * i + 1] = @truncate(bits >> 16);
        }
        const want_y = try a.alloc(f64, y_elems);
        errdefer a.free(want_y);
        const want_state = try a.alloc(f64, state_elems);
        errdefer a.free(want_state);
        var max_state: f64 = 0;
        var want_snap: ?[]f64 = null;
        errdefer if (want_snap) |ws| a.free(ws);
        if (chunk > 0 and chunk < tokens) {
            // chunks >= 2: exactly ONE interior boundary (after `chunk` tokens)
            const ws = try a.alloc(f64, state_elems);
            want_snap = ws;
            RecurWalk.run(soa_bits, g_pairs, beta_bits, h0_bits, want_y, want_state, &max_state, chunk, ws);
        } else {
            RecurWalk.run(soa_bits, g_pairs, beta_bits, h0_bits, want_y, want_state, &max_state, null, null);
        }
        return .{ .a = a, .h0_bits = h0_bits, .want_y = want_y, .want_state = want_state, .max_state = max_state, .want_snap = want_snap };
    }

    pub fn deinit(f: Recur) void {
        f.a.free(f.h0_bits);
        f.a.free(f.want_y);
        f.a.free(f.want_state);
        if (f.want_snap) |ws| f.a.free(ws);
    }

    // Snapshot slot vs boundary reference; poison and partial writes are refused by the same finite/budget gates as hf.
    pub fn checkSnap(f: Recur, snap_pairs: []const u16, violation: ?*Violation) !Report {
        const ws = f.want_snap orelse return error.NoSnapshotExpected;
        var r: Report = .{};
        for (ws, 0..) |w, i| {
            const got = f32Of(snap_pairs, i);
            if (!std.math.isFinite(got)) {
                violated(violation, .snapshot, i, w, got, 0);
                return error.NonFiniteOutput;
            }
            const d = @abs(got - w);
            const bound = budget(130 * 4 + 2, 2, 0, w, f.max_state); // chunk tokens deep
            if (d > bound) {
                violated(violation, .snapshot, i, w, got, bound);
                return error.OverBound;
            }
            const ratio = d / bound;
            if (ratio > r.worst) r = .{ .worst = ratio, .where = i };
        }
        return r;
    }

    pub fn check(f: Recur, y_bits: []const u16, hf_pairs: []const u16, violation: ?*Violation) !Report {
        var r: Report = .{};
        for (f.want_y, 0..) |w, i| {
            const got = c.fromBf16(y_bits[i]);
            if (!std.math.isFinite(got)) {
                violated(violation, .y, i, w, got, 0);
                return error.NonFiniteOutput;
            }
            const d = @abs(got - w);
            const bound = budget(130 * @as(f64, tokens) + 130, 4, 0, w, f.max_state);
            if (d > bound) {
                violated(violation, .y, i, w, got, bound);
                return error.OverBound;
            }
            const ratio = d / bound;
            if (ratio > r.worst) r = .{ .worst = ratio, .where = i };
        }
        for (f.want_state, 0..) |w, i| {
            const got = f32Of(hf_pairs, i);
            if (!std.math.isFinite(got)) {
                violated(violation, .state, i, w, got, 0);
                return error.NonFiniteOutput;
            }
            const d = @abs(got - w);
            const bound = budget(130 * @as(f64, tokens) + 2, 2, 0, w, f.max_state);
            if (d > bound) {
                violated(violation, .state, i, w, got, bound);
                return error.OverBound;
            }
            const ratio = d / bound;
            if (ratio > r.worst) r = .{ .worst = ratio, .where = i };
        }
        return r;
    }
};

// Decode gate: one raw-gates token from state_in; g/beta reference comes from a/b/A_log/dt (kernel prologue inputs).
pub const Decode = struct {
    a: std.mem.Allocator,
    soa_bits: []u16, // [3*2048]
    a_bits: []u16,
    b_bits: []u16,
    a_log_bits: []u16, // f32 pairs
    dt_bits: []u16,
    state_in_bits: []u16, // f32 pairs
    want_y: []f64,
    want_state: []f64,
    max_state: f64,

    pub fn init(a: std.mem.Allocator, seed: u64) !Decode {
        const state_elems: usize = @intCast(Model.state_elements);
        const soa_bits = try poisonedShell(a, @intCast(3 * Model.linear_dim));
        errdefer a.free(soa_bits);
        const a_bits = try poisonedShell(a, @intCast(Model.heads));
        errdefer a.free(a_bits);
        const b_bits = try poisonedShell(a, @intCast(Model.heads));
        errdefer a.free(b_bits);
        const dt_bits = try poisonedShell(a, @intCast(Model.heads));
        errdefer a.free(dt_bits);
        const a_log_bits = try poisonedShell(a, @intCast(2 * Model.heads));
        errdefer a.free(a_log_bits);
        const state_in_bits = try poisonedShell(a, 2 * state_elems);
        errdefer a.free(state_in_bits);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (soa_bits[0..@intCast(3 * Model.linear_dim)]) |*x| x.* = sampleDyadic(rng, -4, 2);
        for (a_bits[0..@intCast(Model.heads)]) |*x| x.* = sampleDyadic(rng, -3, 2);
        for (b_bits[0..@intCast(Model.heads)]) |*x| x.* = sampleDyadic(rng, -3, 2);
        for (dt_bits[0..@intCast(Model.heads)]) |*x| x.* = sampleDyadic(rng, -4, 1);
        for (0..@intCast(Model.heads)) |h| {
            const bits: u32 = @bitCast(sampleDyadicF32(rng, -4, 1));
            a_log_bits[2 * h] = @truncate(bits);
            a_log_bits[2 * h + 1] = @truncate(bits >> 16);
        }
        for (0..state_elems) |i| {
            const bits: u32 = @bitCast(sampleDyadicF32(rng, -6, -2));
            state_in_bits[2 * i] = @truncate(bits);
            state_in_bits[2 * i + 1] = @truncate(bits >> 16);
        }
        // serial reference
        const want_y = try a.alloc(f64, @intCast(Model.linear_dim));
        errdefer a.free(want_y);
        const want_state = try a.alloc(f64, state_elems);
        errdefer a.free(want_state);
        var max_state: f64 = 0;
        const section: usize = @intCast(Model.linear_dim);
        var st: [16 * 128 * 128]f64 = @splat(0);
        for (0..16 * 128 * 128) |i| st[i] = f32Of(state_in_bits, i);
        for (0..16) |h| {
            // HOST Ash chain: the exact bf16 rounds the decode prologue does
            const gref = gateRefAsh(
                a_bits[h],
                b_bits[h],
                f32Of(a_log_bits, h),
                dt_bits[h],
            );
            const g = gref.g;
            const beta = gref.beta;
            for (0..128) |row| {
                var kv: f64 = 0;
                for (0..128) |i| {
                    const s = st[h * 128 * 128 + row * 128 + i];
                    max_state = @max(max_state, @abs(s));
                    kv += (s * g) * c.fromBf16(soa_bits[1 * section + h * 128 + i]);
                }
                const delta = (c.fromBf16(soa_bits[2 * section + h * 128 + row]) - kv) * beta;
                var o: f64 = 0;
                for (0..128) |i| {
                    const idx = h * 128 * 128 + row * 128 + i;
                    const ns = (st[idx] * g) + delta * c.fromBf16(soa_bits[1 * section + h * 128 + i]);
                    st[idx] = ns;
                    o += ns * c.fromBf16(soa_bits[0 * section + h * 128 + i]);
                }
                want_y[h * 128 + row] = o;
            }
        }
        @memcpy(want_state[0..state_elems], st[0..state_elems]);
        return .{
            .a = a,
            .soa_bits = soa_bits,
            .a_bits = a_bits,
            .b_bits = b_bits,
            .a_log_bits = a_log_bits,
            .dt_bits = dt_bits,
            .state_in_bits = state_in_bits,
            .want_y = want_y,
            .want_state = want_state,
            .max_state = max_state,
        };
    }

    pub fn deinit(f: Decode) void {
        f.a.free(f.soa_bits);
        f.a.free(f.a_bits);
        f.a.free(f.b_bits);
        f.a.free(f.dt_bits);
        f.a.free(f.a_log_bits);
        f.a.free(f.state_in_bits);
        f.a.free(f.want_y);
        f.a.free(f.want_state);
    }

    pub fn check(f: Decode, y_bits: []const u16, state_out_pairs: []const u16, violation: ?*Violation) !Report {
        var r: Report = .{};
        for (f.want_y, 0..) |w, i| {
            const got = c.fromBf16(y_bits[i]);
            if (!std.math.isFinite(got)) {
                violated(violation, .y, i, w, got, 0);
                return error.NonFiniteOutput;
            }
            const d = @abs(got - w);
            const bound = budget(260, 4, 0, w, f.max_state);
            if (d > bound) {
                violated(violation, .y, i, w, got, bound);
                return error.OverBound;
            }
            const ratio = d / bound;
            if (ratio > r.worst) r = .{ .worst = ratio, .where = i };
        }
        for (f.want_state, 0..) |w, i| {
            const got = f32Of(state_out_pairs, i);
            if (!std.math.isFinite(got)) {
                violated(violation, .state, i, w, got, 0);
                return error.NonFiniteOutput;
            }
            const d = @abs(got - w);
            const bound = budget(132, 2, 0, w, f.max_state);
            if (d > bound) {
                violated(violation, .state, i, w, got, bound);
                return error.OverBound;
            }
            const ratio = d / bound;
            if (ratio > r.worst) r = .{ .worst = ratio, .where = i };
        }
        return r;
    }
};

// ---- Gated norm ----

// gatedNorm tokens=4: PER-HEAD 128-wide RMS (Ash gdn_post), [4,16,128]; out=bf16(silu_div(z)*bf16(w*(x*inv))).
pub const Norm = struct {
    a: std.mem.Allocator,
    y_bits: []u16, // norm input
    z_bits: []u16, // gate
    w_bits: []u16, // [128]: per-head weight, repeated for every head
    eps: f64,

    pub fn init(a: std.mem.Allocator, seed: u64) !Norm {
        const total: usize = @intCast(4 * Model.linear_dim);
        const y_bits = try poisonedShell(a, total);
        errdefer a.free(y_bits);
        const z_bits = try poisonedShell(a, total);
        errdefer a.free(z_bits);
        const w_bits = try poisonedShell(a, @intCast(Model.head_dim));
        errdefer a.free(w_bits);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (y_bits[0..total]) |*x| x.* = sampleDyadic(rng, -4, 2);
        for (z_bits[0..total]) |*x| x.* = sampleDyadic(rng, -4, 2);
        for (w_bits[0..@intCast(Model.head_dim)]) |*x| x.* = sampleDyadic(rng, -3, 1);
        return .{ .a = a, .y_bits = y_bits, .z_bits = z_bits, .w_bits = w_bits, .eps = 1e-6 };
    }

    pub fn deinit(f: Norm) void {
        f.a.free(f.y_bits);
        f.a.free(f.z_bits);
        f.a.free(f.w_bits);
    }

    // Per-head-row ref (tokens=4, 16 heads, width 128): each row of 128 has its OWN mean square; weight = column index.
    pub fn check(f: Norm, bits: []const u16) !Report {
        var r: Report = .{};
        const width: usize = @intCast(Model.head_dim);
        for (0..4 * @as(usize, @intCast(Model.heads))) |row| {
            var ss: f64 = 0;
            var xmax: f64 = 0;
            for (0..width) |i| {
                const v = c.fromBf16(f.y_bits[row * width + i]);
                ss += v * v;
                xmax = @max(xmax, @abs(v));
            }
            const inv: f64 = 1.0 / @sqrt(ss / @as(f64, width) + f.eps);
            for (0..width) |i| {
                const x = c.fromBf16(f.y_bits[row * width + i]);
                const z = c.fromBf16(f.z_bits[row * width + i]);
                const scaled = x * inv;
                const normed = c.fromBf16(c.toBf16(@floatCast(c.fromBf16(f.w_bits[i]) * scaled)));
                const silu = z / (1.0 + @exp(-z));
                const want = silu * normed;
                const got = c.fromBf16(bits[row * width + i]);
                if (!std.math.isFinite(got)) return error.NonFiniteOutput;
                const d = @abs(got - want);
                const bound = budget(128, 3, 2, want, xmax * xmax * @as(f64, width));
                if (d > bound) return error.OverBound;
                const ratio = d / bound;
                if (ratio > r.worst) r = .{ .worst = ratio, .where = row * width + i };
            }
        }
        return r;
    }
};

// Replay the decode seed through the production gate checker; report old-reference drift, not a GPU-failure cause.
test "gateRefAsh replays the fixture seed and diagnoses g and beta separately" {
    const a = testing.allocator;
    const f = try SplitOne.init(a, 0x57ac);
    defer f.deinit();
    // what the shader computes, on the host: g f32 words, beta bf16
    const g_pairs = try a.alloc(u16, 2 * @as(usize, @intCast(Model.heads)));
    defer a.free(g_pairs);
    const beta_bits = try a.alloc(u16, @intCast(Model.heads));
    defer a.free(beta_bits);
    var old_worst: f64 = 0;
    for (0..@intCast(Model.heads)) |hv| {
        const ash = gateRefAsh(f.a_bits[hv], f.b_bits[hv], f32Of(f.a_log_bits, hv), f.dt_bits[hv]);
        const gbits: u32 = @bitCast(@as(f32, @floatCast(ash.g)));
        g_pairs[2 * hv] = @truncate(gbits);
        g_pairs[2 * hv + 1] = @truncate(gbits >> 16);
        beta_bits[hv] = c.toBf16(@floatCast(ash.beta));
        // evidence: the OLD unrounded f64 reference vs the shader chain
        const old = gateRef(
            c.fromBf16(f.a_bits[hv]),
            c.fromBf16(f.b_bits[hv]),
            f32Of(f.a_log_bits, hv),
            c.fromBf16(f.dt_bits[hv]),
        );
        const gd = @abs(old.g - ash.g);
        const gbound = budget(1, 4, 2, ash.g, @abs(ash.g));
        old_worst = @max(old_worst, gd / gbound);
    }
    std.debug.print("HOST gate old-reference drift seed=0x57ac max_ratio={e} (not a GPU diagnosis)\n", .{old_worst});
    // the exact chain replays CLEAN through the same checker
    const rep = try f.checkGates(g_pairs, beta_bits, null);
    try testing.expect(rep.worst <= 1);
    var diag: Violation = undefined;
    const g0 = g_pairs[0];
    const g1 = g_pairs[1];
    g_pairs[0] = 0;
    g_pairs[1] = 0x4f00;
    try testing.expectError(error.GateOverBound, f.checkGates(g_pairs, beta_bits, &diag));
    try testing.expectEqual(Output.g, diag.output);
    try testing.expectEqual(@as(usize, 0), diag.index);
    g_pairs[0] = g0;
    g_pairs[1] = g1;
    const beta0 = beta_bits[0];
    beta_bits[0] = 0x4f00;
    try testing.expectError(error.GateOverBound, f.checkGates(g_pairs, beta_bits, &diag));
    try testing.expectEqual(Output.beta, diag.output);
    try testing.expectEqual(@as(usize, 0), diag.index);
    beta_bits[0] = beta0;
    // subnormal census (log-only): a_log<=3, sp<=~4.5 => g~exp(-20.085*4.5)~5.6e-40; FTZ=>g=0; host ref:no FTZ.
    var min_g: f32 = std.math.floatMax(f32);
    var subnormals: usize = 0;
    for (0..@intCast(Model.heads)) |hv| {
        const ash = gateRefAsh(f.a_bits[hv], f.b_bits[hv], f32Of(f.a_log_bits, hv), f.dt_bits[hv]);
        const g: f32 = @floatCast(ash.g);
        if (g > 0) min_g = @min(min_g, g);
        const bits: u32 = @bitCast(g);
        const exponent = (bits >> 23) & 0xff;
        if (exponent == 0 and (bits & 0x7fffff) != 0) subnormals += 1; // f32 subnormal
        if (g == 0) subnormals += 1; // flushed/underflowed to zero
    }
    std.debug.print("HOST gate g census (exact seed): min_positive_g={e} subnormal_or_zero_g={d}/16\n", .{ min_g, subnormals });
}

const testing = std.testing;

test "budget grows with depth and stays finite on zeros" {
    try testing.expect(budget(300, 6, 2, 1, 1) > budget(2, 6, 1, 1, 1));
    try testing.expect(budget(2, 6, 1, 0, 0) > 0);
}

test "gateRef behaves like exp(-exp(A_log) softplus) and a sigmoid" {
    const g0 = gateRef(0, 0, 0, 0);
    try testing.expectApproxEqAbs(@as(f64, @exp(-@exp(0.0) * std.math.log1p(@as(f64, 1.0)))), g0.g, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.5), g0.beta, 1e-12);
}

test "sampleDyadic is exactly bf16" {
    var prng = std.Random.DefaultPrng.init(9);
    const rng = prng.random();
    for (0..64) |_| {
        const bits = sampleDyadic(rng, -5, 3);
        const v = c.fromBf16(bits);
        try testing.expectEqual(v, c.fromBf16(c.toBf16(v)));
    }
}

// Negative plants vs PRODUCTION Norm checker: honest output passes; a wrong finite and every nonfinite class refused.
test "norm checker refuses planted outputs" {
    const a = testing.allocator;
    const f = try Norm.init(a, 0x18);
    defer f.deinit();
    const width: usize = @intCast(Model.head_dim);
    const head_rows: usize = 4 * @as(usize, @intCast(Model.heads));
    const good = try a.alloc(u16, head_rows * width);
    defer a.free(good);
    for (0..head_rows) |row| {
        var ss: f64 = 0;
        for (0..width) |i| {
            const v = c.fromBf16(f.y_bits[row * width + i]);
            ss += v * v;
        }
        const inv: f64 = 1.0 / @sqrt(ss / @as(f64, width) + f.eps);
        for (0..width) |i| {
            const x = c.fromBf16(f.y_bits[row * width + i]);
            const z = c.fromBf16(f.z_bits[row * width + i]);
            // the same Metal gdn_post association the checker references
            const normed = c.fromBf16(c.toBf16(@floatCast(c.fromBf16(f.w_bits[i]) * (x * inv))));
            const silu = z / (1.0 + @exp(-z));
            good[row * width + i] = c.toBf16(@floatCast(silu * normed));
        }
    }
    const nrep = try f.check(good);
    try testing.expect(nrep.worst <= 1);
    good[0] = 0x4f00; // 2^31: an absolutely wrong finite, far past any budget
    try testing.expectError(error.OverBound, f.check(good));
    good[0] = 0x7f80;
    try testing.expectError(error.NonFiniteOutput, f.check(good));
}

// Carry-out self-tests: rounded state PASSES; poisoned (0xffff=NaN) carry-out REJECTED; dead store (all-poison) fails.
test "recur checker accepts rounded state and refuses poisoned carry-out" {
    const a = testing.allocator;
    // fabricate the split's outputs as dyadic bits (the walker reads them)
    const soa_len: usize = @intCast(3 * tokens * Model.linear_dim);
    const soa_bits = try a.alloc(u16, soa_len);
    defer a.free(soa_bits);
    var prng = std.Random.DefaultPrng.init(0x51);
    const rng = prng.random();
    for (soa_bits) |*x| x.* = sampleDyadic(rng, -3, 1);
    const g_len: usize = @intCast(tokens * Model.heads * 2);
    const g_pairs = try a.alloc(u16, g_len);
    defer a.free(g_pairs);
    for (g_pairs) |*x| {
        const bits: u32 = @bitCast(sampleDyadicF32(rng, -4, 0));
        x.* = @truncate(bits);
    }
    const beta_len: usize = @intCast(tokens * Model.heads);
    const beta_bits = try a.alloc(u16, beta_len);
    defer a.free(beta_bits);
    for (beta_bits) |*x| x.* = sampleDyadic(rng, -3, 1);
    const f = try Recur.init(a, soa_bits, g_pairs, beta_bits, 0x52, tokens); // single chunk: no interior snapshot
    defer f.deinit();
    const state_elems: usize = @intCast(Model.state_elements);
    // correctly reference-rounded state must pass
    const hf_good = try a.alloc(u16, 2 * state_elems);
    defer a.free(hf_good);
    for (f.want_state, 0..) |w, i| {
        const bits: u32 = @bitCast(@as(f32, @floatCast(w)));
        hf_good[2 * i] = @truncate(bits);
        hf_good[2 * i + 1] = @truncate(bits >> 16);
    }
    const y_good = try a.alloc(u16, @intCast(tokens * Model.linear_dim));
    defer a.free(y_good);
    for (f.want_y, 0..) |w, i| y_good[i] = c.toBf16(@floatCast(w));
    const rep = try f.check(y_good, hf_good, null);
    try testing.expect(rep.worst <= 1);
    // poisoned carry-out (NaN word) must be rejected
    const hf_bad = try a.dupe(u16, hf_good);
    defer a.free(hf_bad);
    hf_bad[0] = 0xffff;
    hf_bad[1] = 0xffff;
    try testing.expectError(error.NonFiniteOutput, f.check(y_good, hf_bad, null));
    // a dead store: all-poison state must be rejected
    @memset(hf_bad, 0xffff);
    try testing.expectError(error.NonFiniteOutput, f.check(y_good, hf_bad, null));
    // single-chunk Recur expects NO snapshot
    try testing.expect(f.want_snap == null);
    // a chunked Recur has exactly one interior boundary reference; the correctly-rounded slot passes, poison is refused
    const fc = try Recur.init(a, soa_bits, g_pairs, beta_bits, 0x54, 4);
    defer fc.deinit();
    const ws = fc.want_snap.?;
    try testing.expectEqual(@as(usize, @intCast(Model.state_elements)), ws.len);
    const snap_good = try a.alloc(u16, 2 * @as(usize, @intCast(Model.state_elements)));
    defer a.free(snap_good);
    for (ws, 0..) |w, i| {
        const bits: u32 = @bitCast(@as(f32, @floatCast(w)));
        snap_good[2 * i] = @truncate(bits);
        snap_good[2 * i + 1] = @truncate(bits >> 16);
    }
    const snap_rep = try fc.checkSnap(snap_good, null);
    try testing.expect(snap_rep.worst <= 1);
    snap_good[2] = 0xffff;
    snap_good[3] = 0xffff;
    try testing.expectError(error.NonFiniteOutput, fc.checkSnap(snap_good, null));
    @memset(snap_good, 0xffff);
    try testing.expectError(error.NonFiniteOutput, fc.checkSnap(snap_good, null));
}

test "decode checker refuses poisoned carry-out state" {
    const a = testing.allocator;
    const f = try Decode.init(a, 0x53);
    defer f.deinit();
    const state_elems: usize = @intCast(Model.state_elements);
    const y_good = try a.alloc(u16, @intCast(Model.linear_dim));
    defer a.free(y_good);
    for (f.want_y, 0..) |w, i| y_good[i] = c.toBf16(@floatCast(w));
    const souts_good = try a.alloc(u16, 2 * state_elems);
    defer a.free(souts_good);
    for (f.want_state, 0..) |w, i| {
        const bits: u32 = @bitCast(@as(f32, @floatCast(w)));
        souts_good[2 * i] = @truncate(bits);
        souts_good[2 * i + 1] = @truncate(bits >> 16);
    }
    const rep = try f.check(y_good, souts_good, null);
    try testing.expect(rep.worst <= 1);
    const souts_bad = try a.dupe(u16, souts_good);
    defer a.free(souts_bad);
    souts_bad[100] = 0xffff;
    souts_bad[101] = 0xffff;
    try testing.expectError(error.NonFiniteOutput, f.check(y_good, souts_bad, null));
}

test "fixture initialization fails cleanly under allocation failure" {
    var failing = std.testing.FailingAllocator.init(testing.allocator, .{ .fail_index = 1 });
    try testing.expectError(error.OutOfMemory, Norm.init(failing.allocator(), 0x19));
    try testing.expectError(error.OutOfMemory, Decode.init(failing.allocator(), 0x1a));
}

test "decode seed replays the measured wrong raw-gates branch at y zero" {
    const f = try Decode.init(testing.allocator, 0x57a8);
    defer f.deinit();
    const replay = struct {
        fn first(d: Decode, g: f32, beta: f32) u16 {
            var kv: f32 = 0;
            for (0..128) |i| {
                const state: f32 = @floatCast(f32Of(d.state_in_bits, i));
                const decayed: f32 = state * g;
                const term: f32 = decayed * c.fromBf16(d.soa_bits[2048 + i]);
                kv += term;
            }
            const delta: f32 = (c.fromBf16(d.soa_bits[4096]) - kv) * beta;
            var out: f32 = 0;
            for (0..128) |i| {
                const state: f32 = @floatCast(f32Of(d.state_in_bits, i));
                const decayed: f32 = state * g;
                const update: f32 = delta * c.fromBf16(d.soa_bits[2048 + i]);
                const next: f32 = decayed + update;
                const term: f32 = next * c.fromBf16(d.soa_bits[i]);
                out += term;
            }
            return c.toBf16(out);
        }
    }.first;
    const wrong = replay(f, c.fromBf16(f.a_bits[0]), c.fromBf16(f.b_bits[0]));
    try testing.expectEqual(c.toBf16(892), wrong);
    const gates = gateRefAsh(f.a_bits[0], f.b_bits[0], f32Of(f.a_log_bits, 0), f.dt_bits[0]);
    const corrected = replay(f, @floatCast(gates.g), @floatCast(gates.beta));
    const bound = budget(260, 4, 0, f.want_y[0], f.max_state);
    try testing.expect(@abs(c.fromBf16(corrected) - f.want_y[0]) <= bound);
    try testing.expect(@abs(c.fromBf16(wrong) - f.want_y[0]) > bound);
    std.debug.print("HOST decode first-element replay seed=0x57a8 skipped-raw={e} corrected={e} want={e} bound={e}\n", .{ c.fromBf16(wrong), c.fromBf16(corrected), f.want_y[0], bound });
}
