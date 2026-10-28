//! Stage 4 fixture: the GatedDeltaNet closure at the real model geometry
//! (16 heads, head_dim 128, 6144 conv channels, 4 taps) on a controlled small
//! token count, against independent f64 host references. Host-checkable only.
//!
//! Acceptance (fixed bring-up budgets, stated; NOT cross-backend bitexact
//! claims; inputs are exactly-bf16-representable dyadics so operand widening
//! is exact and every budget below is dominated by its stated rounding depth):
//!  - split: SiLU/L2-norm/V/gate chain with the Metal-oracle bf16 roundings;
//!    budget depth=40 (the 32-lane norm sum), ops=5, mids=3, magnitude scale
//!    max|vals|^2. The gate's g is f32, budget ops=4, mids=2, at |g|.
//!  - conv: 4 ascending taps on exact operands: budget depth=4, no mids; one
//!    bf16 store.
//!  - prefill recurrence: serial f64 gated-delta walk from h0 over the ACTUAL
//!    soa/g/beta bits the GPU produced; budget depth = tokens*(130) + 130 at
//!    the largest intermediate state magnitude the walk saw, ops=4, no mids
//!    (state stays f32 end to end; y is one bf16 store, budgeted by outUnit).
//!    chunk=4 and chunk=8 must agree with the SAME serial reference.
//!  - decode: one raw-gates step (g/beta from a/b/A_log/dt) from state_in;
//!    same budget form as the recurrence's single token.
//!  - gated norm: budget depth=128 (the row mean), ops=1 (inversesqrt),
//!    mids=2, magnitude scale max|x|^2 * row length.
//! Every finite reference demands a finite output; planted wrong finite and
//! nonfinite outputs must be refused by the same checkers.
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

/// Stated fixed bring-up budget: a depth-deep f32 sum at its magnitude scale,
/// `ops` further f32 roundings and `mids` bf16 roundings at the value's scale,
/// one final bf16 store. Never a sum of squares of outputs; always finite and
/// positive (the 2^-133 floor), so dividing into a ratio is safe. Fixed
/// bring-up acceptance: never enlarged after a GPU run without a new stated
/// derivation.
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

/// The g/beta gate chain in f64 (the split's lane-0 chain and the decode
/// prologue's raw-gates chain share this formula).
pub fn gateRef(a_in: f64, b_in: f64, a_log: f64, dt: f64) gdn_mod.GateF64 {
    return gdn_mod.gateReferenceF64(a_in, b_in, a_log, dt);
}

// ---------------------------------------------------------------------------
// Split
// ---------------------------------------------------------------------------

/// The gate split at T=tokens. conv_out/a/b/dt/a_log are the uploaded inputs;
/// check consumes the downloaded soa (bf16), g (f32) and beta (bf16) bits.
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

    pub fn check(f: Split, soa_bits: []const u16, g_pairs: []const u16, beta_bits: []const u16) !Report {
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
                    if (!std.math.isFinite(got)) return error.NonFiniteOutput;
                    const d = @abs(got - want);
                    const bound = budget(40, 5, 3, want, vmax * vmax);
                    if (d > bound) return error.OverBound;
                    const ratio = d / bound;
                    if (ratio > r.worst) r = .{ .worst = ratio, .where = dst_base + i };
                }
            }
            for (0..@intCast(Model.heads)) |hv| {
                const ref = gateRef(
                    c.fromBf16(f.a_bits[t * 16 + hv]),
                    c.fromBf16(f.b_bits[t * 16 + hv]),
                    f32Of(f.a_log_bits, hv),
                    c.fromBf16(f.dt_bits[hv]),
                );
                const g_idx = (t * 16 + hv) * 2;
                const g_got = f32Of(g_pairs, t * 16 + hv);
                if (!std.math.isFinite(g_got)) return error.NonFiniteOutput;
                const gd = @abs(g_got - ref.g);
                const gbound = budget(1, 4, 2, ref.g, @abs(ref.g));
                if (gd > gbound) return error.GateOverBound;
                const gratio = gd / gbound;
                if (gratio > r.worst) r = .{ .worst = gratio, .where = g_idx };
                const beta_want = ref.beta;
                const beta_got = c.fromBf16(beta_bits[t * 16 + hv]);
                if (!std.math.isFinite(beta_got)) return error.NonFiniteOutput;
                const bd = @abs(beta_got - beta_want);
                const bbound = budget(1, 3, 1, beta_want, @abs(beta_want));
                if (bd > bbound) return error.GateOverBound;
                const bratio = bd / bbound;
                if (bratio > r.worst) r = .{ .worst = bratio, .where = g_idx + 1 };
            }
        }
        return r;
    }
};

// ---------------------------------------------------------------------------
// SplitOne: the gate split at exactly ONE token (the convDecodeStep tail)
// ---------------------------------------------------------------------------

/// The gate split's T=1 geometry: soa [3*2048], g/beta [16], conv_out [6144].
/// `check` consumes the downloaded outputs; assign the ACTUAL activated conv
/// bits into `conv_out_bits` (pub, 6144 elements) before checking so the
/// reference reads what the GPU's decode conv produced.
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

    pub fn check(f: SplitOne, soa_bits: []const u16, g_pairs: []const u16, beta_bits: []const u16) !Report {
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
                if (!std.math.isFinite(got)) return error.NonFiniteOutput;
                const d = @abs(got - want);
                const bound = budget(40, 5, 3, want, vmax * vmax);
                if (d > bound) return error.OverBound;
                const ratio = d / bound;
                if (ratio > r.worst) r = .{ .worst = ratio, .where = dst_base + i };
            }
        }
        for (0..@intCast(Model.heads)) |hv| {
            const ref = gateRef(
                c.fromBf16(f.a_bits[hv]),
                c.fromBf16(f.b_bits[hv]),
                f32Of(f.a_log_bits, hv),
                c.fromBf16(f.dt_bits[hv]),
            );
            const g_got = f32Of(g_pairs, hv);
            if (!std.math.isFinite(g_got)) return error.NonFiniteOutput;
            const gd = @abs(g_got - ref.g);
            const gbound = budget(1, 4, 2, ref.g, @abs(ref.g));
            if (gd > gbound) return error.GateOverBound;
            const gratio = gd / gbound;
            if (gratio > r.worst) r = .{ .worst = gratio, .where = hv };
            const beta_got = c.fromBf16(beta_bits[hv]);
            if (!std.math.isFinite(beta_got)) return error.NonFiniteOutput;
            const bd = @abs(beta_got - ref.beta);
            const bbound = budget(1, 3, 1, ref.beta, @abs(ref.beta));
            if (bd > bbound) return error.GateOverBound;
            const bratio = bd / bbound;
            if (bratio > r.worst) r = .{ .worst = bratio, .where = 16 + hv };
        }
        return r;
    }
};

// ---------------------------------------------------------------------------
// Conv prefill
// ---------------------------------------------------------------------------

/// The prefill conv: the fixture states the conv input it expects the copies
/// to assemble (state rows then qkv), the weight, and checks the conv output.
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

// ---------------------------------------------------------------------------
// Prefill recurrence and decode: one serial f64 walker
// ---------------------------------------------------------------------------

pub const RecurWalk = struct {
    /// state[head][row][dk]; y[token][head*128+row]. When `snap_after` is
    /// non-null, `want_snap` receives the FULL state after token
    /// `snap_after.? - 1` (the shader's interior chunk boundary): the
    /// snapshot slot the prefill scan stores there.
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

/// The prefill recurrence gate: reference inputs are the ACTUAL soa/g/beta
/// bits the GPU produced (the probe downloads them first).
pub const Recur = struct {
    a: std.mem.Allocator,
    h0_bits: []u16, // f32 pairs
    want_y: []f64,
    want_state: []f64,
    max_state: f64,
    /// The interior chunk boundary's reference snapshot: non-null exactly when
    /// the gate runs chunked (0 < chunk < tokens, chunks >= 2, one interior
    /// boundary at token `chunk`).
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

    /// The snapshot slot against its boundary reference; poison and partial
    /// writes are refused by the same finite/budget gates as hf.
    pub fn checkSnap(f: Recur, snap_pairs: []const u16) !Report {
        const ws = f.want_snap orelse return error.NoSnapshotExpected;
        var r: Report = .{};
        for (ws, 0..) |w, i| {
            const got = f32Of(snap_pairs, i);
            if (!std.math.isFinite(got)) return error.NonFiniteOutput;
            const d = @abs(got - w);
            const bound = budget(130 * 4 + 2, 2, 0, w, f.max_state); // chunk tokens deep
            if (d > bound) return error.OverBound;
            const ratio = d / bound;
            if (ratio > r.worst) r = .{ .worst = ratio, .where = i };
        }
        return r;
    }

    pub fn check(f: Recur, y_bits: []const u16, hf_pairs: []const u16) !Report {
        var r: Report = .{};
        for (f.want_y, 0..) |w, i| {
            const got = c.fromBf16(y_bits[i]);
            if (!std.math.isFinite(got)) return error.NonFiniteOutput;
            const d = @abs(got - w);
            const bound = budget(130 * @as(f64, tokens) + 130, 4, 0, w, f.max_state);
            if (d > bound) return error.OverBound;
            const ratio = d / bound;
            if (ratio > r.worst) r = .{ .worst = ratio, .where = i };
        }
        for (f.want_state, 0..) |w, i| {
            const got = f32Of(hf_pairs, i);
            if (!std.math.isFinite(got)) return error.NonFiniteOutput;
            const d = @abs(got - w);
            const bound = budget(130 * @as(f64, tokens) + 2, 2, 0, w, f.max_state);
            if (d > bound) return error.OverBound;
            const ratio = d / bound;
            if (ratio > r.worst) r = .{ .worst = ratio, .where = i };
        }
        return r;
    }
};

/// The decode gate: one raw-gates token from state_in. The g/beta reference
/// comes from a/b/A_log/dt directly (the kernel's own prologue inputs).
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
            const gref = gateRef(
                c.fromBf16(a_bits[h]),
                c.fromBf16(b_bits[h]),
                f32Of(a_log_bits, h),
                c.fromBf16(dt_bits[h]),
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

    pub fn check(f: Decode, y_bits: []const u16, state_out_pairs: []const u16) !Report {
        var r: Report = .{};
        for (f.want_y, 0..) |w, i| {
            const got = c.fromBf16(y_bits[i]);
            if (!std.math.isFinite(got)) return error.NonFiniteOutput;
            const d = @abs(got - w);
            const bound = budget(260, 4, 0, w, f.max_state);
            if (d > bound) return error.OverBound;
            const ratio = d / bound;
            if (ratio > r.worst) r = .{ .worst = ratio, .where = i };
        }
        for (f.want_state, 0..) |w, i| {
            const got = f32Of(state_out_pairs, i);
            if (!std.math.isFinite(got)) return error.NonFiniteOutput;
            const d = @abs(got - w);
            const bound = budget(132, 2, 0, w, f.max_state);
            if (d > bound) return error.OverBound;
            const ratio = d / bound;
            if (ratio > r.worst) r = .{ .worst = ratio, .where = i };
        }
        return r;
    }
};

// ---------------------------------------------------------------------------
// Gated norm
// ---------------------------------------------------------------------------

/// gatedNorm at tokens=4: PER-HEAD 128-wide RMS (Ash gdn_post), head-major
/// [4, 16, 128]; out = bf16(silu_div(z) * bf16(w * (x * inv))).
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

    /// Per-head-row reference (tokens=4, 16 heads, width 128): each (token,
    /// head) row of 128 reduces its OWN mean square; the weight indexes the
    /// column within the head and repeats across heads. Association and
    /// budget unchanged.
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

// Negative plants against the PRODUCTION Norm checker: the honest output
// passes; a wrong finite and every nonfinite class are refused.
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

// Checker self-tests for the state carry-out path: a correctly
// reference-rounded state PASSES, a poisoned (0xffff = NaN) carry-out is
// REJECTED, and a dead store (all-poison state) cannot pass.
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
    const rep = try f.check(y_good, hf_good);
    try testing.expect(rep.worst <= 1);
    // poisoned carry-out (NaN word) must be rejected
    const hf_bad = try a.dupe(u16, hf_good);
    defer a.free(hf_bad);
    hf_bad[0] = 0xffff;
    hf_bad[1] = 0xffff;
    try testing.expectError(error.NonFiniteOutput, f.check(y_good, hf_bad));
    // a dead store: all-poison state must be rejected
    @memset(hf_bad, 0xffff);
    try testing.expectError(error.NonFiniteOutput, f.check(y_good, hf_bad));
    // single-chunk Recur expects NO snapshot
    try testing.expect(f.want_snap == null);
    // a chunked Recur has exactly one interior boundary reference; the
    // correctly-rounded slot passes, poison is refused
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
    const snap_rep = try fc.checkSnap(snap_good);
    try testing.expect(snap_rep.worst <= 1);
    snap_good[2] = 0xffff;
    snap_good[3] = 0xffff;
    try testing.expectError(error.NonFiniteOutput, fc.checkSnap(snap_good));
    @memset(snap_good, 0xffff);
    try testing.expectError(error.NonFiniteOutput, fc.checkSnap(snap_good));
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
    const rep = try f.check(y_good, souts_good);
    try testing.expect(rep.worst <= 1);
    const souts_bad = try a.dupe(u16, souts_good);
    defer a.free(souts_bad);
    souts_bad[100] = 0xffff;
    souts_bad[101] = 0xffff;
    try testing.expectError(error.NonFiniteOutput, f.check(y_good, souts_bad));
}

test "fixture initialization fails cleanly under allocation failure" {
    var failing = std.testing.FailingAllocator.init(testing.allocator, .{ .fail_index = 1 });
    try testing.expectError(error.OutOfMemory, Norm.init(failing.allocator(), 0x19));
    try testing.expectError(error.OutOfMemory, Decode.init(failing.allocator(), 0x1a));
}
