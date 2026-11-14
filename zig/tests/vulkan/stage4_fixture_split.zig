// Stage 4 fixture: the split at T tokens, the split at one token and the conv prefill fixture.

const std = @import("std");
const core = @import("stage4_fixture_core.zig");

const c = core.c;
const Model = core.Model;
const tokens = core.tokens;
const budget = core.budget;
const gateRefOracle = core.gateRefOracle;
const f32Of = core.f32Of;
const poisonedShell = core.poisonedShell;
const violated = core.violated;
const Report = core.Report;
const Output = core.Output;
const Violation = core.Violation;
const sampleDyadic = core.sampleDyadic;
const sampleDyadicF32 = core.sampleDyadicF32;

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
                // HOST oracle chain: the exact bf16 rounds the shader does
                const ref = gateRefOracle(
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
            // HOST oracle chain: the exact bf16 rounds the shader does
            const ref = gateRefOracle(
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
