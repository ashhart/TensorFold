// Stage 4 fixture: the GatedDeltaNet closure, 16 heads, head_dim 128, 6144 conv channels, 4 taps, vs f64 host refs.

// depth/ops/mids budgets, mag: split max|vals|^2, gate |g|, recur max state, norm max|x|^2*len; chunk=4/8 one ref

const std = @import("std");
const core = @import("stage4_fixture_core.zig");
const split_mod = @import("stage4_fixture_split.zig");
const state_mod = @import("stage4_fixture_state.zig");

pub const c = core.c;
pub const tail = core.tail;
pub const poison16 = core.poison16;
pub const Model = core.Model;
pub const tokens = core.tokens;
pub const Report = core.Report;
pub const Output = core.Output;
pub const Violation = core.Violation;
pub const budget = core.budget;
pub const sampleDyadic = core.sampleDyadic;
pub const sampleDyadicF32 = core.sampleDyadicF32;
pub const gateRef = core.gateRef;
pub const gateRefAsh = core.gateRefAsh;
pub const Split = split_mod.Split;
pub const SplitOne = split_mod.SplitOne;
pub const Conv = split_mod.Conv;
pub const RecurWalk = state_mod.RecurWalk;
pub const Recur = state_mod.Recur;
pub const Decode = state_mod.Decode;
pub const Norm = state_mod.Norm;
const f32Of = core.f32Of;

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
