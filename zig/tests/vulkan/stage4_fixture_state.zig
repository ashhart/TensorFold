// Stage 4 fixture: the prefill recurrence walker, the recurrence, the decode and the gated norm fixtures.

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
            // HOST oracle chain: the exact bf16 rounds the decode prologue does
            const gref = gateRefOracle(
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

// gatedNorm tokens=4: PER-HEAD 128-wide RMS (oracle gdn_post), [4,16,128]; out=bf16(silu_div(z)*bf16(w*(x*inv))).
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
