// The GDN contract's host layer: the fills, the references and the upload order, none of it on a device.
const std = @import("std");
const contract = @import("contract");
const fx = @import("stage4_fixture.zig");
const refs = @import("contract_gdn_ref.zig");
const ops = @import("contract_gdn_ops.zig");

const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;

const Gdn = ops.Gdn;
const Which = refs.Which;
const heads = refs.heads;
const dk = refs.dk;
const linear_dim = refs.linear_dim;
const channels = refs.channels;
const taps = refs.taps;
const state_elems = refs.state_elems;
const chunk_fixed = refs.chunk_fixed;
const poison = refs.poison;
const putF32 = refs.putF32;
const guard_words = refs.guard_words;
const max_tokens = refs.max_tokens;
const splitReference = refs.splitReference;
const convReference = refs.convReference;
const decodeGates = refs.decodeGates;
const f32Of = refs.f32Of;
const bf = refs.bf;
const walk = refs.walk;
const PrefillInput = refs.PrefillInput;
const DecodeInput = refs.DecodeInput;

// The fixture's dyadic sampler so gate 3's inputs sit inside the ranges the stated budgets were derived over.
fn fillDyadic(rng: std.Random, v: []u16, lo: i32, hi: i32) void {
    for (v) |*x| x.* = fx.sampleDyadic(rng, lo, hi);
}

fn fillDyadicF32(rng: std.Random, pairs: []u16, lo: i32, hi: i32) void {
    for (0..pairs.len / 2) |i| {
        const bits: u32 = @bitCast(fx.sampleDyadicF32(rng, lo, hi));
        pairs[2 * i] = @truncate(bits);
        pairs[2 * i + 1] = @truncate(bits >> 16);
    }
}

// Gate 1 plants NaN and ±inf in token row 1; finite per-head `a_log`/`dt`/`weight` stop one class masking outputs.
pub fn plantNonfinite(f: *Gdn, rng: std.Random) !void {
    const T = f.tokens;
    const word = [_]u16{ 0x7fc0, 0x7f80, 0xff80 }; // NaN, +inf, -inf
    switch (f.which) {
        .split_soa, .split_g => {
            f.conv_out_bits[1 * channels + 5] = word[0];
            f.conv_out_bits[1 * channels + 6] = word[1];
            f.conv_out_bits[1 * channels + 7] = word[2];
            f.a_bits[1 * heads + 3] = word[0];
            f.a_bits[1 * heads + 11] = word[1];
            f.b_bits[1 * heads + 9] = word[2];
        },
        .conv_prefill => {
            f.qkv_bits[5] = word[0]; // qkv row 0 = assembled input row 3
            f.state_bits[3] = word[1]; // state row 0: +inf and -inf one row
            f.state_bits[9] = word[2];
        },
        .conv_decode => {
            f.qkv_bits[3 * channels + 5] = word[0]; // token 3's x
            f.state_bits[3] = word[1];
            f.state_bits[9] = word[2];
        },
        .prefill_y, .prefill_state => {
            const section = T * linear_dim;
            f.soa_bits[0 * section + (1 * heads + 3) * dk + 5] = word[0]; // q
            f.soa_bits[1 * section + (1 * heads + 3) * dk + 7] = word[1]; // k
            f.soa_bits[2 * section + (1 * heads + 3) * dk + 9] = word[2]; // v
            putF32(f.g_pairs, 1 * heads + 3, std.math.inf(f32));
            f.beta_bits[1 * heads + 7] = word[0];
            f.h0_bits[2 * (11 * dk * dk)] = word[0];
            f.h0_bits[2 * (11 * dk * dk) + 1] = word[0];
        },
        .decode_y, .decode_state => {
            f.soa_bits[0 * linear_dim + 1] = word[0]; // q head 0
            f.soa_bits[1 * linear_dim + 2] = word[1]; // k head 1
            f.soa_bits[2 * linear_dim + 3] = word[2]; // v head 2
            if (T > 1) {
                f.soa_bits[(1 * 3 + 0) * linear_dim + 130] = word[0];
                f.soa_bits[(1 * 3 + 1) * linear_dim + 131] = word[1];
                f.soa_bits[(1 * 3 + 2) * linear_dim + 132] = word[2];
                f.a_bits[1 * heads + 5] = word[1];
                f.b_bits[1 * heads + 6] = word[2];
            }
            f.state_bits[2 * (4 * dk * dk)] = word[0];
            f.state_bits[2 * (4 * dk * dk) + 1] = word[0];
        },
    }
    _ = rng;
}

// F32 dyadic sums are order-exact below 2^24; fp16 misses by thousands, while bf16 `v` pins state with tiny delta.
pub fn fillStress(f: *Gdn) void {
    const T = f.tokens;
    switch (f.which) {
        .split_soa, .split_g => {
            for (0..T * channels) |i| {
                const mag: i64 = 128 + @as(i64, @intCast(i % 8));
                f.conv_out_bits[i] = c.toBf16(@floatFromInt(if (i % 2 == 0) mag else -mag));
            }
            for (0..T * heads) |i| {
                f.a_bits[i] = c.toBf16(0);
                f.b_bits[i] = c.toBf16(0);
            }
            for (0..heads) |h| {
                f.dt_bits[h] = c.toBf16(0);
                putF32(f.a_log_bits, h, 0);
            }
        },
        .conv_prefill => {
            for (f.state_bits, 0..) |*x, i| {
                const sgn: f32 = if (i % 2 == 0) 1 else -1;
                x.* = c.toBf16(sgn * @as(f32, @floatFromInt(16384 + i % 64)));
            }
            for (f.qkv_bits, 0..) |*x, i| {
                const sgn: f32 = if (i % 2 == 0) 1 else -1;
                x.* = c.toBf16(sgn * @as(f32, @floatFromInt(16384 + i % 64)));
            }
            for (f.weight_bits, 0..) |*x, i| {
                const sgn: f32 = if (i % 3 == 0) 1 else -1;
                x.* = c.toBf16(sgn * @as(f32, @floatFromInt(2 + i % 3)));
            }
        },
        .conv_decode => {
            for (f.state_bits, 0..) |*x, i| {
                const sgn: f32 = if (i % 2 == 0) 1 else -1;
                x.* = c.toBf16(sgn * @as(f32, @floatFromInt(16384 + i % 64)));
            }
            for (f.qkv_bits, 0..) |*x, i| {
                const sgn: f32 = if (i % 2 == 0) 1 else -1;
                x.* = c.toBf16(sgn * @as(f32, @floatFromInt(16384 + i % 64)));
            }
            for (f.weight_bits, 0..) |*x, i| {
                const sgn: f32 = if (i % 3 == 0) 1 else -1;
                x.* = c.toBf16(sgn * @as(f32, @floatFromInt(2 + i % 3)));
            }
        },
        .prefill_y, .prefill_state => {
            // State 2^12+small; k=1, q=±1, g=1, beta=2^-10. Gain 1.0 blows 2^24; bf16 v keeps delta tiny.
            for (0..state_elems) |i| putF32(f.h0_bits, i, @floatFromInt(4096 + i % 7));
            for (0..3 * T * linear_dim) |i| f.soa_bits[i] = c.toBf16(1); // q, k
            for (0..T * linear_dim) |i| {
                if (i % dk % 2 == 1) f.soa_bits[i] = c.toBf16(-1); // q alternates
            }
            // With k=1 and state varying only by i%7, every head/row kv dot is the same ~524667 integer.
            var kv: f64 = 0;
            for (0..dk) |i| kv += @as(f64, @floatFromInt(4096 + i % 7));
            for (0..T * linear_dim) |i| f.soa_bits[2 * T * linear_dim + i] = c.toBf16(@floatCast(kv));
            for (0..T * heads) |i| {
                putF32(f.g_pairs, i, 1.0);
                f.beta_bits[i] = c.toBf16(0x1p-10);
            }
        },
        .decode_y, .decode_state => {
            for (0..state_elems) |i| putF32(f.state_bits, i, @floatFromInt(4096 + i % 7));
            for (f.soa_bits) |*x| x.* = c.toBf16(1); // q, k = 1
            for (0..T * linear_dim) |i| {
                if (i % dk % 2 == 1) f.soa_bits[i] = c.toBf16(-1);
            }
            // v mirrors the kv dot (state pinned, k = 1)
            for (0..heads) |h| for (0..dk) |row| {
                var kv: f64 = 0;
                for (0..dk) |i| kv += @as(f64, @floatFromInt(4096 + @as(i64, @intCast((h * dk * dk + row * dk + i) % 7))));
                for (0..T) |t| f.soa_bits[(t * 3 + 2) * linear_dim + h * dk + row] = c.toBf16(@floatCast(kv));
            };
            for (0..T * heads) |i| {
                f.a_bits[i] = c.toBf16(0);
                f.b_bits[i] = c.toBf16(-10); // sigmoid(-10): a small convergent beta gain
            }
            for (0..heads) |h| {
                f.dt_bits[h] = c.toBf16(0);
                putF32(f.a_log_bits, h, -88.0); // exp(-88) softplus gates g to f32 1.0
            }
        },
    }
}

// `prepareHost` refuses shapes, generates fills, and computes references without a device for host tests.
pub fn prepareHost(f: *Gdn, shape: c.Shape, fill: g.Fill, seed: u64) !?usize {
    if (shape.rows == 0 or shape.n != heads or shape.k != dk or shape.rows > max_tokens) return null;
    f.reset();
    _ = f.arena.reset(.retain_capacity);
    f.shape = shape;
    f.tokens = shape.rows;
    const T = f.tokens;
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    f.chunks = @intCast((T + @as(usize, @intCast(chunk_fixed)) - 1) / @as(usize, @intCast(chunk_fixed)));

    // host inputs per which
    switch (f.which) {
        .split_soa, .split_g => {
            f.conv_out_bits = try f.allocU16(T * channels, poison);
            f.a_bits = try f.allocU16(T * heads, poison);
            f.b_bits = try f.allocU16(T * heads, poison);
            f.dt_bits = try f.allocU16(heads, poison);
            f.a_log_bits = try f.allocU16(2 * heads, poison);
        },
        .conv_prefill => {
            f.state_bits = try f.allocU16(3 * channels, poison);
            f.qkv_bits = try f.allocU16(T * channels, poison);
            f.weight_bits = try f.allocU16(taps * channels, poison);
            f.a_bits = try f.allocU16(T * heads, poison);
            f.b_bits = try f.allocU16(T * heads, poison);
            f.dt_bits = try f.allocU16(heads, poison);
            f.a_log_bits = try f.allocU16(2 * heads, poison);
        },
        .conv_decode => {
            f.state_bits = try f.allocU16(3 * channels, poison);
            f.state_init_bits = try f.allocU16(3 * channels, poison);
            f.state_swap_bits = try f.allocU16(3 * channels, poison);
            f.qkv_bits = try f.allocU16(T * channels, poison); // x rows
            f.weight_bits = try f.allocU16(taps * channels, poison);
            f.a_bits = try f.allocU16(T * heads, poison);
            f.b_bits = try f.allocU16(T * heads, poison);
            f.dt_bits = try f.allocU16(heads, poison);
            f.a_log_bits = try f.allocU16(2 * heads, poison);
        },
        .prefill_y, .prefill_state => {
            f.soa_bits = try f.allocU16(3 * T * linear_dim, poison);
            f.g_pairs = try f.allocU16(2 * T * heads, poison);
            f.beta_bits = try f.allocU16(T * heads, poison);
            f.h0_bits = try f.allocU16(2 * state_elems, poison);
            f.walk_st = try f.allocF64(state_elems);
        },
        .decode_y, .decode_state => {
            f.soa_bits = try f.allocU16(3 * T * linear_dim, poison);
            f.a_bits = try f.allocU16(T * heads, poison);
            f.b_bits = try f.allocU16(T * heads, poison);
            f.a_log_bits = try f.allocU16(2 * heads, poison);
            f.dt_bits = try f.allocU16(heads, poison);
            f.state_bits = try f.allocU16(2 * state_elems, poison);
            f.state_init_bits = try f.allocU16(2 * state_elems, poison);
            f.state_swap_bits = try f.allocU16(2 * state_elems, poison);
            f.walk_st = try f.allocF64(state_elems);
            @memcpy(f.state_init_bits, f.state_bits);
        },
    }

    // fills: the fixture's dyadic ranges for gate 3, engineered integers for gate 4, planted classes for gate 1
    switch (fill) {
        .random => switch (f.which) {
            .split_soa, .split_g => {
                fillDyadic(rng, f.conv_out_bits, -4, 2);
                fillDyadic(rng, f.a_bits, -3, 2);
                fillDyadic(rng, f.b_bits, -3, 2);
                fillDyadic(rng, f.dt_bits, -4, 1);
                // Served `a_log` keeps g normal; above ~2.4, devices flush subnormal g (census, not fill failure).
                fillDyadicF32(rng, f.a_log_bits, -6, 0);
            },
            .conv_prefill, .conv_decode => {
                fillDyadic(rng, f.state_bits, -4, 2);
                fillDyadic(rng, f.qkv_bits, -4, 2);
                fillDyadic(rng, f.weight_bits, -3, 1);
                fillDyadic(rng, f.a_bits, -3, 2);
                fillDyadic(rng, f.b_bits, -3, 2);
                fillDyadic(rng, f.dt_bits, -4, 1);
                fillDyadicF32(rng, f.a_log_bits, -6, 0);
            },
            .prefill_y, .prefill_state => {
                // For 130 tokens, g, beta, soa <=2^-3 keep feedback <1; f32 state decays to the delta floor.
                fillDyadic(rng, f.soa_bits, -6, -3);
                fillDyadicF32(rng, f.g_pairs, -6, -3);
                fillDyadic(rng, f.beta_bits, -6, -4);
                fillDyadicF32(rng, f.h0_bits, -6, -2);
            },
            .decode_y, .decode_state => {
                fillDyadic(rng, f.soa_bits, -4, 2);
                fillDyadic(rng, f.a_bits, -3, 2);
                fillDyadic(rng, f.b_bits, -3, 2);
                fillDyadic(rng, f.dt_bits, -4, 1);
                fillDyadicF32(rng, f.a_log_bits, -4, 1);
                fillDyadicF32(rng, f.state_bits, -6, -2);
            },
        },
        .stress => fillStress(f),
        .nonfinite => switch (f.which) {
            .split_soa, .split_g => {
                fillDyadic(rng, f.conv_out_bits, -4, 2);
                fillDyadic(rng, f.a_bits, -3, 2);
                fillDyadic(rng, f.b_bits, -3, 2);
                fillDyadic(rng, f.dt_bits, -4, 1);
                // Served `a_log` keeps g normal; above ~2.4, devices flush subnormal g (census, not fill failure).
                fillDyadicF32(rng, f.a_log_bits, -6, 0);
            },
            .conv_prefill, .conv_decode => {
                fillDyadic(rng, f.state_bits, -4, 2);
                fillDyadic(rng, f.qkv_bits, -4, 2);
                fillDyadic(rng, f.weight_bits, -3, 1);
                fillDyadic(rng, f.a_bits, -3, 2);
                fillDyadic(rng, f.b_bits, -3, 2);
                fillDyadic(rng, f.dt_bits, -4, 1);
                fillDyadicF32(rng, f.a_log_bits, -6, 0);
            },
            .prefill_y, .prefill_state => {
                // For 130 tokens, g, beta, soa <=2^-3 keep feedback <1; f32 state decays to the delta floor.
                fillDyadic(rng, f.soa_bits, -6, -3);
                fillDyadicF32(rng, f.g_pairs, -6, -3);
                fillDyadic(rng, f.beta_bits, -6, -4);
                fillDyadicF32(rng, f.h0_bits, -6, -2);
            },
            .decode_y, .decode_state => {
                fillDyadic(rng, f.soa_bits, -4, 2);
                fillDyadic(rng, f.a_bits, -3, 2);
                fillDyadic(rng, f.b_bits, -3, 2);
                fillDyadic(rng, f.dt_bits, -4, 1);
                fillDyadicF32(rng, f.a_log_bits, -4, 1);
                fillDyadicF32(rng, f.state_bits, -6, -2);
            },
        },
    }
    if (fill == .nonfinite) try plantNonfinite(f, rng);
    // the decode loops' uploaded initial state: the FINAL generated bits
    if (f.state_init_bits.len != 0) @memcpy(f.state_init_bits, f.state_bits);

    // `out` holds guarded u16 poison (2 words/f32); `spec.out` reads `count` output elements.
    const out_len: usize = switch (f.which) {
        .split_soa => 3 * T * linear_dim + T * heads,
        .split_g => 2 * T * heads,
        .conv_prefill, .conv_decode => T * channels + 3 * channels,
        .prefill_y => T * linear_dim,
        .prefill_state => 2 * state_elems * f.chunks,
        .decode_y => T * linear_dim,
        .decode_state => 2 * state_elems,
    };
    const count_len: usize = switch (f.which) {
        .split_g => T * heads,
        .prefill_state => state_elems * f.chunks,
        .decode_state => state_elems,
        else => out_len,
    };
    f.out = try f.allocU16(out_len + guard_words, poison);
    f.out_words = out_len;
    f.want = try f.allocF64(count_len);
    f.mag = try f.allocF64(count_len);
    try computeWant(f);
    if (fill != .nonfinite) for (f.want, f.mag) |w, m| {
        if (!std.math.isFinite(w) or !std.math.isFinite(m)) return error.NonFiniteReference;
    };
    return out_len;
}

/// The uploaded inputs in the fixed order the unchanged-check compares.
pub fn inputList(f: *Gdn, buf: *[8][]u16) []const []u16 {
    switch (f.which) {
        .split_soa, .split_g => {
            buf[0..5].* = .{ f.conv_out_bits, f.a_bits, f.b_bits, f.dt_bits, f.a_log_bits };
            return buf[0..5];
        },
        .conv_prefill => {
            buf[0..7].* = .{ f.state_bits, f.qkv_bits, f.weight_bits, f.a_bits, f.b_bits, f.dt_bits, f.a_log_bits };
            return buf[0..7];
        },
        // Decode initial state uses an unwritten buffer so unchanged checks see the exact first-step input.
        .conv_decode => {
            buf[0..7].* = .{ f.state_init_bits, f.qkv_bits, f.weight_bits, f.a_bits, f.b_bits, f.dt_bits, f.a_log_bits };
            return buf[0..7];
        },
        .prefill_y, .prefill_state => {
            buf[0..4].* = .{ f.soa_bits, f.g_pairs, f.beta_bits, f.h0_bits };
            return buf[0..4];
        },
        .decode_y, .decode_state => {
            buf[0..6].* = .{ f.soa_bits, f.a_bits, f.b_bits, f.a_log_bits, f.dt_bits, f.state_init_bits };
            return buf[0..6];
        },
    }
}

// `outWords` uses fetched outputs' `out` slice or poison scratch; bad bounds report gate errors, not panics.
pub fn outWords(f: *Gdn, checked: bool, start: usize, words: usize) ![]u16 {
    if (!checked) return f.allocU16(words, poison);
    if (start > f.out_words or words > f.out_words - start) return error.OutputRegionOutOfRange;
    return f.out[start .. start + words];
}

// `uploadWords` orders input bits/output poison; checked outputs use sized `out`, others fresh scratch.
pub fn uploadWords(f: *Gdn, buf: *[16][]u16) !usize {
    const T = f.tokens;
    var n: usize = 0;
    switch (f.which) {
        .split_soa, .split_g => {
            // 0 conv_out, 1 a, 2 b, 3 dt, 4 a_log, 5 soa, 6 g, 7 beta
            buf[0..5].* = .{ f.conv_out_bits, f.a_bits, f.b_bits, f.dt_bits, f.a_log_bits };
            n = 5;
            buf[n] = try outWords(f, f.which == .split_soa, 0, 3 * T * linear_dim);
            buf[n + 1] = try outWords(f, f.which == .split_g, 0, 2 * T * heads);
            buf[n + 2] = try outWords(f, f.which == .split_soa, 3 * T * linear_dim, T * heads);
            n += 3;
        },
        .conv_prefill => {
            // Slots 0-12 are state_in, qkv, weight, a, b, dt, a_log, conv_input, conv_out, soa, g, beta, state_out.
            buf[0..7].* = .{ f.state_bits, f.qkv_bits, f.weight_bits, f.a_bits, f.b_bits, f.dt_bits, f.a_log_bits };
            n = 7;
            buf[n] = try outWords(f, false, 0, (T + 3) * channels);
            buf[n + 1] = try outWords(f, true, 0, T * channels);
            buf[n + 2] = try outWords(f, false, 0, 3 * T * linear_dim);
            buf[n + 3] = try outWords(f, false, 0, 2 * T * heads);
            buf[n + 4] = try outWords(f, false, 0, T * heads);
            buf[n + 5] = try outWords(f, true, T * channels, 3 * channels);
            n += 6;
        },
        .conv_decode => {
            // Slots 0-12 are state_init, state_a, state_b, x, weight, a, b, dt, a_log, activated, soa, g, beta.
            buf[0..9].* = .{ f.state_init_bits, f.state_swap_bits, f.state_swap_bits, f.qkv_bits, f.weight_bits, f.a_bits, f.b_bits, f.dt_bits, f.a_log_bits };
            n = 9;
            buf[n] = try outWords(f, true, 0, T * channels);
            buf[n + 1] = try outWords(f, false, 0, 3 * T * linear_dim);
            buf[n + 2] = try outWords(f, false, 0, 2 * T * heads);
            buf[n + 3] = try outWords(f, false, 0, T * heads);
            n += 4;
        },
        .prefill_y, .prefill_state => {
            // 0 soa, 1 g, 2 beta, 3 h0, 4 y, 5 hf, 6 snap?
            buf[0..4].* = .{ f.soa_bits, f.g_pairs, f.beta_bits, f.h0_bits };
            n = 4;
            buf[n] = try outWords(f, f.which == .prefill_y, 0, T * linear_dim);
            buf[n + 1] = try outWords(f, f.which == .prefill_state, 0, 2 * state_elems);
            n += 2;
            if (f.chunks > 1) {
                buf[n] = try outWords(f, f.which == .prefill_state, 2 * state_elems, 2 * state_elems * (f.chunks - 1));
                n += 1;
            }
        },
        .decode_y, .decode_state => {
            // 0 soa, 1 a, 2 b, 3 a_log, 4 dt, 5 state_init, 6 state_a, 7 state_b, 8 y
            buf[0..8].* = .{ f.soa_bits, f.a_bits, f.b_bits, f.a_log_bits, f.dt_bits, f.state_init_bits, f.state_swap_bits, f.state_swap_bits };
            n = 8;
            buf[n] = try outWords(f, f.which == .decode_y, 0, T * linear_dim);
            n += 1;
        },
    }
    return n;
}

// `computeWant` builds f64 `want/mag` from bits before device work; `reference()` returns the prepared copy.
pub fn computeWant(f: *Gdn) !void {
    const T = f.tokens;
    switch (f.which) {
        .split_soa => splitReference(T, f.conv_out_bits, f.a_bits, f.b_bits, f.a_log_bits, f.dt_bits, f.want, f.mag),
        .split_g => {
            for (0..T * heads) |i| {
                const ref = fx.gateRefAsh(f.a_bits[i], f.b_bits[i], f32Of(f.a_log_bits, i % heads), f.dt_bits[i % heads]);
                f.want[i] = ref.g;
                f.mag[i] = @abs(ref.g);
            }
        },
        .conv_prefill => {
            const ctx: PrefillInput = .{ .state_bits = f.state_bits, .qkv_bits = f.qkv_bits };
            convReference(T, f.weight_bits, ctx, f.want[0 .. T * channels], f.mag[0 .. T * channels], f.want[T * channels ..]);
            @memset(f.mag[T * channels ..], 0);
        },
        .conv_decode => {
            // the chain: each step taps the carried state then shifts it
            for (0..T) |t| {
                const ctx: DecodeInput = .{ .state_bits = f.state_bits, .x_bits = f.qkv_bits, .t = t };
                convReference(1, f.weight_bits, ctx, f.want[t * channels ..][0..channels], f.mag[t * channels ..][0..channels], null);
                for (0..2) |r| @memcpy(f.state_bits[r * channels ..][0..channels], f.state_bits[(r + 1) * channels ..][0..channels]);
                @memcpy(f.state_bits[2 * channels ..][0..channels], f.qkv_bits[t * channels ..][0..channels]);
            }
            // the final carry is the input rows T..T+2 of the walked chain
            for (0..3 * channels) |i| {
                f.want[T * channels + i] = bf(f.state_bits[i]);
                f.mag[T * channels + i] = 0;
            }
        },
        .prefill_y, .prefill_state => {
            f.max_state = 0;
            const snap_chunk: ?usize = if (f.chunks > 1) @intCast(chunk_fixed) else null;
            const y: ?[]f64 = if (f.which == .prefill_y) f.want else null;
            const hf: ?[]f64 = if (f.which == .prefill_state) f.want[0..state_elems] else null;
            const snap: ?[]f64 = if (f.which == .prefill_state and f.chunks > 1) f.want[state_elems..] else null;
            walk(T, f.soa_bits, f.g_pairs, f.beta_bits, f.h0_bits, f.walk_st, y, hf, &f.max_state, snap_chunk, snap, null, null);
            @memset(f.mag, f.max_state);
        },
        .decode_y, .decode_state => {
            f.max_state = 0;
            const g_t = try f.allocU16(2 * heads, 0);
            const beta_t = try f.allocU16(heads, 0);
            const y_t = try f.allocF64(linear_dim);
            const st_scratch = try f.allocF64(state_elems);
            const st_out = try f.allocF64(state_elems);
            for (0..T) |t| {
                decodeGates(f.a_bits, f.b_bits, f.a_log_bits, f.dt_bits, t, g_t, beta_t);
                walk(1, f.soa_bits[t * 3 * linear_dim ..][0 .. 3 * linear_dim], g_t, beta_t, f.state_bits, st_scratch, y_t, st_out, &f.max_state, null, null, null, null);
                if (f.which == .decode_y) {
                    @memcpy(f.want[t * linear_dim ..][0..linear_dim], y_t);
                } else {
                    // only the final carry is checked; the chain still walks every step
                    @memcpy(f.want[0..state_elems], st_out);
                }
                // the next step reads the f32 state this one left
                for (0..state_elems) |i| putF32(f.state_bits, i, @floatCast(st_out[i]));
            }
            @memset(f.mag, f.max_state);
        },
    }
}

/// The device buffer index of every uploaded input, in inputList's order.
pub fn inputBufIndices(f: *Gdn, buf: *[8]usize) []const usize {
    switch (f.which) {
        .split_soa, .split_g => {
            buf[0..5].* = .{ 0, 1, 2, 3, 4 };
            return buf[0..5];
        },
        .conv_prefill => {
            buf[0..7].* = .{ 0, 1, 2, 3, 4, 5, 6 };
            return buf[0..7];
        },
        .conv_decode => {
            buf[0..7].* = .{ 0, 3, 4, 5, 6, 7, 8 };
            return buf[0..7];
        },
        .prefill_y, .prefill_state => {
            buf[0..4].* = .{ 0, 1, 2, 3 };
            return buf[0..4];
        },
        .decode_y, .decode_state => {
            buf[0..6].* = .{ 0, 1, 2, 3, 4, 5 };
            return buf[0..6];
        },
    }
}
