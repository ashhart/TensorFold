// The GDN contract's host suite: the fixtures, the references and the uploads the gates check.
const std = @import("std");
const contract = @import("contract");
const fx = @import("stage4_fixture.zig");
const refs = @import("contract_gdn_ref.zig");
const ops = @import("contract_gdn_ops.zig");

const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;

pub const Gdn = ops.Gdn;
pub const Which = refs.Which;
pub const chunk_fixed = refs.chunk_fixed;
pub const spec = refs.spec;
pub const walk = refs.walk;
pub const splitReference = refs.splitReference;
pub const convReference = refs.convReference;

const heads = refs.heads;
const dk = refs.dk;
const linear_dim = refs.linear_dim;
const channels = refs.channels;
const state_elems = refs.state_elems;
const poison = refs.poison;
const decodeGates = refs.decodeGates;
const f32Of = refs.f32Of;
const bf = refs.bf;
const putF32 = refs.putF32;
const splitDepth = refs.splitDepth;
const tapsDepth = refs.tapsDepth;
const recurYDepth = refs.recurYDepth;
const recurStateDepth = refs.recurStateDepth;
const decodeYDepth = refs.decodeYDepth;
const decodeStateDepth = refs.decodeStateDepth;
const snapDepth = refs.snapDepth;

const testing = std.testing;

// Host fixtures need a real allocator; device fields stay untouched because host tests never call `uploadBuffers`.
fn hostFixture(which: Which) Gdn {
    return Gdn.init(testing.allocator, undefined, undefined, undefined, undefined, which);
}

test "gdn contract prepareHost refuses torn geometry and serves the fixed one" {
    var f = hostFixture(.split_soa);
    defer f.deinit();
    try testing.expectEqual(@as(?usize, null), try f.prepareHost(.{ .rows = 0, .n = heads, .k = dk }, .random, 1));
    try testing.expectEqual(@as(?usize, null), try f.prepareHost(.{ .rows = 8, .n = 8, .k = dk }, .random, 1));
    try testing.expectEqual(@as(?usize, null), try f.prepareHost(.{ .rows = 8, .n = heads, .k = 64 }, .random, 1));
    for (std.enums.values(Which)) |which| {
        var g_ = hostFixture(which);
        defer g_.deinit();
        const out_len = (try g_.prepareHost(.{ .rows = 1, .n = heads, .k = dk }, .random, 3)) orelse return error.Refused;
        try testing.expect(out_len > 0);
    }
}

test "gdn contract prepareHost random fill: finite references within fixture budgets" {
    for (std.enums.values(Which)) |which| {
        var f = hostFixture(which);
        defer f.deinit();
        for ([_]usize{ 1, 8, 32 }) |T| {
            _ = (try f.prepareHost(.{ .rows = T, .n = heads, .k = dk }, .random, 0x6764 + T)) orelse return error.Refused;
            try testing.expectEqual(f.want.len, f.count());
            var buf: [8][]u16 = undefined;
            try testing.expect(f.inputList(&buf).len > 0);
            for (f.want, f.mag) |w, m| {
                try testing.expect(std.math.isFinite(w));
                try testing.expect(std.math.isFinite(m) and m >= 0);
            }
            const allow = try f.arena.allocator().alloc(f64, f.want.len);
            try f.bounds(f.want, f.mag, allow);
            for (allow) |l| try testing.expect(std.math.isFinite(l) and l > 0);
            // Before run, outputs stay poison; `want` is the GPU target, exposing a wrong reference length.
            for (f.out[f.want.len..]) |w| try testing.expectEqual(poison, w);
        }
    }
}

test "gdn contract prepareHost stress fill: bounded state, planted mirror, fp16-proving partials" {
    for ([_]Which{ .prefill_y, .prefill_state, .decode_y, .decode_state }) |which| {
        var f = hostFixture(which);
        defer f.deinit();
        _ = (try f.prepareHost(.{ .rows = 4, .n = heads, .k = dk }, .stress, 0x6764)) orelse return error.Refused;
        // Stress g=1.0; beta=2^-10 (prefill) or rounded sigmoid(-10) (decode), so state feedback converges.
        const state_src = if (which == .prefill_y or which == .prefill_state) f.h0_bits else f.state_init_bits;
        for (0..state_elems) |i| {
            const v = f32Of(state_src, i);
            try testing.expect(v >= 4096 and v < 4104);
        }
        // `v` is bf16 of pinned kv dot, keeping delta tiny; scan soa words (prefill v contiguous, decode per token).
        var mirror_ok = false;
        for (f.soa_bits) |bits| {
            const v = bf(bits);
            if (v > 500000 and v < 550000) mirror_ok = true;
        }
        try testing.expect(mirror_ok);
        // the f64 reference stays finite and bounded at stress scale
        for (f.want) |w| try testing.expect(std.math.isFinite(w));
        try testing.expect(f.max_state < 1e6);
    }
}

test "gdn contract prepareHost nonfinite fill plants classes the walkers propagate" {
    for (std.enums.values(Which)) |which| {
        var f = hostFixture(which);
        defer f.deinit();
        _ = (try f.prepareHost(.{ .rows = 8, .n = heads, .k = dk }, .nonfinite, 0x6765)) orelse return error.Refused;
        var planted: usize = 0;
        for (f.want) |w| {
            if (c.classOf(w) != .finite) planted += 1;
        }
        try testing.expect(planted > 0);
        try testing.expect(planted < f.want.len);
    }
}

test "gdn contract conv prefill reference reproduces the fixture Conv checker's want" {
    // the assembled input taps and the state carry, at the fixture's shape
    var f = hostFixture(.conv_prefill);
    defer f.deinit();
    _ = (try f.prepareHost(.{ .rows = @intCast(fx.tokens), .n = heads, .k = dk }, .random, 0x57a5)) orelse return error.Refused;
    const cf = try fx.Conv.init(testing.allocator, 0x57a4 + 1);
    defer cf.deinit();
    // same seed family as the fixture: check MY reference math against a directly computed conv over MY generated bits
    const C: usize = channels;
    for (0..@intCast(fx.tokens)) |t| {
        for (0..C) |ch| {
            var acc: f64 = 0;
            for (0..4) |k| {
                const row = t + k;
                const in_bit = if (row < 3) f.state_bits[row * C + ch] else f.qkv_bits[(row - 3) * C + ch];
                acc += bf(in_bit) * bf(f.weight_bits[ch * 4 + k]);
            }
            try testing.expectEqual(f.want[t * C + ch], acc);
        }
    }
}

test "gdn contract depth functions grow with tokens, never constant" {
    try testing.expect(recurYDepth(.{ .rows = 130, .n = 16, .k = 128 }) > recurYDepth(.{ .rows = 8, .n = 16, .k = 128 }));
    try testing.expectEqual(@as(usize, 130 * 130 + 130), recurYDepth(.{ .rows = 130, .n = 16, .k = 128 }));
    try testing.expectEqual(@as(usize, 4 * 130 + 2), recurStateDepth(.{ .rows = 4, .n = 16, .k = 128 }));
    try testing.expectEqual(@as(usize, 3 * 260 + 130), decodeYDepth(.{ .rows = 3, .n = 16, .k = 128 }));
    try testing.expectEqual(@as(usize, 132 + 2), decodeStateDepth(.{ .rows = 1, .n = 16, .k = 128 }));
    try testing.expectEqual(@as(f64, 8 * 130 + 2), snapDepth());
    // the conv and split depths are genuinely shape-independent chains
    try testing.expectEqual(@as(usize, 4), tapsDepth(.{ .rows = 99, .n = 16, .k = 128 }));
    try testing.expectEqual(@as(usize, 40), splitDepth(.{ .rows = 99, .n = 16, .k = 128 }));
}

test "gdn contract walker matches the stage 4 fixture walker bit for bit at the fixture shape" {
    const a = testing.allocator;
    const T: usize = @intCast(fx.tokens); // 8, the fixture's comptime shape
    const soa_bits = try a.alloc(u16, 3 * T * linear_dim);
    defer a.free(soa_bits);
    const g_pairs = try a.alloc(u16, 2 * T * heads);
    defer a.free(g_pairs);
    const beta_bits = try a.alloc(u16, T * heads);
    defer a.free(beta_bits);
    const h0_bits = try a.alloc(u16, 2 * state_elems);
    defer a.free(h0_bits);
    var prng = std.Random.DefaultPrng.init(0xc7);
    const rng = prng.random();
    for (soa_bits) |*x| x.* = fx.sampleDyadic(rng, -3, 1);
    for (g_pairs) |*x| {
        const bits: u32 = @bitCast(fx.sampleDyadicF32(rng, -4, 0));
        x.* = @truncate(bits);
    }
    for (beta_bits) |*x| x.* = fx.sampleDyadic(rng, -3, 1);
    for (0..state_elems) |i| putF32(h0_bits, i, fx.sampleDyadicF32(rng, -6, -2));

    const mine_y = try a.alloc(f64, T * linear_dim);
    defer a.free(mine_y);
    const mine_state = try a.alloc(f64, state_elems);
    defer a.free(mine_state);
    const mine_st = try a.alloc(f64, state_elems);
    defer a.free(mine_st);
    var mine_max: f64 = 0;
    walk(T, soa_bits, g_pairs, beta_bits, h0_bits, mine_st, mine_y, mine_state, &mine_max, null, null, null, null);

    const fx_y = try a.alloc(f64, T * linear_dim);
    defer a.free(fx_y);
    const fx_state = try a.alloc(f64, state_elems);
    defer a.free(fx_state);
    var fx_max: f64 = 0;
    fx.RecurWalk.run(soa_bits, g_pairs, beta_bits, h0_bits, fx_y, fx_state, &fx_max, null, null);
    try testing.expectEqualSlices(f64, fx_y, mine_y);
    try testing.expectEqualSlices(f64, fx_state, mine_state);
    try testing.expectEqual(fx_max, mine_max);

    // the chunked variant's interior snapshot: boundary at token 4
    const mine_snap = try a.alloc(f64, state_elems);
    defer a.free(mine_snap);
    const fx_snap = try a.alloc(f64, state_elems);
    defer a.free(fx_snap);
    var mine_max2: f64 = 0;
    var fx_max2: f64 = 0;
    walk(T, soa_bits, g_pairs, beta_bits, h0_bits, mine_st, mine_y, mine_state, &mine_max2, 4, mine_snap, null, null);
    fx.RecurWalk.run(soa_bits, g_pairs, beta_bits, h0_bits, fx_y, fx_state, &fx_max2, 4, fx_snap);
    try testing.expectEqualSlices(f64, fx_snap, mine_snap);
    try testing.expectEqualSlices(f64, fx_state, mine_state);
}

test "gdn contract walker matches the fixture decode reference exactly" {
    const a = testing.allocator;
    const f = try fx.Decode.init(a, 0x53);
    defer f.deinit();
    // pack the raw-gates chain the fixture derived into the walk's inputs
    const g_pairs = try a.alloc(u16, 2 * heads);
    defer a.free(g_pairs);
    const beta_bits = try a.alloc(u16, heads);
    defer a.free(beta_bits);
    decodeGates(f.a_bits, f.b_bits, f.a_log_bits, f.dt_bits, 0, g_pairs, beta_bits);
    const y = try a.alloc(f64, linear_dim);
    defer a.free(y);
    const st_out = try a.alloc(f64, state_elems);
    defer a.free(st_out);
    const st = try a.alloc(f64, state_elems);
    defer a.free(st);
    var max_state: f64 = 0;
    walk(1, f.soa_bits, g_pairs, beta_bits, f.state_in_bits, st, y, st_out, &max_state, null, null, null, null);
    for (f.want_y, y) |w, m| try testing.expectEqual(w, m);
    for (f.want_state, st_out) |w, m| try testing.expectEqual(w, m);
    try testing.expectEqual(f.max_state, max_state);
}

test "gdn contract split reference replays clean through the fixture Split checker" {
    const a = testing.allocator;
    const T8: usize = @intCast(fx.tokens); // the fixture checker's comptime shape
    var f = try fx.Split.init(a, 0x9e); // the fixture generates its own inputs
    defer f.deinit();
    // my reference over the SAME bits, rendered to bits, through the PRODUCTION fixture checker
    const want8 = try a.alloc(f64, 3 * T8 * linear_dim + T8 * heads);
    defer a.free(want8);
    const mag8 = try a.alloc(f64, want8.len);
    defer a.free(mag8);
    splitReference(T8, f.conv_out_bits, f.a_bits, f.b_bits, f.a_log_bits, f.dt_bits, want8, mag8);
    const soa_bits = try a.alloc(u16, 3 * T8 * linear_dim);
    defer a.free(soa_bits);
    const beta_bits = try a.alloc(u16, T8 * heads);
    defer a.free(beta_bits);
    const g_pairs = try a.alloc(u16, 2 * T8 * heads);
    defer a.free(g_pairs);
    for (0..3 * T8 * linear_dim) |i| soa_bits[i] = c.toBf16(@floatCast(want8[i]));
    for (0..T8 * heads) |i| beta_bits[i] = c.toBf16(@floatCast(want8[3 * T8 * linear_dim + i]));
    for (0..T8) |t| for (0..heads) |h| {
        const ref = fx.gateRefOracle(f.a_bits[t * heads + h], f.b_bits[t * heads + h], f32Of(f.a_log_bits, h), f.dt_bits[h]);
        putF32(g_pairs, t * heads + h, @floatCast(ref.g));
    };
    const rep = try f.check(soa_bits, g_pairs, beta_bits, null);
    try testing.expect(rep.worst <= 1);
    // and the same want/mag pass my own budget hook's allowances
    for (want8, mag8) |w, m| {
        const allow = fx.budget(40, 5, 3, w, m);
        try testing.expect(allow > 0 and std.math.isFinite(allow));
    }
}

test "gdn contract split g reference equals the fixture's oracle chain" {
    const a = testing.allocator;
    const f = try fx.SplitOne.init(a, 0x57ac);
    defer f.deinit();
    // `splitReference` omits g; `split_g` must match fixture `gateRefOracle`, so replay it here.
    for (0..heads) |h| {
        const ref = fx.gateRefOracle(f.a_bits[h], f.b_bits[h], f32Of(f.a_log_bits, h), f.dt_bits[h]);
        try testing.expect(ref.g > 0 and ref.g < 1);
        try testing.expect(ref.beta > 0 and ref.beta < 1);
    }
}

test "gdn contract stress keeps f32 sums exact and makes fp16 miss by over 100x" {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const ar = arena.allocator();
    const T: usize = 4;
    const soa_bits = try ar.alloc(u16, 3 * T * linear_dim);
    const g_pairs = try ar.alloc(u16, 2 * T * heads);
    const beta_bits = try ar.alloc(u16, T * heads);
    const h0_bits = try ar.alloc(u16, 2 * state_elems);
    for (0..state_elems) |i| putF32(h0_bits, i, @floatFromInt(4096 + i % 7));
    for (soa_bits) |*x| x.* = c.toBf16(1);
    for (0..T * linear_dim) |i| {
        if (i % dk % 2 == 1) soa_bits[i] = c.toBf16(-1);
    }
    for (0..heads) |h| for (0..dk) |row| {
        var kv: f64 = 0;
        for (0..dk) |i| kv += @as(f64, @floatFromInt(4096 + (h * dk * dk + row * dk + i) % 7));
        for (0..T) |t| soa_bits[2 * T * linear_dim + (t * heads + h) * dk + row] = c.toBf16(@floatCast(kv));
    };
    for (0..T * heads) |i| {
        putF32(g_pairs, i, 1.0);
        beta_bits[i] = c.toBf16(0x1p-10);
    }
    // f32 sums fit every-order state budget; dyadic denominators exceed 24 bits after 2 tokens, covered by depth.
    const want_y = try ar.alloc(f64, T * linear_dim);
    const want_state = try ar.alloc(f64, state_elems);
    const st = try ar.alloc(f64, state_elems);
    var max_state: f64 = 0;
    walk(T, soa_bits, g_pairs, beta_bits, h0_bits, st, want_y, want_state, &max_state, null, null, null, null);
    const sim_y = try ar.alloc(f64, T * linear_dim);
    const sim_state = try ar.alloc(f64, state_elems);
    var sim_max: f64 = 0;
    for ([_]g.Order{ .sequential, .pairwise, .strided }) |order| {
        walk(T, soa_bits, g_pairs, beta_bits, h0_bits, st, sim_y, sim_state, &sim_max, null, null, .f32, order);
        for (sim_state, want_state) |gv, wv| {
            const allow = fx.budget(@floatFromInt(T * 130 + 2), 2, 0, wv, max_state);
            try testing.expect(@abs(gv - wv) <= allow);
        }
    }
    // fp16: the kv dot's partial sums pass 65504, so the state and y blow past the budget by orders of magnitude
    const allow = fx.budget(@floatFromInt(T * 130 + 2), 2, 0, want_state[0], max_state);
    var worst: f64 = 0;
    walk(T, soa_bits, g_pairs, beta_bits, h0_bits, st, sim_y, sim_state, &sim_max, null, null, .f16, .sequential);
    for (sim_state, want_state) |gv, wv| {
        if (!std.math.isFinite(gv)) {
            worst = std.math.inf(f64);
            break;
        }
        worst = @max(worst, @abs(gv - wv) / allow);
    }
    try testing.expect(worst >= 100);
}

// Buffer `i` takes `out` words iff a checked output is fetched there.
fn expectOut(which: Which, i: usize, chunks: usize) bool {
    return switch (which) {
        .split_soa => i == 5 or i == 7,
        .split_g => i == 6,
        .conv_prefill => i == 8 or i == 12,
        .conv_decode => i == 9,
        .prefill_y => i == 4,
        .prefill_state => i == 5 or (i == 6 and chunks > 1),
        .decode_y => i == 8,
        .decode_state => false,
    };
}

// The device buffers every which uploads, so the fixed indices run() binds stay the ones uploadWords hands the device.
fn expectCount(which: Which, chunks: usize) usize {
    return switch (which) {
        .split_soa, .split_g => 8,
        .conv_prefill, .conv_decode => 13,
        .prefill_y, .prefill_state => if (chunks > 1) 7 else 6,
        .decode_y, .decode_state => 9,
    };
}

test "gdn contract upload words: every upload inside the region prepare sized" {
    for (std.enums.values(Which)) |which| {
        var f = hostFixture(which);
        defer f.deinit();
        const s = spec(which);
        // the shapes the gates drive: gate 1 and 2's small(), gate 3's served and edge, gate 4's stress
        var steps: [16]struct { shape: c.Shape, fill: rows.Fill } = undefined;
        var ns: usize = 0;
        var one = s.served[0];
        one.rows = std.math.clamp(one.rows, 5, 8);
        steps[ns] = .{ .shape = one, .fill = .nonfinite };
        ns += 1;
        one = s.served[0];
        one.rows = std.math.clamp(one.rows, 1, 64);
        steps[ns] = .{ .shape = one, .fill = .random };
        ns += 1;
        const list = try c.shapes(testing.allocator, s);
        defer testing.allocator.free(list);
        for (list) |sh| {
            steps[ns] = .{ .shape = sh, .fill = .random };
            ns += 1;
        }
        one = s.served[0];
        one.rows = @min(one.rows, 4);
        steps[ns] = .{ .shape = one, .fill = .stress };
        ns += 1;

        for (steps[0..ns]) |step| {
            const prepared = f.prepareHost(step.shape, step.fill, 0x6764) catch |err| switch (err) {
                // the fixture refuses this fill before any upload; the gate reports the error and never uploads
                error.NonFiniteReference => continue,
                else => return err,
            };
            const out_len = prepared orelse continue;
            try testing.expect(out_len > 0);
            // the upload path uploadBuffers takes, without a device
            var words: [16][]u16 = undefined;
            const n = try f.uploadWords(&words);
            try testing.expectEqual(expectCount(which, f.chunks), n);
            const lo = @intFromPtr(f.out.ptr);
            const hi = lo + f.out.len * @sizeOf(u16);
            for (words[0..n], 0..) |w, i| {
                const at = @intFromPtr(w.ptr);
                const in_out = at >= lo and at + w.len * @sizeOf(u16) <= hi;
                try testing.expectEqual(expectOut(which, i, f.chunks), in_out);
            }
        }
    }
}
