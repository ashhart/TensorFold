// Preserve the omitted contract, precision, and host-interface facts from the original comment.
const std = @import("std");
const contract = @import("contract");
const c = contract.spec;
// The cases live in stage3_fixture_cases.zig; the tests stay here so every importer of this file runs them.
const cases = @import("stage3_fixture_cases.zig");

pub const tail = cases.tail;
pub const poison16 = cases.poison16;
pub const heads = cases.heads;
pub const kv_heads = cases.kv_heads;
pub const head_dim = cases.head_dim;
pub const rotary = cases.rotary;
pub const theta = cases.theta;
pub const scale = cases.scale;
pub const Report = cases.Report;
pub const budget = cases.budget;
pub const sampleDyadic = cases.sampleDyadic;
pub const ropeRef = cases.ropeRef;
pub const RopeCase = cases.RopeCase;
pub const rope_cases = cases.rope_cases;
pub const Rope = cases.Rope;
pub const CacheCase = cases.CacheCase;
pub const cache_case = cases.cache_case;
pub const Cache = cases.Cache;
pub const AttendCase = cases.AttendCase;
pub const attend_case = cases.attend_case;
pub const visibleFrom = cases.visibleFrom;
pub const Attend = cases.Attend;
pub const GateCase = cases.GateCase;
pub const gate_case = cases.gate_case;
pub const GateDiag = cases.GateDiag;
pub const Gate = cases.Gate;

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
