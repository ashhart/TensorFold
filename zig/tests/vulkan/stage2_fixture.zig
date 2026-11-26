// ORACLE_QWEN35 add/SwiGLU/RMSNorm fixtures under fixed budgets; nonfinite by class (drivers vary NaN payloads).
const std = @import("std");
const contract = @import("contract");
const c = contract.spec;
// The cases live in stage2_fixture_ops.zig; the tests stay here so every importer of this file runs them.
const ops = @import("stage2_fixture_ops.zig");
const f32ops = @import("stage2_fixture_f32.zig");
comptime {
    _ = f32ops;
}

pub const tail = ops.tail;
pub const poison16 = ops.poison16;
pub const classOfBits = ops.classOfBits;
pub const classify = ops.classify;
pub const Report = ops.Report;
pub const Offsets = ops.Offsets;
pub const AddFill = ops.AddFill;
pub const AddCase = ops.AddCase;
pub const add_cases = ops.add_cases;
pub const Add = ops.Add;
pub const SwigluCase = ops.SwigluCase;
pub const swiglu_cases = ops.swiglu_cases;
pub const swiglu_gate_controls = ops.swiglu_gate_controls;
pub const Swiglu = ops.Swiglu;
pub const RmsFill = ops.RmsFill;
pub const RmsCase = ops.RmsCase;
pub const RmsOffsets = ops.RmsOffsets;
pub const rms_cases = ops.rms_cases;
pub const rms_eps = ops.rms_eps;
pub const Rms = ops.Rms;

const testing = std.testing;

test "add wants are the shader pipeline: f32 add then one RNE bf16 store" {
    for (add_cases) |case| {
        const f = try Add.init(testing.allocator, case, 0x5702);
        defer f.deinit();
        for (0..f.case.elems) |i| {
            const want = c.toBf16(c.fromBf16(f.a_bits[f.case.at.a + i]) + c.fromBf16(f.b_bits[f.case.at.b + i]));
            try testing.expectEqual(want, f.want[i]);
        }
    }
}

test "add cancel case produces exact tiny sums and the classes case plants every nonfinite class" {
    const f = try Add.init(testing.allocator, add_cases[1], 0x5703);
    defer f.deinit();
    var zeros: usize = 0;
    for (0..f.case.elems) |i| {
        const sum = c.fromBf16(f.a_bits[f.case.at.a + i]) + c.fromBf16(f.b_bits[f.case.at.b + i]);
        if (@abs(c.fromBf16(f.want[i]) - sum) == 0) zeros += 1;
    }
    try testing.expect(zeros > 0); // v + -(v - 0*ulp) pairs cancel exactly
    const g = try Add.init(testing.allocator, add_cases[2], 0x5704);
    defer g.deinit();
    var classes: std.EnumSet(c.Class) = .empty;
    for (g.want) |w| classes.insert(classOfBits(w));
    try testing.expect(classes.contains(.nan));
    try testing.expect(classes.contains(.pos_inf));
    try testing.expect(classes.contains(.neg_inf));
    try testing.expect(classes.contains(.finite));
}

test "swiglu controls plant nan, both infinities and signed zeros, and references stay finite elsewhere" {
    for (swiglu_cases) |case| {
        const f = try Swiglu.init(testing.allocator, case, 0x5705);
        defer f.deinit();
        for (f.want, 0..) |w, i| {
            if (case.controls and i < swiglu_gate_controls.len) {
                const gate = c.fromBf16(f.gate[f.case.at.a + i]);
                if (!std.math.isFinite(gate) or gate == 0) continue; // class/plant entries
            }
            if (!case.controls) try testing.expect(std.math.isFinite(w));
        }
    }
    const g = try Swiglu.init(testing.allocator, swiglu_cases[1], 0x5706);
    defer g.deinit();
    try testing.expect(std.math.isNan(g.want[0])); // nan gate
    try testing.expect(std.math.isInf(g.want[1])); // +inf gate, up 1
    try testing.expect(std.math.isNan(g.want[2])); // -inf gate -> -inf/inf
    try testing.expect(g.want[3] == 0); // +0 gate -> exactly zero
    try testing.expect(std.math.signbit(g.want[4])); // -0 gate keeps its zero sign
}

test "rms references are finite and the zero row wants exact zeros" {
    for (rms_cases) |case| {
        const f = try Rms.init(testing.allocator, case, 0x5707);
        defer f.deinit();
        for (f.want) |w| try testing.expect(std.math.isFinite(w));
        if (case.zero_row) {
            const zero_at = (case.rows - 1) * case.width;
            for (f.want[zero_at..]) |w| try testing.expectEqual(@as(f64, 0), w);
        }
    }
}

test "classify names the class of raw bf16 bits with the sign visible" {
    try testing.expectEqualStrings("+zero", classify(0x0000));
    try testing.expectEqualStrings("-zero", classify(0x8000));
    try testing.expectEqualStrings("+subnormal", classify(0x0001));
    try testing.expectEqualStrings("-subnormal", classify(0x807f));
    try testing.expectEqualStrings("+normal", classify(0x0080));
    try testing.expectEqualStrings("-normal", classify(0xbf80));
    try testing.expectEqualStrings("+inf", classify(0x7f80));
    try testing.expectEqualStrings("-inf", classify(0xff80));
    try testing.expectEqualStrings("+nan", classify(0x7fc0));
    try testing.expectEqualStrings("-nan", classify(0xffc1));
}

test "add firstMismatch reports the first bad element with both inputs and want/got words" {
    const f = try Add.init(testing.allocator, add_cases[2], 0x5717);
    defer f.deinit();
    const bits = try testing.allocator.dupe(u16, f.out);
    defer testing.allocator.free(bits);
    @memcpy(bits[f.case.at.out..][0..f.want.len], f.want);
    try testing.expectEqual(@as(?Add.Mismatch, null), try f.firstMismatch(bits));
    // Two corruptions at finite-want elements: the helper must report only the FIRST, zero-based.
    bits[f.case.at.out + 10] = 0x0000;
    bits[f.case.at.out + 11] = 0x0000;
    const m = (try f.firstMismatch(bits)).?;
    try testing.expectEqual(error.BitMismatch, m.kind);
    try testing.expectEqual(@as(usize, 10), m.index);
    try testing.expectEqual(f.a_bits[f.case.at.a + 10], m.a);
    try testing.expectEqual(f.b_bits[f.case.at.b + 10], m.b);
    try testing.expectEqual(f.want[10], m.want);
    try testing.expectEqual(@as(u16, 0x0000), m.got);
    try testing.expectError(error.BitMismatch, f.check(bits));
    // A class break at a nonfinite-want element: NonFiniteMismatch, still first index.
    @memcpy(bits[f.case.at.out..][0..f.want.len], f.want);
    const inf_at: usize = blk: {
        for (f.want, 0..) |w, i| if (classOfBits(w) == .pos_inf) break :blk i;
        unreachable;
    };
    bits[f.case.at.out + inf_at] = 0x3f80; // +inf wanted, +1 got: class break
    const nf = (try f.firstMismatch(bits)).?;
    try testing.expectEqual(error.NonFiniteMismatch, nf.kind);
    try testing.expectEqual(inf_at, nf.index);
    try testing.expectError(error.NonFiniteMismatch, f.check(bits));
}

test "add firstMismatch keeps the named length and guard failures" {
    const f = try Add.init(testing.allocator, add_cases[2], 0x5718);
    defer f.deinit();
    try testing.expectError(error.LengthMismatch, f.firstMismatch(f.out[0 .. f.out.len - 1]));
    const bits = try testing.allocator.dupe(u16, f.out);
    defer testing.allocator.free(bits);
    @memcpy(bits[f.case.at.out..][0..f.want.len], f.want);
    bits[0] = 0x1234; // guard poison clobbered before the data window
    try testing.expectError(error.GuardClobbered, f.firstMismatch(bits));
}

test "only unsupported-device subnormal classes may flush; preserving devices stay strict" {
    const f_base = try Add.init(testing.allocator, add_cases[2], 0x5719);
    defer f_base.deinit();
    const bits = try testing.allocator.dupe(u16, f_base.out);
    defer testing.allocator.free(bits);
    for ([_]bool{ true, false }) |preserve32| {
        var f = f_base;
        f.allow_subnormal_flush = !preserve32;
        @memcpy(bits[f.case.at.out..][0..f.want.len], f.want);
        try f.check(bits);
        for ([_]usize{ 12, 13, 14 }) |i| bits[f.case.at.out + i] = f.want[i] & 0x8000;
        if (preserve32) {
            try testing.expectError(error.BitMismatch, f.check(bits));
        } else {
            try f.check(bits);
            bits[f.case.at.out + 12] = 0x8000;
            try testing.expectError(error.BitMismatch, f.check(bits));
            bits[f.case.at.out + 12] = 0;
            for ([_]usize{ 4, 5, 10, 15, 16 }) |i| {
                const old = bits[f.case.at.out + i];
                bits[f.case.at.out + i] = old ^ 0x8000;
                try testing.expectError(error.BitMismatch, f.check(bits));
                bits[f.case.at.out + i] = old;
            }
            bits[f.case.at.out] = 0;
            try testing.expectError(error.NonFiniteMismatch, f.check(bits));
        }
    }
    // The policy is scoped to the classes gate, not arbitrary finite outputs.
    var normal = try Add.init(testing.allocator, add_cases[0], 0x571a);
    defer normal.deinit();
    normal.allow_subnormal_flush = true;
    const normal_bits = try testing.allocator.dupe(u16, normal.out);
    defer testing.allocator.free(normal_bits);
    @memcpy(normal_bits[normal.case.at.out..][0..normal.want.len], normal.want);
    normal.want[0] = 0x0001;
    normal_bits[normal.case.at.out] = 0;
    try testing.expectError(error.BitMismatch, normal.check(normal_bits));
}

test "the checkers reject a deliberately wrong finite output, not only accept good ones" {
    // RMS dynamic range: want + 0.1 is far outside the output-relative budget.
    {
        const f = try Rms.init(testing.allocator, rms_cases[2], 0x5710);
        defer f.deinit();
        const bits = try testing.allocator.dupe(u16, f.out);
        defer testing.allocator.free(bits);
        for (f.want, 0..) |w, i| bits[f.case.at.out + i] = c.toBf16(@floatCast(w));
        _ = try f.check(bits);
        const at = f.case.width / 2;
        bits[f.case.at.out + at] = c.toBf16(@floatCast(f.want[at] + 0.1));
        try testing.expectError(error.OutOfBound, f.check(bits));
    }
    // RMS zero row: one bf16 subnormal unit of output must be refused outright.
    {
        const f = try Rms.init(testing.allocator, rms_cases[1], 0x5711);
        defer f.deinit();
        const bits = try testing.allocator.dupe(u16, f.out);
        defer testing.allocator.free(bits);
        for (f.want, 0..) |w, i| bits[f.case.at.out + i] = c.toBf16(@floatCast(w));
        _ = try f.check(bits);
        bits[f.case.at.out + (f.case.rows - 1) * f.case.width] = 0x0001;
        try testing.expectError(error.OutOfBound, f.check(bits));
    }
    // SwiGLU: a wrong finite output must be refused too.
    {
        const f = try Swiglu.init(testing.allocator, swiglu_cases[0], 0x5712);
        defer f.deinit();
        const bits = try testing.allocator.dupe(u16, f.out);
        defer testing.allocator.free(bits);
        for (f.want, 0..) |w, i| bits[f.case.at.out + i] = c.toBf16(@floatCast(w));
        _ = try f.check(bits);
        bits[f.case.at.out] = c.toBf16(@floatCast(f.want[0] + 0.1));
        try testing.expectError(error.OutOfBound, f.check(bits));
    }
    // Residual add: a wrong finite output is a bit mismatch.
    {
        const f = try Add.init(testing.allocator, add_cases[0], 0x5713);
        defer f.deinit();
        const bits = try testing.allocator.dupe(u16, f.out);
        defer testing.allocator.free(bits);
        @memcpy(bits[f.case.at.out..][0..f.want.len], f.want);
        try f.check(bits);
        bits[f.case.at.out] = c.toBf16(@floatCast(c.fromBf16(f.want[0]) + 0.1));
        try testing.expectError(error.BitMismatch, f.check(bits));
    }
}

fn addAllocRun(a: std.mem.Allocator, case: AddCase) !void {
    const f = try Add.init(a, case, 0x5714);
    f.deinit();
}

fn swigluAllocRun(a: std.mem.Allocator, case: SwigluCase) !void {
    const f = try Swiglu.init(a, case, 0x5715);
    f.deinit();
}

fn rmsAllocRun(a: std.mem.Allocator, case: RmsCase) !void {
    const f = try Rms.init(a, case, 0x5716);
    f.deinit();
}

test "every fixture init survives each single allocation failure without leaking" {
    // checkAllAllocationFailures fails each init allocation; error.OutOfMemory required; errdefers free earlier bytes.
    try std.testing.checkAllAllocationFailures(testing.allocator, addAllocRun, .{add_cases[0]});
    try std.testing.checkAllAllocationFailures(testing.allocator, addAllocRun, .{add_cases[2]});
    try std.testing.checkAllAllocationFailures(testing.allocator, swigluAllocRun, .{swiglu_cases[0]});
    try std.testing.checkAllAllocationFailures(testing.allocator, swigluAllocRun, .{swiglu_cases[1]});
    for (rms_cases) |case| try std.testing.checkAllAllocationFailures(testing.allocator, rmsAllocRun, .{case});
}

test "guards sit poisoned outside every data window" {
    const f = try Rms.init(testing.allocator, rms_cases[1], 0x5708);
    defer f.deinit();
    for (f.x, 0..) |g, i| {
        if (i >= f.case.at.x and i < f.case.at.x + f.case.rows * f.case.width) continue;
        try testing.expectEqual(poison16, g);
    }
    for (f.out, 0..) |g, i| {
        if (i >= f.case.at.out and i < f.case.at.out + f.case.rows * f.case.width) continue;
        try testing.expectEqual(poison16, g);
    }
}
