//! Default-off fixed-split NVFP4 16x128 experiment; only the two dense MLP shapes on sm_12x.
const std = @import("std");
const q = @import("qmmf.zig");
const abi = @import("abi.zig");
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const Driver = @import("driver.zig").Driver;

pub const shared: u32 = 26624;
pub const control_symbol = "tf_qmmf_fp4_wide_order_control";
pub const Set = struct { functions: [2][2]Function, sm: u32 };

pub fn symbol(comptime out: q.Out, comptime cluster: bool) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN7tf_qmmf11qmmf_kernelILi0ELi16ELi128ELi1ELi4ELi4ELb{d}ELb{d}ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .{ @intFromBool(out == .f32), @intFromBool(cluster) });
}

pub fn supported(sm: u32) bool {
    return sm == 120 or sm == 121;
}

/// An absent optional symbol leaves the old lane intact; other driver failures are not hidden.
pub fn load(mod: Module, sm: u32) !?Set {
    if (!supported(sm)) return null;
    var set: Set = .{ .functions = undefined, .sm = sm };
    inline for (.{ q.Out.bf16, q.Out.f32 }, 0..) |out, oi| inline for (.{ false, true }, 0..) |cluster, ci| {
        var handle: abi.Function = null;
        const result = mod.d.api.cuModuleGetFunction(&handle, mod.handle, symbol(out, cluster).ptr);
        if (result == abi.error_not_found) return null;
        try mod.d.check(result, "cuModuleGetFunction");
        const f: Function = .{ .d = mod.d, .handle = handle };
        try f.allowDynamicShared(shared);
        set.functions[oi][ci] = f;
    };
    return set;
}

pub fn apply(l: *q.Lane, d: *const Driver) !void {
    if (!try @import("sys").environment.read(bool, "TF_QMMF_EXACT_TILE", false)) return;
    const sm = try d.currentCapability();
    l.fp4_wide = try load(l.mod, sm);
}

pub fn eligible(m: usize, w: q.Weight) bool {
    return m > 0 and m <= 16 and w.mode == .fp4 and w.npad >= w.n and w.npad % 128 == 0 and
        ((w.n == 17408 and w.k == 5120) or (w.n == 5120 and w.k == 17408));
}

pub fn pick(l: q.Lane, m: usize, w: q.Weight, out: q.Out) ?Function {
    if (!eligible(m, w)) return null;
    const set = l.fp4_wide orelse return null;
    if (!supported(set.sm) or l.half_n != null) return null;
    const p = q.planFor(m, w, l.clusters);
    if (p.fused or p.bm != 16) return null;
    return set.functions[@backingInt(out)][@intFromBool(p.cluster)];
}

const Span = struct {
    first: u64,
    end: u64,
    fn init(first: u64, bytes: usize) !Span {
        if (first == 0) return error.BadWideBuffer;
        return .{ .first = first, .end = std.math.add(u64, first, bytes) catch return error.BadWideBuffer };
    }
    fn overlaps(a: Span, b: Span) bool {
        return a.first < b.end and b.first < a.end;
    }
};

/// Extra refusals apply only to the experiment; the null-set path leaves main's launcher untouched.
pub fn checked(l: q.Lane, x: u64, ldx: usize, m: usize, w: q.Weight, y: u64, out: q.Out, part: u64) !?Function {
    const f = pick(l, m, w, out) orelse return null;
    const p = q.planFor(m, w, l.clusters);
    _ = try q.pinned(w, p.sk);
    const stride = if (m == 1) w.k else ldx;
    if (stride < w.k or stride % 8 != 0 or x % 16 != 0 or w.codes % 16 != 0 or w.scales % 16 != 0)
        return error.BadWideBuffer;
    const words = std.math.mul(usize, m - 1, stride) catch return error.BadWideBuffer;
    const length = std.math.add(usize, words, w.k) catch return error.BadWideBuffer;
    const input_bytes = std.math.mul(usize, length, 2) catch return error.BadWideBuffer;
    const inputs = [_]Span{ try Span.init(x, input_bytes), try Span.init(w.codes, @as(usize, w.npad) * w.k / 2), try Span.init(w.scales, @as(usize, w.npad) * (w.k / 16)) };
    const output = try Span.init(y, m * w.n * (if (out == .f32) @as(usize, 4) else 2));
    if (y % (if (out == .f32) @as(u64, 4) else 2) != 0) return error.BadWideBuffer;
    for (inputs) |input| if (output.overlaps(input)) return error.WideAlias;
    if (p.reduce) {
        if (part % 4 != 0) return error.BadWideBuffer;
        const partial = try Span.init(part, @as(usize, p.sk) * m * w.n * 4);
        if (partial.overlaps(output)) return error.WideAlias;
        for (inputs) |input| if (partial.overlaps(input)) return error.WideAlias;
    }
    return f;
}

test "qmmf exact tile: the actual environment row keeps unset and zero off and refuses other spellings" {
    const env = @import("sys").environment;
    try std.testing.expectEqual(@as(?u64, null), try env.parse("TF_QMMF_EXACT_TILE", null));
    try std.testing.expectEqual(@as(?u64, 0), try env.parse("TF_QMMF_EXACT_TILE", "0"));
    try std.testing.expectEqual(@as(?u64, 1), try env.parse("TF_QMMF_EXACT_TILE", "1"));
    for ([_][]const u8{ "", "2", "true", "on", " 1", "1 " }) |bad|
        try std.testing.expectError(error.InvalidEnvironment, env.parse("TF_QMMF_EXACT_TILE", bad));
}
