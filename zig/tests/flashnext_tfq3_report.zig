//! The TFQ3 parity test's checks: a failure names its mode, width and comparison, the index off and both values.
const std = @import("std");

/// How `same` prints a value: an expert id, or a float from its bits (u32 an f32, u16 a bf16).
pub const Kind = enum { id, f32, bf16 };

/// One parity run: the source it compiled and each output slot's width.
pub const Run = struct {
    units: bool,
    /// The checks' own test: return the errors without printing, so a passing run shows no failure text.
    quiet: bool = false,
    /// widths[row * 11 + slot]: slots 0..9 the B=1 pick's bits, slot 10 the shared expert's 8.
    widths: [44]u32 = @splat(0),

    pub fn mode(r: *const Run) []const u8 {
        return if (r.units) "tensor-unit" else "simdgroup";
    }

    /// A command buffer's own error, named with the dispatch that ran.
    pub fn ran(r: *const Run, what: []const u8, n: usize, failure: ?[*:0]const u8) !void {
        const why = failure orelse return;
        if (!r.quiet) std.debug.print("TFQ3 parity, {s} mode: {s} {d}: the command buffer failed: {s}\n", .{ r.mode(), what, n, why });
        return error.Tfq3CommandFailed;
    }

    /// Exact: the first element off, with its row, slot, width and both values; `per` elements make one slot.
    pub fn same(r: *const Run, comptime T: type, kind: Kind, what: []const u8, per: usize, slots: usize, want: []const T, got: []const T) !void {
        const i = std.mem.indexOfDiff(T, want, got) orelse return;
        const row = i / per / slots;
        const slot = i / per % slots;
        if (r.quiet) return error.TestExpectedEqual;
        std.debug.print("TFQ3 parity, {s} mode: {s} differ first at index {d} (row {d}, slot {d}, {d}-bit): ", .{ r.mode(), what, i, row, slot, r.widths[row * 11 + slot] });
        switch (kind) {
            .id => std.debug.print("want {d}, got {d}\n", .{ want[i], got[i] }),
            .f32, .bf16 => std.debug.print("want {e} (0x{x}), got {e} (0x{x})\n", .{ bits(want[i]), want[i], bits(got[i]), got[i] }),
        }
        return error.TestExpectedEqual;
    }

    /// Within its bound: the first element past it (NaN included), how many are, and the largest difference.
    pub fn near(r: *const Run, what: []const u8, slot: usize, width: u32, n: Near) !void {
        const f = n.first orelse return;
        if (!r.quiet) std.debug.print("TFQ3 parity, {s} mode: {s}, slot {d} ({d}-bit): {d} values not within {d}, first at index {d}: want {e}, got {e}; largest difference {e}\n", .{ r.mode(), what, slot, width, n.past, n.bound, f.at, f.want, f.got, n.worst });
        return error.TestExpectedApproxEqAbs;
    }
};

fn bits(v: anytype) f32 {
    return @bitCast(if (@TypeOf(v) == u16) @as(u32, v) << 16 else @as(u32, v));
}

/// Absolute differences against a bound; a NaN difference counts as past it.
pub const Near = struct {
    bound: f64,
    first: ?struct { at: usize, want: f64, got: f64 } = null,
    past: usize = 0,
    worst: f64 = 0,

    pub fn see(n: *Near, at: usize, want: f64, got: f64) void {
        const d = @abs(got - want);
        if (!(d < n.bound)) {
            n.past += 1;
            if (n.first == null) n.first = .{ .at = at, .want = want, .got = got };
        }
        if (!std.math.isNan(n.worst) and !(d <= n.worst)) n.worst = d;
    }
};

test "TFQ3 parity checks: equal and near pass; a difference, a NaN and a failed buffer are named" {
    var r: Run = .{ .units = true, .quiet = true };
    r.widths[2] = 4;
    const a = [_]u16{ 1, 2, 3, 4 };
    try r.same(u16, .bf16, "B=1 and B=4 outputs", 1, 11, &a, &a);
    var b = a;
    b[2] = 0x3f80;
    try std.testing.expectError(error.TestExpectedEqual, r.same(u16, .bf16, "B=1 and B=4 outputs", 1, 11, &a, &b));
    var n: Near = .{ .bound = 0.02 };
    n.see(0, 0.01, 0.015);
    try r.near("routed expert vs CPU reference", 0, 3, n);
    n.see(1, 0.01, std.math.nan(f64));
    try std.testing.expectEqual(@as(usize, 1), n.past);
    try std.testing.expectError(error.TestExpectedApproxEqAbs, r.near("routed expert vs CPU reference", 0, 3, n));
    try r.ran("decode B=1 row", 0, null);
    try std.testing.expectError(error.Tfq3CommandFailed, r.ran("prefill", 0, "Internal Error"));
}
