//! Shared pieces of the LFM3 op tests: bf16 data, launches, the FP64 comparison and its control.

const std = @import("std");
pub const rt = @import("rig.zig");

pub const gpa = std.heap.page_allocator;
pub const Buf = rt.Buffer;
/// Largest |device - FP64| over the largest |FP64| a bf16 output may show; a control must land above it.
pub const tol: f64 = 1e-2;

pub fn bf(v: u16) f64 {
    return @as(f32, @bitCast(@as(u32, v) << 16));
}

/// Round to nearest even, as the kernels' to_bf.
pub fn toBf(f: f64) u16 {
    var u: u32 = @bitCast(@as(f32, @floatCast(f)));
    u +%= 0x7fff + ((u >> 16) & 1);
    return @intCast(u >> 16);
}

pub fn randBf(buf: []u16, rnd: std.Random, sd: f64) void {
    for (buf) |*v| v.* = toBf(rnd.floatNorm(f64) * sd);
}

pub fn widen(gpa_: std.mem.Allocator, h: []const u16) ![]f64 {
    const out = try gpa_.alloc(f64, h.len);
    for (h, out) |v, *o| o.* = bf(v);
    return out;
}

pub fn run(k: *rt.Kernel, groups: [3]u32, a: anytype) !void {
    inline for (a, 0..) |v, i| {
        const T = @TypeOf(v);
        if (T == Buf) {
            try k.setBuffer(i, v);
        } else if (T == f32 or T == comptime_float) {
            try k.setF32(i, v);
        } else try k.setU32(i, @as(u32, v));
    }
    try k.launch(groups);
}

pub fn upload(r: *rt.Runtime, h: []const u16) !Buf {
    const b = try r.alloc(h.len * 2);
    try r.upload(b, std.mem.sliceAsBytes(h));
    return b;
}

pub fn fetch(r: *rt.Runtime, b: Buf, n: usize) ![]u16 {
    const h = try gpa.alloc(u16, n);
    try r.sync();
    try r.download(std.mem.sliceAsBytes(h), b);
    try r.sync();
    return h;
}

/// max|got - want| / max|want|.
pub fn relErr(got: []const u16, want: []const f64) f64 {
    var e: f64 = 0;
    var m: f64 = 0;
    for (got, want) |g, w| {
        e = @max(e, @abs(bf(g) - w));
        m = @max(m, @abs(w));
    }
    return if (m == 0) e else e / m;
}

/// Tallies the checks of one test program: real checks must stay within tol, planted controls must exceed it.
pub const Tally = struct {
    name: []const u8,
    bad: bool = false,

    pub fn check(t: *Tally, what: []const u8, err: f64) void {
        const ok = err <= tol;
        if (!ok) t.bad = true;
        std.debug.print("  {s}: rel err {e:.2} (tol {e:.0}) {s}\n", .{ what, err, tol, if (ok) "pass" else "FAIL" });
    }

    pub fn control(t: *Tally, what: []const u8, err: f64) void {
        const fails = err > tol;
        if (!fails) t.bad = true;
        std.debug.print("  control {s}: rel err {e:.2} {s}\n", .{ what, err, if (fails) "fails as planted" else "PASSES (control broken)" });
    }

    pub fn same(t: *Tally, what: []const u8, eq: bool) void {
        if (!eq) t.bad = true;
        std.debug.print("  {s}: {s}\n", .{ what, if (eq) "bitwise identical" else "DIFFERENT" });
    }

    pub fn finish(t: *Tally) !void {
        std.debug.print("{s}: {s}\n", .{ t.name, if (t.bad) "FAILED" else "ok" });
        if (t.bad) return error.TestFailed;
    }
};
