//! Shared helpers for the GPU test runner: the device bundle, pass/fail lines and byte comparison.

const std = @import("std");
const cuda = @import("cuda");

pub const Gpu = struct {
    d: *const cuda.Driver,
    ctx: *const cuda.Context,
    gpa: std.mem.Allocator,
    io: std.Io,
};

pub const Failed = error{TestFailed};

pub fn expect(ok: bool, comptime fmt: []const u8, args: anytype) Failed!void {
    if (ok) return;
    std.debug.print("FAIL " ++ fmt ++ "\n", args);
    return error.TestFailed;
}

pub fn pass(comptime fmt: []const u8, args: anytype) void {
    std.debug.print("PASS " ++ fmt ++ "\n", args);
}

/// Equal bytes, or the first difference and how many 4-byte words differ.
pub fn sameBytes(what: []const u8, got: []const u8, want: []const u8) Failed!void {
    if (got.len != want.len) {
        std.debug.print("FAIL {s}: {d} bytes, expected {d}\n", .{ what, got.len, want.len });
        return error.TestFailed;
    }
    if (std.mem.eql(u8, got, want)) return;
    const first = std.mem.indexOfDiff(u8, got, want).?;
    var words: usize = 0;
    var i: usize = 0;
    while (i + 4 <= got.len) : (i += 4) {
        if (!std.mem.eql(u8, got[i..][0..4], want[i..][0..4])) words += 1;
    }
    std.debug.print("FAIL {s}: first difference at byte {d}, {d} of {d} words differ\n", .{ what, first, words, got.len / 4 });
    return error.TestFailed;
}

/// Monotonic nanoseconds for host timing.
pub fn now(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

pub fn ptr(b: cuda.DeviceBuffer) u64 {
    return b.ptr;
}

pub fn download(gpu: Gpu, b: cuda.DeviceBuffer) ![]u8 {
    const out = try gpu.gpa.alloc(u8, b.len);
    errdefer gpu.gpa.free(out);
    try b.download(0, out);
    return out;
}

pub fn median(xs: []f64) f64 {
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    const n = xs.len;
    return if (n % 2 == 1) xs[n / 2] else (xs[n / 2 - 1] + xs[n / 2]) / 2;
}
