//! Canary-banded device storage and production/control launch handles shared by GLP qualification cases.

const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41_kernels");
const check = @import("../check.zig");

pub const glp = dsv41.glp;
pub const band: usize = 512;

fn canary(i: usize) u8 {
    return @truncate(i *% 167 +% 0x5b);
}

/// One payload between two canary bands (256-byte aligned, so the payload is 16-byte aligned).
pub const Buf = struct {
    b: cuda.DeviceBuffer,
    len: usize,

    pub fn init(gpu: check.Gpu, payload: []const u8) !Buf {
        const a = gpu.gpa;
        const all = try a.alloc(u8, payload.len + 2 * band);
        defer a.free(all);
        for (all[0..band], 0..) |*v, i| v.* = canary(i);
        @memcpy(all[band..][0..payload.len], payload);
        for (all[band + payload.len ..], 0..) |*v, i| v.* = canary(i + 7);
        return .{ .b = try cuda.DeviceBuffer.fromHost(gpu.d, all), .len = payload.len };
    }

    pub fn ptr(s: Buf) u64 {
        return s.b.ptr + band;
    }

    pub fn write(s: Buf, data: []const u8) !void {
        try s.b.upload(band, data);
    }

    pub fn read(s: Buf, a: std.mem.Allocator) ![]align(16) u8 {
        const out = try a.alignedAlloc(u8, .@"16", s.len);
        errdefer a.free(out);
        try s.b.download(band, out);
        return out;
    }

    /// Both bands still hold their canary.
    pub fn guards(s: Buf, a: std.mem.Allocator) !bool {
        const all = try a.alloc(u8, s.len + 2 * band);
        defer a.free(all);
        try s.b.download(0, all);
        for (all[0..band], 0..) |v, i| if (v != canary(i)) return false;
        for (all[band + s.len ..], 0..) |v, i| if (v != canary(i + 7)) return false;
        return true;
    }

    pub fn free(s: *Buf) void {
        s.b.free();
    }
};

/// The GPU, a blocking stream (ordered after the synchronous uploads) and the production launch handle.
pub const Ctx = struct {
    gpu: check.Gpu,
    stream: cuda.Stream,
    ops: glp.Ops,

    pub fn sync(c: Ctx) !void {
        try c.stream.synchronize();
    }
};

pub fn bytes(s: anytype) []const u8 {
    return std.mem.sliceAsBytes(s);
}

/// The words of a downloaded buffer (the allocator's alignment makes the cast legal).
pub fn words(b: []align(2) const u8) []const u16 {
    return @alignCast(std.mem.bytesAsSlice(u16, b));
}

pub fn fail(comptime fmt: []const u8, args: anytype) check.Failed {
    std.debug.print("FAIL glp: " ++ fmt ++ "\n", args);
    return error.TestFailed;
}

/// A fold call on `ops`: `xs` / `ts` the row strides in elements, `alpha` any strength the checks admit.
pub fn fold(ops: glp.Ops, x: u64, xs: usize, tap: u64, ts: usize, dir: u64, alpha: f32, rows: usize, width: usize) !void {
    try ops.stream(.{
        .x = x,
        .xs = @intCast(xs),
        .tap = tap,
        .ts = @intCast(ts),
        .dir = dir,
        .alpha = alpha,
        .R = @intCast(rows),
        .width = @intCast(width),
    });
}

/// A writer call: gathered partials g [world, rows, width] with rank stride `gr` and row stride `gs` (elements).
pub fn writer(ops: glp.Ops, g: u64, gr: usize, gs: usize, dir: u64, alpha: f32, rows: usize, width: usize, world: usize) !void {
    try ops.writer(.{
        .g = g,
        .gr = @intCast(gr),
        .gs = @intCast(gs),
        .dir = dir,
        .alpha = alpha,
        .R = @intCast(rows),
        .width = @intCast(width),
        .world = @intCast(world),
    });
}
