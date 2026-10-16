//! The Flash Next glue tests' shared rig: device buffers, bf16 values, float64 rounding points and tolerances.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");

pub fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

pub fn bf(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}

/// A float64 value at a kernel's bf16 rounding point (through fp32, as the kernel rounds).
pub fn rbf(x: f64) f64 {
    return bf(bf16(@floatCast(x)));
}

pub fn sigmoid(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}

/// `n` bf16 ulps of |want| (one ulp: 2^-7 of it), with a floor for zeros.
pub fn ulps(want: f64, n: f64) f64 {
    return n * @abs(want) * std.math.pow(f64, 2, -7) + 1e-6;
}

/// An fp32 sum's tolerance: `terms` roundings of at most 2^-24 of the sum of |terms|.
pub fn sumTol(abs_sum: f64, terms: usize) f64 {
    return abs_sum * @as(f64, @floatFromInt(terms)) * std.math.pow(f64, 2, -24) + 1e-30;
}

pub const Rig = struct {
    gpu: check.Gpu,
    a: std.mem.Allocator,
    s: cuda.Stream,
    rng: std.Random,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    /// The largest |got - want| / tolerance seen, by check name.
    worst: std.StringArrayHashMapUnmanaged(f64) = .empty,

    pub fn init(gpu: check.Gpu, a: std.mem.Allocator, rng: std.Random) !Rig {
        return .{ .gpu = gpu, .a = a, .s = try cuda.Stream.init(gpu.d, false), .rng = rng };
    }

    pub fn deinit(r: *Rig) void {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(r.gpu.gpa);
        r.s.deinit();
    }

    pub fn dev(r: *Rig, bytes: []const u8) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(r.gpu.d, bytes);
        try r.bufs.append(r.gpu.gpa, b);
        return b.ptr;
    }

    /// A device buffer of `bytes`, every byte 0xff (a value the kernel misses reads NaN).
    pub fn out(r: *Rig, bytes: usize) !u64 {
        const b = try cuda.DeviceBuffer.alloc(r.gpu.d, @max(bytes, 16));
        try r.bufs.append(r.gpu.gpa, b);
        try b.fill8(0xff, r.s.handle);
        return b.ptr;
    }

    pub fn back(r: *Rig, comptime T: type, ptr: u64, n: usize) ![]T {
        const o = try r.a.alloc(T, n);
        try r.s.synchronize();
        const b: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = ptr, .len = n * @sizeOf(T) };
        try b.download(0, std.mem.sliceAsBytes(o));
        return o;
    }

    /// bf16 values within [-scale, scale).
    pub fn bfs(r: *Rig, n: usize, scale: f32) []u16 {
        const v = r.a.alloc(u16, n) catch unreachable;
        for (v) |*x| x.* = bf16((r.rng.float(f32) * 2 - 1) * scale);
        return v;
    }

    /// fp32 values within [lo, hi).
    pub fn f32s(r: *Rig, n: usize, lo: f32, hi: f32) []f32 {
        const v = r.a.alloc(f32, n) catch unreachable;
        for (v) |*x| x.* = lo + r.rng.float(f32) * (hi - lo);
        return v;
    }

    pub fn words(r: *Rig, n: usize) []u32 {
        const v = r.a.alloc(u32, n) catch unreachable;
        for (v) |*x| x.* = r.rng.int(u32);
        return v;
    }

    pub fn within(r: *Rig, what: []const u8, got: f64, want: f64, tol: f64) !void {
        const q = @abs(got - want) / tol;
        if (!(q <= 1)) try check.expect(false, "{s}: within float64 tolerance (got {d}, want {d}, {d:.3} tolerances)", .{ what, got, want, q });
        const gop = try r.worst.getOrPut(r.a, what);
        if (!gop.found_existing) gop.value_ptr.* = 0;
        gop.value_ptr.* = @max(gop.value_ptr.*, q);
    }

    /// Bytes of a launch over fewer rows against the same rows of the widest launch.
    pub fn sameRows(r: *Rig, what: []const u8, comptime T: type, ptr: u64, want: []const T) !void {
        try check.sameBytes(what, std.mem.sliceAsBytes(try r.back(T, ptr, want.len)), std.mem.sliceAsBytes(want));
    }

    pub fn report(r: *Rig, what: []const u8) void {
        var it = r.worst.iterator();
        var line: std.ArrayList(u8) = .empty;
        while (it.next()) |e| line.print(r.a, " {s} {d:.3};", .{ e.key_ptr.*, e.value_ptr.* }) catch {};
        check.pass("{s}: every row count's bits equal; worst tolerances used:{s}", .{ what, line.items });
    }
};

/// q (4 bits) of an MLX row's value i: word i / 8, nibble i % 8.
pub fn nibble(w: []const u32, row_words: usize, row: usize, i: usize) f64 {
    return @floatFromInt((w[row * row_words + i / 8] >> @intCast(4 * (i % 8))) & 0xF);
}
