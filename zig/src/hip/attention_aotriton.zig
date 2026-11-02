//! AOTriton 0.11.1's version-1 forward ABI, from upstream d34f3b6c, without a libtorch or Python dependency.
const std = @import("std");
const platform = @import("sys");
const hip = @import("runtime.zig");
const abi = @import("abi.zig");
const Stream = @import("stream.zig").Stream;
pub const Error = hip.Error || platform.optional_libraries.Failure || error{BadAttentionShape};

pub fn View(comptime rank: usize) type {
    return extern struct {
        data: u64 = 0,
        sizes: [rank]u64 = @splat(0),
        strides: [rank]u64 = @splat(0),
        dtype: c_int = 0,
    };
}
pub const T4 = View(4);
pub const T2 = View(2);
const T1 = View(1);
const T0 = extern struct { data: u64 = 0, dtype: c_int = 0 };
const NativeStream = extern struct { handle: abi.Stream };
const Params = extern struct {
    q: T4,
    k: T4,
    v: T4,
    b: T4,
    a: T2,
    scale: f32,
    lse: T2,
    out: T4,
    cu_q: T1,
    cu_k: T1,
    max_q: i32,
    max_k: i32,
    dropout: f32,
    seed: T0,
    offset1: T0,
    offset2: u64,
    seed_out: T0,
    offset_out: T0,
    encoded: T4,
    counter: T0,
    causal: i8,
    varlen: i8,
    left: i32,
    right: i32,
};

const Forward = *const fn (*const Params, i32, NativeStream, ?*const anyopaque) callconv(.c) c_int;
const Check = *const fn (NativeStream) callconv(.c) c_int;
const forward_name = "_ZN8aotriton2v35flash8attn_fwdERKNS1_15attn_fwd_paramsEiNS_14StreamTemplateIP12ihipStream_tEEPKNS1_12attn_optionsE";
const check_name = "_ZN8aotriton2v25flash9check_gpuENS_14StreamTemplateIP12ihipStream_tEE";

pub const Library = struct {
    lib: platform.DynLib,
    forward_fn: Forward,

    pub fn open(stream: Stream) Error!Library {
        if (platform.getenv("TENSORFOLD_AOTRITON")) |path| return openPath(path, stream);
        for (platform.optional_libraries.spec(.aotriton).names) |path| {
            return openPath(path, stream) catch |err| switch (err) {
                error.AotritonUnavailable => continue,
                else => return err,
            };
        }
        return platform.optional_libraries.unavailable(.aotriton);
    }

    pub fn openPath(path: []const u8, stream: Stream) Error!Library {
        var lib = platform.DynLib.open(path) catch return platform.optional_libraries.unavailable(.aotriton);
        errdefer lib.close();
        const forward_fn = lib.lookup(Forward, forward_name) orelse return platform.optional_libraries.unavailable(.aotriton);
        const check = lib.lookup(Check, check_name) orelse return platform.optional_libraries.unavailable(.aotriton);
        try hip.check(check(.{ .handle = stream.handle }));
        return .{ .lib = lib, .forward_fn = forward_fn };
    }

    /// All commands must complete before the library's GPU modules are released.
    pub fn close(l: *Library, stream: Stream) void {
        stream.synchronize() catch {};
        l.lib.close();
        l.* = undefined;
    }

    /// BF16 Q/K/V and output in [batch=1, heads, rows, dim] views, with caller-owned FP32 LSE and one int32 counter.
    pub fn forward(l: *const Library, stream: Stream, q: T4, k: T4, v: T4, out: T4, lse: u64, counter: u64, causal: bool) Error!void {
        const span = try validate(q, k, v, out);
        if (lse == 0 or counter == 0 or lse % 4 != 0 or counter % 4 != 0) return error.BadAttentionShape;
        const lse_size = q.sizes[1] * q.sizes[2] * 4;
        for (span) |region| {
            if (overlap(region[0], region[1], lse, lse_size) or overlap(region[0], region[1], counter, 4)) return error.BadAttentionShape;
        }
        if (overlap(lse, lse_size, counter, 4)) return error.BadAttentionShape;
        var p = std.mem.zeroes(Params);
        p.q = q;
        p.k = k;
        p.v = v;
        p.out = out;
        p.scale = 1 / @sqrt(@as(f32, @floatFromInt(q.sizes[3])));
        p.lse = .{ .data = lse, .sizes = .{ q.sizes[1], q.sizes[2] }, .strides = .{ q.sizes[2], 1 }, .dtype = 1 };
        p.counter = .{ .data = counter, .dtype = 12 };
        p.causal = if (causal) 3 else 0;
        p.left = if (causal) -2147483646 else 0;
        p.right = 0;
        p.dropout = 0;
        try hip.check(stream.r.api.hipMemsetAsync(@ptrFromInt(counter), 0, 4, stream.handle));
        try hip.check(l.forward_fn(&p, 1, .{ .handle = stream.handle }, null));
    }
};

fn overlap(a: u64, an: u64, b: u64, bn: u64) bool {
    const ae = std.math.add(u64, a, an) catch return true;
    const be = std.math.add(u64, b, bn) catch return true;
    return a < be and b < ae;
}

fn range(t: T4) Error![2]u64 {
    if (t.data == 0 or t.data % 2 != 0 or t.dtype != 3 or t.sizes[0] != 1 or
        t.sizes[1] == 0 or t.sizes[1] > 64 or t.sizes[2] == 0 or t.sizes[2] > 131072 or
        (t.sizes[3] != 64 and t.sizes[3] != 128) or t.strides[3] != 1) return error.BadAttentionShape;
    var last: u64 = 0;
    for (t.sizes, t.strides) |dim, stride| {
        const offset = std.math.mul(u64, dim - 1, stride) catch return error.BadAttentionShape;
        last = std.math.add(u64, last, offset) catch return error.BadAttentionShape;
    }
    const count = std.math.add(u64, last, 1) catch return error.BadAttentionShape;
    const bytes = std.math.mul(u64, count, 2) catch return error.BadAttentionShape;
    _ = std.math.add(u64, t.data, bytes) catch return error.BadAttentionShape;
    return .{ t.data, bytes };
}

fn validate(q: T4, k: T4, v: T4, out: T4) Error![4][2]u64 {
    const views = [_]T4{ q, k, v, out };
    var spans: [4][2]u64 = undefined;
    for (views, &spans) |t, *r| r.* = try range(t);
    if (!std.mem.eql(u64, &q.sizes, &out.sizes) or !std.mem.eql(u64, &k.sizes, &v.sizes) or
        q.sizes[3] != k.sizes[3] or q.sizes[1] % k.sizes[1] != 0 or q.sizes[2] > k.sizes[2]) return error.BadAttentionShape;
    const token_major = out.strides[1] == out.sizes[3] and out.strides[2] >= out.sizes[1] * out.sizes[3];
    const head_major = out.strides[2] == out.sizes[3] and out.strides[1] >= out.sizes[2] * out.sizes[3];
    if (!token_major and !head_major) return error.BadAttentionShape;
    for (spans[0..3]) |s| if (overlap(s[0], s[1], spans[3][0], spans[3][1])) return error.BadAttentionShape;
    return spans;
}

/// Dense token-major head rows as a head-major view, without a physical transpose.
pub fn rows(data: u64, heads: usize, tokens: usize, dim: usize) T4 {
    return .{ .data = data, .sizes = .{ 1, heads, tokens, dim }, .strides = .{ tokens * heads * dim, dim, heads * dim, 1 }, .dtype = 3 };
}

test "HIP AOTriton: pinned ABI sizes and public member offsets match the C++ header" {
    try std.testing.expectEqual(@as(usize, 80), @sizeOf(T4));
    try std.testing.expectEqual(@as(usize, 768), @sizeOf(Params));
    try std.testing.expectEqual(@as(usize, 368), @offsetOf(Params, "scale"));
    try std.testing.expectEqual(@as(usize, 424), @offsetOf(Params, "out"));
    try std.testing.expectEqual(@as(usize, 736), @offsetOf(Params, "counter"));
    try std.testing.expectEqual(@as(usize, 756), @offsetOf(Params, "left"));
}

test "HIP AOTriton: malformed views, overflow and output aliases are refused without runtime calls" {
    const q = rows(0x1000, 16, 4, 64);
    const k = rows(0x10000, 8, 4, 64);
    const v = rows(0x20000, 8, 4, 64);
    _ = try validate(q, k, v, rows(0x30000, 16, 4, 64));
    try std.testing.expectError(error.BadAttentionShape, validate(q, k, v, q));
    var bad = q;
    bad.strides[2] = std.math.maxInt(u64);
    try std.testing.expectError(error.BadAttentionShape, validate(bad, k, v, rows(0x30000, 16, 4, 64)));
}
