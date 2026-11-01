//! EXL3 pack helpers for a rank's load: a device arena, trellis reads, TP cuts, the strips layout and fp32 reads.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const config = @import("config.zig");
const exl3 = @import("exl3.zig");
const fmt = core.exl3_format;

/// Device memory in 64 MiB blocks handed out first-fit (256-byte aligned); a block or more gets its own buffer.
pub const Arena = struct {
    d: *const cuda.Driver,
    gpa: std.mem.Allocator,
    blocks: std.ArrayList(cuda.DeviceBuffer) = .empty,
    own: std.ArrayList(cuda.DeviceBuffer) = .empty, // requests of a block or more, each its own buffer
    fill: std.ArrayList(usize) = .empty, // bytes handed out of each block
    first_open: usize = 0, // the first block with 4 KiB or more left
    total: usize = 0,
    pub const block_bytes: usize = 64 << 20;

    /// Device bytes the arena holds (blocks and own buffers), against `total` handed out.
    pub fn reserved(a: *const Arena) usize {
        var r: usize = 0;
        for (a.blocks.items) |b| r += b.len;
        for (a.own.items) |b| r += b.len;
        return r;
    }

    pub fn alloc(a: *Arena, bytes: usize) !u64 {
        const n = std.mem.alignForward(usize, @max(bytes, 1), 256);
        if (n >= block_bytes) {
            try a.own.ensureUnusedCapacity(a.gpa, 1);
            const buf = try cuda.DeviceBuffer.alloc(a.d, n);
            a.own.appendAssumeCapacity(buf);
            a.total += n;
            return buf.ptr;
        }
        var i = a.first_open;
        while (i < a.blocks.items.len) : (i += 1) {
            if (a.fill.items[i] + n <= a.blocks.items[i].len) break;
        }
        if (i == a.blocks.items.len) {
            try a.fill.ensureUnusedCapacity(a.gpa, 1);
            try a.blocks.append(a.gpa, try cuda.DeviceBuffer.alloc(a.d, block_bytes));
            a.fill.appendAssumeCapacity(0);
        }
        const at = a.blocks.items[i].ptr + a.fill.items[i];
        a.fill.items[i] += n;
        a.total += n;
        while (a.first_open < a.blocks.items.len and a.blocks.items[a.first_open].len - a.fill.items[a.first_open] < 4096) a.first_open += 1;
        return at;
    }
    pub fn upload(a: *Arena, bytes: []const u8) !u64 {
        const at = try a.alloc(bytes.len);
        try check(a.d, a.d.api.cuMemcpyHtoD_v2(at, bytes.ptr, bytes.len));
        return at;
    }
    pub fn zeros(a: *Arena, bytes: usize) !u64 {
        const at = try a.alloc(bytes);
        try check(a.d, a.d.api.cuMemsetD8_v2(at, 0, std.mem.alignForward(usize, @max(bytes, 1), 256)));
        return at;
    }
    pub fn deinit(a: *Arena) void {
        for (a.blocks.items) |*b| b.free();
        a.blocks.deinit(a.gpa);
        for (a.own.items) |*b| b.free();
        a.own.deinit(a.gpa);
        a.fill.deinit(a.gpa);
    }
};

fn check(d: *const cuda.Driver, r: anytype) !void {
    try d.check(r, "muse_spark load");
}

/// An EXL3 matrix on the host: trellis int16 [tk, tn, 8 * k2], suh fp16 [K], svh fp16 [N].
pub const Mat = struct {
    trellis: []i16,
    suh: []f16,
    svh: []f16,
    tk: usize,
    tn: usize,
    k2: u32,

    pub fn deinit(m: Mat, a: std.mem.Allocator) void {
        a.free(m.trellis);
        a.free(m.suh);
        a.free(m.svh);
    }
    pub fn k(m: Mat) usize {
        return m.tk * 16;
    }
    pub fn n(m: Mat) usize {
        return m.tn * 16;
    }
};

pub fn readMat(a: std.mem.Allocator, ck: *core.Checkpoint, prefix: []const u8) !Mat {
    var nb: [256]u8 = undefined;
    if ((ck.get(try std.fmt.bufPrint(&nb, "{s}.mul1", .{prefix})) catch null) == null) return error.NotMul1;
    const t = try ck.get(try std.fmt.bufPrint(&nb, "{s}.trellis", .{prefix}));
    if (t.dtype != .i16 or t.rank != 3 or t.shape[2] % 8 != 0) return error.BadTrellis;
    const tr = try a.alloc(i16, t.numel());
    errdefer a.free(tr);
    @memcpy(std.mem.sliceAsBytes(tr), t.bytes);
    const suh = try scales(a, ck, prefix, "suh", t.shape[0] * 16);
    errdefer a.free(suh);
    return .{ .trellis = tr, .suh = suh, .svh = try scales(a, ck, prefix, "svh", t.shape[1] * 16), .tk = t.shape[0], .tn = t.shape[1], .k2 = @intCast(t.shape[2] / 8) };
}

fn scales(a: std.mem.Allocator, ck: *core.Checkpoint, prefix: []const u8, part: []const u8, size: usize) ![]f16 {
    var nb: [256]u8 = undefined;
    const t = try ck.get(try std.fmt.bufPrint(&nb, "{s}.{s}", .{ prefix, part }));
    const out = try a.alloc(f16, size);
    errdefer a.free(out);
    if (t.dtype == .f16 and t.numel() == size) {
        @memcpy(std.mem.sliceAsBytes(out), t.bytes);
    } else if (t.dtype == .i16 and t.numel() * 16 == size) {
        const w = try a.alloc(i16, t.numel());
        defer a.free(w);
        @memcpy(std.mem.sliceAsBytes(w), t.bytes);
        fmt.unpackSigns(w, out);
    } else return error.BadScales;
    return out;
}

/// weights.py cut(): outputs [lo, hi) (trellis columns, svh) or inputs [lo, hi) (trellis rows, suh); `m` is freed.
pub fn cut(a: std.mem.Allocator, m: Mat, c: config.Cut) !Mat {
    if (c.axis == .whole) return m;
    defer m.deinit(a);
    const w = 8 * @as(usize, m.k2);
    if (c.lo % 128 != 0 or c.hi % 128 != 0) return error.CutNotOnBlock;
    if (c.axis == .n) {
        const tn = (c.hi - c.lo) / 16;
        const tr = try a.alloc(i16, m.tk * tn * w);
        for (0..m.tk) |kt| @memcpy(tr[kt * tn * w ..][0 .. tn * w], m.trellis[(kt * m.tn + c.lo / 16) * w ..][0 .. tn * w]);
        return .{ .trellis = tr, .suh = try a.dupe(f16, m.suh), .svh = try a.dupe(f16, m.svh[c.lo..c.hi]), .tk = m.tk, .tn = tn, .k2 = m.k2 };
    }
    const tk = (c.hi - c.lo) / 16;
    return .{ .trellis = try a.dupe(i16, m.trellis[(c.lo / 16) * m.tn * w ..][0 .. tk * m.tn * w]), .suh = try a.dupe(f16, m.suh[c.lo..c.hi]), .svh = try a.dupe(f16, m.svh), .tk = tk, .tn = m.tn, .k2 = m.k2 };
}

/// linear.strips: trellis int32 words [K/16, N/16, W] -> [N/128, K/16, 8, W].
pub fn strips(a: std.mem.Allocator, m: Mat) ![]u32 {
    const w32 = 4 * @as(usize, m.k2);
    const words: []align(2) const u32 = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(m.trellis));
    const out = try a.alloc(u32, words.len);
    for (0..m.tk) |kt| for (0..m.tn / 8) |nb8| for (0..8) |j| {
        @memcpy(out[((nb8 * m.tk + kt) * 8 + j) * w32 ..][0..w32], words[((kt * m.tn) + nb8 * 8 + j) * w32 ..][0..w32]);
    };
    return out;
}

/// Exl3Linear.from_tensors (strips) on the device, with its zeroed split counters; the host matrix `m` is freed.
pub fn linear(a: std.mem.Allocator, ar: *Arena, m: Mat) !exl3.Linear {
    defer m.deinit(a);
    if (m.k() % 128 != 0 or m.n() % 128 != 0) return error.NotBlocked;
    const st = try strips(a, m);
    defer a.free(st);
    return .{
        .words = try ar.upload(std.mem.sliceAsBytes(st)),
        .suh = try ar.upload(std.mem.sliceAsBytes(m.suh)),
        .svh = try ar.upload(std.mem.sliceAsBytes(m.svh)),
        .counters = try ar.zeros(8 * (m.n() / 128) * 4),
        .k = m.k(),
        .n = m.n(),
        .k2 = m.k2,
        .split = exl3.plan(m.k(), m.n()),
    };
}

/// torch's float -> bfloat16 (round to nearest even; NaN kept quiet).
pub fn bf16Round(x: f32) f32 {
    const u: u32 = @bitCast(x);
    if (std.math.isNan(x)) return @bitCast((u | 0x0040_0000) & 0xFFFF_0000);
    const r = u + 0x7FFF + ((u >> 16) & 1);
    return @bitCast(r & 0xFFFF_0000);
}

/// A tensor as fp32 (`.float()`): bf16, fp16 or fp32 storage.
pub fn asF32(a: std.mem.Allocator, t: core.safetensors.Tensor) ![]f32 {
    const n = t.numel();
    const out = try a.alloc(f32, n);
    switch (t.dtype) {
        .f32 => @memcpy(std.mem.sliceAsBytes(out), t.bytes),
        .bf16 => for (out, 0..) |*o, i| {
            o.* = @bitCast(@as(u32, std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little)) << 16);
        },
        .f16 => for (out, 0..) |*o, i| {
            o.* = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little))));
        },
        else => return error.BadDType,
    }
    return out;
}

test "bf16 rounding is torch's: nearest even, NaN kept quiet" {
    try std.testing.expectEqual(@as(f32, 1.0), bf16Round(1.0));
    try std.testing.expectEqual(@as(u32, 0x3F80_0000), @as(u32, @bitCast(bf16Round(@bitCast(@as(u32, 0x3F80_8000))))));
    try std.testing.expectEqual(@as(u32, 0x3F82_0000), @as(u32, @bitCast(bf16Round(@bitCast(@as(u32, 0x3F81_8000))))));
    try std.testing.expect(std.math.isNan(bf16Round(std.math.nan(f32))));
}
