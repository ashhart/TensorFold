//! Uploading core exl3_load groups to the device as the exl3 kernels' weights (the engine's load path).
const std = @import("std");
const driver = @import("driver.zig");
const memory = @import("memory.zig");
const exl3 = @import("exl3.zig");
const exl3e = @import("exl3_experts.zig");
const Driver = driver.Driver;

/// The tensors one upload holds: appended to `bufs` and freed together.
pub const Uploader = struct {
    d: *const Driver,
    gpa: std.mem.Allocator,
    bufs: std.ArrayList(memory.DeviceBuffer) = .empty,
    max_buffers: usize = 0, // caller supplies a finite tensor budget; zero refuses uploads
    bytes: usize = 0, // the tensors' total device bytes

    pub fn deinit(u: *Uploader) void {
        for (u.bufs.items) |*b| b.free();
        u.bufs.deinit(u.gpa);
    }

    pub fn dev(u: *Uploader, bytes: []const u8) !u64 {
        if (u.bufs.items.len >= u.max_buffers) return error.UploadLimit;
        const total = std.math.add(usize, u.bytes, bytes.len) catch return error.UploadLimit;
        try u.bufs.ensureUnusedCapacity(u.gpa, 1);
        const b = try memory.DeviceBuffer.fromHost(u.d, bytes);
        u.bufs.appendAssumeCapacity(b);
        u.bytes = total;
        return b.ptr;
    }

    /// One dense group: `parts` (exl3_load.Parts) and `g` (exl3_load.Group) as the Weight; the scales must be fp16.
    pub fn dense(u: *Uploader, parts: anytype, g: anytype) !exl3.Weight {
        const mark = u.bufs.items.len;
        errdefer u.rollback(mark);
        const st = exl3.strides(g.k, g.n, g.halves, false);
        return .{
            .trellis = try u.dev(parts.trellis.bytes),
            .suh = try u.scale(parts.suh.bytes, parts.suh_packed),
            .svh = try u.scale(parts.svh.bytes, parts.suh_packed),
            .bias = if (parts.bias) |b| try u.dev(b.bytes) else 0,
            .k = @intCast(g.k),
            .n = @intCast(g.n),
            .halves = g.halves,
            .codebook = switch (g.codebook) {
                .inst3 => .inst3,
                .mcg => .mcg,
                .mul1 => .mul1,
            },
            .stride_k = st.k,
            .stride_nb = st.nb,
        };
    }
    pub fn rollback(u: *Uploader, mark: usize) void {
        while (u.bufs.items.len > mark) {
            var b = u.bufs.pop().?;
            u.bytes -= b.len;
            b.free();
        }
    }

    fn scale(u: *Uploader, bytes: []const u8, sign_words: bool) !u64 {
        if (!sign_words) return u.dev(bytes);
        if (bytes.len % 2 != 0) return error.BadScales;
        const expanded = try u.gpa.alloc(f16, try std.math.mul(usize, bytes.len, 8));
        defer u.gpa.free(expanded);
        try expandSigns(bytes, std.mem.sliceAsBytes(expanded));
        return u.dev(std.mem.sliceAsBytes(expanded));
    }
};

/// One expert projection's tensors (fp16 scales) and half-bit width: the exl3_load Parts' bytes, the group's halves.
pub const Proj = struct { trellis: []const u8, suh: []const u8, svh: []const u8, k2: u32, signs: bool = false };

/// One layer's routed experts: trellises in place (pointers stacked), scales stacked [E, n] fp16, widths int32 [E].
pub fn layer(u: *Uploader, a: std.mem.Allocator, dims: usize, width: usize, cb: exl3.Codebook, limit: f32, act_mode: i32, gate: []const Proj, up: []const Proj, down: []const Proj) !exl3e.Layer {
    const mark = u.bufs.items.len;
    errdefer u.rollback(mark);
    const e = gate.len;
    if (up.len != e or down.len != e or e == 0) return error.BadExperts;
    const gt = try pointers(u, a, gate);
    defer a.free(gt);
    const ut = try pointers(u, a, up);
    defer a.free(ut);
    const dt = try pointers(u, a, down);
    defer a.free(dt);
    return .{
        .gate = try u.dev(std.mem.sliceAsBytes(gt)),
        .up = try u.dev(std.mem.sliceAsBytes(ut)),
        .down = try u.dev(std.mem.sliceAsBytes(dt)),
        .gate_k2 = try k2s(u, a, gate),
        .up_k2 = try k2s(u, a, up),
        .down_k2 = try k2s(u, a, down),
        .suh_g = try stacked(u, a, gate, dims, "suh"),
        .suh_u = try stacked(u, a, up, dims, "suh"),
        .svh_g = try stacked(u, a, gate, width, "svh"),
        .svh_u = try stacked(u, a, up, width, "svh"),
        .suh_d = try stacked(u, a, down, width, "suh"),
        .svh_d = try stacked(u, a, down, dims, "svh"),
        .experts = e,
        .dims = dims,
        .width = width,
        .codebook = cb,
        .gu_stride = 0,
        .limit = limit,
        .act_mode = act_mode,
    };
}

fn pointers(u: *Uploader, a: std.mem.Allocator, projs: []const Proj) ![]u64 {
    const out = try a.alloc(u64, projs.len);
    errdefer a.free(out);
    for (projs, 0..) |p, i| out[i] = try u.dev(p.trellis);
    return out;
}

fn k2s(u: *Uploader, a: std.mem.Allocator, projs: []const Proj) !u64 {
    const host = try a.alloc(i32, projs.len);
    defer a.free(host);
    for (projs, 0..) |p, i| host[i] = @intCast(p.k2);
    return u.dev(std.mem.sliceAsBytes(host));
}

/// Every expert's fp16 scale of `name` ("suh" or "svh"), `n` values each, stacked [E, n].
fn stacked(u: *Uploader, a: std.mem.Allocator, projs: []const Proj, n: usize, comptime name: []const u8) !u64 {
    const host = try a.alloc(u8, try std.math.mul(usize, projs.len, try std.math.mul(usize, n, 2)));
    defer a.free(host);
    for (projs, 0..) |p, i| {
        const s = @field(p, name);
        const dest = host[i * n * 2 ..][0 .. n * 2];
        if (p.signs) {
            try expandSigns(s, dest);
        } else {
            if (s.len != dest.len) return error.BadScales;
            @memcpy(dest, s);
        }
    }
    return u.dev(host);
}

test "the upload budget bounds repeated appends" {
    const mock = @import("exl3_mock.zig");
    var d = mock.init();
    var u: Uploader = .{ .d = &d, .gpa = std.testing.allocator };
    if (comptime @hasField(Uploader, "max_buffers")) u.max_buffers = 2;
    defer u.deinit();
    _ = try u.dev(&.{1});
    _ = try u.dev(&.{2});
    try std.testing.expectError(error.UploadLimit, u.dev(&.{3}));
    try std.testing.expectEqual(@as(usize, 2), mock.allocations);
}

test "a failed list allocation leaves no device buffer behind" {
    const mock = @import("exl3_mock.zig");
    var d = mock.init();
    var bytes: [0]u8 = .{};
    var fba = std.heap.FixedBufferAllocator.init(&bytes);
    var u: Uploader = .{ .d = &d, .gpa = fba.allocator() };
    u.max_buffers = 1;
    defer u.deinit();
    const before = memory.usage(false).device;
    try std.testing.expectError(error.OutOfMemory, u.dev(&.{1}));
    const after = memory.usage(false).device;
    if (after > before) {
        var leaked: memory.DeviceBuffer = .{ .d = &d, .ptr = 256, .len = 1 };
        leaked.free();
    }
    try std.testing.expectEqual(before, after);
}

test "packed scales expand in little endian bit order and release their temporary owner" {
    const mock = @import("exl3_mock.zig");
    var d = mock.init();
    var u: Uploader = .{ .d = &d, .gpa = std.testing.allocator, .max_buffers = 1 };
    defer u.deinit();
    _ = try u.scale(&.{ 1, 128 }, true);
    try std.testing.expectEqual(@as(usize, 32), mock.copy_len);
    for (0..16) |i| {
        const value = std.mem.readInt(u16, mock.copied[i * 2 ..][0..2], .little);
        try std.testing.expectEqual(@as(u16, if (i == 0 or i == 15) 0xbc00 else 0x3c00), value);
    }
}

test "a partially uploaded dense group rolls back to its entry mark" {
    const mock = @import("exl3_mock.zig");
    var d = mock.init();
    var u: Uploader = .{ .d = &d, .gpa = std.testing.allocator, .max_buffers = 2 };
    defer u.deinit();
    const Tensor = struct { bytes: []const u8 };
    const Part = struct { trellis: Tensor, suh: Tensor, svh: Tensor, bias: ?Tensor = null, suh_packed: bool = true };
    const parts: Part = .{ .trellis = .{ .bytes = &.{ 0, 0 } }, .suh = .{ .bytes = &.{ 0, 0 } }, .svh = .{ .bytes = &.{ 0, 0 } } };
    const Book = enum { inst3, mcg, mul1 };
    const g = .{ .k = @as(usize, 128), .n = @as(usize, 128), .halves = @as(u16, 8), .codebook = Book.mul1 };
    try std.testing.expectError(error.UploadLimit, u.dense(parts, g));
    try std.testing.expectEqual(@as(usize, 0), u.bufs.items.len);
    try std.testing.expectEqual(@as(usize, 0), u.bytes);
    try std.testing.expectEqual(mock.allocations, mock.frees);
}

fn expandSigns(words: []const u8, dest: []u8) !void {
    if (words.len % 2 != 0 or words.len != dest.len / 16 or dest.len % 32 != 0) return error.BadScales;
    for (0..dest.len / 2) |i| {
        const word = std.mem.readInt(u16, words[i / 16 * 2 ..][0..2], .little);
        const value: u16 = if (word & (@as(u16, 1) << @intCast(i % 16)) != 0) 0xbc00 else 0x3c00;
        std.mem.writeInt(u16, dest[i * 2 ..][0..2], value, .little);
    }
}

test "expert upload retains independent projection widths and expands packed scales" {
    const mock = @import("exl3_mock.zig");
    var d = mock.init();
    var u: Uploader = .{ .d = &d, .gpa = std.testing.allocator, .max_buffers = 18 };
    defer u.deinit();
    const signs: [16]u8 = @splat(255);
    var gate: [2]Proj = @splat(.{ .trellis = &.{ 0, 0 }, .suh = &signs, .svh = &signs, .k2 = 8, .signs = true });
    gate[1].k2 = 10;
    var up = gate;
    up[0].k2 = 12;
    up[1].k2 = 16;
    var down = gate;
    down[0].k2 = 9;
    down[1].k2 = 13;
    const l = try layer(&u, std.testing.allocator, 128, 128, .mul1, 0, 0, &gate, &up, &down);
    try std.testing.expect(l.gate_k2 != l.up_k2 and l.up_k2 != l.down_k2);
    for ([_][2]u32{ .{ 8, 10 }, .{ 12, 16 }, .{ 9, 13 } }, 9..) |expected, j| {
        for (expected, 0..) |v, i| try std.testing.expectEqual(v, std.mem.readInt(u32, mock.copies[j][i * 4 ..][0..4], .little));
    }
    try std.testing.expectEqual(@as(usize, 18), u.bufs.items.len);
    try std.testing.expectEqual(@as(u16, 0xbc00), std.mem.readInt(u16, mock.copied[0..2], .little));
}
