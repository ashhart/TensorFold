//! RMSNorm and the 4-bit embedding lookup on Nemotron data, compared with the fp32 reference to one bf16 ulp.

const std = @import("std");
const xpu = @import("xpu");
const rt = @import("rt.zig");
const fixture = @import("fixture.zig");

const spv = xpu.kernels.basic;
var rms_w: []const u8 = &.{};
var rms_x: []const u8 = &.{};
var rms_y: []const u8 = &.{};
var emb_w: []const u8 = &.{};
var emb_s: []const u8 = &.{};
var emb_b: []const u8 = &.{};
var emb_y: []const u8 = &.{};

/// Largest bf16 bit distance and the count of mismatching values.
fn compare(name: []const u8, got: []const u16, want: []const u16) !void {
    var worst: u32 = 0;
    var off: usize = 0;
    for (got, want) |g, w| {
        const d: u32 = @abs(@as(i32, g) - @as(i32, w));
        if (d > 0) off += 1;
        worst = @max(worst, d);
    }
    std.debug.print("{s}: {d} values, {d} differ, worst {d} ulp\n", .{ name, got.len, off, worst });
    if (worst > 1) return error.TooInaccurate;
}

/// Embedded bytes carry no alignment guarantee; copies them into u16 storage.
fn aligned(bytes: []const u8) ![]u16 {
    const out = try std.heap.page_allocator.alloc(u16, bytes.len / 2);
    @memcpy(std.mem.sliceAsBytes(out), bytes);
    return out;
}

pub fn run() !void {
    try loadFixtures();
    var r = try rt.Runtime.init();
    defer r.deinit();
    var m = try r.module(spv);
    defer m.deinit();

    // rmsnorm: 3 rows of 2688
    {
        const n: u32 = 2688;
        var w = try r.alloc(rms_w.len);
        defer w.free();
        var x = try r.alloc(rms_x.len);
        defer x.free();
        var y = try r.alloc(rms_x.len);
        defer y.free();
        try r.upload(w, rms_w);
        try r.upload(x, rms_x);
        var k = try m.kernel("rmsnorm", .{ 64, 1, 1 });
        defer k.deinit();
        try k.setBuffer(0, x);
        try k.setBuffer(1, w);
        try k.setBuffer(2, y);
        try k.setU32(3, n);
        const eps: f32 = 1e-5;
        try k.setF32(4, eps);
        try k.launch(.{ 3, 1, 1 });
        const out = try std.heap.page_allocator.alloc(u16, 3 * n);
        try r.download(std.mem.sliceAsBytes(out), y);
        try r.sync();
        try compare("rmsnorm", out, try aligned(rms_y));
    }

    // embed4: token rows 0..2 of the extracted rows
    {
        const dim: u32 = 2688;
        const ids = [_]u32{ 0, 1, 2 };
        var w = try r.alloc(emb_w.len);
        defer w.free();
        var s = try r.alloc(emb_s.len);
        defer s.free();
        var b = try r.alloc(emb_b.len);
        defer b.free();
        var t = try r.alloc(ids.len * 4);
        defer t.free();
        var y = try r.alloc(3 * dim * 2);
        defer y.free();
        try r.upload(w, emb_w);
        try r.upload(s, emb_s);
        try r.upload(b, emb_b);
        try r.upload(t, std.mem.sliceAsBytes(&ids));
        var k = try m.kernel("embed4", .{ 64, 1, 1 });
        defer k.deinit();
        try k.setBuffer(0, w);
        try k.setBuffer(1, s);
        try k.setBuffer(2, b);
        try k.setBuffer(3, t);
        try k.setBuffer(4, y);
        try k.setU32(5, dim);
        try k.launch(.{ 3, 1, 1 });
        const out = try std.heap.page_allocator.alloc(u16, 3 * dim);
        try r.download(std.mem.sliceAsBytes(out), y);
        try r.sync();
        try compare("embed4", out, try aligned(emb_y));
    }
}

fn loadFixtures() !void {
    rms_w = try fixture.load("rms_w");
    rms_x = try fixture.load("rms_x");
    rms_y = try fixture.load("rms_y");
    emb_w = try fixture.load("emb_w");
    emb_s = try fixture.load("emb_s");
    emb_b = try fixture.load("emb_b");
    emb_y = try fixture.load("emb_y");
}
