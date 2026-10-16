//! Runs the 4-bit matvec kernel on 256 real rows of the Nemotron checkpoint and compares with the fp64 reference.

const std = @import("std");
const xpu = @import("xpu");
const rt = @import("rt.zig");
const fixture = @import("fixture.zig");

const spv = xpu.kernels.qmv4;
var fw: []const u8 = &.{};
var fs: []const u8 = &.{};
var fb: []const u8 = &.{};
var fx: []const u8 = &.{};
var fy: []const u8 = &.{};

pub fn run() !void {
    try loadFixtures();
    const rows: u32 = 256;
    const in_dim: u32 = 4096;
    var r = try rt.Runtime.init();
    defer r.deinit();
    var w = try r.alloc(fw.len);
    defer w.free();
    var s = try r.alloc(fs.len);
    defer s.free();
    var b = try r.alloc(fb.len);
    defer b.free();
    var x = try r.alloc(fx.len);
    defer x.free();
    var y = try r.alloc(rows * 4);
    defer y.free();
    try r.upload(w, fw);
    try r.upload(s, fs);
    try r.upload(b, fb);
    try r.upload(x, fx);
    var m = try r.module(spv);
    defer m.deinit();
    var k = try m.kernel("qmv4", .{ 16, 1, 1 });
    defer k.deinit();
    try k.setBuffer(0, w);
    try k.setBuffer(1, s);
    try k.setBuffer(2, b);
    try k.setBuffer(3, x);
    try k.setBuffer(4, y);
    try k.setU32(5, in_dim);
    try k.launch(.{ rows, 1, 1 });
    var out: [rows]f32 = undefined;
    try r.download(std.mem.sliceAsBytes(&out), y);
    try r.sync();
    const want = std.mem.bytesAsSlice(f32, fy);
    var max_rel: f32 = 0;
    for (out, want) |got, ref| max_rel = @max(max_rel, @abs(got - ref) / @max(@abs(ref), 1e-3));
    std.debug.print("qmv4 {d} rows x {d}: max relative error {e}\n", .{ rows, in_dim, max_rel });
    if (max_rel > 1e-4) return error.TooInaccurate;
}

fn loadFixtures() !void {
    fw = try fixture.load("qmv_w");
    fs = try fixture.load("qmv_s");
    fb = try fixture.load("qmv_b");
    fx = try fixture.load("qmv_x");
    fy = try fixture.load("qmv_y");
}
