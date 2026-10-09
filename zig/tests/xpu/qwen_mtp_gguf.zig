//! Loads the GSQ GGUF and its MTP block, runs the head fc projection on a real activation. usage: ... MODEL.gguf

const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");
const ld = @import("xpu").loader;
const cfgm = @import("qwen_xpu").config;
const model = @import("qwen_xpu").model;
const qg = @import("qwen_xpu").gguf;
const mg = @import("qwen_xpu").mtp_gguf;

const gpa = std.heap.page_allocator;

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var r = try rt.open();
    defer r.deinit();
    var l = try ld.Loader.initBare(gpa, &r);
    const meta = try qg.readHeader(gpa, &l, args[1]);
    const cfg = try qg.config(gpa, &l, meta);
    var m = try model.Model.load(gpa, &r, &l, cfg, 512);
    const before = l.total;
    const w = try mg.loadWeights(gpa, &m, &l, 512);
    std.debug.print("MTP block: {d:.3} GB on the device, fc {d} -> {d} ({s}), q {s}, down {s}\n", .{ @as(f64, @floatFromInt(l.total - before)) / 1e9, w.fc.in, w.fc.rows, @tagName(w.fc.format), @tagName(w.attn.q.format), @tagName(w.mlp.down.format) });
    // fc on [embedding of token 760 | the same] as a smoke test
    const x = try r.alloc(2 * 5120 * 2);
    try m.ops.embed(m.emb, 760);
    var row: [5120]u16 = undefined;
    try r.download(std.mem.sliceAsBytes(&row), m.ops.s.x);
    try r.sync();
    var both: [2 * 5120]u16 = undefined;
    @memcpy(both[0..5120], &row);
    @memcpy(both[5120..], &row);
    try r.upload(x, std.mem.sliceAsBytes(&both));
    const y = try r.alloc(5120 * 2);
    try m.ops.matvec(w.fc, x, y, 0);
    var out: [5120]u16 = undefined;
    try r.download(std.mem.sliceAsBytes(&out), y);
    try r.sync();
    var ss: f64 = 0;
    var bad: usize = 0;
    for (out) |v| {
        const f: f32 = @bitCast(@as(u32, v) << 16);
        if (!std.math.isFinite(f)) bad += 1 else ss += @as(f64, f) * f;
    }
    std.debug.print("fc output rms {d:.4}, non-finite {d}\n", .{ @sqrt(ss / 5120.0), bad });
    if (bad != 0) return error.NonFinite;
}
