//! Grouped experts (fn_gx.metal): a wide window's pairs by expert, the recorded kernels' bits at any width.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const replay = @import("replay.zig");
const variants = @import("variants.zig");
const Run = replay.Run;
const Buf = replay.Buf;

/// fn_gx.metal's LIST words: gate/up's, the down twin's and the down tiles' threadgroup counts, and the whole list.
const L_GU = 4;
const L_DNT = 8;
const L_DNM = 12;
pub const rmax = 32;
pub const list_bytes = 4 * (16 + rmax * 10 + 2 * (rmax * 10 + rmax) + 4 * (rmax * 10 + rmax / 8));
/// The kernels in launch order: picks, the list, gate/up, down a pair, down a tile.
pub const functions = [_][:0]const u8{ "gx_pick", "gx_list", "gx_gateup", "gx_down_one", "gx_down_tile" };

pub const Gx = struct {
    pipes: [functions.len]mtl.Pipeline,
    buf: mtl.Buffer,
    min_rows: usize, // windows from this many rows take the grouped kernels

    /// The routed and shared experts for `rows` rows (at most rmax; the kernels read the count from `rows_buf`).
    pub fn encode(g: *const Gx, r: *Run, x: Buf, lg: Buf, e: [18]Buf, act: Buf, pick: Buf, wts: Buf, rows_buf: Buf, y: Buf, rows: usize) void {
        if (r.skip & (1 << 2) != 0) return;
        var bound = e;
        if (r.sb_index) @import("scale_index.zig").bind(r, &bound, true, true) catch return; // init checked each set
        g.launch(r.enc, !r.serial, x, lg, bound, act, pick, wts, rows_buf, y, rows);
    }

    /// The five launches on `enc`; `barriers` for a concurrent encoder, where each waits for the one it reads.
    pub fn launch(g: *const Gx, enc: mtl.ComputeEncoder, barriers: bool, x: Buf, lg: Buf, e: [18]Buf, act: Buf, pick: Buf, wts: Buf, rows_buf: Buf, y: Buf, rows: usize) void {
        enc.setPipeline(g.pipes[0]);
        for ([_]Buf{ lg, rows_buf, pick, wts }, 0..) |b, j| enc.setBuffer(b.b, b.off, j);
        enc.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(32, 1, 1));
        if (barriers) enc.barrier();
        enc.setPipeline(g.pipes[1]);
        for ([_]Buf{ pick, rows_buf }, 0..) |b, j| enc.setBuffer(b.b, b.off, j);
        enc.setBuffer(g.buf, 0, 2);
        enc.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(512, 1, 1));
        if (barriers) enc.barrier();
        enc.setPipeline(g.pipes[2]);
        enc.setBuffer(x.b, x.off, 0);
        enc.setBuffer(g.buf, 0, 1);
        for (e[0..12], 2..) |b, j| enc.setBuffer(b.b, b.off, j);
        enc.setBuffer(act.b, act.off, 14);
        enc.dispatchIndirect(g.buf, L_GU * 4, mtl.Size.of(64, 1, 1));
        if (barriers) enc.barrier();
        for (g.pipes[3..5], [_]usize{ L_DNT, L_DNM }) |p, at| { // independent: no barrier between
            enc.setPipeline(p);
            enc.setBuffer(act.b, act.off, 0);
            enc.setBuffer(g.buf, 0, 1);
            for (e[12..18], 2..) |b, j| enc.setBuffer(b.b, b.off, j);
            enc.setBuffer(y.b, y.off, 8);
            enc.dispatchIndirect(g.buf, at * 4, mtl.Size.of(64, 1, 1));
        }
        if (barriers) enc.barrier();
    }
};

/// The grouped kernels read 4-bit group 32 routed and shared experts, gate/up and down.
pub fn applies(w: variants.Experts) bool {
    const four = [4]usize{ 4, 32, 4, 32 };
    return std.mem.eql(usize, &w.gu, &four) and std.mem.eql(usize, &w.down, &four);
}

/// fn_gx.metal compiled as the loader compiles it; `indexed` reads scales and biases through scale_index's tables.
pub fn library(a: std.mem.Allocator, device: mtl.Device, indexed: bool) !mtl.Library {
    const text = if (indexed) try @import("experts4.zig").indexedGrouped(a, ks.flashnext_gx) else ks.flashnext_gx;
    return mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
}

/// The grouped kernels for windows of `min_rows` rows or more (at most rmax).
pub fn init(r: *Run, min_rows: usize) !*Gx {
    const lib = try r.ownResource(try library(r.arena, r.device, r.sb_index));
    const g = try r.arena.create(Gx);
    var pipes: [functions.len]mtl.Pipeline = undefined;
    for (functions, &pipes) |name, *p| p.* = try r.ownResource(try mtl.Pipeline.init(r.device, lib, name, false));
    g.* = .{ .pipes = pipes, .buf = try r.buffer(list_bytes), .min_rows = @min(min_rows, rmax) };
    return g;
}

test "the grouped kernels take 4-bit group 32 experts only" {
    try std.testing.expect(applies(.{ .gu = .{ 4, 32, 4, 32 }, .down = .{ 4, 32, 4, 32 } }));
    try std.testing.expect(!applies(.{ .gu = .{ 6, 32, 6, 32 }, .down = .{ 4, 32, 4, 32 } }));
}
