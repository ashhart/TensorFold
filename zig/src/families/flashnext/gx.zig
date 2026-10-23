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
const RMAX = 32;
const LIST_WORDS = 16 + RMAX * 10 + 2 * (RMAX * 10 + RMAX) + 4 * (RMAX * 10 + RMAX / 8);

pub const Gx = struct {
    pick: mtl.Pipeline,
    list: mtl.Pipeline,
    gu: mtl.Pipeline,
    one: mtl.Pipeline,
    tile: mtl.Pipeline,
    buf: mtl.Buffer,
    min_rows: usize, // windows from this many rows take the grouped kernels

    /// The routed and shared experts for `rows` rows (at most RMAX; the kernels read the count from `rows_buf`).
    pub fn encode(g: *const Gx, r: *Run, x: Buf, lg: Buf, e: [18]Buf, act: Buf, pick: Buf, wts: Buf, rows_buf: Buf, y: Buf, rows: usize) void {
        if (r.skip & (1 << 2) != 0) return;
        r.enc.setPipeline(g.pick);
        for ([_]Buf{ lg, rows_buf, pick, wts }, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
        r.enc.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(32, 1, 1));
        if (!r.serial) r.enc.barrier();
        r.enc.setPipeline(g.list);
        for ([_]Buf{ pick, rows_buf }, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
        r.enc.setBuffer(g.buf, 0, 2);
        r.enc.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(512, 1, 1));
        if (!r.serial) r.enc.barrier();
        r.enc.setPipeline(g.gu);
        r.enc.setBuffer(x.b, x.off, 0);
        r.enc.setBuffer(g.buf, 0, 1);
        for (e[0..12], 2..) |b, j| r.enc.setBuffer(b.b, b.off, j);
        r.enc.setBuffer(act.b, act.off, 14);
        r.enc.dispatchIndirect(g.buf, L_GU * 4, mtl.Size.of(64, 1, 1));
        if (!r.serial) r.enc.barrier();
        for ([_]mtl.Pipeline{ g.one, g.tile }, [_]usize{ L_DNT, L_DNM }) |p, at| { // independent: no barrier between
            r.enc.setPipeline(p);
            r.enc.setBuffer(act.b, act.off, 0);
            r.enc.setBuffer(g.buf, 0, 1);
            for (e[12..18], 2..) |b, j| r.enc.setBuffer(b.b, b.off, j);
            r.enc.setBuffer(y.b, y.off, 8);
            r.enc.dispatchIndirect(g.buf, at * 4, mtl.Size.of(64, 1, 1));
        }
        if (!r.serial) r.enc.barrier();
    }
};

/// The grouped kernels read 4-bit group 32 routed and shared experts, gate/up and down.
pub fn applies(w: variants.Experts) bool {
    const four = [4]usize{ 4, 32, 4, 32 };
    return std.mem.eql(usize, &w.gu, &four) and std.mem.eql(usize, &w.down, &four);
}

/// The grouped kernels for windows of `min_rows` rows or more (at most RMAX).
pub fn init(r: *Run, min_rows: usize) !*Gx {
    const lib = try r.ownResource(try mtl.Library.fromSource(r.device, ks.flashnext_gx, mtl.CompileOptions.mlx()));
    const g = try r.arena.create(Gx);
    g.* = .{
        .pick = try r.ownResource(try mtl.Pipeline.init(r.device, lib, "gx_pick", false)),
        .list = try r.ownResource(try mtl.Pipeline.init(r.device, lib, "gx_list", false)),
        .gu = try r.ownResource(try mtl.Pipeline.init(r.device, lib, "gx_gateup", false)),
        .one = try r.ownResource(try mtl.Pipeline.init(r.device, lib, "gx_down_one", false)),
        .tile = try r.ownResource(try mtl.Pipeline.init(r.device, lib, "gx_down_tile", false)),
        .buf = try r.buffer(LIST_WORDS * 4),
        .min_rows = @min(min_rows, RMAX),
    };
    return g;
}

test "the grouped kernels take 4-bit group 32 experts only" {
    try std.testing.expect(applies(.{ .gu = .{ 4, 32, 4, 32 }, .down = .{ 4, 32, 4, 32 } }));
    try std.testing.expect(!applies(.{ .gu = .{ 6, 32, 6, 32 }, .down = .{ 4, 32, 4, 32 } }));
}
