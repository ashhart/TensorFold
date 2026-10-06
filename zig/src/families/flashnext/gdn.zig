//! Flash Next's DeltaNet window step on fz_gdn (kernels/metal/decode/fn_gdn.metal): the recorded q4_gdn's arithmetic
//! with every row's conv, norms and gates at once and the state recurrence free of barriers between rows.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const replay = @import("replay.zig");
const Run = replay.Run;
const Buf = replay.Buf;

/// Bytes of one window's replay record for one layer: k and v of every row and head (bf16), then four floats each.
pub const RECORD = 2 * replay.MAXR * 48 * 128 * 2 + replay.MAXR * 48 * 4 * 4;

/// fz_gdn after the recorded kernels' helpers: every row's state stored, or (`kept`) one state a layer kept in place
/// with the previous window's kept rows replayed into it.
pub fn compile(r: *Run, kept: bool) !mtl.Pipeline {
    const text = try std.mem.concat(r.arena, u8, &.{ r.xnew_header, if (kept) "#define FZ_GDN_REPLAY 1\n" else "#define FZ_GDN_REPLAY 0\n", ks.flashnext_gdn });
    const lib = try mtl.Library.fromSource(r.device, text, mtl.CompileOptions.mlx());
    return mtl.Pipeline.init(r.device, lib, "fz_gdn", false);
}

/// The step over value heads [h0, h0 + heads) for the rows `ins`' rows buffer holds; inputs and outputs as q4_gdn's.
pub fn step(r: *Run, pipe: mtl.Pipeline, ins: []const Buf, outs: []const Buf, h0: u32, heads: usize) void {
    if (r.skip & Run.class("gdn") != 0) return;
    r.enc.setPipeline(pipe);
    for (ins, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
    for (outs, 0..) |b, j| r.enc.setBuffer(b.b, b.off, ins.len + j);
    r.enc.setBytes(std.mem.asBytes(&h0), ins.len + outs.len);
    r.enc.dispatchThreads(mtl.Size.of(heads * 1024, 1, 1), mtl.Size.of(1024, 1, 1));
    if (!r.serial) r.enc.barrier();
}

/// The kept-state step: `ins` as q4_gdn's with its state read and written in place, outputs and conv windows in
/// `outs`, the previous window's record `prev` (its first ar[0] rows replayed) and this window's `cur`.
pub fn stepKept(r: *Run, pipe: mtl.Pipeline, ins: []const Buf, outs: []const Buf, prev: Buf, cur: Buf, ar: Buf, h0: u32, heads: usize) void {
    if (r.skip & Run.class("gdn") != 0) return;
    r.enc.setPipeline(pipe);
    for (ins, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
    for (outs, 0..) |b, j| r.enc.setBuffer(b.b, b.off, 9 + j);
    r.enc.setBuffer(prev.b, prev.off, 11);
    r.enc.setBytes(std.mem.asBytes(&h0), 12);
    r.enc.setBuffer(cur.b, cur.off, 13);
    r.enc.setBuffer(ar.b, ar.off, 14);
    r.enc.dispatchThreads(mtl.Size.of(heads * 1024, 1, 1), mtl.Size.of(1024, 1, 1));
    if (!r.serial) r.enc.barrier();
}
