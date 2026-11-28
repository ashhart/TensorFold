//! Flash Next TFQ3 Metal compute dispatchers for MoE decode (B=1 and B=2..16 dedup) and prompt prefill.
const std = @import("std");
const mtl = @import("metal");
const replay = @import("replay.zig");
const variants = @import("variants.zig");
const tfq3 = @import("tfq3.zig");

const Buf = replay.Buf;
const Run = replay.Run;
const Pipes = tfq3.Pipes;
const Dims = tfq3.Dims;

pub fn sharedWidths(xw: ?variants.Experts) [4]u32 {
    if (xw) |w| return .{ @intCast(w.gu[2]), @intCast(w.gu[3]), @intCast(w.down[2]), @intCast(w.down[3]) };
    return .{ 6, 32, 6, 32 };
}

fn barrier(r: *Run) void {
    if (!r.serial) r.enc.barrier();
}

pub fn moeDecode(
    p: *const Pipes,
    r: *Run,
    tl: *const tfq3.Layer,
    x: Buf,
    lg: Buf,
    ex: [18]Buf,
    act: Buf,
    pick: Buf,
    wts: Buf,
    rows_buf: Buf,
    y: Buf,
    rows: usize,
    xw: ?variants.Experts,
) void {
    if (r.skip & (1 << 2) != 0) return;
    const e = r.enc;
    const r_u32: u32 = @intCast(rows);
    const slots: u32 = r_u32 * tfq3.topk;
    const sh = sharedWidths(xw);
    const d: Dims = .{
        .rows = r_u32,
        .slots = slots,
        .topk = tfq3.topk,
        .experts = tfq3.num_experts,
        .sh_gu_bits = sh[0],
        .sh_gu_group = sh[1],
        .sh_dn_bits = sh[2],
        .sh_dn_group = sh[3],
        .flags = if (rows == 1) 2 else if (p.serial_route) 0 else 4, // 4: fwht_x_fwd routes every live row
    };

    e.setPipeline(p.get(.fwht_x_fwd));
    e.setBuffer(x.b, x.off, 0);
    e.setBuffer(p.xrot_buf, 0, 1);
    e.setBuffer(rows_buf.b, rows_buf.off, 2);
    e.setValue(d, 3);
    e.setBuffer(lg.b, lg.off, 4);
    e.setBuffer(pick.b, pick.off, 5);
    e.setBuffer(wts.b, wts.off, 6);
    e.dispatchGroups(mtl.Size.of(if (rows == 1 or !p.serial_route) 21 else 20, rows, 1), mtl.Size.of(32, 1, 1));
    barrier(r);

    if (rows == 1) {
        e.setPipeline(if (tl.has_5b) p.get(.dec_gu_b1_543) else p.get(.dec_gu_b1_43));
        e.setBuffer(tl.weights, 0, 0);
        e.setBuffer(tl.scales, 0, 1);
        e.setBuffer(tl.meta, 0, 2);
        e.setBuffer(tl.tables, 0, 3);
        e.setBuffer(p.xrot_buf, 0, 4);
        e.setBuffer(pick.b, pick.off, 5);
        e.setBuffer(p.gu_buf, 0, 6);
        e.setValue(d, 7);
        e.setBuffer(x.b, x.off, 8);
        for (ex[6..12], 9..) |b, j| e.setBuffer(b.b, b.off, j);
        e.setBuffer(act.b, act.off, 15);
        e.setBuffer(rows_buf.b, rows_buf.off, 16);
        e.dispatchGroups(mtl.Size.of(20, 18, 2), mtl.Size.of(64, 1, 1));
        barrier(r);

        e.setPipeline(p.get(.fwht_glu_fwht));
        e.setBuffer(p.gu_buf, 0, 0);
        e.setBuffer(p.act_rot_buf, 0, 1);
        e.setBuffer(rows_buf.b, rows_buf.off, 2);
        e.setValue(d, 3);
        e.dispatchGroups(mtl.Size.of(5, 10, 1), mtl.Size.of(32, 1, 1));
        barrier(r);

        e.setPipeline(if (tl.has_5b) p.get(.dec_down_b1_543) else p.get(.dec_down_b1_43));
        e.setBuffer(tl.weights, 0, 0);
        e.setBuffer(tl.scales, 0, 1);
        e.setBuffer(tl.meta, 0, 2);
        e.setBuffer(tl.tables, 0, 3);
        e.setBuffer(p.act_rot_buf, 0, 4);
        e.setBuffer(pick.b, pick.off, 5);
        e.setBuffer(p.yr_buf, 0, 6);
        e.setValue(d, 7);
        e.setBuffer(act.b, act.off, 8);
        for (ex[15..18], 9..) |b, j| e.setBuffer(b.b, b.off, j);
        e.setBuffer(y.b, y.off, 12);
        e.setBuffer(rows_buf.b, rows_buf.off, 13);
        e.dispatchGroups(mtl.Size.of(80, 14, 1), mtl.Size.of(64, 1, 1));
        barrier(r);
    } else {
        if (p.serial_route) {
            e.setPipeline(p.get(.route_dedup));
            e.setBuffer(lg.b, lg.off, 0);
            e.setBuffer(rows_buf.b, rows_buf.off, 1);
            e.setBuffer(pick.b, pick.off, 2);
            e.setBuffer(wts.b, wts.off, 3);
            e.setBuffer(p.uids_buf, 0, 4);
            e.setBuffer(p.umem_buf, 0, 5);
            e.setBuffer(p.ucount_buf, 0, 6);
            e.setValue(d, 7);
            e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(32, 1, 1));
        } else { // fwht_x_fwd's group 20 routed every live row; the lists as route_dedup writes them, in parallel
            e.setPipeline(p.get(.dedup_par));
            e.setBuffer(pick.b, pick.off, 0);
            e.setBuffer(rows_buf.b, rows_buf.off, 1);
            e.setBuffer(p.uids_buf, 0, 2);
            e.setBuffer(p.umem_buf, 0, 3);
            e.setBuffer(p.ucount_buf, 0, 4);
            e.setValue(d, 5);
            e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(tfq3.max_decode_slots, 1, 1));
        }
        barrier(r);

        e.setPipeline(if (tl.has_5b) p.get(.dec_gu_dedup_543) else p.get(.dec_gu_dedup_43));
        e.setBuffer(tl.weights, 0, 0);
        e.setBuffer(tl.scales, 0, 1);
        e.setBuffer(tl.meta, 0, 2);
        e.setBuffer(tl.tables, 0, 3);
        e.setBuffer(p.xrot_buf, 0, 4);
        e.setBuffer(p.uids_buf, 0, 5);
        e.setBuffer(p.umem_buf, 0, 6);
        e.setBuffer(p.ucount_buf, 0, 7);
        e.setBuffer(p.gu_buf, 0, 8);
        e.setValue(d, 9);
        e.setBuffer(x.b, x.off, 10);
        for (ex[6..12], 11..) |b, j| e.setBuffer(b.b, b.off, j);
        e.setBuffer(act.b, act.off, 17);
        e.setBuffer(rows_buf.b, rows_buf.off, 18);
        e.dispatchGroups(mtl.Size.of(20, slots + 8, 2), mtl.Size.of(64, 1, 1));
        barrier(r);

        e.setPipeline(p.get(.fwht_glu_fwht));
        e.setBuffer(p.gu_buf, 0, 0);
        e.setBuffer(p.act_rot_buf, 0, 1);
        e.setBuffer(rows_buf.b, rows_buf.off, 2);
        e.setValue(d, 3);
        e.dispatchGroups(mtl.Size.of(5, slots, 1), mtl.Size.of(32, 1, 1));
        barrier(r);

        e.setPipeline(if (tl.has_5b) p.get(.dec_down_dedup_543) else p.get(.dec_down_dedup_43));
        e.setBuffer(tl.weights, 0, 0);
        e.setBuffer(tl.scales, 0, 1);
        e.setBuffer(tl.meta, 0, 2);
        e.setBuffer(tl.tables, 0, 3);
        e.setBuffer(p.act_rot_buf, 0, 4);
        e.setBuffer(p.uids_buf, 0, 5);
        e.setBuffer(p.umem_buf, 0, 6);
        e.setBuffer(p.ucount_buf, 0, 7);
        e.setBuffer(p.yr_buf, 0, 8);
        e.setValue(d, 9);
        e.setBuffer(act.b, act.off, 10);
        for (ex[15..18], 11..) |b, j| e.setBuffer(b.b, b.off, j);
        e.setBuffer(y.b, y.off, 14);
        e.setBuffer(rows_buf.b, rows_buf.off, 15);
        e.dispatchGroups(mtl.Size.of(80, slots + 4, 1), mtl.Size.of(64, 1, 1));
        barrier(r);
    }

    e.setPipeline(p.get(.fwht_y_inv));
    e.setBuffer(p.yr_buf, 0, 0);
    e.setBuffer(y.b, y.off, 1);
    e.setBuffer(rows_buf.b, rows_buf.off, 2);
    e.setValue(d, 3);
    e.dispatchGroups(mtl.Size.of(20, slots, 1), mtl.Size.of(32, 1, 1));
    barrier(r);
}

pub fn moePrefill(
    p: *const Pipes,
    r: *Run,
    tl: *const tfq3.Layer,
    xs: Buf,
    off: Buf,
    act_rot: Buf,
    ds: Buf,
    tile_meta: Buf,
    tile_cnt: Buf,
    pairs: usize,
) void {
    const e = r.enc;
    const slots: u32 = @intCast(pairs);
    const d: Dims = .{
        .rows = slots / tfq3.topk,
        .slots = slots,
        .topk = tfq3.topk,
        .experts = tfq3.num_experts,
    };

    e.setPipeline(p.get(.prefill_fwht_x));
    e.setBuffer(xs.b, xs.off, 0);
    e.setBuffer(ds.b, ds.off, 1);
    e.setValue(d, 2);
    e.dispatchGroups(mtl.Size.of(20, pairs, 1), mtl.Size.of(32, 1, 1));
    barrier(r);

    e.setPipeline(p.get(.tile_scan));
    e.setBuffer(off.b, off.off, 0);
    e.setBuffer(tile_meta.b, tile_meta.off, 1);
    e.setBuffer(tile_cnt.b, tile_cnt.off, 2);
    e.setValue(d, 3);
    e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(512, 1, 1));
    barrier(r);

    const max_tiles = @min(pairs / 32 + tfq3.num_experts, tfq3.max_prefill_tiles);
    e.setPipeline(if (tl.has_5b) p.get(.prefill_gu_543) else p.get(.prefill_gu_43));
    e.setBuffer(tl.weights, 0, 0);
    e.setBuffer(tl.scales, 0, 1);
    e.setBuffer(tl.meta, 0, 2);
    e.setBuffer(tl.tables, 0, 3);
    e.setBuffer(ds.b, ds.off, 4);
    e.setBuffer(off.b, off.off, 5);
    e.setBuffer(act_rot.b, act_rot.off, 6);
    e.setValue(d, 7);
    e.setBuffer(tile_meta.b, tile_meta.off, 8);
    e.setBuffer(tile_cnt.b, tile_cnt.off, 9);
    e.dispatchGroups(mtl.Size.of(5, max_tiles, 1), mtl.Size.of(256, 1, 1));
    barrier(r);

    e.setPipeline(if (tl.has_5b) p.get(.prefill_down_543) else p.get(.prefill_down_43));
    e.setBuffer(tl.weights, 0, 0);
    e.setBuffer(tl.scales, 0, 1);
    e.setBuffer(tl.meta, 0, 2);
    e.setBuffer(tl.tables, 0, 3);
    e.setBuffer(act_rot.b, act_rot.off, 4);
    e.setBuffer(off.b, off.off, 5);
    e.setBuffer(ds.b, ds.off, 6);
    e.setValue(d, 7);
    e.setBuffer(tile_meta.b, tile_meta.off, 8);
    e.setBuffer(tile_cnt.b, tile_cnt.off, 9);
    e.dispatchGroups(mtl.Size.of(20, max_tiles, 1), mtl.Size.of(128, 1, 1));
    barrier(r);
}
