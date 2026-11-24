//! GLM-5.3-Flash TFQ3 Metal compute dispatchers for dense projections, KDA input, dense GLU, and MoE layers.
const std = @import("std");
const mtl = @import("metal");
const affine = @import("affine.zig");
const tfq3 = @import("tfq3.zig");

const Ref = affine.Ref;
const Affine = affine.Affine;
const Pipes = tfq3.Pipes;
const Dims = tfq3.Dims;
const DenseDims = tfq3.DenseDims;

fn dispatchDenseRot(
    p: *const Pipes,
    e: mtl.ComputeEncoder,
    w: Ref,
    s: Ref,
    tables: Ref,
    bits: u8,
    xrot: Ref,
    yr_out: Ref,
    dd_in: DenseDims,
) DenseDims {
    const nb = dd_in.rows;
    const n_bc = dd_in.k >> 7;
    var dd = dd_in;
    if (dd.n < 8192 and n_bc >= 8 and (n_bc % 8 == 0)) {
        dd.ksplit = 8;
        const pipe = if (nb == 1)
            (if (bits == 4) p.get(.dense_dec_2dsk8_1_4b) else p.get(.dense_dec_2dsk8_1_5b))
        else if (nb == 2)
            (if (bits == 4) p.get(.dense_dec_2dsk8_2_4b) else p.get(.dense_dec_2dsk8_2_5b))
        else
            (if (bits == 4) p.get(.dense_dec_2dsk8_4_4b) else p.get(.dense_dec_2dsk8_4_5b));
        e.setPipeline(pipe);
        e.setBuffer(w.buf, w.off, 0);
        e.setBuffer(s.buf, s.off, 1);
        e.setBuffer(tables.buf, tables.off, 2);
        e.setBuffer(xrot.buf, xrot.off, 3);
        e.setBuffer(yr_out.buf, yr_out.off, 4);
        e.setValue(dd, 5);
        e.dispatchGroups(mtl.Size.of(dd.n / 32, 8, 1), mtl.Size.of(64, 1, 1));
    } else {
        dd.ksplit = 1;
        const pipe = if (nb == 1)
            (if (bits == 4) p.get(.dense_dec_nosplit1_4b) else p.get(.dense_dec_nosplit1_5b))
        else if (nb == 2)
            (if (bits == 4) p.get(.dense_dec_nosplit2_4b) else p.get(.dense_dec_nosplit2_5b))
        else
            (if (bits == 4) p.get(.dense_dec_nosplit4_4b) else p.get(.dense_dec_nosplit4_5b));
        e.setPipeline(pipe);
        e.setBuffer(w.buf, w.off, 0);
        e.setBuffer(s.buf, s.off, 1);
        e.setBuffer(tables.buf, tables.off, 2);
        e.setBuffer(xrot.buf, xrot.off, 3);
        e.setBuffer(yr_out.buf, yr_out.off, 4);
        e.setValue(dd, 5);
        e.dispatchGroups(mtl.Size.of(dd.n / 32, 1, 1), mtl.Size.of(64, 1, 1));
    }
    return dd;
}

pub fn dense(
    p: *const Pipes,
    e: mtl.ComputeEncoder,
    w: Ref,
    s: Ref,
    tables: Ref,
    bits: u8,
    in: Ref,
    out: Ref,
    n: u32,
    k: u32,
    rows: u32,
    ldx: u32,
    ldo: u32,
) void {
    const max_r: u32 = @max(1, tfq3.max_xrot_elems / k);
    var row0: u32 = 0;
    while (row0 < rows) {
        const nb = @min(max_r, rows - row0);
        const in_off = in.at(@as(usize, row0) * ldx * 2);
        const out_off = out.at(@as(usize, row0) * ldo * 2);
        const dd: DenseDims = .{ .rows = nb, .n = n, .k = k, .ldx = ldx, .ldo = ldo, .ksplit = 1 };

        e.setPipeline(p.get(.fwht_x_gen));
        e.setBuffer(in_off.buf, in_off.off, 0);
        e.setBuffer(p.xrot_buf, 0, 1);
        e.setValue(dd, 2);
        e.dispatchGroups(mtl.Size.of(k / 128, nb, 1), mtl.Size.of(32, 1, 1));

        if (nb >= 8) {
            e.setPipeline(if (bits == 4) p.get(.dense_prefill_nax_4b) else p.get(.dense_prefill_nax_5b));
            e.setBuffer(w.buf, w.off, 0);
            e.setBuffer(s.buf, s.off, 1);
            e.setBuffer(tables.buf, tables.off, 2);
            e.setBuffer(p.xrot_buf, 0, 3);
            e.setBuffer(out_off.buf, out_off.off, 4);
            e.setValue(dd, 5);
            e.dispatchGroups(mtl.Size.of(n / 128, (nb + 31) / 32, 1), mtl.Size.of(128, 1, 1));
        } else if (n > 65536 or (n >= 16384 and k <= 2048)) {
            const pipe = if (nb == 1)
                (if (bits == 4) p.get(.dense_dec_fused1_4b) else p.get(.dense_dec_fused1_5b))
            else if (nb == 2)
                (if (bits == 4) p.get(.dense_dec_fused2_4b) else p.get(.dense_dec_fused2_5b))
            else
                (if (bits == 4) p.get(.dense_dec_fused4_4b) else p.get(.dense_dec_fused4_5b));
            e.setPipeline(pipe);
            e.setBuffer(w.buf, w.off, 0);
            e.setBuffer(s.buf, s.off, 1);
            e.setBuffer(tables.buf, tables.off, 2);
            e.setBuffer(p.xrot_buf, 0, 3);
            e.setBuffer(out_off.buf, out_off.off, 4);
            e.setValue(dd, 5);
            e.dispatchGroups(mtl.Size.of(n / 128, 1, 1), mtl.Size.of(256, 1, 1));
        } else {
            const dd_out = dispatchDenseRot(p, e, w, s, tables, bits, .{ .buf = p.xrot_buf }, .{ .buf = p.yr_buf }, dd);
            e.setPipeline(p.get(.fwht_y_gen));
            e.setBuffer(p.yr_buf, 0, 0);
            e.setBuffer(out_off.buf, out_off.off, 1);
            e.setValue(dd_out, 2);
            e.dispatchGroups(mtl.Size.of(n / 128, nb, 1), mtl.Size.of(32, 1, 1));
        }
        row0 += nb;
    }
}

/// Fused KDA input path: FWHT(X) + 320-row 8-bit/64 gates in one launch, then TG=64 NO_SPLIT q/k/v + inv-FWHT.
pub fn tryKdaIn(p: *const Pipes, e: mtl.ComputeEncoder, q: Affine, in: Ref, out: Ref, rows: u32, ldx: u32, ldo: u32) bool {
    if (q.more != 1 or rows > 7) return false;
    const r0 = q.run(0);
    const r1 = q.run(1);
    if (r0.group != 512 or r0.n != 24576 or q.k != 4096 or r1.group != 64 or r1.bits != 8 or r1.n != 320 or r1.row != 24576) {
        return false;
    }
    const dd: DenseDims = .{ .rows = rows, .n = 24576, .k = 4096, .ldx = ldx, .ldo = ldo, .ksplit = 1 };
    e.setPipeline(p.get(.fwht_x_plus_q8));
    e.setBuffer(in.buf, in.off, 0);
    e.setBuffer(p.xrot_buf, 0, 1);
    e.setBuffer(r1.w.buf, r1.w.off, 2);
    e.setBuffer(r1.s.buf, r1.s.off, 3);
    e.setBuffer(r1.b.buf, r1.b.off, 4);
    e.setBuffer(out.buf, out.off, 5);
    e.setValue(dd, 6);
    e.dispatchGroups(mtl.Size.of(352, rows, 1), mtl.Size.of(32, 1, 1));

    const dd_out = dispatchDenseRot(p, e, r0.w, r0.s, r0.b, r0.bits, .{ .buf = p.xrot_buf }, .{ .buf = p.yr_buf }, dd);
    e.setPipeline(p.get(.fwht_y_gen));
    e.setBuffer(p.yr_buf, 0, 0);
    e.setBuffer(out.buf, out.off, 1);
    e.setValue(dd_out, 2);
    e.dispatchGroups(mtl.Size.of(24576 / 128, rows, 1), mtl.Size.of(32, 1, 1));
    return true;
}

pub fn tryDenseGlu(p: ?*const Pipes, e: mtl.ComputeEncoder, gu: Affine, x_in: Ref, out: Ref, n: u32, rows: u32, lim: f32) bool {
    const pw = p orelse return false;
    if (rows > 7 or gu.more != 1 or gu.group != 512 or gu.runs[0].group != 512 or gu.bits != gu.runs[0].bits) return false;
    denseGlu(pw, e, gu.w, gu.s, gu.b, gu.bits, x_in, out, n, gu.k, rows, lim);
    return true;
}

pub fn denseGlu(
    p: *const Pipes,
    e: mtl.ComputeEncoder,
    w: Ref,
    s: Ref,
    tables: Ref,
    bits: u8,
    in: Ref,
    out: Ref,
    n: u32,
    k: u32,
    rows: u32,
    lim: f32,
) void {
    std.debug.assert(rows * 2 * n <= tfq3.max_yr_elems);
    const dd: DenseDims = .{ .rows = rows, .n = 2 * n, .k = k, .ldx = k, .ldo = n, .lim = lim, .ksplit = 1 };
    e.setPipeline(p.get(.fwht_x_gen));
    e.setBuffer(in.buf, in.off, 0);
    e.setBuffer(p.xrot_buf, 0, 1);
    e.setValue(dd, 2);
    e.dispatchGroups(mtl.Size.of(k / 128, rows, 1), mtl.Size.of(32, 1, 1));

    const dd_out = dispatchDenseRot(p, e, w, s, tables, bits, .{ .buf = p.xrot_buf }, .{ .buf = p.yr_buf }, dd);

    e.setPipeline(p.get(.fwht_glu_unrot));
    e.setBuffer(p.yr_buf, 0, 0);
    e.setBuffer(out.buf, out.off, 1);
    e.setValue(dd_out, 2);
    e.dispatchGroups(mtl.Size.of(n / 128, rows, 1), mtl.Size.of(32, 1, 1));
}

pub const FusedArgs = struct {
    x_in: Ref,
    logits_r: Ref,
    bias: Ref,
    pick: Ref,
    uids: Ref,
    umem: Ref,
    ucount: Ref,
    wts: Ref,
    sh_gu_w: Ref,
    sh_gu_s: Ref,
    sh_dn_w: Ref,
    sh_dn_s: Ref,
    act: Ref,
    ye: Ref,
    ys: Ref,
    branch: Ref,
    routed_scale: f32,
    has_5b: bool,
    fuse_pw_shared: bool = false,
    sh_bits: u8 = 5,
    fuse_combine: bool,
    fuse_route: bool,
};

pub fn moeDecode(
    p: *const Pipes,
    e: mtl.ComputeEncoder,
    tl: *const tfq3.Layer,
    a: FusedArgs,
    rows: u32,
    topk: u32,
    experts: u32,
    lim: f32,
) void {
    const slots = rows * topk;
    const do_route = a.fuse_route and rows == 1;
    const do_pw_sh = a.fuse_pw_shared and a.fuse_combine and ((a.has_5b and a.sh_bits == 5) or (!a.has_5b and a.sh_bits == 4));
    const d: Dims = .{
        .rows = rows,
        .slots = slots,
        .topk = topk,
        .experts = experts,
        .lim = lim,
        .routed_scale = a.routed_scale,
        .flags = (if (do_route) @as(u32, 2) else 0) | (if (do_pw_sh) @as(u32, 4) else 0),
    };

    e.setPipeline(p.get(.fwht_x_fwd));
    e.setBuffer(a.x_in.buf, a.x_in.off, 0);
    e.setBuffer(p.xrot_buf, 0, 1);
    e.setValue(d, 2);
    e.setBuffer(a.logits_r.buf, a.logits_r.off, 3);
    e.setBuffer(a.bias.buf, a.bias.off, 4);
    e.setBuffer(a.pick.buf, a.pick.off, 5);
    e.setBuffer(a.wts.buf, a.wts.off, 6);
    e.dispatchGroups(mtl.Size.of(if (do_route) 33 else 32, rows, 1), mtl.Size.of(32, 1, 1));

    const sh_gu_w_buf = if (do_pw_sh) a.sh_gu_w.buf else tl.weights;
    const sh_gu_w_off = if (do_pw_sh) a.sh_gu_w.off else 0;
    const sh_gu_s_buf = if (do_pw_sh) a.sh_gu_s.buf else tl.scales;
    const sh_gu_s_off = if (do_pw_sh) a.sh_gu_s.off else 0;
    const gu_groups = slots + (if (do_pw_sh) @as(u32, 1) else 0);
    if (rows == 1) {
        e.setPipeline(if (a.has_5b) p.get(.dec_gu_b1_543) else p.get(.dec_gu_b1_43));
        e.setBuffer(tl.weights, 0, 0);
        e.setBuffer(tl.scales, 0, 1);
        e.setBuffer(tl.meta, 0, 2);
        e.setBuffer(tl.tables, 0, 3);
        e.setBuffer(p.xrot_buf, 0, 4);
        e.setBuffer(a.pick.buf, a.pick.off, 5);
        e.setBuffer(p.gu_buf, 0, 6);
        e.setValue(d, 7);
        e.setBuffer(sh_gu_w_buf, sh_gu_w_off, 8);
        e.setBuffer(sh_gu_s_buf, sh_gu_s_off, 9);
        e.dispatchGroups(mtl.Size.of(64, gu_groups, 2), mtl.Size.of(64, 1, 1));
    } else {
        e.setPipeline(if (a.has_5b) p.get(.dec_gu_dedup_543) else p.get(.dec_gu_dedup_43));
        e.setBuffer(tl.weights, 0, 0);
        e.setBuffer(tl.scales, 0, 1);
        e.setBuffer(tl.meta, 0, 2);
        e.setBuffer(tl.tables, 0, 3);
        e.setBuffer(p.xrot_buf, 0, 4);
        e.setBuffer(a.uids.buf, a.uids.off, 5);
        e.setBuffer(a.umem.buf, a.umem.off, 6);
        e.setBuffer(a.ucount.buf, a.ucount.off, 7);
        e.setBuffer(p.gu_buf, 0, 8);
        e.setValue(d, 9);
        e.setBuffer(sh_gu_w_buf, sh_gu_w_off, 10);
        e.setBuffer(sh_gu_s_buf, sh_gu_s_off, 11);
        e.dispatchGroups(mtl.Size.of(64, gu_groups, 2), mtl.Size.of(64, 1, 1));
    }

    e.setPipeline(p.get(.fwht_glu_fwht));
    e.setBuffer(p.gu_buf, 0, 0);
    e.setBuffer(a.act.buf, a.act.off, 1);
    e.setValue(d, 2);
    e.setBuffer(p.sact_buf, 0, 3);
    e.dispatchGroups(mtl.Size.of(16, slots + (if (do_pw_sh) rows else 0), 1), mtl.Size.of(32, 1, 1));

    const sh_dn_w_buf = if (do_pw_sh) a.sh_dn_w.buf else tl.weights;
    const sh_dn_w_off = if (do_pw_sh) a.sh_dn_w.off else 0;
    const sh_dn_s_buf = if (do_pw_sh) a.sh_dn_s.buf else tl.scales;
    const sh_dn_s_off = if (do_pw_sh) a.sh_dn_s.off else 0;
    if (rows == 1) {
        e.setPipeline(if (a.has_5b) p.get(.dec_down_b1_543) else p.get(.dec_down_b1_43));
        e.setBuffer(tl.weights, 0, 0);
        e.setBuffer(tl.scales, 0, 1);
        e.setBuffer(tl.meta, 0, 2);
        e.setBuffer(tl.tables, 0, 3);
        e.setBuffer(a.act.buf, a.act.off, 4);
        e.setBuffer(a.pick.buf, a.pick.off, 5);
        e.setBuffer(p.yr_buf, 0, 6);
        e.setValue(d, 7);
        e.setBuffer(sh_dn_w_buf, sh_dn_w_off, 8);
        e.setBuffer(sh_dn_s_buf, sh_dn_s_off, 9);
        e.setBuffer(p.sact_buf, 0, 10);
        e.dispatchGroups(mtl.Size.of(128, gu_groups, 1), mtl.Size.of(64, 1, 1));
    } else {
        e.setPipeline(if (a.has_5b) p.get(.dec_down_dedup_543) else p.get(.dec_down_dedup_43));
        e.setBuffer(tl.weights, 0, 0);
        e.setBuffer(tl.scales, 0, 1);
        e.setBuffer(tl.meta, 0, 2);
        e.setBuffer(tl.tables, 0, 3);
        e.setBuffer(a.act.buf, a.act.off, 4);
        e.setBuffer(a.uids.buf, a.uids.off, 5);
        e.setBuffer(a.umem.buf, a.umem.off, 6);
        e.setBuffer(a.ucount.buf, a.ucount.off, 7);
        e.setBuffer(p.yr_buf, 0, 8);
        e.setValue(d, 9);
        e.setBuffer(sh_dn_w_buf, sh_dn_w_off, 10);
        e.setBuffer(sh_dn_s_buf, sh_dn_s_off, 11);
        e.setBuffer(p.sact_buf, 0, 12);
        e.dispatchGroups(mtl.Size.of(128, gu_groups, 1), mtl.Size.of(64, 1, 1));
    }

    if (a.fuse_combine) {
        e.setPipeline(p.get(.fwht_y_combine));
        e.setBuffer(p.yr_buf, 0, 0);
        e.setBuffer(a.ys.buf, a.ys.off, 1);
        e.setBuffer(a.wts.buf, a.wts.off, 2);
        e.setBuffer(a.branch.buf, a.branch.off, 3);
        e.setValue(d, 4);
        e.dispatchGroups(mtl.Size.of(32, rows, 1), mtl.Size.of(32, 1, 1));
    } else {
        e.setPipeline(p.get(.fwht_y_inv));
        e.setBuffer(p.yr_buf, 0, 0);
        e.setBuffer(a.ye.buf, a.ye.off, 1);
        e.setValue(d, 2);
        e.dispatchGroups(mtl.Size.of(32, slots, 1), mtl.Size.of(32, 1, 1));
    }
}

fn moeUnfused(p: *const Pipes, e: mtl.ComputeEncoder, tl: *const tfq3.Layer, x_in: Ref, pick: Ref, act: Ref, ye: Ref, rows: u32, topk: u32, experts: u32, lim: f32) void {
    const slots = rows * topk;
    const d: Dims = .{ .rows = rows, .slots = slots, .topk = topk, .experts = experts, .lim = lim, .flags = 0 };
    e.setPipeline(p.get(.fwht_x_fwd));
    e.setBuffer(x_in.buf, x_in.off, 0);
    e.setBuffer(p.xrot_buf, 0, 1);
    e.setValue(d, 2);
    e.setBuffer(x_in.buf, x_in.off, 3);
    e.setBuffer(x_in.buf, x_in.off, 4);
    e.setBuffer(pick.buf, pick.off, 5);
    e.setBuffer(x_in.buf, x_in.off, 6);
    e.dispatchGroups(mtl.Size.of(32, rows, 1), mtl.Size.of(32, 1, 1));

    e.setPipeline(if (tl.has_5b) p.get(.dec_gu_b1_543) else p.get(.dec_gu_b1_43));
    e.setBuffer(tl.weights, 0, 0);
    e.setBuffer(tl.scales, 0, 1);
    e.setBuffer(tl.meta, 0, 2);
    e.setBuffer(tl.tables, 0, 3);
    e.setBuffer(p.xrot_buf, 0, 4);
    e.setBuffer(pick.buf, pick.off, 5);
    e.setBuffer(p.gu_buf, 0, 6);
    e.setValue(d, 7);
    e.setBuffer(tl.weights, 0, 8);
    e.setBuffer(tl.scales, 0, 9);
    e.dispatchGroups(mtl.Size.of(64, slots, 2), mtl.Size.of(64, 1, 1));

    e.setPipeline(p.get(.fwht_glu_fwht));
    e.setBuffer(p.gu_buf, 0, 0);
    e.setBuffer(act.buf, act.off, 1);
    e.setValue(d, 2);
    e.setBuffer(p.sact_buf, 0, 3);
    e.dispatchGroups(mtl.Size.of(16, slots, 1), mtl.Size.of(32, 1, 1));

    e.setPipeline(if (tl.has_5b) p.get(.dec_down_b1_543) else p.get(.dec_down_b1_43));
    e.setBuffer(tl.weights, 0, 0);
    e.setBuffer(tl.scales, 0, 1);
    e.setBuffer(tl.meta, 0, 2);
    e.setBuffer(tl.tables, 0, 3);
    e.setBuffer(act.buf, act.off, 4);
    e.setBuffer(pick.buf, pick.off, 5);
    e.setBuffer(p.yr_buf, 0, 6);
    e.setValue(d, 7);
    e.setBuffer(tl.weights, 0, 8);
    e.setBuffer(tl.scales, 0, 9);
    e.setBuffer(p.sact_buf, 0, 10);
    e.dispatchGroups(mtl.Size.of(128, slots, 1), mtl.Size.of(64, 1, 1));

    e.setPipeline(p.get(.fwht_y_inv));
    e.setBuffer(p.yr_buf, 0, 0);
    e.setBuffer(ye.buf, ye.off, 1);
    e.setValue(d, 2);
    e.dispatchGroups(mtl.Size.of(32, slots, 1), mtl.Size.of(32, 1, 1));
}

pub fn moeSorted(
    p: *const Pipes,
    e: mtl.ComputeEncoder,
    tl: *const tfq3.Layer,
    x_in: Ref,
    pick: Ref,
    offsets: Ref,
    order: Ref,
    act: Ref,
    ye: Ref,
    rows: u32,
    topk: u32,
    experts: u32,
    lim: f32,
) void {
    if (rows < 4) return moeUnfused(p, e, tl, x_in, pick, act, ye, rows, topk, experts, lim);
    const slots = rows * topk;
    const d: Dims = .{ .rows = rows, .slots = slots, .topk = topk, .experts = experts, .lim = lim, .flags = 0 };
    const max_tiles = @min(slots, (slots + 31) / 32 + experts - 1);

    e.setPipeline(p.get(.fwht_x_fwd));
    e.setBuffer(x_in.buf, x_in.off, 0);
    e.setBuffer(p.xrot_buf, 0, 1);
    e.setValue(d, 2);
    e.setBuffer(x_in.buf, x_in.off, 3);
    e.setBuffer(x_in.buf, x_in.off, 4);
    e.setBuffer(pick.buf, pick.off, 5);
    e.setBuffer(x_in.buf, x_in.off, 6);
    e.dispatchGroups(mtl.Size.of(32, rows, 1), mtl.Size.of(32, 1, 1));

    e.setPipeline(p.get(.tile_scan));
    e.setBuffer(offsets.buf, offsets.off, 0);
    e.setBuffer(p.tile_meta_buf, 0, 1);
    e.setBuffer(p.tile_cnt_buf, 0, 2);
    e.setValue(d, 3);
    e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(320, 1, 1));

    e.setPipeline(if (tl.has_5b) p.get(.prefill_nax_glu_543) else p.get(.prefill_nax_glu_43));
    e.setBuffer(tl.weights, 0, 0);
    e.setBuffer(tl.scales, 0, 1);
    e.setBuffer(tl.meta, 0, 2);
    e.setBuffer(tl.tables, 0, 3);
    e.setBuffer(p.xrot_buf, 0, 4);
    e.setBuffer(offsets.buf, offsets.off, 5);
    e.setBuffer(order.buf, order.off, 6);
    e.setBuffer(act.buf, act.off, 7);
    e.setValue(d, 8);
    e.setBuffer(p.tile_meta_buf, 0, 9);
    e.setBuffer(p.tile_cnt_buf, 0, 10);
    e.dispatchGroups(mtl.Size.of(16, max_tiles, 1), mtl.Size.of(256, 1, 1));

    e.setPipeline(if (tl.has_5b) p.get(.prefill_nax_down_543) else p.get(.prefill_nax_down_43));
    e.setBuffer(tl.weights, 0, 0);
    e.setBuffer(tl.scales, 0, 1);
    e.setBuffer(tl.meta, 0, 2);
    e.setBuffer(tl.tables, 0, 3);
    e.setBuffer(act.buf, act.off, 4);
    e.setBuffer(offsets.buf, offsets.off, 5);
    e.setBuffer(order.buf, order.off, 6);
    e.setBuffer(ye.buf, ye.off, 7);
    e.setValue(d, 8);
    e.setBuffer(p.tile_meta_buf, 0, 9);
    e.setBuffer(p.tile_cnt_buf, 0, 10);
    e.dispatchGroups(mtl.Size.of(32, max_tiles, 1), mtl.Size.of(128, 1, 1));
}
