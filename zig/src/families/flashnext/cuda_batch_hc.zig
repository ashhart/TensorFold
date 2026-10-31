//! A layer's writeback with its read-out's normalize in one launch, and the native read-out (TF_FLASHNEXT_HC_FUSE).
const std = @import("std");
const wt = @import("cuda_prompt_wt.zig");
const parts = @import("cuda_batch_state.zig");

const dims = wt.dims;
const streams = wt.streams;
const wide = wt.wide;
const low = wt.low;
const slots = wt.slots;
const eps = wt.eps;
const pss_n = (dims / 256) * streams;

/// On by default (the same bits either way); TF_FLASHNEXT_HC_FUSE=0: the writeback and the normalize as two launches.
pub fn on() bool {
    const v = std.c.getenv("TF_FLASHNEXT_HC_FUSE") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// Whether hc_w's read-out takes its normed rows from the writeback: native, nothing kept, 64 rows or more, our glue.
pub fn fuses(pass: anytype, hc_w: wt.HcW, n: usize, keep: parts.HcKeep) bool {
    return n >= 64 and hc_w.down.native != null and std.meta.eql(keep, parts.HcKeep{}) and dims == 2560 and
        streams == 4 and pass.tri.writebackOwned() and on();
}

/// The pass's pending writeback with hc_w's normalize in the same launch: the normed rows readoutNative reads.
pub fn writebackNorm(pass: anytype, hc_w: wt.HcW, r0: usize, n: usize) !void {
    const s = pass.s;
    const dummy = s.dummy;
    const h = s.h + r0 * wide * 2;
    const pss = s.pss + r0 * pss_n * 4;
    const inj = s.inj + r0 * streams * 2;
    const norm = s.norm + r0 * wide * 2;
    const rs = pass.rows * dims;
    const scale = hc_w.scale.ptr;
    if (pass.pending == null) return pass.tri.hcWritebackNorm(h, h, pss, dummy, dummy, dummy, dummy, scale, norm, eps, rs, n, dims, streams, 0, 1, 1, 1);
    pass.pending = null;
    if (pass.pending_f32) {
        pass.pending_f32 = false;
        return pass.tri.hcWritebackNorm(h, h, pss, s.moe_f32 + r0 * dims * 4, inj, dummy, dummy, scale, norm, eps, rs, n, dims, streams, 4, 1, 1, 1);
    }
    const y = s.y + r0 * slots * dims * 4;
    const wts = s.wts + r0 * slots * 4;
    try pass.tri.hcWritebackNorm(h, h, pss, dummy, inj, y, wts, scale, norm, eps, rs, n, dims, streams, 2, 10, 11, 1);
}

/// A native read-out over rows [r0, r0 + n) whose normed rows are in: down, act and gates, up, the mix.
pub fn readoutNative(pass: anytype, hc_w: wt.HcW, r0: usize, n: usize) !void {
    const s = pass.s;
    const inj = s.inj + r0 * streams * 2;
    const norm = s.norm + r0 * wide * 2;
    const act = s.act + r0 * low * 2;
    const xs_act = s.xs_act + r0 * (low / 32) * 4;
    const down = hc_w.down.native orelse return error.UnexpectedTensor;
    const up = hc_w.up.native orelse return error.UnexpectedTensor;
    if (n * hc_w.n_down * 4 > s.hc_part_bytes) return error.UnexpectedTensor; // one slice within hc_part (hcPartRows)
    try pass.k.nativeMatmul(pass.stream, norm, wide, n, down, s.hc_part, s.native_part, .f32);
    const inj_ptr = if (hc_w.has_inj == 1) inj else act;
    try pass.tri.hcReduceAct(s.hc_part, act, xs_act, inj_ptr, 1, n, streams, low, hc_w.n_down, hc_w.has_inj);
    try pass.k.nativeMatmul(pass.stream, act, low, n, up, s.hc_up, s.native_part, .bf16);
    try pass.tri.hcDenseMix(s.hc_up, norm, s.mixed + r0 * dims * 2, s.xsm + r0 * (dims / 32) * 4, n, dims, streams);
}
