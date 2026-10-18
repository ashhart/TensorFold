//! Maxima from the binding launch sites and their public shape tables, without compiled-usage guesses.
const dk = @import("dsv41_kernels");
const cuda = @import("cuda");
const catalog = @import("catalog.zig");
fn note(f: cuda.Function, threads: u32, shared: u32, cluster: u32) void {
    const e = &catalog.entries[@intFromPtr(f.handle.?) - 1];
    e.threads = @max(e.threads, threads);
    e.shared = @max(e.shared, shared);
    e.cluster = @max(e.cluster, cluster);
}
pub fn operators(f: dk.ops.Functions) void {
    const o = dk.ops;
    for (f.attn) |fun| note(fun, o.attn.NT, o.attn.SMEM, 1);
    for (f.attn_split, 0..) |row, pi| for (row) |fun| note(fun, o.attn.NT, o.attn.splitSmem(if (pi == 0) 1 else 4), if (pi == 0) 1 else 4);
    note(f.topk, o.topk.NT, o.topk.smem(o.topk.MAXEPT), o.topk.MAXCL);
    note(f.topk_b, o.topk.NT, o.topk.smem(o.topk.MAXEPT), o.topk.MAXCL);
    for (f.mhc) |row| for (row) |fun| note(fun, o.mhc.NT, o.mhc.MAXR * o.mhc.ROWB, 1);
    note(f.mhc_coef, o.mhc.NT, 0, 1);
    for (f.mhc_pf, 0..) |fun, i| note(fun, o.mhc_pf.NT, o.mhc_pf.smem(o.mhc_pf.instances[i].fn32), 1);
    for (f.gemv) |row| for (row, 0..) |fun, i| note(fun, o.rg.wide[i][0] * 32, 0, 1);
    for (f.narrow) |row| for (row, 0..) |fun, i| note(fun, o.rg.narrow[i] * 32, 0, 1);
    for (f.pfd, 0..) |width, ki| for (width) |row| for (row, 0..) |fun, i| if (fun) |value| {
        note(value, o.pfd.threads(i), o.pfd.smem(i, o.pfd.k2s[ki]), 1);
    };
    for (f.pfd_dequant) |width| for (width) |fun| if (fun) |value| {
        note(value, 128, 0, 1);
    };
    note(f.gate_copy, 256, 0, 1);
    note(f.paced, 32, 0, 1);
    for (f.segments) |fun| note(fun, 256, 0, 1);
    note(f.topk_keys, 1024, 0, 1);
    inline for (.{ "swiglu", "add", "add_bf16", "kv_dense", "kv_gather" }) |name| note(@field(f, name), 256, 0, 1);
    if (f.kv_pack) |fun| note(fun, 256, 0, 1);
    if (f.kv_dedup) |row| for (row) |fun| {
        note(fun, 256, 0, 1);
    };
    if (f.kv_hist) |row| for (row) |fun| {
        note(fun, 256, 0, 1);
    };
    for (f.glue) |fun| note(fun, 256, 0, 1);
    for (f.pfglue, 0..) |fun, i| note(fun, if (i == 0) 1024 else 256, 0, 1);
    if (f.ds_cands) |fun| note(fun, 1024, 0, 1);
    inline for (.{ "pick_pack", "ds_stage", "ds_stage_rows", "vs_choose" }) |name| if (@field(f, name)) |fun| {
        note(fun, 256, 0, 1);
    };
    if (f.pick_merge) |fun| note(fun, 1024, 0, 1);
    if (f.ds_accept_rows) |fun| note(fun, 32, 0, 1);
}
fn gm(f: dk.exl3.Gm) void {
    note(f.f, f.threads, f.smem, 1);
    const e = &catalog.entries[@intFromPtr(f.f.handle.?) - 1];
    e.optional_shared = true;
    e.available = f.per_sm > 0;
}
pub fn experts(f: dk.exl3.Functions) void {
    for (f.ld) |probes| for (probes) |cfg| for (cfg) |fun| if (fun) |value| {
        note(value, 128, 0, 1);
    };
    for (f.pf) |cfg| for (cfg) |fun| note(fun, 128, 0, 1);
    for (f.epi) |kind| for (kind) |cfg| for (cfg) |fun| note(fun, 128, 0, 1);
    for (f.grouped) |cfg| for (cfg) |fun| note(fun, 128, 0, 1);
    for (f.dequant) |fun| note(fun, 32, 0, 1);
    note(f.group, 1024, 49152, 1);
    note(f.group_rot, 1024, 0, 1);
    for (f.rot_in) |fun| note(fun, 32, 0, 1);
    note(f.gateup_epilogue, 32, 0, 1);
    note(f.down_epilogue, 32, 0, 1);
    note(f.combine, 256, 0, 1);
    for (f.down_combine) |fun| note(fun, 1024, 0, 1); // Ops permits up to 32 routed slots.
    for (f.gm_dequant) |fun| note(fun, 32, 0, 1);
    for (f.gm_rot) |row| for (row) |fun| note(fun, 256, 0, 1);
    note(f.gm_plan, 512, 0, 1);
    for (f.gm_gu) |row| for (row) |cfg| for (cfg) |fun| gm(fun);
    for (f.gm_dn) |row| for (row) |fun| gm(fun);
    for (f.gm2_gu) |row| for (row) |fun| gm(fun);
    gm(f.gm2_dn);
    gm(f.gm2_gu2);
    for (f.gm3) |fun| gm(fun);
}
pub fn dense(f: dk.dense.Functions) void {
    note(f.rot_in, 128, 0, 1);
    note(f.seg_rot_in, 128, 0, 1);
    for (f.linear) |row| for (row, 0..) |fun, i| {
        const wk: u32 = if (i == 0) 2 else if (i == 1) 4 else 8;
        note(fun, wk * 32, wk * 4096, 1);
    };
    for (f.unpack) |fun| note(fun, 32, 0, 1);
    for (f.seg) |row| for (row, 0..) |fun, i| {
        const wk = dk.dense.seg_cfgs[i][0];
        note(fun, wk * 32, wk * 4096, 1);
    };
    for (f.lanes) |row| for (row, 0..) |fun, i| {
        const wk = dk.dense.lanes_cfgs[i][0];
        note(fun, wk * 32, wk * 4096, 1);
    };
    for (f.lanes_unpack) |fun| note(fun, 32, 0, 1);
    note(f.to_lanes, 256, 0, 1);
    note(f.to_strips, 256, 0, 1);
}
pub fn lin2(f: dk.dense.Lin2) void {
    for (f.lanes) |row| for (row, 0..) |fun, i| {
        const wk: u32 = if (i == 0) 4 else 8;
        note(fun, wk * 32, wk * 4096, 1);
    };
    for (f.linear) |row| for (row, 0..) |fun, i| {
        const wk: u32 = if (i == 0) 2 else if (i == 1) 4 else 8;
        note(fun, wk * 32, wk * 4096, 1);
    };
    if (f.mode.cluster) for (f.lanesc) |row| for (row, 0..) |fun, i| {
        const cfg = dk.dense.lanesc_cfgs[i];
        note(fun, cfg[0] * 32, cfg[0] * 4096 + 16 * cfg[2] * 128 * 4, 8);
    };
}
pub fn sampling(f: dk.sampling.Functions) void {
    note(f.stats_fn, 1024, 0, 1);
    note(f.pack_fn, 256, 0, 1);
    if (f.split) |split| {
        note(split.max, 256, 0, 1);
        note(split.sum, 256, 0, 1);
        note(split.fin, 32, 0, 1);
    }
}
pub fn base(image: []const u8, symbol: []const u8) void {
    const std = @import("std");
    for (catalog.entries[0..catalog.count], 0..) |e, i| {
        if (!std.mem.eql(u8, e.image, image) or !std.mem.eql(u8, e.name[0..e.len], symbol)) continue;
        const fun: cuda.Function = .{ .d = undefined, .handle = @ptrFromInt(i + 1) };
        if (std.mem.eql(u8, image, "exl3_linear")) {
            if (std.mem.eql(u8, symbol, cuda.exl3.rot_in_symbol)) note(fun, 128, 0, 1);
            inline for (cuda.exl3.variants) |v| if (std.mem.eql(u8, symbol, cuda.exl3.symbol(v.halves, v.codebook, v.warps))) {
                note(fun, 32 * v.warps, 4096 * v.warps, 1);
            };
        } else if (std.mem.eql(u8, image, "exl3_candidate")) {
            const threads: u32 = if (std.mem.startsWith(u8, symbol, "tf_exl3_rotate") or std.mem.eql(u8, symbol, "tf_exl3_unpack")) 32 else 256;
            note(fun, threads, 0, 1);
        } else if (std.mem.eql(u8, image, "exl3_experts")) {
            const threads: u32 = if (std.mem.eql(u8, symbol, "tf_exl3_group") or std.mem.eql(u8, symbol, "tf_exl3_down_combine")) 1024 else if (std.mem.startsWith(u8, symbol, "tf_exl3_grouped_")) 128 else 32;
            note(fun, threads, if (std.mem.eql(u8, symbol, "tf_exl3_group")) 49152 else 0, 1);
        }
    }
}
