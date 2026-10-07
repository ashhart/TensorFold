//! One fused decode row: hyper-connection, linear or full attention, experts, and the n-gram step.

const std = @import("std");
const cuda = @import("cuda");
const qmm = @import("cuda_qmm.zig");
const gdn = @import("cuda_gdn.zig");
const ple = @import("cuda_ple.zig");
const wt = @import("cuda_prompt_wt.zig");

const dims = wt.dims;
const streams = wt.streams;
const wide = wt.wide;
const low = wt.low;
const eps = wt.eps;
const experts_n = wt.experts_n;
const slots = wt.slots;
const moe_w = wt.moe_w;
const block_bytes = wt.block_bytes;
const tile = wt.tile;
const q_heads = wt.q_heads;
const kv_heads = wt.kv_heads;
const head_dim = wt.head_dim;
const index_heads = wt.index_heads;
const index_dim = wt.index_dim;
const half = wt.half;
const proj_n = wt.proj_n;
const out_k = wt.out_k;
const nch = wt.nch;
const ple_tail_n = wt.ple_tail_n;
const state_n = wt.state_n;
const hc_sk = wt.hc_sk;
const out_sk = wt.out_sk;
const inj_stride = wt.inj_stride;
const wts_stride = wt.wts_stride;
const vocab = wt.vocab;
const cint = wt.cint;
const toBf16 = wt.toBf16;
const pack_symbol = wt.pack_symbol;
const plan_symbol = wt.plan_symbol;
const up_symbol = wt.up_symbol;
const down_symbol = wt.down_symbol;
const Face = wt.Face;
const Q4 = wt.Q4;
const HcW = wt.HcW;

pub const Scratch = struct {
    h: cuda.DeviceBuffer,
    y: cuda.DeviceBuffer,
    wts: cuda.DeviceBuffer,
    inj: cuda.DeviceBuffer,
    pss: cuda.DeviceBuffer,
    norm: cuda.DeviceBuffer,
    dn: cuda.DeviceBuffer,
    hc_part: cuda.DeviceBuffer,
    act: cuda.DeviceBuffer,
    xs_act: cuda.DeviceBuffer,
    mixed: cuda.DeviceBuffer,
    xsm: cuda.DeviceBuffer,
    proj: cuda.DeviceBuffer,
    part: cuda.DeviceBuffer,
    gout: cuda.DeviceBuffer,
    gxs: cuda.DeviceBuffer,
    st: cuda.DeviceBuffer,
    ple_emb: cuda.DeviceBuffer,
    ple_xs: cuda.DeviceBuffer,
    ple_keys: cuda.DeviceBuffer,
    ple_vals: cuda.DeviceBuffer,
    ple_gated: cuda.DeviceBuffer,
    ple_pss: cuda.DeviceBuffer,
    ple_nrow: cuda.DeviceBuffer,
    ple_part: cuda.DeviceBuffer,
    pa: cuda.DeviceBuffer,
    q: cuda.DeviceBuffer,
    iq: cuda.DeviceBuffer,
    po: cuda.DeviceBuffer,
    pm: cuda.DeviceBuffer,
    pl: cuda.DeviceBuffer,
    ao: cuda.DeviceBuffer,
    gated: cuda.DeviceBuffer,
    gxs_attn: cuda.DeviceBuffer,
    dummy: cuda.DeviceBuffer,
    inv: cuda.DeviceBuffer,
    pos: cuda.DeviceBuffer,
    ids: cuda.DeviceBuffer,
    logits: cuda.DeviceBuffer,
    router_l: cuda.DeviceBuffer,
    picks: cuda.DeviceBuffer,
    moe_members: cuda.DeviceBuffer,
    moe_items: cuda.DeviceBuffer,
    moe_counts: cuda.DeviceBuffer,
    moe_act: cuda.DeviceBuffer,

    pub fn alloc(driver: *cuda.Driver, rows: usize) !Scratch {
        var s: Scratch = undefined;
        s.h = try cuda.DeviceBuffer.alloc(driver, rows * wide * 2);
        s.y = try cuda.DeviceBuffer.alloc(driver, rows * slots * dims * 4);
        s.wts = try cuda.DeviceBuffer.alloc(driver, rows * wts_stride);
        s.inj = try cuda.DeviceBuffer.alloc(driver, rows * inj_stride);
        s.pss = try cuda.DeviceBuffer.alloc(driver, (dims / 256) * streams * 4);
        s.norm = try cuda.DeviceBuffer.alloc(driver, wide * 2);
        s.dn = try cuda.DeviceBuffer.alloc(driver, (low + 4) * 2);
        s.hc_part = try cuda.DeviceBuffer.alloc(driver, hc_sk * (low + 4) * 4);
        s.act = try cuda.DeviceBuffer.alloc(driver, low * 2);
        s.xs_act = try cuda.DeviceBuffer.alloc(driver, (low / 32) * 4);
        s.mixed = try cuda.DeviceBuffer.alloc(driver, dims * 2);
        s.xsm = try cuda.DeviceBuffer.alloc(driver, (dims / 32) * 4);
        s.proj = try cuda.DeviceBuffer.alloc(driver, gdn.proj_width * 2);
        s.part = try cuda.DeviceBuffer.alloc(driver, out_sk * dims * 4);
        s.gout = try cuda.DeviceBuffer.alloc(driver, gdn.value_dim * 2);
        s.gxs = try cuda.DeviceBuffer.alloc(driver, (gdn.value_dim / 32) * 4);
        s.st = try cuda.DeviceBuffer.alloc(driver, state_n * 4);
        s.ple_emb = try cuda.DeviceBuffer.alloc(driver, dims * 2);
        s.ple_xs = try cuda.DeviceBuffer.alloc(driver, (dims / 32) * 4);
        s.ple_keys = try cuda.DeviceBuffer.alloc(driver, wide * 2);
        s.ple_vals = try cuda.DeviceBuffer.alloc(driver, dims * 2);
        s.ple_gated = try cuda.DeviceBuffer.alloc(driver, wide * 2);
        s.ple_pss = try cuda.DeviceBuffer.alloc(driver, streams * 4);
        s.ple_nrow = try cuda.DeviceBuffer.alloc(driver, wide * 2);
        s.ple_part = try cuda.DeviceBuffer.alloc(driver, 4 * dims * 4);
        s.pa = try cuda.DeviceBuffer.alloc(driver, proj_n * 2);
        s.q = try cuda.DeviceBuffer.alloc(driver, q_heads * head_dim * 2);
        s.iq = try cuda.DeviceBuffer.alloc(driver, index_heads * index_dim * 2);
        s.po = try cuda.DeviceBuffer.alloc(driver, nch * q_heads * head_dim * 4);
        s.pm = try cuda.DeviceBuffer.alloc(driver, nch * q_heads * 4);
        s.pl = try cuda.DeviceBuffer.alloc(driver, nch * q_heads * 4);
        s.ao = try cuda.DeviceBuffer.alloc(driver, out_k * 2);
        s.gated = try cuda.DeviceBuffer.alloc(driver, out_k * 2);
        s.gxs_attn = try cuda.DeviceBuffer.alloc(driver, (out_k / 32) * 4);
        s.dummy = try cuda.DeviceBuffer.alloc(driver, 4096);
        var inv: [half]f32 = undefined;
        for (&inv, 0..) |*o, i| o.* = @floatCast(std.math.pow(f64, 10_000_000.0, -@as(f64, @floatFromInt(i)) / @as(f64, half)));
        s.inv = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(&inv));
        s.pos = try cuda.DeviceBuffer.alloc(driver, 4);
        s.ids = try cuda.DeviceBuffer.alloc(driver, 4);
        s.logits = try cuda.DeviceBuffer.alloc(driver, vocab * 2);
        s.router_l = try cuda.DeviceBuffer.alloc(driver, (experts_n + 1) * 4);
        s.picks = try cuda.DeviceBuffer.alloc(driver, slots * 4);
        s.moe_members = try cuda.DeviceBuffer.alloc(driver, slots * 4);
        s.moe_items = try cuda.DeviceBuffer.alloc(driver, slots * 3 * 4);
        s.moe_counts = try cuda.DeviceBuffer.alloc(driver, 8);
        s.moe_act = try cuda.DeviceBuffer.alloc(driver, slots * moe_w * 2);
        return s;
    }

    pub fn free(self: *Scratch) void {
        self.h.free();
        self.y.free();
        self.wts.free();
        self.inj.free();
        self.pss.free();
        self.norm.free();
        self.dn.free();
        self.hc_part.free();
        self.act.free();
        self.xs_act.free();
        self.mixed.free();
        self.xsm.free();
        self.proj.free();
        self.part.free();
        self.gout.free();
        self.gxs.free();
        self.st.free();
        self.ple_emb.free();
        self.ple_xs.free();
        self.ple_keys.free();
        self.ple_vals.free();
        self.ple_gated.free();
        self.ple_pss.free();
        self.ple_nrow.free();
        self.ple_part.free();
        self.pa.free();
        self.q.free();
        self.iq.free();
        self.po.free();
        self.pm.free();
        self.pl.free();
        self.ao.free();
        self.gated.free();
        self.gxs_attn.free();
        self.dummy.free();
        self.inv.free();
        self.pos.free();
        self.ids.free();
        self.logits.free();
        self.router_l.free();
        self.picks.free();
        self.moe_members.free();
        self.moe_items.free();
        self.moe_counts.free();
        self.moe_act.free();
    }
};

pub fn writeback(tri: anytype, s: *Scratch, h: u64, mode: usize, inj: u64, y: u64, wts: u64) !void {
    const dummy = s.dummy.ptr;
    if (mode == 0) {
        try tri.hcWriteback(h, h, s.pss.ptr, dummy, "*bf16", dummy, dummy, "*bf16", dummy, "*bf16", dims, 1, dims, streams, 0, 1, 1, 1);
    } else if (mode == 2) {
        try tri.hcWriteback(h, h, s.pss.ptr, dummy, "*bf16", inj, y, "*fp32", wts, "*fp32", dims, 1, dims, streams, 2, 10, 11, 1);
    } else if (mode == 4) {
        try tri.hcWriteback(h, h, s.pss.ptr, s.part.ptr, "*fp32", inj, dummy, "*bf16", dummy, "*bf16", dims, 1, dims, streams, 4, 1, 1, out_sk);
    } else return error.UnexpectedTensor;
}

pub fn readout(tri: anytype, s: *Scratch, hc_w: HcW, h: u64, inj: u64) !void {
    try tri.hcDown(h, s.pss.ptr, hc_w.scale.ptr, s.norm.ptr, hc_w.down.w.ptr, hc_w.down.s.ptr, hc_w.down.b.ptr, s.dn.ptr, s.hc_part.ptr, eps, 1, hc_w.n_down, wide, dims, dims / 256, streams, hc_sk);
    const inj_ptr = if (hc_w.has_inj == 1) inj else s.act.ptr;
    try tri.hcReduceAct(s.hc_part.ptr, s.act.ptr, s.xs_act.ptr, inj_ptr, hc_sk, 1, streams, low, hc_w.n_down, hc_w.has_inj);
    try tri.hcUpmix(s.act.ptr, s.xs_act.ptr, hc_w.up.w.ptr, hc_w.up.s.ptr, hc_w.up.b.ptr, s.norm.ptr, s.mixed.ptr, s.xsm.ptr, 1, wide, low, dims, streams);
}

fn shiftTail(gpa: std.mem.Allocator, stream: *cuda.Stream, window: cuda.DeviceBuffer, taps_n: usize, row_bytes: usize, fresh: []const u8) !void {
    const bytes = taps_n * row_bytes;
    const raw = try gpa.alloc(u8, bytes);
    defer gpa.free(raw);
    try stream.synchronize();
    try window.download(0, raw);
    const next = try gpa.alloc(u8, bytes);
    defer gpa.free(next);
    const keep = taps_n - 1;
    @memcpy(next[0 .. keep * row_bytes], raw[row_bytes..]);
    @memcpy(next[keep * row_bytes ..], fresh);
    try window.upload(0, next);
}

pub fn gdnStep(gpa: std.mem.Allocator, driver: *cuda.Driver, stream: *cuda.Stream, s: *Scratch, proj_q: Q4, out_q: Q4, conv: cuda.DeviceBuffer, cw: u64, a_log: u64, dt: u64, norm: u64, state: *cuda.DeviceBuffer) !void {
    try qmm.matmul(driver, stream.*, s.mixed.ptr, s.xsm.ptr, proj_q.w.ptr, proj_q.s.ptr, proj_q.b.ptr, s.proj.ptr, 1, proj_q.n, proj_q.k);
    try gdn.launch(driver, stream.*, s.proj.ptr, conv.ptr, cw, state.ptr, a_log, dt, norm, eps, s.gout.ptr, s.gxs.ptr, s.st.ptr);
    try state.copyFrom(0, s.st.ptr, state_n * 4, stream.handle);
    const row = try gpa.alloc(u8, gdn.conv_dim * 2);
    defer gpa.free(row);
    try stream.synchronize();
    try s.proj.download(0, row);
    try shiftTail(gpa, stream, conv, gdn.taps - 1, gdn.conv_dim * 2, row);
    try qmm.partials(driver, stream.*, s.gout.ptr, s.gxs.ptr, out_q.w.ptr, out_q.s.ptr, out_q.b.ptr, s.dummy.ptr, s.part.ptr, 1, dims, gdn.value_dim, out_sk);
}

pub fn attnStep(comptime Tri: type, driver: *cuda.Driver, stream: *cuda.Stream, tri: Tri, s: *Scratch, proj_q: Q4, out_q: Q4, q_scale: u64, k_scale: u64, i_scale: u64, kc: u64, vc: u64, ikc: u64, pos: i32) !void {
    try qmm.matmul(driver, stream.*, s.mixed.ptr, s.xsm.ptr, proj_q.w.ptr, proj_q.s.ptr, proj_q.b.ptr, s.pa.ptr, 1, proj_n, dims);
    try s.pos.upload(0, std.mem.asBytes(&pos));
    try tri.attnPrep(s.pa.ptr, s.pos.ptr, q_scale, k_scale, i_scale, s.inv.ptr, s.q.ptr, kc, vc, s.dummy.ptr, s.dummy.ptr, s.iq.ptr, ikc, s.dummy.ptr, s.dummy.ptr, eps, 1, 0);
    try tri.attnChunks(s.q.ptr, kc, vc, s.dummy.ptr, s.dummy.ptr, s.pos.ptr, s.po.ptr, s.pm.ptr, s.pl.ptr, s.dummy.ptr, s.dummy.ptr, s.dummy.ptr, 1, kv_heads, 1);
    try tri.attnMerge(s.po.ptr, s.pm.ptr, s.pl.ptr, s.pos.ptr, s.ao.ptr, s.dummy.ptr, s.dummy.ptr, 1, kv_heads);
    try tri.attnGate(s.ao.ptr, s.pa.ptr, s.gated.ptr, s.gxs_attn.ptr, 1, proj_n, q_heads, head_dim);
    try qmm.partials(driver, stream.*, s.gated.ptr, s.gxs_attn.ptr, out_q.w.ptr, out_q.s.ptr, out_q.b.ptr, s.dummy.ptr, s.part.ptr, 1, dims, out_k, out_sk);
}

fn launchExpert(f: cuda.Function, stream: cuda.Stream, x: u64, x_stride: usize, slot_n: usize, w: u64, kg: usize, nb: usize, items: u64, counts: u64, members: u64, out: u64, n: usize, units: usize) !void {
    var a: cuda.Args = .{};
    a.add(x);
    a.add(cint(x_stride));
    a.add(cint(slot_n));
    a.add(w);
    a.add(cint(kg));
    a.add(cint(nb));
    a.add(items);
    a.add(counts);
    a.add(members);
    a.add(out);
    a.add(cint(n));
    a.add(@as(f32, 0));
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((units + 3) / 4) }, .block = .{ .x = 128 } }, stream, &a);
}

pub fn moeStep(comptime Tri: type, tri: Tri, s: *Scratch, stream: *cuda.Stream, plan_fn: cuda.Function, up_fn: cuda.Function, down_fn: cuda.Function, router_w: u64, up: u64, down: u64, y: u64, wts: u64) !void {
    try tri.router(s.mixed.ptr, router_w, s.router_l.ptr, dims);
    try tri.topkRows(s.router_l.ptr, s.picks.ptr, wts);
    var plan_a: cuda.Args = .{};
    plan_a.add(s.picks.ptr);
    plan_a.add(cint(slots));
    plan_a.add(cint(experts_n + 1));
    plan_a.add(cint(tile));
    plan_a.add(s.moe_members.ptr);
    plan_a.add(s.moe_items.ptr);
    plan_a.add(s.moe_counts.ptr);
    try cuda.launch.launch(plan_fn, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, stream.*, &plan_a);
    try launchExpert(up_fn, stream.*, s.mixed.ptr, dims, slots, up, dims / 32, moe_w / 32, s.moe_items.ptr, s.moe_counts.ptr, s.moe_members.ptr, s.moe_act.ptr, moe_w, slots * (moe_w / 32));
    try launchExpert(down_fn, stream.*, s.moe_act.ptr, moe_w, 0, down, moe_w / 32, dims / 32, s.moe_items.ptr, s.moe_counts.ptr, s.moe_members.ptr, y, dims, slots * (dims / 32));
}

fn reduceBf16(gpa: std.mem.Allocator, stream: *cuda.Stream, part: cuda.DeviceBuffer, sk: usize, n: usize, dst: cuda.DeviceBuffer) !void {
    const raw = try gpa.alloc(u8, sk * n * 4);
    defer gpa.free(raw);
    try stream.synchronize();
    try part.download(0, raw);
    const row = try gpa.alloc(u16, n);
    defer gpa.free(row);
    for (0..n) |i| {
        var acc: f32 = @bitCast(std.mem.readInt(u32, raw[4 * i ..][0..4], .little));
        for (1..sk) |s| acc += @as(f32, @bitCast(std.mem.readInt(u32, raw[4 * (s * n + i) ..][0..4], .little)));
        row[i] = toBf16(acc);
    }
    try dst.upload(0, std.mem.sliceAsBytes(row));
}

pub fn pleStep(comptime Tri: type, gpa: std.mem.Allocator, io: std.Io, driver: *cuda.Driver, stream: *cuda.Stream, tri: Tri, s: *Scratch, model_dir: []const u8, history: *[2]i64, token: u32, h: u64, key_q: Q4, val_q: Q4, nk: u64, nq: u64, nc: u64, cw: u64, tail: *cuda.DeviceBuffer) !void {
    const emb = try gpa.alloc(u16, dims);
    defer gpa.free(emb);
    const xs = try gpa.alloc(f32, dims / 32);
    defer gpa.free(xs);
    try ple.embedding(gpa, io, model_dir, history, token, emb, xs);
    try s.ple_emb.upload(0, std.mem.sliceAsBytes(emb));
    try s.ple_xs.upload(0, std.mem.sliceAsBytes(xs));
    try qmm.matmul(driver, stream.*, s.ple_emb.ptr, s.ple_xs.ptr, key_q.w.ptr, key_q.s.ptr, key_q.b.ptr, s.ple_keys.ptr, 1, wide, dims);
    try qmm.partials(driver, stream.*, s.ple_emb.ptr, s.ple_xs.ptr, val_q.w.ptr, val_q.s.ptr, val_q.b.ptr, s.ple_vals.ptr, s.ple_part.ptr, 1, dims, dims, 4);
    try reduceBf16(gpa, stream, s.ple_part, 4, dims, s.ple_vals);
    try tri.pleGate(s.ple_keys.ptr, s.ple_vals.ptr, h, nk, nq, s.ple_gated.ptr, s.ple_pss.ptr, eps, 1, dims, streams);
    try tri.pleConv(s.ple_gated.ptr, s.ple_pss.ptr, nc, tail.ptr, cw, h, h, s.ple_nrow.ptr, eps, 1, dims, streams, 4, 3);
    const nrow = try gpa.alloc(u8, wide * 2);
    defer gpa.free(nrow);
    try stream.synchronize();
    try s.ple_nrow.download(0, nrow);
    try shiftTail(gpa, stream, tail.*, ple_tail_n, wide * 2, nrow);
    history[0] = history[1];
    history[1] = token;
}
