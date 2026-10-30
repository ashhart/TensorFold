//! flashnext-exl3: one decoder layer through the family pass against the float64 layer (gate 5).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const check = @import("../../check.zig");
const fwd = @import("exl3_fwd_cpu.zig");
const hc = @import("exl3_hc.zig");
const moe = @import("exl3_moe.zig");
const router_cpu = @import("exl3_router.zig");
const route_cpu = @import("exl3_route.zig");
const scan = @import("../pack/exl3_scan.zig");
const place = @import("../pack/exl3_place.zig");
const gdn_cpu = @import("exl3_gdn_cpu.zig");
const attn_cpu = @import("exl3_attn_cpu.zig");

const X = flash.exl3;
const wt = X.prompt_wt;
const pl = X.prompt_load;
const batch = X.batch;
const Gpu = check.Gpu;
const st = core.safetensors;

const dims = wt.dims;
const wide = wt.wide;
const streams = wt.streams;
const slots = wt.slots;
const topk = slots - 1;
const experts_n = wt.experts_n;
const shared = @import("flashnext_exl3_rig.zig");
const typed = shared.typed;
const valueOf = shared.valueOf;
const f64Of = shared.f64Of;
const f32Of = shared.f32Of;
const f16Of = shared.f16Of;
const qmat = shared.qmat;
const bf16Bits = shared.bf16Bits;
const bfValue = shared.bfValue;
const Err = shared.Err;
const errOf = shared.errOf;
const download = shared.download;
const uploadTo = shared.uploadTo;
const RawPack = shared.RawPack;
const Pass = shared.Pass;
const Rig = shared.Rig;
const bf16Rows = shared.bf16Rows;
const default_ids = shared.default_ids;
const projections = @import("flashnext_exl3_proj_tests.zig").projections;

// -- the float64 layer --------------------------------------------------------------------------------------------

const HcHost = struct {
    norm: []f32,
    down: []f32,
    up: []f32,
    inject: ?[]f32,

    fn load(a: std.mem.Allocator, raw: *const RawPack, comptime base: []const u8, args: anytype, with_inject: bool) !HcHost {
        return .{
            .norm = try f32Of(a, try raw.get(base ++ ".hc_norm.weight", args)),
            .down = try f32Of(a, try raw.get(base ++ ".input_mix_weight_down.weight", args)),
            .up = try f32Of(a, try raw.get(base ++ ".input_mix_weight_up.weight", args)),
            .inject = if (with_inject) try f32Of(a, try raw.get(base ++ ".block_inject_weight.weight", args)) else null,
        };
    }

    fn weights(h: HcHost) hc.Weights {
        return .{ .norm = h.norm, .down = h.down, .up = h.up, .inject = h.inject };
    }
};

const ExpertCtx = struct {
    gpa: std.mem.Allocator,
    raw: *const RawPack,
    layer: usize,

    fn load(ctx: *anyopaque, e: usize, out: *[3]fwd.QMat) anyerror!void {
        const self: *ExpertCtx = @ptrCast(@alignCast(ctx));
        const projs = [3][]const u8{ "gate_proj", "up_proj", "down_proj" };
        var made: usize = 0;
        errdefer for (0..made) |i| out[i].deinit(self.gpa);
        while (made < 3) : (made += 1) {
            var nb: [160]u8 = undefined;
            out[made] = try qmat(self.gpa, self.raw, try std.fmt.bufPrint(&nb, "model.language_model.layers.{d}.mlp.experts.{d}.{s}", .{ self.layer, e, projs[made] }), 1);
        }
    }
};

/// The oracle's layer (exl3_fwd_cpu.LayerW) from the pack's stored tensors, decoded by the Zig f64 decoder.
const Oracle = struct {
    attn_hc: HcHost,
    mlp_hc: HcHost,
    gdn: fwd.GdnW = undefined,
    attn: fwd.AttnW = undefined,
    moe: fwd.MoeW,
    ctx: ExpertCtx,
    state: fwd.LayerState,
    /// The mixer's state for the staged comparison (fed the device's own inputs), apart from the chained one.
    staged: fwd.LayerState,
    linear: bool,

    fn create(a: std.mem.Allocator, raw: *const RawPack, l: usize, offset: f32, cap: usize) !*Oracle {
        const o = try a.create(Oracle);
        o.linear = wt.linearLayer(l);
        o.attn_hc = try HcHost.load(a, raw, "model.language_model.layers.{d}.attn_hyper_connection", .{l}, true);
        o.mlp_hc = try HcHost.load(a, raw, "model.language_model.layers.{d}.mlp_hyper_connection", .{l}, true);
        o.ctx = .{ .gpa = a, .raw = raw, .layer = l };
        var nb: [4][160]u8 = undefined;
        o.moe = .{
            .gate16 = try f16Of(a, try raw.get("model.language_model.layers.{d}.mlp.gate.weight", .{l})),
            .sgate16 = try f16Of(a, try raw.get("model.language_model.layers.{d}.mlp.shared_expert_gate.weight", .{l})),
            .shared = .{
                try qmat(a, raw, try std.fmt.bufPrint(&nb[0], "model.language_model.layers.{d}.mlp.shared_expert.gate_proj", .{l}), 48),
                try qmat(a, raw, try std.fmt.bufPrint(&nb[1], "model.language_model.layers.{d}.mlp.shared_expert.up_proj", .{l}), 48),
                try qmat(a, raw, try std.fmt.bufPrint(&nb[2], "model.language_model.layers.{d}.mlp.shared_expert.down_proj", .{l}), 48),
            },
            .source = .{ .ctx = &o.ctx, .load = ExpertCtx.load },
        };
        if (o.linear) {
            const stem = "model.language_model.layers.{d}.linear_attn";
            o.gdn = .{
                .qkv = try qmat(a, raw, try std.fmt.bufPrint(&nb[0], stem ++ ".in_proj_qkv", .{l}), 48),
                .z = try qmat(a, raw, try std.fmt.bufPrint(&nb[1], stem ++ ".in_proj_z", .{l}), 48),
                .out = try qmat(a, raw, try std.fmt.bufPrint(&nb[2], stem ++ ".out_proj", .{l}), 48),
                .a_w = try f64Of(a, try raw.get(stem ++ ".in_proj_a.weight", .{l})),
                .b_w = try f64Of(a, try raw.get(stem ++ ".in_proj_b.weight", .{l})),
                .params = .{
                    .conv_w = try f64Of(a, try raw.get(stem ++ ".conv1d.weight", .{l})),
                    .a_log = try f64Of(a, try raw.get(stem ++ ".A_log", .{l})),
                    .dt_bias = try f64Of(a, try raw.get(stem ++ ".dt_bias", .{l})),
                    .norm_w = try f64Of(a, try raw.get(stem ++ ".norm.weight", .{l})),
                },
            };
            o.state = .{ .gdn = try gdn_cpu.State.init(a, gdn_cpu.qwen) };
            o.staged = .{ .gdn = try gdn_cpu.State.init(a, gdn_cpu.qwen) };
        } else {
            const stem = "model.language_model.layers.{d}.self_attn";
            var sc: [4][]f32 = undefined;
            for ([_][]const u8{ "q_norm", "k_norm", "indexer.q_layernorm", "indexer.k_layernorm" }, 0..) |nm, i| {
                var buf: [160]u8 = undefined;
                const name = try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.self_attn.{s}.weight", .{ l, nm });
                sc[i] = try f32Of(a, try raw.get("{s}", .{name}));
                for (sc[i]) |*v| v.* += offset;
            }
            o.attn = .{
                .q = try qmat(a, raw, try std.fmt.bufPrint(&nb[0], stem ++ ".q_proj", .{l}), 48),
                .k = try qmat(a, raw, try std.fmt.bufPrint(&nb[1], stem ++ ".k_proj", .{l}), 48),
                .v = try qmat(a, raw, try std.fmt.bufPrint(&nb[2], stem ++ ".v_proj", .{l}), 48),
                .o = try qmat(a, raw, try std.fmt.bufPrint(&nb[3], stem ++ ".o_proj", .{l}), 48),
                .idx = try qmat(a, raw, try std.fmt.bufPrint(&nb[0], stem ++ ".indexer.index_qk_proj", .{l}), 48),
                .norms = .{ .q = sc[0], .k = sc[1], .iq = sc[2], .ik = sc[3], .inv = attn_cpu.invFreq() },
            };
            o.state = .{ .attn = try attn_cpu.Cache.init(a, cap) };
            o.staged = .{ .attn = try attn_cpu.Cache.init(a, cap) };
        }
        return o;
    }

    fn weights(o: *Oracle) fwd.LayerW {
        return .{ .attn_hc = o.attn_hc.weights(), .mlp_hc = o.mlp_hc.weights(), .mixer = if (o.linear) .{ .gdn = &o.gdn } else .{ .attn = &o.attn }, .moe = &o.moe };
    }
};

// -- one decoder layer through the family's pass (gate 5) ----------------------------------------------------------

const Stage = struct { mixer: f64 = 0, streams: f64 = 0, moe: f64 = 0, out: f64 = 0, picks_same: usize = 0, rows: usize = 0 };

/// What the device fed each block: the read-outs' mixed rows and inject gates (bf16), attention's then the MoE's.
pub const Taps = struct { attn_mixed: []u16, attn_inj: []u16, mlp_mixed: []u16, mlp_inj: []u16 };

/// The pass's per-layer sequence (cuda_batch.zig Pass.chunk's loop body), stepped with read-backs between blocks.
pub fn deviceLayer(gpu: Gpu, p: *Pass, lw: *const pl.Layer, l: usize, ids: []const u32, pos0: usize, h_in: []const u16, tr: fwd.Trace, h_out: []u16, taps: Taps) !void {
    const s = p.s;
    const n = ids.len;
    p.rows = n;
    p.pos0 = pos0;
    p.pending = null;
    try p.stage(ids);
    try uploadTo(gpu, p.stream, s.h, std.mem.sliceAsBytes(h_in));
    try p.writeback(0, n);
    try p.readout(lw.attn_hc, 0, n);
    try tap(gpu, p, s.mixed, taps.attn_mixed[0 .. n * dims]);
    try tap(gpu, p, s.inj, taps.attn_inj[0 .. n * streams]);
    if (lw.linear) try p.gdnBlock(lw, l, wt.linearIndex(l), .{}) else try p.attnBlock(lw, wt.attnIndex(l));
    const branch = try download(f32, gpu, p.stream, s.part, n * dims);
    defer gpu.gpa.free(branch);
    for (tr.attn_branch[0 .. n * dims], branch) |*d, v| d.* = v;
    try p.tri.hcWriteback(s.h, s.h, s.pss, s.part, "*fp32", s.inj, s.dummy, "*bf16", s.dummy, "*bf16", n * dims, n, dims, streams, 4, 1, 1, wt.out_sk);
    const mid = try download(u16, gpu, p.stream, s.h, n * wide);
    defer gpu.gpa.free(mid);
    for (tr.h_mid[0 .. n * wide], mid) |*d, v| d.* = bfValue(v);
    try p.readout(lw.mlp_hc, 0, n);
    try tap(gpu, p, s.mixed, taps.mlp_mixed[0 .. n * dims]);
    try tap(gpu, p, s.inj, taps.mlp_inj[0 .. n * streams]);
    try p.moeBlock(lw);
    const picks = try download(i32, gpu, p.stream, s.picks, n * slots);
    defer gpu.gpa.free(picks);
    const wts = try download(f32, gpu, p.stream, s.wts, n * slots);
    defer gpu.gpa.free(wts);
    const y = try download(f32, gpu, p.stream, s.y, n * slots * dims);
    defer gpu.gpa.free(y);
    for (0..n) |r| {
        for (0..topk) |k| tr.picks[r * topk + k] = @intCast(picks[r * slots + k]);
        for (0..dims) |d| {
            var acc: f64 = 0;
            for (0..slots) |k| acc += @as(f64, wts[r * slots + k]) * y[(r * slots + k) * dims + d];
            tr.mlp_branch[r * dims + d] = acc;
        }
    }
    p.pending = l;
    try p.writeback(0, n);
    const out = try download(u16, gpu, p.stream, s.h, n * wide);
    defer gpu.gpa.free(out);
    @memcpy(h_out, out);
}

fn tap(gpu: Gpu, p: *Pass, ptr: u64, out: []u16) !void {
    const got = try download(u16, gpu, p.stream, ptr, out.len);
    defer gpu.gpa.free(got);
    @memcpy(out, got);
}

fn toF64(a: std.mem.Allocator, bits: []const u16) ![]f64 {
    const out = try a.alloc(f64, bits.len);
    for (out, bits) |*o, b| o.* = bfValue(b);
    return out;
}

/// Each block against the float64 oracle fed the DEVICE's own input to that block: its error alone.
const Staged = struct { attn_mixed: f64 = 0, attn_inj: f64 = 0, mixer: f64 = 0, mlp_mixed: f64 = 0, mlp_inj: f64 = 0, moe: f64 = 0, writeback: f64 = 0 };

pub fn layerCheck(gpu: Gpu, r: *Rig, raw: *const RawPack, lw: *const pl.Layer, l: usize, tokens: []const u32) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const prefill = 8;
    var t0 = check.now(gpu.io);
    const o = try Oracle.create(a, raw, l, r.store.norm_offset, r.cap);
    std.debug.print("  float64 layer {d} decoded in {d:.0} ms (offset {d})\n", .{ l, @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e6, r.store.norm_offset });
    try r.reset();
    var p = r.pass((lw)[0..1]);
    const R = prefill;
    const mk = struct {
        fn trace(al: std.mem.Allocator) !fwd.Trace {
            return .{ .attn_branch = try al.alloc(f64, R * dims), .mlp_branch = try al.alloc(f64, R * dims), .picks = try al.alloc(u16, R * topk), .h_mid = try al.alloc(f64, R * wide) };
        }
    };
    const dt = try mk.trace(a);
    const ot = try mk.trace(a);
    const table = try raw.get("model.language_model.embed_tokens.weight", .{});
    const calls = [_][2]usize{ .{ 0, prefill }, .{ 8, 1 }, .{ 9, 1 }, .{ 10, 1 }, .{ 11, 1 } };
    var worst: Stage = .{};
    var alone: Staged = .{};
    var device_ms: f64 = 0;
    const taps: Taps = .{ .attn_mixed = try a.alloc(u16, R * dims), .attn_inj = try a.alloc(u16, R * streams), .mlp_mixed = try a.alloc(u16, R * dims), .mlp_inj = try a.alloc(u16, R * streams) };
    for (calls) |c| {
        const toks = tokens[c[0] .. c[0] + c[1]];
        const n = c[1];
        // the layer's input: each token's embedding row in all four streams (bf16, as the pass's expand writes it)
        const hb = try a.alloc(u16, n * wide);
        const ho = try a.alloc(f64, n * wide);
        for (toks, 0..) |tk, t| for (0..dims) |i| {
            const bits = std.mem.readInt(u16, table.bytes[(tk * dims + i) * 2 ..][0..2], .little);
            for (0..streams) |sidx| {
                hb[t * wide + sidx * dims + i] = bits;
                ho[t * wide + sidx * dims + i] = bfValue(bits);
            }
        };
        const hg = try a.alloc(u16, n * wide);
        t0 = check.now(gpu.io);
        try deviceLayer(gpu, &p, lw, l, toks, c[0], hb, dt, hg, taps);
        device_ms += @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e6;
        // staged: every block from the device's own input (before the chained oracle updates `ho` in place)
        {
            const w = o.weights();
            const off = r.store.norm_offset;
            const mixed = try a.alloc(f64, n * dims);
            const inj = try a.alloc(f64, n * streams);
            try fwd.hcPreRows(a, w.attn_hc, off, ho, n, mixed, inj);
            const e_am = errOf(@as([]const u16, taps.attn_mixed[0 .. n * dims]), mixed);
            const e_ai = errOf(@as([]const u16, taps.attn_inj[0 .. n * streams]), inj);
            const dev_mixed = try toF64(a, taps.attn_mixed[0 .. n * dims]);
            const branch = try a.alloc(f64, n * dims);
            switch (w.mixer) {
                .gdn => |g| try fwd.gdnBranch(a, g, o.staged.gdn, dev_mixed, n, branch),
                .attn => |at| try fwd.attnBranch(a, at, &o.staged.attn, c[0], dev_mixed, n, branch),
            }
            const e_mx = errOf(dt.attn_branch[0 .. n * dims], branch);
            const dev_mid = dt.h_mid[0 .. n * wide];
            try fwd.hcPreRows(a, w.mlp_hc, off, dev_mid, n, mixed, inj);
            const e_mm = errOf(@as([]const u16, taps.mlp_mixed[0 .. n * dims]), mixed);
            const e_mi = errOf(@as([]const u16, taps.mlp_inj[0 .. n * streams]), inj);
            const dev_mlp = try toF64(a, taps.mlp_mixed[0 .. n * dims]);
            const moe_out = try a.alloc(f64, n * dims);
            const picks = try a.alloc(u16, n * topk);
            try fwd.moeBranch(a, w.moe, dev_mlp, n, moe_out, picks);
            var same_picks = true;
            for (0..n) |t| {
                var x = dt.picks[t * topk ..][0..topk].*;
                var y = picks[t * topk ..][0..topk].*;
                std.mem.sort(u16, &x, {}, std.sort.asc(u16));
                std.mem.sort(u16, &y, {}, std.sort.asc(u16));
                same_picks = same_picks and std.mem.eql(u16, &x, &y);
            }
            const e_moe = errOf(dt.mlp_branch[0 .. n * dims], moe_out);
            // the write-back alone: the device's streams after the mixer plus its MoE branch times its inject gates
            const wb = try a.alloc(f64, n * wide);
            for (0..n) |t| for (0..streams) |sidx| for (0..dims) |d| {
                const at = t * wide + sidx * dims + d;
                wb[at] = dev_mid[at] + dt.mlp_branch[t * dims + d] * bfValue(taps.mlp_inj[t * streams + sidx]);
            };
            const e_wb = errOf(@as([]const u16, hg), wb);
            std.debug.print("    staged (device input to each block): attn readout mixed rel {e:.3}, inject rel {e:.3}; mixer rel {e:.3}; mlp readout mixed rel {e:.3}, inject rel {e:.3}; MoE rel {e:.3} (picks {s}); write-back rel {e:.3}\n", .{ e_am.rel, e_ai.rel, e_mx.rel, e_mm.rel, e_mi.rel, e_moe.rel, if (same_picks) "equal" else "DIFFER", e_wb.rel });
            alone.attn_mixed = @max(alone.attn_mixed, e_am.rel);
            alone.attn_inj = @max(alone.attn_inj, e_ai.rel);
            alone.mixer = @max(alone.mixer, e_mx.rel);
            alone.mlp_mixed = @max(alone.mlp_mixed, e_mm.rel);
            alone.mlp_inj = @max(alone.mlp_inj, e_mi.rel);
            if (same_picks) alone.moe = @max(alone.moe, e_moe.rel);
            alone.writeback = @max(alone.writeback, e_wb.rel);
        }
        try fwd.layerRows(a, o.weights(), &o.state, c[0], r.store.norm_offset, ho, n, ot);
        const e_mix = errOf(dt.attn_branch[0 .. n * dims], ot.attn_branch[0 .. n * dims]);
        const e_mid = errOf(dt.h_mid[0 .. n * wide], ot.h_mid[0 .. n * wide]);
        var same: usize = 0;
        for (0..n) |t| {
            var x = dt.picks[t * topk ..][0..topk].*;
            var y = ot.picks[t * topk ..][0..topk].*;
            std.mem.sort(u16, &x, {}, std.sort.asc(u16));
            std.mem.sort(u16, &y, {}, std.sort.asc(u16));
            same += @intFromBool(std.mem.eql(u16, &x, &y));
        }
        const e_moe = errOf(dt.mlp_branch[0 .. n * dims], ot.mlp_branch[0 .. n * dims]);
        const e_out = errOf(@as([]const u16, hg), ho);
        std.debug.print("  rows {d}..{d}: mixer branch rel {e:.3} (peak {d:.3}); streams after mixer rel {e:.3}; MoE picks same set {d}/{d}, MoE branch rel {e:.3} (peak {d:.3}); layer output streams rel {e:.3} (peak {d:.2}, max_abs {e:.2})\n", .{ c[0], c[0] + n - 1, e_mix.rel, e_mix.peak, e_mid.rel, same, n, e_moe.rel, e_moe.peak, e_out.rel, e_out.peak, e_out.max_abs });
        worst.mixer = @max(worst.mixer, e_mix.rel);
        worst.streams = @max(worst.streams, e_mid.rel);
        worst.picks_same += same;
        worst.rows += n;
        if (same == n) {
            worst.moe = @max(worst.moe, e_moe.rel);
            worst.out = @max(worst.out, e_out.rel);
        }
    }
    std.debug.print("  device time for the 5 calls (read-backs between blocks included): {d:.0} ms\n", .{device_ms});
    std.debug.print("  worst staged: attn readout mixed {e:.3} inject {e:.3}; mixer {e:.3}; mlp readout mixed {e:.3} inject {e:.3}; MoE {e:.3}; write-back {e:.3}\n", .{ alone.attn_mixed, alone.attn_inj, alone.mixer, alone.mlp_mixed, alone.mlp_inj, alone.moe, alone.writeback });
    std.debug.print("  worst chained: mixer {e:.3}, streams {e:.3}, MoE {e:.3}, output {e:.3}\n", .{ worst.mixer, worst.streams, worst.moe, worst.out });
    // Each block alone, fed the device's own input: its bf16 output and the roundings inside it.
    const staged = [_]f64{ alone.attn_mixed, alone.attn_inj, alone.mixer, alone.mlp_mixed, alone.mlp_inj, alone.moe, alone.writeback };
    for (staged) |v| try check.expect(v <= 0.01, "layer {d}: a block alone differs from float64 by {d:.4} of its peak (> 0.01)", .{ l, v });
    // Chained from the embedding: the old engine's gates (mixer 1% GDN / 3% attention, streams 1%, MoE 2%) hold; its
    const mixer_tol: f64 = if (wt.linearLayer(l)) 0.01 else 0.03;
    try check.expect(worst.mixer <= mixer_tol, "layer {d}: mixer branch differs from the oracle ({d:.4} > {d})", .{ l, worst.mixer, mixer_tol });
    try check.expect(worst.streams <= 0.01, "layer {d}: streams after the mixer differ ({d:.4})", .{ l, worst.streams });
    try check.expect(worst.moe <= 0.02, "layer {d}: MoE branch differs from the oracle ({d:.4})", .{ l, worst.moe });
    try check.expect(worst.out <= 0.02, "layer {d}: layer output streams differ ({d:.4})", .{ l, worst.out });
    if (worst.out > 0.01) std.debug.print("  NOTE layer {d}: chained output {e:.3} is above the old engine's 1% gate (f32 streams there)\n", .{ l, worst.out });
    check.pass("layer {d} ({s}) through the family's pass (writeback, readout, {s}, write-back, readout, EXL3 moeBlock, write-back) vs float64: mixer rel {e:.3}, streams rel {e:.3}, MoE rel {e:.3}, output rel {e:.3}; picks equal on {d}/{d} rows", .{ l, if (wt.linearLayer(l)) "GDN" else "attention", if (wt.linearLayer(l)) "gdnBlock" else "attnBlock", worst.mixer, worst.streams, worst.moe, worst.out, worst.picks_same, worst.rows });
}

pub fn layerMode(gpu: Gpu, dir: []const u8, which: []const usize) !void {
    var rig: Rig = undefined;
    try rig.init(gpu, dir, 8, 64);
    defer rig.deinit();
    var raw = try RawPack.open(gpu, dir);
    defer raw.close(gpu);
    var prng = std.Random.DefaultPrng.init(0xe713_c03e);
    var failed: usize = 0;
    for (which) |l| {
        const t0 = check.now(gpu.io);
        var lw = try pl.loadLayerRank(gpu.gpa, @constCast(gpu.d), &rig.stream, &rig.store, rig.pack_fn, l, 1, 0, null);
        defer lw.free(gpu.gpa);
        std.debug.print("== layer {d} ({s}): the family's loader (cuda_prompt_load.loadLayerRank) in {d:.0} ms, every expert resident\n", .{ l, if (wt.linearLayer(l)) "GDN" else "attention", @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e6 });
        try check.expect(lw.exl3 != null, "layer {d}: experts did not load as EXL3", .{l});
        var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
        defer arena.deinit();
        // every layer runs, a failed check reported and counted
        projections(gpu, &rig, &raw, &lw, l, arena.allocator(), prng.random()) catch |err| if (err == error.TestFailed) {
            failed += 1;
        } else return err;
        layerCheck(gpu, &rig, &raw, &lw, l, &default_ids) catch |err| if (err == error.TestFailed) {
            failed += 1;
        } else return err;
    }
    try check.expect(failed == 0, "flashnext-exl3 layer: {d} checks failed", .{failed});
}
