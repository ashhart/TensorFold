//! flashnext-exl3: tensor names, norm statistics and the router against the float64 oracle.
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

// -- names ----------------------------------------------------------------------------------------------------------

const Names = struct {
    gpa: std.mem.Allocator,
    store: *wt.Store,
    claimed: std.StringHashMapUnmanaged(void) = .empty,
    missing: usize = 0,
    matrices: usize = 0,
    dense: usize = 0,
    widened: usize = 0,
    by_halves: [17]usize = @splat(0),

    fn claim(n: *Names, requested: []const u8) !void {
        var buf: [512]u8 = undefined;
        const key = n.store.resolveKey(requested, &buf) catch {
            n.missing += 1;
            if (n.missing <= 20) std.debug.print("  missing: {s}\n", .{requested});
            return;
        };
        if (n.claimed.contains(key)) return;
        try n.claimed.put(n.gpa, try n.gpa.dupe(u8, key), {});
    }

    /// An EXL3 matrix: its four parts claimed and the matrix admitted by the family's own reader (marker, widths).
    fn matrix(n: *Names, comptime fmt: []const u8, args: anytype) !void {
        var buf: [256]u8 = undefined;
        const prefix = try std.fmt.bufPrint(&buf, fmt, args);
        var pb: [280]u8 = undefined;
        for ([_][]const u8{ "trellis", "suh", "svh", "mul1" }) |part| try n.claim(try std.fmt.bufPrint(&pb, "{s}.{s}", .{ prefix, part }));
        if (!X.load.has(n.store, prefix)) {
            n.missing += 1;
            std.debug.print("  not EXL3: {s}\n", .{prefix});
            return;
        }
        const m = try X.load.mat(n.store, prefix);
        n.by_halves[m.halves] += 1;
        n.matrices += 1;
    }

    /// A dense tensor as the loaders read it: through Store.tensor, so an F16 `.weight` is widened (exact or refused).
    fn tensor(n: *Names, comptime fmt: []const u8, args: anytype) !void {
        var buf: [256]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, fmt, args);
        try n.claim(name);
        if (!n.store.has(name)) return;
        const before = n.store.widened.items.len;
        _ = try n.store.tensor(name);
        if (n.store.widened.items.len > before) n.widened += 1;
        n.dense += 1;
    }

    fn hcw(n: *Names, base: []const u8, inject: bool) !void {
        try n.tensor("{s}.hc_norm.weight", .{base});
        try n.tensor("{s}.input_mix_weight_down.weight", .{base});
        try n.tensor("{s}.input_mix_weight_up.weight", .{base});
        if (inject) try n.tensor("{s}.block_inject_weight.weight", .{base});
    }

    /// What cuda_prompt_load.loadLayerAtRank (and the EXL3 hooks it reaches) read for one decoder layer at `base`.
    fn layer(n: *Names, base: []const u8, linear: bool, with_ple: bool) !void {
        var hb: [128]u8 = undefined;
        try n.hcw(try std.fmt.bufPrint(&hb, "{s}.attn_hyper_connection", .{base}), true);
        try n.hcw(try std.fmt.bufPrint(&hb, "{s}.mlp_hyper_connection", .{base}), true);
        try n.tensor("{s}.mlp.gate.weight", .{base});
        try n.tensor("{s}.mlp.shared_expert_gate.weight", .{base});
        if (linear) {
            for ([_][]const u8{ "in_proj_qkv", "in_proj_z", "out_proj" }) |p| try n.matrix("{s}.linear_attn.{s}", .{ base, p });
            for ([_][]const u8{ "in_proj_b.weight", "in_proj_a.weight", "conv1d.weight", "A_log", "dt_bias", "norm.weight" }) |p| try n.tensor("{s}.linear_attn.{s}", .{ base, p });
        } else {
            for ([_][]const u8{ "q_proj", "k_proj", "v_proj", "indexer.index_qk_proj", "o_proj" }) |p| try n.matrix("{s}.self_attn.{s}", .{ base, p });
            for ([_][]const u8{ "q_norm", "k_norm", "indexer.q_layernorm", "indexer.k_layernorm" }) |p| try n.tensor("{s}.self_attn.{s}.weight", .{ base, p });
        }
        if (with_ple) {
            for ([_][]const u8{ "key_proj", "value_proj", "norm_key", "norm_query", "norm_conv", "conv1d" }) |p| try n.tensor("{s}.ple.{s}.weight", .{ base, p });
        }
        for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |p| try n.matrix("{s}.mlp.shared_expert.{s}", .{ base, p });
        for (0..experts_n) |e| {
            for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |p| try n.matrix("{s}.mlp.experts.{d}.{s}", .{ base, e, p });
            if (e % 64 == 63) n.store.release();
        }
        n.store.release();
    }
};

pub fn names(gpu: Gpu, dir: []const u8) !void {
    const gpa = gpu.gpa;
    var store = try wt.Store.open(gpa, gpu.io, dir);
    defer store.deinit();
    var n: Names = .{ .gpa = gpa, .store = &store };
    defer {
        var it = n.claimed.keyIterator();
        while (it.next()) |k| gpa.free(k.*);
        n.claimed.deinit(gpa);
    }
    const t0 = check.now(gpu.io);
    // the engine (cuda_prompt.zig Engine.initRank): embedding, the final mixer, the head (EXL3 here: plan step 9)
    try n.tensor("language_model.model.embed_tokens.weight", .{});
    try n.hcw("language_model.model.hyper_connection_mixer", false);
    try n.matrix("language_model.lm_head", .{});
    for (0..wt.layers_n) |l| {
        var bb: [64]u8 = undefined;
        try n.layer(try std.fmt.bufPrint(&bb, "language_model.model.layers.{d}", .{l}), wt.linearLayer(l), l == wt.ple_layer);
    }
    // the MTP head (cuda_mtp.zig Head.init): norms, two EXL3 fc projections, its mixer (the patch file), one layer
    try n.tensor("language_model.mtp.pre_fc_norm_embedding.weight", .{});
    try n.tensor("language_model.mtp.pre_fc_norm_hidden.weight", .{});
    try n.matrix("language_model.mtp.fc_embedding", .{});
    try n.matrix("language_model.mtp.fc_hidden", .{});
    try n.hcw("language_model.mtp.hyper_connection_mixer", false);
    try n.layer("language_model.mtp.layers.0", false, false);
    // everything the index (and the patch) names: claimed, or the vision tower; and the header rules' homes agree
    const wm = store.parsed.value.object.get("weight_map").?.object;
    var vision: usize = 0;
    var unclaimed: usize = 0;
    var homes: [8]usize = @splat(0);
    var disagree: usize = 0;
    var it = wm.iterator();
    while (it.next()) |kv| {
        const name = kv.key_ptr.*;
        const home = place.home(.{ .name = name, .rank = if (std.mem.endsWith(u8, name, ".trellis")) 3 else 1, .file = kv.value_ptr.string });
        homes[@backingInt(home)] += 1;
        if (n.claimed.contains(name)) {
            // a routed expert's part must be the routed bank's, a shared expert's part the resident one's
            if (std.mem.indexOf(u8, name, ".mlp.experts.") != null and home != .routed_host) disagree += 1;
            if (std.mem.indexOf(u8, name, ".shared_expert.") != null and home != .shared_resident) disagree += 1;
            continue;
        }
        if (std.mem.startsWith(u8, name, "model.visual.")) {
            vision += 1;
            continue;
        }
        unclaimed += 1;
        if (unclaimed <= 20) std.debug.print("  unclaimed: {s}\n", .{name});
    }
    const ms = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e6;
    std.debug.print("names: {d} indexed (+ the mixer patch), {d} claimed, {d} vision, {d} unclaimed, {d} missing; {d} EXL3 matrices admitted (mul1 marker, widths from the trellis), {d} dense tensors read, {d} F16 weights widened to BF16 exactly; {d:.0} ms\n", .{ wm.count(), n.claimed.count(), vision, unclaimed, n.missing, n.matrices, n.dense, n.widened, ms });
    std.debug.print("  EXL3 widths (halves: count):", .{});
    for (n.by_halves, 0..) |c, h| if (c > 0) std.debug.print(" {d}:{d}", .{ h, c });
    std.debug.print("\n  header homes (exl3_place):", .{});
    for (homes, 0..) |c, i| if (c > 0) std.debug.print(" {s}={d}", .{ @tagName(@as(place.Home, @fromBackingInt(@intCast(i)))), c });
    std.debug.print("\n", .{});
    try check.expect(n.missing == 0, "names: {d} tensors the family reads do not resolve", .{n.missing});
    try check.expect(unclaimed == 0, "names: {d} indexed tensors no family loader reads (vision excepted)", .{unclaimed});
    try check.expect(disagree == 0, "names: {d} expert tensors whose exl3_place home disagrees", .{disagree});
    // the family's device sizing (cuda_exl3_memory.zig) on the same pack
    const gib = 1024.0 * 1024.0 * 1024.0;
    const on_host = try X.memory.deviceBytes(gpa, gpu.io, dir, .{});
    const resident = try X.memory.deviceBytes(gpa, gpu.io, dir, .{ .routed_on_host = false });
    const head = try X.memory.headBytes(gpa, gpu.io, dir, .{});
    std.debug.print("  device bytes: {d:.2} GiB with routed experts on the host, {d:.2} GiB all resident; MTP head {d:.2} GiB (routed on the host)\n", .{ @as(f64, @floatFromInt(on_host)) / gib, @as(f64, @floatFromInt(resident)) / gib, @as(f64, @floatFromInt(head)) / gib });
    check.pass("names: every tensor the family reads resolves ({d}), nothing unclaimed but {d} vision tensors; {d} EXL3 matrices admitted; {d} F16 weights widened exactly", .{ n.claimed.count(), vision, n.matrices, n.widened });
}

// -- norms ----------------------------------------------------------------------------------------------------------

fn meanOf(t: st.Tensor) f64 {
    var s: f64 = 0;
    const count = t.bytes.len / 2;
    for (0..count) |i| s += valueOf(t, i);
    return s / @as(f64, @floatFromInt(count));
}

pub fn norms(gpu: Gpu, dir: []const u8) !void {
    const gpa = gpu.gpa;
    var store = try wt.Store.open(gpa, gpu.io, dir);
    defer store.deinit();
    try store.detectNorms();
    var raw = try RawPack.open(gpu, dir);
    defer raw.close(gpu);
    var means: [wt.layers_n]f64 = undefined;
    for (&means, 0..) |*m, l| m.* = meanOf(try raw.get("model.language_model.layers.{d}.attn_hyper_connection.hc_norm.weight", .{l}));
    var sorted = means;
    std.mem.sort(f64, &sorted, {}, std.sort.asc(f64));
    std.debug.print("norms: 48 attn hc_norm means min {d:.4} median {d:.4} max {d:.4} -> offset {d} (cuda_norms.offset via Store.detectNorms)\n", .{ sorted[0], (sorted[23] + sorted[24]) / 2, sorted[47], store.norm_offset });
    try check.expect(store.norm_offset == 1, "norms: the pack stores gamma - 1; the convention came out {d}", .{store.norm_offset});
    // the independent cross-check: the indexer norms' own 24 means locate the same storage
    var idx: [24]f32 = undefined;
    var c: usize = 0;
    for (0..wt.layers_n) |l| {
        if (wt.linearLayer(l)) continue;
        for ([_][]const u8{ "q_layernorm", "k_layernorm" }) |nm| {
            idx[c] = @floatCast(meanOf(try raw.get("model.language_model.layers.{d}.self_attn.indexer.{s}.weight", .{ l, nm })));
            c += 1;
        }
    }
    const indexer = try scan.centredOffset(&idx);
    std.debug.print("norms: indexer q/k_layernorm means over the 12 attention layers -> exl3_scan.centredOffset {d}\n", .{indexer});
    try check.expect(indexer == store.norm_offset, "norms: the indexer norms locate offset {d}, the hc_norm means {d}", .{ indexer, store.norm_offset });
    // every centred norm the family scales, MTP's included, loads through Store.normScale with that one offset
    const mtp_names = [_][]const u8{ "language_model.mtp.pre_fc_norm_embedding.weight", "language_model.mtp.pre_fc_norm_hidden.weight", "language_model.mtp.hyper_connection_mixer.hc_norm.weight", "language_model.mtp.layers.0.attn_hyper_connection.hc_norm.weight", "language_model.mtp.layers.0.mlp_hyper_connection.hc_norm.weight", "language_model.mtp.layers.0.self_attn.q_norm.weight", "language_model.mtp.layers.0.self_attn.k_norm.weight", "language_model.mtp.layers.0.self_attn.indexer.q_layernorm.weight", "language_model.mtp.layers.0.self_attn.indexer.k_layernorm.weight", "language_model.model.layers.1.ple.norm_key.weight", "language_model.model.layers.1.ple.norm_query.weight", "language_model.model.layers.1.ple.norm_conv.weight", "language_model.model.layers.3.self_attn.q_norm.weight", "language_model.model.layers.3.self_attn.k_norm.weight", "language_model.model.hyper_connection_mixer.hc_norm.weight" };
    for (mtp_names) |name| {
        const scales = try store.normScale(gpa, try store.tensor(name));
        defer gpa.free(scales);
        var sum: f64 = 0;
        var lo: f32 = std.math.inf(f32);
        for (scales) |v| {
            sum += v;
            lo = @min(lo, v);
        }
        std.debug.print("  {s:<72} stored mean {d:>7.4} -> scale mean {d:.4} (min {d:.4})\n", .{ name, sum / @as(f64, @floatFromInt(scales.len)) - store.norm_offset, sum / @as(f64, @floatFromInt(scales.len)), lo });
    }
    store.release();
    check.pass("norms: one convention (offset {d}) from the 48 hc_norm means, confirmed by the indexer norms; every centred norm incl. MTP's scales finite", .{store.norm_offset});
}

// -- router ---------------------------------------------------------------------------------------------------------

pub fn router(gpu: Gpu, dir: []const u8, layers: []const usize, rows: usize) !void {
    const gpa = gpu.gpa;
    var store = try wt.Store.open(gpa, gpu.io, dir);
    defer store.deinit();
    var raw = try RawPack.open(gpu, dir);
    defer raw.close(gpu);
    var glue = try flash.Glue.load(gpu.d);
    defer glue.deinit();
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0x5eed_2026);
    const rng = prng.random();
    var worst_rows_diff: usize = 0;
    for (layers) |l| {
        // the family's router buffer, as cuda_prompt_load builds it: mlp.gate's 512 rows then the shared gate.
        var nb: [128]u8 = undefined;
        const gate = try store.tensor(try std.fmt.bufPrint(&nb, "language_model.model.layers.{d}.mlp.gate.weight", .{l}));
        try check.expect(gate.is(.bf16, &.{ experts_n, dims }), "router: layer {d} gate is not the loaders' bf16 [512, 2560]", .{l});
        const w = try gpa.alloc(u8, (experts_n + 1) * dims * 2);
        defer gpa.free(w);
        @memcpy(w[0 .. experts_n * dims * 2], gate.bytes);
        const sg = try store.tensor(try std.fmt.bufPrint(&nb, "language_model.model.layers.{d}.mlp.shared_expert_gate.weight", .{l}));
        @memcpy(w[experts_n * dims * 2 ..], sg.bytes);
        store.release();
        var wd = try cuda.DeviceBuffer.fromHost(gpu.d, w);
        defer wd.free();
        // the oracle's weights: the pack's F16 bits, never the widened copy
        const g16 = try f16Of(gpa, try raw.get("model.language_model.layers.{d}.mlp.gate.weight", .{l}));
        defer gpa.free(g16);
        const s16 = try f16Of(gpa, try raw.get("model.language_model.layers.{d}.mlp.shared_expert_gate.weight", .{l}));
        defer gpa.free(s16);
        // rows: N(0, sigma) per row, sigma in {0.25, 1, 4}, rounded to bf16 as the pass's `mixed` is
        const xb = try gpa.alloc(u16, rows * dims);
        defer gpa.free(xb);
        for (0..rows) |r| {
            const sigma: f32 = switch (r % 3) {
                0 => 0.25,
                1 => 1,
                else => 4,
            };
            for (0..dims) |i| xb[r * dims + i] = bf16Bits(rng.floatNorm(f32) * sigma);
        }
        var xd = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xb));
        defer xd.free();
        var ld = try cuda.DeviceBuffer.alloc(gpu.d, rows * (experts_n + 1) * 4);
        defer ld.free();
        var pd = try cuda.DeviceBuffer.alloc(gpu.d, rows * slots * 4);
        defer pd.free();
        var wdd = try cuda.DeviceBuffer.alloc(gpu.d, rows * slots * 4);
        defer wdd.free();
        try glue.routerMany(s, xd.ptr, wd.ptr, ld.ptr, dims, rows);
        try glue.topkMany(s, ld.ptr, pd.ptr, wdd.ptr, rows);
        const gl = try download(f32, gpu, s, ld.ptr, rows * (experts_n + 1));
        defer gpa.free(gl);
        const gp = try download(i32, gpu, s, pd.ptr, rows * slots);
        defer gpa.free(gp);
        const gw = try download(f32, gpu, s, wdd.ptr, rows * slots);
        defer gpa.free(gw);
        var diff_rows: usize = 0;
        var logit_err: f64 = 0;
        var min_gap: f64 = std.math.inf(f64);
        var worst_w: f64 = 0;
        var worst_sw: f64 = 0;
        var f32_agree: usize = 0;
        const x = try gpa.alloc(f64, dims);
        defer gpa.free(x);
        const xf = try gpa.alloc(f32, dims);
        defer gpa.free(xf);
        var logits: [experts_n + 1]f64 = undefined;
        for (0..rows) |r| {
            for (x, xf, 0..) |*v, *vf, i| {
                v.* = bfValue(xb[r * dims + i]);
                vf.* = @floatCast(v.*);
            }
            try moe.logitsF16(g16, experts_n, s16, x, &logits);
            var picks: [topk]u16 = undefined;
            var wts: [topk]f64 = undefined;
            var sw: f64 = 0;
            try moe.routeFromLogits(&logits, topk, &picks, &wts, &sw);
            for (0..experts_n + 1) |i| logit_err = @max(logit_err, @abs(@as(f64, gl[r * (experts_n + 1) + i]) - logits[i]));
            var sorted = logits[0..experts_n].*;
            std.mem.sort(f64, &sorted, {}, std.sort.desc(f64));
            const gap = sorted[topk - 1] - sorted[topk];
            min_gap = @min(min_gap, gap);
            var a: [topk]u16 = undefined;
            for (&a, 0..) |*v, k| v.* = @intCast(gp[r * slots + k]);
            var b = picks;
            std.mem.sort(u16, &a, {}, std.sort.asc(u16));
            std.mem.sort(u16, &b, {}, std.sort.asc(u16));
            if (!std.mem.eql(u16, &a, &b)) {
                diff_rows += 1;
                if (diff_rows <= 10) std.debug.print("  layer {d} row {d}: picks differ; f64 logit gap between the 10th and 11th {e:.3}\n", .{ l, r, gap });
            } else {
                for (0..topk) |k| {
                    // the same expert's weight: the device's bf16 against the float64 softmax
                    const id = gp[r * slots + k];
                    for (picks, wts) |pk, wv| if (pk == id) {
                        worst_w = @max(worst_w, @abs(@as(f64, gw[r * slots + k]) - wv));
                    };
                }
                try check.expect(gp[r * slots + topk] == experts_n, "router: the shared slot holds {d}", .{gp[r * slots + topk]});
                worst_sw = @max(worst_sw, @abs(@as(f64, gw[r * slots + topk]) - sw));
            }
            // the f32 host reference (exl3_router.launchDtype(.f16)) on the same row: its picks as a set
            var l32: [experts_n + 1]f32 = undefined;
            var p32: [topk + 1]u16 = undefined;
            var w32: [topk + 1]f32 = undefined;
            var scratch: [experts_n]f32 = undefined;
            try router_cpu.launchDtype(.f16, 0, std.mem.sliceAsBytes(g16), xf, std.mem.sliceAsBytes(s16), xf, topk, &l32, &p32, &w32, &scratch);
            var c32 = p32[0..topk].*;
            std.mem.sort(u16, &c32, {}, std.sort.asc(u16));
            f32_agree += @intFromBool(std.mem.eql(u16, &c32, &b));
        }
        std.debug.print("router layer {d}: {d} rows, picks equal to float64 on {d}/{d} (f32 host reference equal on {d}/{d}); max |device logit - f64| {e:.3}; smallest f64 10th/11th gap {e:.3}; weights max |bf16 - f64| {e:.3}, shared {e:.3}\n", .{ l, rows, rows - diff_rows, rows, f32_agree, rows, logit_err, min_gap, worst_w, worst_sw });
        worst_rows_diff = @max(worst_rows_diff, diff_rows);
        try check.expect(diff_rows == 0 or min_gap <= 4 * logit_err, "router layer {d}: {d} rows pick differently with every gap above the logit error", .{ l, diff_rows });
    }
    try check.expect(worst_rows_diff == 0, "router: picks differ from float64 on {d} rows (see the gaps above)", .{worst_rows_diff});
    check.pass("router: the exactly widened F16 router's top-10 equals the float64 oracle on every row of layers {any}", .{layers});
}
