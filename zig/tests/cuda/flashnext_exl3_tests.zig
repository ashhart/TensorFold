//! flashnext-exl3: the Flash Next CUDA family on a real EXL3 pack (Qwen3.8-Flash-Next 3.05bpw), against our float64
//! oracles (exl3_fwd_cpu.zig and the per-op ones it assembles), through the family's OWN loaders and pass code.
//! Usage: tf-cuda-test flashnext-exl3 <pack dir> names                 every tensor the family reads resolves; nothing
//!                                                                     indexed is left unclaimed but the vision tower
//!        tf-cuda-test flashnext-exl3 <pack dir> norms                 the one centred-norm convention and its cross-checks
//!        tf-cuda-test flashnext-exl3 <pack dir> router [layers] [rows] router picks (device) vs float64 (default 0,3,17)
//!        tf-cuda-test flashnext-exl3 <pack dir> ple                   PLE key/value from fp16 n-gram rows vs float64
//!        tf-cuda-test flashnext-exl3 <pack dir> layer [layers]        projections, then one whole decoder layer (prefill
//!                                                                     8 rows + 4 decode rows) vs the float64 layer (0,3)
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const check = @import("check.zig");
const fwd = @import("exl3_fwd_cpu.zig");
const hc = @import("exl3_hc.zig");
const moe = @import("exl3_moe.zig");
const router_cpu = @import("exl3_router.zig");
const route_cpu = @import("exl3_route.zig");
const scan = @import("exl3_scan.zig");
const place = @import("exl3_place.zig");
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

/// "The capital of France is Paris. It is known for the Eiffel Tower." (exl3_fwd_tests.zig's default ids)
pub const default_ids = [_]u32{ 760, 6511, 314, 9338, 369, 11751, 13, 1049, 369, 3750, 364, 279, 242476, 300, 21262, 13 };

// -- the raw pack (the oracle's view: tensors exactly as stored, no widening) -------------------------------------

pub const RawPack = struct {
    files: std.ArrayList(st.File) = .empty,

    pub fn open(gpu: Gpu, dir: []const u8) !RawPack {
        var p: RawPack = .{};
        var buf: [512]u8 = undefined;
        for (0..7) |i| {
            const path = try std.fmt.bufPrint(&buf, "{s}/model-{d:0>5}-of-00007.safetensors", .{ dir, i + 1 });
            try p.files.append(gpu.gpa, try st.File.open(gpu.gpa, gpu.io, path));
        }
        return p;
    }

    pub fn close(p: *RawPack, gpu: Gpu) void {
        for (p.files.items) |*f| f.close(gpu.io);
        p.files.deinit(gpu.gpa);
    }

    pub fn get(p: *const RawPack, comptime fmt: []const u8, args: anytype) !st.Tensor {
        var buf: [256]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, fmt, args);
        for (p.files.items) |*f| if (f.get(name)) |t| return t;
        std.debug.print("raw pack: no {s}\n", .{name});
        return error.MissingTensor;
    }
};

fn typed(comptime T: type, a: std.mem.Allocator, bytes: []const u8) ![]T {
    const out = try a.alloc(T, bytes.len / @sizeOf(T));
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. out.len * @sizeOf(T)]);
    return out;
}

fn valueOf(t: st.Tensor, i: usize) f64 {
    const h = std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little);
    return switch (t.dtype) {
        .f16 => @floatCast(@as(f16, @bitCast(h))),
        .bf16 => @as(f32, @bitCast(@as(u32, h) << 16)),
        else => unreachable,
    };
}

fn f64Of(a: std.mem.Allocator, t: st.Tensor) ![]f64 {
    if (t.dtype != .f16 and t.dtype != .bf16) return error.UnexpectedTensor;
    const out = try a.alloc(f64, t.bytes.len / 2);
    for (out, 0..) |*o, i| o.* = valueOf(t, i);
    return out;
}

fn f32Of(a: std.mem.Allocator, t: st.Tensor) ![]f32 {
    if (t.dtype != .f16 and t.dtype != .bf16) return error.UnexpectedTensor;
    const out = try a.alloc(f32, t.bytes.len / 2);
    for (out, 0..) |*o, i| o.* = @floatCast(valueOf(t, i));
    return out;
}

fn f16Of(a: std.mem.Allocator, t: st.Tensor) ![]f16 {
    if (t.dtype != .f16) return error.UnexpectedTensor;
    return typed(f16, a, t.bytes);
}

/// The EXL3 matrix `prefix` as the float64 oracle's QMat (W = diag(suh) H Wq H diag(svh), exl3_format's decoder).
fn qmat(a: std.mem.Allocator, raw: *const RawPack, prefix: []const u8, threads: usize) !fwd.QMat {
    const t = try raw.get("{s}.trellis", .{prefix});
    const trellis = try typed(i16, a, t.bytes);
    defer a.free(trellis);
    const suh = try typed(f16, a, (try raw.get("{s}.suh", .{prefix})).bytes);
    defer a.free(suh);
    const svh = try typed(f16, a, (try raw.get("{s}.svh", .{prefix})).bytes);
    defer a.free(svh);
    return fwd.QMat.init(a, @intCast(t.dim(2) / 8), trellis, t.dim(0), t.dim(1), suh, svh, threads);
}

fn bf16Bits(v: f32) u16 {
    const b: u32 = @bitCast(v);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

pub fn bfValue(bits: u16) f64 {
    return @as(f32, @bitCast(@as(u32, bits) << 16));
}

pub const Err = struct { max_abs: f64 = 0, peak: f64 = 0, rel: f64 = 0 };

pub fn errOf(got: anytype, want: []const f64) Err {
    var e: Err = .{};
    for (want, 0..) |w, i| {
        const g: f64 = switch (@TypeOf(got)) {
            []const u16, []u16 => bfValue(got[i]),
            else => @floatCast(got[i]),
        };
        e.max_abs = @max(e.max_abs, @abs(g - w));
        e.peak = @max(e.peak, @abs(w));
    }
    e.rel = if (e.peak > 0) e.max_abs / e.peak else 0;
    return e;
}

pub fn download(comptime T: type, gpu: Gpu, s: cuda.Stream, ptr: u64, n: usize) ![]T {
    try s.synchronize();
    const out = try gpu.gpa.alloc(T, n);
    const b: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = ptr, .len = n * @sizeOf(T) };
    try b.download(0, std.mem.sliceAsBytes(out));
    return out;
}

/// A pageable copy into the pass's scratch: the legacy-stream copy is ordered against nothing, so drain both sides.
pub fn uploadTo(gpu: Gpu, s: cuda.Stream, ptr: u64, bytes: []const u8) !void {
    try s.synchronize();
    const b: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = ptr, .len = bytes.len };
    try b.upload(0, bytes);
    try gpu.ctx.synchronize();
}

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

fn names(gpu: Gpu, dir: []const u8) !void {
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

fn norms(gpu: Gpu, dir: []const u8) !void {
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

fn router(gpu: Gpu, dir: []const u8, layers: []const usize, rows: usize) !void {
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
        // the family's router buffer, as cuda_prompt_load builds it: mlp.gate's 512 rows then the shared gate (Store-widened)
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

// -- the pass rig -----------------------------------------------------------------------------------------------------

pub const Pass = batch.Pass(flash.Tri);

pub const Rig = struct {
    gpu: Gpu,
    stream: cuda.Stream,
    glue: flash.Glue,
    k: batch.Kernels,
    s: batch.Scratch,
    store: wt.Store,
    pack_mod: cuda.Module,
    pack_fn: cuda.Function,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    conv: [wt.linear_n]cuda.DeviceBuffer = undefined,
    state: [wt.linear_n]cuda.DeviceBuffer = undefined,
    kc: [wt.attn_n]cuda.DeviceBuffer = undefined,
    vc: [wt.attn_n]cuda.DeviceBuffer = undefined,
    ikc: [wt.attn_n]cuda.DeviceBuffer = undefined,
    pooled: [wt.attn_n]cuda.DeviceBuffer = undefined,
    tail: cuda.DeviceBuffer = undefined,
    history: [2]i64 = .{ 0, 0 },
    scores: cuda.DeviceBuffer = undefined,
    cap: usize,
    /// The ranks' NCCL owner when a check runs two ranks (flashnext_exl3_tp_tests.zig); null: one rank.
    collective: ?*flash.collective.Collective = null,

    pub fn init(r: *Rig, gpu: Gpu, dir: []const u8, rows: usize, cap: usize) !void {
        r.* = .{ .gpu = gpu, .stream = try cuda.Stream.init(gpu.d, true), .glue = undefined, .k = undefined, .s = undefined, .store = undefined, .pack_mod = undefined, .pack_fn = undefined, .cap = cap };
        r.glue = try flash.Glue.load(gpu.d);
        r.k = try batch.Kernels.load(gpu.d);
        r.s = try batch.Scratch.alloc(gpu.gpa, gpu.d, r.stream, rows);
        r.store = try wt.Store.open(gpu.gpa, gpu.io, dir);
        try r.store.detectNorms();
        r.pack_mod = try cuda.Module.load(gpu.d, cuda.kernels.experts_pack);
        r.pack_fn = try r.pack_mod.function(wt.pack_symbol);
        const gdn = flash.gdn;
        const z: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = 0, .len = 0 };
        r.conv = @splat(z);
        r.state = @splat(z);
        r.kc = @splat(z);
        r.vc = @splat(z);
        r.ikc = @splat(z);
        r.pooled = @splat(z);
        // every layer's state, as Engine.newSeq makes them (a check may step any layer)
        for (&r.conv, &r.state) |*c, *x| {
            c.* = try r.zeroed((gdn.taps - 1) * gdn.conv_dim * 2);
            x.* = try r.zeroed(wt.state_n * 4);
        }
        for (&r.kc, &r.vc, &r.ikc, &r.pooled) |*kc, *vc, *ikc, *pooled| {
            kc.* = try r.zeroed(cap * wt.kv_heads * wt.head_dim * 2);
            vc.* = try r.zeroed(cap * wt.kv_heads * wt.head_dim * 2);
            ikc.* = try r.zeroed(cap * wt.index_dim * 2);
            pooled.* = try r.zeroed(flash.prompt.pooledBytes(cap));
        }
        r.tail = try r.zeroed(wt.ple_tail_n * wide * 2);
        r.scores = try r.zeroed(batch.scoreBytes(cap, rows));
    }

    fn zeroed(r: *Rig, bytes: usize) !cuda.DeviceBuffer {
        const b = try cuda.DeviceBuffer.alloc(r.gpu.d, bytes);
        try r.bufs.append(r.gpu.gpa, b);
        try b.fill8(0, r.stream.handle);
        try r.stream.synchronize();
        return b;
    }

    pub fn reset(r: *Rig) !void {
        for (r.bufs.items) |b| try b.fill8(0, r.stream.handle);
        try r.stream.synchronize();
    }

    pub fn deinit(r: *Rig) void {
        r.stream.synchronize() catch {};
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(r.gpu.gpa);
        r.pack_mod.unload();
        r.store.deinit();
        r.s.free(r.gpu.gpa);
        r.k.deinit();
        r.glue.deinit();
        r.stream.deinit();
    }

    pub fn pass(r: *Rig, layers: []const pl.Layer) Pass {
        return .{
            .gpa = r.gpu.gpa,
            .io = r.gpu.io,
            .stream = r.stream,
            .tri = .{ .own = &r.glue, .s = r.stream },
            .k = &r.k,
            .s = &r.s,
            .m = .{ .table_w = 0, .table_s = 0, .table_b = 0, .layers = layers, .mixer = undefined, .head = undefined, .ngram = undefined, .collective = r.collective },
            .seq = .{ .conv = &r.conv, .state = &r.state, .kc = &r.kc, .vc = &r.vc, .ikc = &r.ikc, .pooled = &r.pooled, .tail = r.tail, .history = &r.history, .cap = r.cap, .scores = r.scores.ptr },
            .hooks = .{},
        };
    }
};

fn bf16Rows(a: std.mem.Allocator, rng: std.Random, n: usize, sigma: f32) ![]u16 {
    const v = try a.alloc(u16, n);
    for (v) |*x| x.* = bf16Bits(rng.floatNorm(f32) * sigma);
    return v;
}

// -- projections (gate 4) --------------------------------------------------------------------------------------------

/// One Projection through Pass.lane(): `sk` 1 is bf16 [m, n]; `sk` > 1 fp32 [m, n] in the first slice, others zeroed.
fn laneOut(gpu: Gpu, p: *Pass, w: wt.Q4, x: u64, m: usize, sk: usize) !struct { bf: ?[]u16 = null, f: ?[]f32 = null, rest_zero: bool = true } {
    const s = p.s;
    if (sk == 1) {
        try p.lane(x, s.dummy, w, s.proj, s.dummy, m, 1);
        return .{ .bf = try download(u16, gpu, p.stream, s.proj, m * w.n) };
    }
    // the slices in a buffer of their own: a 12,288-wide part's exceeds the pass's out_sk x dims `part`
    var slices = try cuda.DeviceBuffer.alloc(gpu.d, sk * m * w.n * 4);
    defer slices.free();
    try slices.fill8(0xff, p.stream.handle);
    try p.lane(x, s.dummy, w, s.dummy, slices.ptr, m, sk);
    const all = try download(f32, gpu, p.stream, slices.ptr, sk * m * w.n);
    defer gpu.gpa.free(all);
    var zero = true;
    for (all[m * w.n ..]) |v| zero = zero and v == 0;
    return .{ .f = try gpu.gpa.dupe(f32, all[0 .. m * w.n]), .rest_zero = zero };
}

/// The f64 product of `rows` bf16 rows through the EXL3 matrix `q`.
fn exactRows(a: std.mem.Allocator, q: *const fwd.QMat, xb: []const u16, rows: usize) ![]f64 {
    const y = try a.alloc(f64, rows * q.n);
    const x = try a.alloc(f64, q.k);
    defer a.free(x);
    const tmp = try a.alloc(f64, q.k);
    defer a.free(tmp);
    for (0..rows) |r| {
        for (x, 0..) |*v, i| v.* = bfValue(xb[r * q.k + i]);
        q.apply(x, y[r * q.n ..][0..q.n], tmp);
    }
    return y;
}

/// The f64 product of `rows` bf16 rows through a dense row-major [n, k] weight.
fn denseRows(a: std.mem.Allocator, w: []const f64, n: usize, k: usize, xb: []const u16, rows: usize) ![]f64 {
    const y = try a.alloc(f64, rows * n);
    for (0..rows) |r| for (0..n) |j| {
        var acc: f64 = 0;
        for (0..k) |i| acc += bfValue(xb[r * k + i]) * w[j * k + i];
        y[r * n + j] = acc;
    };
    return y;
}

/// The fp16 rotated-input contract: x * suh after the Hadamard is rounded to fp16 (2^-11 of each term), so a column's
/// error is a few 1e-4 of the row's peak. exl3-pack measured 1.75e-4 .. 2.4e-4 (uniform rows); here N(0, 1) rows give
/// 1.3e-4 .. 3.04e-4 (layer 3 q_proj, one row, peak 4.5).
const proj_f32_tol = 4e-4;
const proj_bf16_tol = 4e-3; // bf16 output rounding: half an ulp is 2^-9 of a value, at most 1.95e-3 of the peak

fn projections(gpu: Gpu, r: *Rig, raw: *const RawPack, lw: *const pl.Layer, l: usize, a: std.mem.Allocator, rng: std.Random) !void {
    var p = r.pass((lw)[0..1]);
    const linear = wt.linearLayer(l);
    const kind = if (linear) "linear_attn" else "self_attn";
    const e = lw.proj.exl3 orelse return error.NotExl3;
    try check.expect(lw.out.exl3 != null and lw.proj.native == null and lw.out.native == null, "layer {d}: projections did not load as EXL3", .{l});
    var worst32: f64 = 0;
    var worst16: f64 = 0;
    for ([_]usize{ 1, 5, 8 }) |m| {
        const xb = try bf16Rows(a, rng, m * dims, 1);
        var xd = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xb));
        defer xd.free();
        // the stitched input projection (bf16, as the pass reads it) and each of its parts alone (fp32)
        const whole = try laneOut(gpu, &p, lw.proj, xd.ptr, m, 1);
        for (e.parts) |pt| {
            const want = switch (pt.kind) {
                .exl3 => blk: {
                    var names_buf: [128]u8 = undefined;
                    const name = partName(&names_buf, l, kind, pt.col, linear);
                    var q = try qmat(a, raw, name, 48);
                    defer q.deinit(a);
                    const y = try exactRows(a, &q, xb, m);
                    // the part alone in fp32: Projection.exl3 over one part
                    var one = e.*;
                    var single = [1]X.load.Part{pt};
                    single[0].col = 0;
                    one.parts = &single;
                    one.n = pt.n;
                    var view = wt.Q4.empty(gpu.d);
                    view.n = pt.n;
                    view.k = dims;
                    view.exl3 = &one;
                    const got = try laneOut(gpu, &p, view, xd.ptr, m, 2);
                    const ef = errOf(got.f.?, y);
                    try check.expect(got.rest_zero, "layer {d}: {s} left the other K slice unzeroed", .{ l, name });
                    std.debug.print("  L{d} {s:<50} rows={d} fp32 out: max|err| {e:.3} peak {d:.3} rel {e:.3}\n", .{ l, name, m, ef.max_abs, ef.peak, ef.rel });
                    worst32 = @max(worst32, ef.rel);
                    break :blk y;
                },
                .native => blk: {
                    // in_proj_b then in_proj_a: the F16 rows the Store widened, the oracle reads the stored F16
                    const b = try f64Of(a, try raw.get("model.language_model.layers.{d}.linear_attn.in_proj_b.weight", .{l}));
                    const av = try f64Of(a, try raw.get("model.language_model.layers.{d}.linear_attn.in_proj_a.weight", .{l}));
                    const w = try std.mem.concat(a, f64, &.{ b, av });
                    break :blk try denseRows(a, w, pt.n, dims, xb, m);
                },
            };
            // the part's columns of the stitched bf16 row
            const got = try a.alloc(u16, m * pt.n);
            for (0..m) |row| @memcpy(got[row * pt.n ..][0..pt.n], whole.bf.?[row * e.n + pt.col ..][0..pt.n]);
            const eb = errOf(got, want);
            std.debug.print("  L{d} stitched in-proj columns {d}..{d} ({s}) rows={d} bf16 out: rel {e:.3}\n", .{ l, pt.col, pt.col + pt.n, @tagName(pt.kind), m, eb.rel });
            worst16 = @max(worst16, eb.rel);
        }
        // the output projection as the pass runs it: fp32 into the first of out_sk K slices
        const ok = lw.out.k;
        const ob = try bf16Rows(a, rng, m * ok, 1);
        var od = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(ob));
        defer od.free();
        var nb: [128]u8 = undefined;
        const oname = try std.fmt.bufPrint(&nb, "model.language_model.layers.{d}.{s}.{s}", .{ l, kind, if (linear) "out_proj" else "o_proj" });
        var q = try qmat(a, raw, oname, 48);
        defer q.deinit(a);
        const want = try exactRows(a, &q, ob, m);
        const got = try laneOut(gpu, &p, lw.out, od.ptr, m, wt.out_sk);
        const eo = errOf(got.f.?, want);
        try check.expect(got.rest_zero, "layer {d}: the output projection left its other K slices unzeroed", .{l});
        std.debug.print("  L{d} {s:<50} rows={d} fp32 out (slice 0 of {d}): max|err| {e:.3} peak {d:.3} rel {e:.3}\n", .{ l, oname, m, wt.out_sk, eo.max_abs, eo.peak, eo.rel });
        worst32 = @max(worst32, eo.rel);
    }
    try check.expect(worst32 <= proj_f32_tol, "layer {d}: an EXL3 projection's fp32 output is {e:.3} of peak from float64 (> {e})", .{ l, worst32, proj_f32_tol });
    try check.expect(worst16 <= proj_bf16_tol, "layer {d}: the stitched bf16 input projection is {e:.3} of peak from float64 (> {e})", .{ l, worst16, proj_bf16_tol });
    check.pass("layer {d} projections through Pass.lane(): fp32 out worst {e:.3} of peak, stitched bf16 in-proj worst {e:.3}, rows 1/5/8", .{ l, worst32, worst16 });
}

/// The EXL3 matrix behind a stitched part, from its first column.
fn partName(buf: []u8, l: usize, kind: []const u8, col: usize, linear: bool) []const u8 {
    const name: []const u8 = if (linear) switch (col) {
        0 => "in_proj_qkv",
        else => "in_proj_z",
    } else switch (col) {
        0 => "q_proj",
        12288 => "k_proj",
        12800 => "v_proj",
        else => "indexer.index_qk_proj",
    };
    return std.fmt.bufPrint(buf, "model.language_model.layers.{d}.{s}.{s}", .{ l, kind, name }) catch unreachable;
}

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

/// The oracle's layer (exl3_fwd_cpu.LayerW) from the pack's stored tensors, decoded by the Zig f64 decoder. Arena-owned.
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
    if (lw.linear) try p.gdnBlock(lw, wt.linearIndex(l), .{}) else try p.attnBlock(lw, wt.attnIndex(l));
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
    // Each block alone, fed the device's own input: its bf16 output and the roundings inside it (the read-out's act and
    // inject, the projections' bf16 rows, the MoE weights' bf16) stay under 1% of the peak.
    const staged = [_]f64{ alone.attn_mixed, alone.attn_inj, alone.mixer, alone.mlp_mixed, alone.mlp_inj, alone.moe, alone.writeback };
    for (staged) |v| try check.expect(v <= 0.01, "layer {d}: a block alone differs from float64 by {d:.4} of its peak (> 0.01)", .{ l, v });
    // Chained from the embedding: the old engine's gates (mixer 1% GDN / 3% attention, streams 1%, MoE 2%) hold; its
    // output gate was 1% with f32 residual streams, and upstream keeps the streams in bf16 between blocks, so the
    // write-back's rounding lands on top of the chained MoE error: the output gate here is 2% (layer 0 measured 1.23e-2).
    const mixer_tol: f64 = if (wt.linearLayer(l)) 0.01 else 0.03;
    try check.expect(worst.mixer <= mixer_tol, "layer {d}: mixer branch differs from the oracle ({d:.4} > {d})", .{ l, worst.mixer, mixer_tol });
    try check.expect(worst.streams <= 0.01, "layer {d}: streams after the mixer differ ({d:.4})", .{ l, worst.streams });
    try check.expect(worst.moe <= 0.02, "layer {d}: MoE branch differs from the oracle ({d:.4})", .{ l, worst.moe });
    try check.expect(worst.out <= 0.02, "layer {d}: layer output streams differ ({d:.4})", .{ l, worst.out });
    if (worst.out > 0.01) std.debug.print("  NOTE layer {d}: chained output {e:.3} is above the old engine's 1% gate (f32 streams there)\n", .{ l, worst.out });
    check.pass("layer {d} ({s}) through the family's pass (writeback, readout, {s}, write-back, readout, EXL3 moeBlock, write-back) vs float64: mixer rel {e:.3}, streams rel {e:.3}, MoE rel {e:.3}, output rel {e:.3}; picks equal on {d}/{d} rows", .{ l, if (wt.linearLayer(l)) "GDN" else "attention", if (wt.linearLayer(l)) "gdnBlock" else "attnBlock", worst.mixer, worst.streams, worst.moe, worst.out, worst.picks_same, worst.rows });
}

fn layerMode(gpu: Gpu, dir: []const u8, which: []const usize) !void {
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

/// The PLE key/value projections of an EXL3 pack take the n-gram rows as fp16 (cuda_exl3_ple.zig's default): through
/// Pass.lane() (exact split into two bf16 rows, two bf16 products, fp32 sums) against float64 of the fp16 rows times the
/// stored F16 weights; beside it, the error the rows would carry if rounded to bf16 first (TF_FLASHNEXT_PLE_EMB=bf16).
fn pleMode(gpu: Gpu, dir: []const u8) !void {
    var rig: Rig = undefined;
    try rig.init(gpu, dir, 8, 64);
    defer rig.deinit();
    var raw = try RawPack.open(gpu, dir);
    defer raw.close(gpu);
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const d: *cuda.Driver = @constCast(gpu.d);
    // an EXL3 pack's store holds the shared Ops once its first EXL3 matrix loads; this check loads none
    (try X.load.opsOf(&rig.store, gpu.d)).release();
    var key = try X.projection_load.load(a, d, &rig.store, "language_model.model.layers.1.ple.key_proj", wide, dims);
    defer key.free();
    var value = try X.projection_load.load(a, d, &rig.store, "language_model.model.layers.1.ple.value_proj", dims, dims);
    defer value.free();
    try X.load.pleRows(a, d, &rig.store, &key, &value);
    try check.expect(key.exl3 != null and key.exl3.?.x == .f16 and value.exl3.?.x == .f16, "ple: key/value did not take fp16 rows", .{});
    var p = rig.pass(&.{});
    const kw = try f64Of(a, try raw.get("model.language_model.layers.1.ple.key_proj.weight", .{}));
    const vw = try f64Of(a, try raw.get("model.language_model.layers.1.ple.value_proj.weight", .{}));
    var prng = std.Random.DefaultPrng.init(0x91e_f16);
    const rng = prng.random();
    var worst_key: f64 = 0;
    var worst_value: f64 = 0;
    var worst_bf16: f64 = 0;
    for ([_]usize{ 1, 5, 8 }) |m| {
        const xh = try a.alloc(f16, m * dims);
        const xv = try a.alloc(f64, m * dims);
        const xr = try a.alloc(u16, m * dims); // the same rows rounded to bf16
        for (xh, xv, xr) |*h, *v, *b| {
            h.* = @floatCast(rng.floatNorm(f32) * 0.5);
            v.* = @floatCast(h.*);
            b.* = bf16Bits(@floatCast(h.*));
        }
        var xd = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xh));
        defer xd.free();
        const want_k = try a.alloc(f64, m * wide);
        const want_v = try a.alloc(f64, m * dims);
        for (0..m) |r| {
            for (0..wide) |j| {
                var acc: f64 = 0;
                for (0..dims) |i| acc += xv[r * dims + i] * kw[j * dims + i];
                want_k[r * wide + j] = acc;
            }
            for (0..dims) |j| {
                var acc: f64 = 0;
                for (0..dims) |i| acc += xv[r * dims + i] * vw[j * dims + i];
                want_v[r * dims + j] = acc;
            }
        }
        const got_k = try laneOut(gpu, &p, key, xd.ptr, m, 1);
        const ek = errOf(got_k.bf.?, want_k);
        // the value projection as pleBlock reads it: the sum of its 4 fp32 K slices
        var slices = try cuda.DeviceBuffer.alloc(gpu.d, 4 * m * dims * 4);
        defer slices.free();
        try slices.fill8(0xff, p.stream.handle);
        try p.lane(xd.ptr, p.s.dummy, value, p.s.dummy, slices.ptr, m, 4);
        const all = try download(f32, gpu, p.stream, slices.ptr, 4 * m * dims);
        defer gpu.gpa.free(all);
        const sum = try a.alloc(f64, m * dims);
        for (sum, 0..) |*o, i| o.* = @as(f64, all[i]) + all[m * dims + i] + all[2 * m * dims + i] + all[3 * m * dims + i];
        const ev = errOf(sum, want_v);
        // the bf16-row alternative, on the host in float64: rows rounded to bf16, weights exact
        const alt = try a.alloc(f64, m * dims);
        for (0..m) |r| for (0..dims) |j| {
            var acc: f64 = 0;
            for (0..dims) |i| acc += bfValue(xr[r * dims + i]) * vw[j * dims + i];
            alt[r * dims + j] = acc;
        };
        const eb = errOf(alt, want_v);
        std.debug.print("  ple rows={d}: key_proj bf16 out rel {e:.3}; value_proj fp32 slices rel {e:.3}; (bf16-rounded rows would give {e:.3})\n", .{ m, ek.rel, ev.rel, eb.rel });
        worst_key = @max(worst_key, ek.rel);
        worst_value = @max(worst_value, ev.rel);
        worst_bf16 = @max(worst_bf16, eb.rel);
    }
    // key: bf16 output rounding (2^-9 of a value); value: fp32 sums of exact products
    try check.expect(worst_key <= 4e-3, "ple: key_proj from fp16 rows is {e:.3} of peak from float64", .{worst_key});
    try check.expect(worst_value <= 1e-5, "ple: value_proj from fp16 rows is {e:.3} of peak from float64", .{worst_value});
    check.pass("ple: fp16 n-gram rows through key/value (exact bf16 split): key {e:.3}, value {e:.3} of peak (bf16-rounded rows: {e:.3})", .{ worst_key, worst_value, worst_bf16 });
}

fn parseList(gpa: std.mem.Allocator, text: []const u8) ![]usize {
    var list: std.ArrayList(usize) = .empty;
    var it = std.mem.splitScalar(u8, text, ',');
    while (it.next()) |tok| if (tok.len > 0) try list.append(gpa, try std.fmt.parseInt(usize, tok, 10));
    return list.toOwnedSlice(gpa);
}

pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 2) return error.MissingArgument;
    const dir = rest[0];
    const mode = rest[1];
    if (std.mem.eql(u8, mode, "names")) return names(gpu, dir);
    if (std.mem.eql(u8, mode, "norms")) return norms(gpu, dir);
    if (std.mem.eql(u8, mode, "ple")) return pleMode(gpu, dir);
    if (std.mem.eql(u8, mode, "router")) {
        const which = try parseList(gpu.gpa, if (rest.len > 2) rest[2] else "0,3,17");
        defer gpu.gpa.free(which);
        const rows = if (rest.len > 3) try std.fmt.parseInt(usize, rest[3], 10) else 4096;
        return router(gpu, dir, which, rows);
    }
    if (std.mem.eql(u8, mode, "layer")) {
        const which = try parseList(gpu.gpa, if (rest.len > 2) rest[2] else "0,3");
        defer gpu.gpa.free(which);
        return layerMode(gpu, dir, which);
    }
    return error.UnknownMode;
}
