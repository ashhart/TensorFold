//! flashnext-exl3: the projections (gate 4) and the PLE key/value projections.
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

// -- projections (gate 4) --------------------------------------------------------------------------------------------

/// One Projection through Pass.lane(): `sk` 1 is bf16 [m, n]; `sk` > 1 fp32 [m, n] in the first slice, others zeroed.
fn laneOut(gpu: Gpu, p: *Pass, w: wt.Q4, x: u64, m: usize, sk: usize) !struct { bf: ?[]u16 = null, f: ?[]f32 = null, rest_zero: bool = true } {
    const s = p.s;
    if (sk == 1) {
        _ = try p.lane(x, s.dummy, w, s.proj, s.dummy, m, 1);
        return .{ .bf = try download(u16, gpu, p.stream, s.proj, m * w.n) };
    }
    // the slices in a buffer of their own: a 12,288-wide part's exceeds the pass's out_sk x dims `part`
    var slices = try cuda.DeviceBuffer.alloc(gpu.d, sk * m * w.n * 4);
    defer slices.free();
    try slices.fill8(0xff, p.stream.handle);
    _ = try p.lane(x, s.dummy, w, s.dummy, slices.ptr, m, sk);
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
const proj_f32_tol = 4e-4;
const proj_bf16_tol = 4e-3; // bf16 output rounding: half an ulp is 2^-9 of a value, at most 1.95e-3 of the peak

pub fn projections(gpu: Gpu, r: *Rig, raw: *const RawPack, lw: *const pl.Layer, l: usize, a: std.mem.Allocator, rng: std.Random) !void {
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

/// The PLE key/value projections of an EXL3 pack take the n-gram rows as fp16 (cuda_exl3_ple.zig's default): through
pub fn pleMode(gpu: Gpu, dir: []const u8) !void {
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
        _ = try p.lane(xd.ptr, p.s.dummy, value, p.s.dummy, slices.ptr, m, 4);
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
