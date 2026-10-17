//! Ling layer 7's MLA on real inputs (an HF fixture): absorb, rope, cache, attention, expand, gate vs HF and float64.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const glm = @import("glm_cuda");
const Gpu = check.Gpu;
const Rig = qr.RigT;
const bf = qr.helpers.bfOf;
const latent = glm.latent;
const Set = glm.glue.Set;

const lw = 512;
const nope = 128;
const dv = 128;

/// The fixture's dims and device buffers: q [t][h][192], lat [t][576] (raw k_rope), wk/wv [h][128][512], g fp32 [t][h].
const Layer = struct { t: usize, h: usize, qk: usize, kw: usize, q: u64, lat: u64, wk: u64, wv: u64, g: u64, freq: u64 };

/// Device buffers one pass writes: the roped cache, qa, latent output, expanded and gated rows.
const Work = struct { cache: u64, qa: u64, ol: u64, vn: u64 };

fn take(comptime T: type, a: std.mem.Allocator, bytes: []const u8, at: *usize, n: usize) ![]T {
    const out = try a.alloc(T, n);
    @memcpy(std.mem.sliceAsBytes(out), bytes[at.*..][0 .. n * @sizeOf(T)]);
    at.* += n * @sizeOf(T);
    return out;
}

/// Rows lo .. hi through absorb, rope, cache write, attention and expand at positions lo ..; out at vn row lo.
fn pass(r: *Rig, set: Set, l: Layer, w: Work, lo: usize, hi: usize) !void {
    const n = hi - lo;
    const p0: i32 = @intCast(lo);
    const pos = try r.dev(std.mem.asBytes(&p0));
    const qa = w.qa + lo * l.h * l.kw * 2;
    try latent.absorbQStrided(set, r.s, l.q + lo * l.h * l.qk * 2, l.qk, l.wk, qa, l.kw, n, l.h, nope, lw);
    try latent.rope(set, r.s, .{ .x = l.q + (lo * l.h * l.qk + nope) * 2, .x_row = l.h * l.qk, .x_head = l.qk, .out = qa + lw * 2, .o_row = l.h * l.kw, .o_head = l.kw }, n, l.h, pos, l.freq);
    const k = try r.zeros(n * l.kw * 2);
    const zero: i32 = 0;
    try latent.latentWrite(set, r.s, l.lat + lo * l.kw * 2, l.kw, k, try r.dev(std.mem.asBytes(&zero)), n, l.kw);
    try latent.rope(set, r.s, .{ .x = k + lw * 2, .x_row = l.kw, .x_head = 0, .out = k + lw * 2, .o_row = l.kw, .o_head = 0 }, n, 1, pos, l.freq);
    try latent.latentWrite(set, r.s, k, l.kw, w.cache, pos, n, l.kw);
    const nch = latent.chunksFor(hi);
    const p: latent.Partials = .{ .po = try r.zeros(nch * n * l.h * lw * 4), .pm = try r.zeros(nch * n * l.h * 4), .pl = try r.zeros(nch * n * l.h * 4) };
    const scale: f32 = @floatCast(1 / @sqrt(@as(f64, @floatFromInt(l.qk))));
    try latent.attentionRoped(set, r.s, qa, w.cache, pos, p, w.ol + lo * l.h * lw * 2, try r.zeros(4), n, l.h, lw, nch, scale);
    try latent.expandV(set, r.s, w.ol + lo * l.h * lw * 2, l.wv, w.vn + lo * l.h * dv * 2, n, l.h, dv, lw);
}

fn work(r: *Rig, l: Layer) !Work {
    return .{ .cache = try r.zeros(l.t * l.kw * 2), .qa = try r.zeros(l.t * l.h * l.kw * 2), .ol = try r.zeros(l.t * l.h * lw * 2), .vn = try r.zeros(l.t * l.h * dv * 2) };
}

/// Relative L2 of bf16 `got` against fp32 `want`, and the largest difference over want's rms.
fn distance(got: []const u8, want: []const f32) [2]f64 {
    var dd: f64 = 0;
    var ww: f64 = 0;
    var most: f64 = 0;
    for (want, 0..) |wv, i| {
        const g = bf(std.mem.bytesToValue(u16, got[2 * i ..][0..2]));
        dd += (g - wv) * (g - wv);
        ww += @as(f64, wv) * wv;
        most = @max(most, @abs(g - wv));
    }
    return .{ @sqrt(dd / ww), most / @sqrt(ww / @as(f64, @floatFromInt(want.len))) };
}

fn report(name: []const u8, got: []const u8, want: []const f32, bound: f64) !void {
    const dist = distance(got, want);
    std.debug.print("RESULT {s}: rel L2 {e:.3}, max |diff| {e:.3} of rms\n", .{ name, dist[0], dist[1] });
    try check.expect(dist[0] <= bound, "{s}: rel L2 within {e}", .{ name, bound });
}

/// `ling-mla-real <fixture>`: 512-token prompt vs HF and float64; decode rows, a window and a cut equal its bytes.
pub fn run(gpu: Gpu, path: []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(7);
    var r: Rig = .{ .gpu = gpu, .a = a, .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer mod.unload();
    const ks = try glm.glue.Kernels.load(mod);
    const set: Set = .{ .glue = &ks };
    const bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, a, .limited(256 << 20));
    var at: usize = 0;
    const dims = try take(i32, a, bytes, &at, 4);
    const t: usize = @intCast(dims[0]);
    const h: usize = @intCast(dims[1]);
    const qk: usize = @intCast(dims[2]);
    const kw: usize = @intCast(dims[3]);
    const q = try take(u16, a, bytes, &at, t * h * qk);
    const lat = try take(u16, a, bytes, &at, t * kw);
    const wk = try take(u16, a, bytes, &at, h * nope * lw);
    const wv = try take(u16, a, bytes, &at, h * dv * lw);
    const g = try take(f32, a, bytes, &at, t * h);
    const freq = try take(f32, a, bytes, &at, latent.rope_width / 2);
    var refs: [4][]f32 = undefined;
    for (&refs) |*x| x.* = try take(f32, a, bytes, &at, t * h * dv);
    try check.expect(at == bytes.len, "fixture: {d} bytes read of {d}", .{ at, bytes.len });
    const ours = latent.ropeFreqs(6000000);
    var fd: f64 = 0;
    for (ours, freq) |o, f| fd = @max(fd, @abs(@as(f64, o) - f) / f);
    std.debug.print("RESULT rope inv_freq: ropeFreqs vs HF's fp32 table, worst relative {e:.3}\n", .{fd});
    const l: Layer = .{
        .t = t,
        .h = h,
        .qk = qk,
        .kw = kw,
        .q = try r.dev(std.mem.sliceAsBytes(q)),
        .lat = try r.dev(std.mem.sliceAsBytes(lat)),
        .wk = try r.dev(std.mem.sliceAsBytes(wk)),
        .wv = try r.dev(std.mem.sliceAsBytes(wv)),
        .g = try r.dev(std.mem.sliceAsBytes(g)),
        .freq = try r.dev(std.mem.sliceAsBytes(freq)),
    };
    const full = try work(&r, l);
    try pass(&r, set, l, full, 0, t);
    const pre = try r.back(full.vn, t * h * dv * 2);
    try latent.headGate(set, r.s, full.vn, l.g, h, t, h, dv);
    const post = try r.back(full.vn, t * h * dv * 2);
    try report("Ling L7 attention pre-gate vs HF bf16 (eager)", pre, refs[0], 2e-2);
    try report("Ling L7 attention post-gate (dense input) vs HF bf16 (eager)", post, refs[1], 2e-2);
    try report("Ling L7 attention pre-gate vs float64 on the kernel inputs", pre, refs[2], 1e-2);
    try report("Ling L7 attention post-gate vs float64 on the kernel inputs", post, refs[3], 1e-2);
    const row = h * dv * 2;
    const dec = try work(&r, l);
    try pass(&r, set, l, dec, 0, 400);
    for (400..432) |i| try pass(&r, set, l, dec, i, i + 1);
    try pass(&r, set, l, dec, 432, 440);
    try latent.headGate(set, r.s, dec.vn, l.g, h, 440, h, dv);
    try check.sameBytes("Ling L7: 400-row prompt, 32 decode rows, an 8-row window equal the 512-row prompt", try r.back(dec.vn, 440 * row), post[0 .. 440 * row]);
    const cut = try work(&r, l);
    try pass(&r, set, l, cut, 0, 300);
    try pass(&r, set, l, cut, 300, t);
    try latent.headGate(set, r.s, cut.vn, l.g, h, t, h, dv);
    try check.sameBytes("Ling L7: the prompt cut at 300 equals uncut", try r.back(cut.vn, t * row), post);
    check.pass("ling-mla-real: layer 7 within rel L2 of HF and float64; decode, verify window and cut prompt equal the prompt", .{});
}
