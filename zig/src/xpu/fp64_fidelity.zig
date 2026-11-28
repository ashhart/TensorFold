//! Teacher-forced top-token fidelity with FP64 GEMM replacement (XPU_FIDELITY), runner only.
const std = @import("std");
const q = @import("qwen27.zig");
const fwd = @import("forward.zig");
const fp64 = @import("fp64_ref.zig");
const bias_oracle = @import("fp64_bias_oracle.zig");
const ze = q.ze;
const Dev = q.Dev;
const Proj = q.Proj;
const alloc = q.alloc;
extern "c" fn memfd_create(name: [*:0]const u8, flags: c_uint) c_int;
extern "c" fn pread(fd: c_int, buf: [*]u8, n: usize, off: i64) isize;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn write(fd: c_int, buf: [*]const u8, n: usize) isize;

var runner: *q.Runner = undefined;

fn addReferenceBias(layer: usize, site: u32, ref: []f64, m: usize, n: usize) !void {
    const bias = try bias_oracle.assemble(&runner.ckpt, layer, site, n, runner.model.biascorr_start);
    if (bias) |b| {
        defer alloc.free(b);
        for (0..m) |row| {
            for (0..n) |c| ref[row * n + c] += b[c];
        }
    }
}

// ---------------------------------------------------------------- teacher-forced fidelity

/// The FP64 forward's GEMMs: each XPU GEMM output replaced by fp16(FP64) of the same int4 weights and input rows.
fn replace(d: *Dev, h: *anyopaque, w: Proj, out: *anyopaque, m: u32) !void {
    const r = runner;
    var layer: usize = q.LAYERS;
    var site: u32 = 4;
    for (&r.model.layers, 0..) |*l, i| {
        for ([_]Proj{ l.mixer_in, l.mixer_out, l.gu, l.down }, 0..) |p, si| if (p.w == w.w) {
            layer = i;
            site = @intCast(si);
        };
    }
    if (site == 4 and w.w != r.model.head.w) return error.UnknownProjection;
    try d.sync();
    const K: usize = w.k;
    const N: usize = w.n;
    const xh = try alloc.alloc(f16, m * K);
    defer alloc.free(xh);
    const o = try alloc.alloc(f16, m * N);
    defer alloc.free(o);
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, xh.ptr, h, xh.len * 2, null, 0, null));
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, o.ptr, out, o.len * 2, null, 0, null));
    try d.sync();
    const ps = try fp64.parts(&r.ckpt, w, layer, site);
    const ref = try alloc.alloc(f64, m * N);
    defer alloc.free(ref);
    try fp64.hostGemm(ps, xh, m, N, ref, &.{});
    try addReferenceBias(layer, site, ref, m, N);
    for (ps) |*p| for (0..p.n) |n| {
        const c = p.col(n);
        for (0..m) |s| o[s * N + c] = @floatCast(ref[s * N + c]);
    };
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, out, o.ptr, o.len * 2, null, 0, null));
    try d.sync();
}

/// The prompt's final normed hidden rows (fp16 [T, D]) through XPU_DUMP_H's path, into memory.
fn hidden(r: *q.Runner, ids: []const u32) ![]f16 {
    const fd = memfd_create("xpu_fidelity", 0);
    if (fd < 0) return error.MemfdFailed;
    defer _ = close(fd);
    try r.reset();
    q.dump_h = fd;
    defer q.dump_h = null;
    try r.prompt(ids, true);
    const h = try alloc.alloc(f16, ids.len * q.D);
    const bytes = std.mem.sliceAsBytes(h);
    if (pread(fd, bytes.ptr, bytes.len, 0) != bytes.len) return error.ShortRead;
    return h;
}

const Top = struct { t1: u32 = 0, v1: f64 = -std.math.inf(f64), t2: u32 = 0, v2: f64 = -std.math.inf(f64) };

fn push(t: *Top, i: u32, v: f64) void {
    if (v > t.v1) {
        t.t2 = t.t1;
        t.v2 = t.v1;
        t.t1 = i;
        t.v1 = v;
    } else if (v > t.v2) {
        t.t2 = i;
        t.v2 = v;
    }
}

/// The two top tokens of each row's logits by the FP64 head (same int4 head weights).
fn headTops(h: []const f16, T: usize) ![]Top {
    const head = runner.model.head;
    const ps = try fp64.parts(&runner.ckpt, head, q.LAYERS, 4);
    const tops = try alloc.alloc(Top, T);
    @memset(tops, .{});
    const step: usize = 64; // rows at a time: the FP64 logits of a block fit in memory
    var r0: usize = 0;
    while (r0 < T) : (r0 += step) {
        const R: usize = @min(step, T - r0);
        const ref = try alloc.alloc(f64, R * head.n);
        defer alloc.free(ref);
        try fp64.hostGemm(ps, h[r0 * q.D ..][0 .. R * q.D], R, head.n, ref, &.{});
        try addReferenceBias(q.LAYERS, 4, ref, R, head.n);
        for (0..R) |s| for (0..q.VOCAB) |c| push(&tops[r0 + s], @intCast(c), ref[s * head.n + c]);
    }
    return tops;
}

/// The reference's top tokens (int4 GEMMs as fp16(FP64), the rest the device's), kept in XPU_FIDELITY_CACHE when set.
fn referenceTops(r: *q.Runner, ids: []const u32) ![]Top {
    const T = ids.len;
    const path = if (std.c.getenv("XPU_FIDELITY_CACHE")) |v| v else null;
    if (path) |p| {
        const fd = open(p, 0);
        if (fd >= 0) {
            defer _ = close(fd);
            const tops = try alloc.alloc(Top, T);
            const bytes = std.mem.sliceAsBytes(tops);
            if (pread(fd, bytes.ptr, bytes.len, 0) == bytes.len) return tops;
        }
    }
    fwd.gemm_hook = replace;
    const h = hidden(r, ids);
    fwd.gemm_hook = null;
    const tops = try headTops(try h, T);
    if (path) |p| {
        const fd = open(p, 0o1101, @as(c_uint, 0o644)); // write, create, truncate
        if (fd >= 0) {
            _ = write(fd, std.mem.sliceAsBytes(tops).ptr, std.mem.sliceAsBytes(tops).len);
            _ = close(fd);
        }
    }
    return tops;
}

/// Top tokens of the prompt path (head in FP64) and the decode path against the FP64-GEMM forward, every position.
pub fn fidelity(r: *q.Runner, ids: []const u32) !void {
    runner = r;
    const T = ids.len;
    if (T > q.CHUNK) return error.PromptLongerThanAChunk;
    const t0 = q.now();
    const ours = try headTops(try hidden(r, ids), T);
    const ref = try referenceTops(r, ids);
    const t1 = q.now();
    const dec = try alloc.alloc(Top, T);
    const logits = try alloc.alloc(f16, q.VOCAB);
    defer alloc.free(logits);
    try r.reset();
    for (0..T) |i| {
        if (i == 0) try r.prompt(ids[0..1], true) else try r.step(ids[i]);
        const api = r.d.g.api;
        const src = r.acts.logits;
        try ze.check(api.zeCommandListAppendMemoryCopy(r.d.list, logits.ptr, src, q.VOCAB * 2, null, 0, null));
        try r.d.sync();
        dec[i] = .{};
        for (logits, 0..) |v, c| push(&dec[i], @intCast(c), v);
        if (dec[i].t1 != r.last) return error.DecodeArgmaxMismatch;
    }
    std.debug.print("fidelity: {d} positions, reference forward (FP64 GEMMs) {d:.0} s, decode {d:.0} s\n", .{ T, t1 - t0, q.now() - t1 });
    var min_margin: f64 = std.math.inf(f64);
    for (ref) |t| min_margin = @min(min_margin, t.v1 - t.v2);
    std.debug.print("fidelity: smallest FP64 margin {e:.2}\n", .{min_margin});
    for ([_][]const Top{ ours, dec }, [_][]const u8{ "prompt", "decode" }) |path, name| {
        var diff: usize = 0;
        for (path, ref, 0..) |a, f, i| if (a.t1 != f.t1) {
            diff += 1;
            std.debug.print("| {s} | {d} | {d} ", .{ name, i, ids[@min(i + 1, T - 1)] });
            std.debug.print("| {d} ({d:.4}) | {d} ({d:.4}) | {e:.2} ", .{ f.t1, f.v1, f.t2, f.v2, f.v1 - f.v2 });
            std.debug.print("| {d} ({d:.4}) | {d} ({d:.4}) | {e:.2} |\n", .{ a.t1, a.v1, a.t2, a.v2, a.v1 - a.v2 });
        };
        std.debug.print("fidelity: {s} path, top-1 differences against FP64: {d} of {d}\n", .{ name, diff, T });
    }
    var pd: usize = 0;
    for (ours, dec) |a, b| pd += @intFromBool(a.t1 != b.t1);
    std.debug.print("fidelity: prompt and decode paths differ at {d} of {d} positions\n", .{ pd, T });
}
