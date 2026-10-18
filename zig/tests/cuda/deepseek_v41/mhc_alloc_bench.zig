//! `tf-dsv41-test mhc-alloc [windows]`: mHC boundary time with per-buffer / arena / VMM buffers (TF_DSV41_ARENA).

const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41_kernels");
const check = @import("../check.zig");

const Gpu = check.Gpu;
const ops = dsv41.ops;

const D = 5120;
const R = 16;
const W = 2;
const SITES = 80;

const Mode = enum { @"per-buffer", plain, vmm };

const Site = struct { fnw: cuda.DeviceBuffer, base: cuda.DeviceBuffer, scale: cuda.DeviceBuffer, nw: cuda.DeviceBuffer, decoy: cuda.DeviceBuffer };

fn stats(xs: []f32) [3]f32 {
    std.mem.sort(f32, xs, {}, std.sort.asc(f32));
    return .{ xs[xs.len / 2], xs[xs.len / 10], xs[xs.len * 9 / 10] };
}

fn one(gpu: Gpu, k: *const dsv41.Kernels, mode: Mode, windows: usize) !void {
    const d = gpu.d;
    // the arena under DeviceBuffer.alloc for this mode's buffers
    var plain: cuda.arena.Plain = .{ .d = d };
    var vmm: cuda.arena.Vmm = undefined;
    var ar: cuda.arena.Arena = undefined;
    var have_vmm = false;
    if (mode != .@"per-buffer") {
        var be = plain.backend();
        if (mode == .vmm) {
            var total: usize = 0;
            try d.check(d.api.cuDeviceTotalMem_v2(&total, gpu.ctx.device), "cuDeviceTotalMem");
            vmm = cuda.arena.Vmm.open(d, gpu.ctx.device, 2 * total) catch |e| {
                std.debug.print("vmm: unavailable ({t}), skipped\n", .{e});
                return;
            };
            have_vmm = true;
            be = vmm.backend();
        }
        ar = cuda.arena.Arena.init(gpu.gpa, be, .{});
        ar.install();
    }
    defer if (mode != .@"per-buffer") {
        cuda.memory.setHook(null);
        ar.deinit();
        if (have_vmm) vmm.close();
    };

    // the window's buffers as one block like run.zig's window arena, then per-layer site weights with decoys between
    const off = struct {
        const x = 0;
        const xout = x + R * 4 * D * 2;
        const g = xout + R * 4 * D * 2;
        const part = g + W * R * D * 2;
        const c = part + R * 4 * 40 * 32 * 4;
        const out = c + R * D * 2;
        const coef = out + R * D * 2; // pre 4, post 4, comb 16 floats a row, twice
        const cnt = coef + 2 * R * 24 * 4;
        const end = cnt + 256;
    };
    var win = try cuda.DeviceBuffer.alloc(d, off.end);
    defer win.free();
    var sites: [SITES]Site = undefined;
    for (&sites) |*s| s.* = .{
        .decoy = try cuda.DeviceBuffer.alloc(d, 24 << 20),
        .fnw = try cuda.DeviceBuffer.alloc(d, 24 * 4 * D * 2),
        .base = try cuda.DeviceBuffer.alloc(d, 24 * 4),
        .scale = try cuda.DeviceBuffer.alloc(d, 3 * 4),
        .nw = try cuda.DeviceBuffer.alloc(d, D * 4),
    };
    defer for (&sites) |*s| {
        s.fnw.free();
        s.base.free();
        s.scale.free();
        s.nw.free();
        s.decoy.free();
    };
    var pollute = try cuda.DeviceBuffer.alloc(d, 512 << 20);
    defer pollute.free();

    // values: bf16 1.0 streams / partials, small fn, coefficients 0.25, norm 1.0 (finite everywhere)
    try win.fill32(0x3f803f80, null);
    try (cuda.DeviceBuffer{ .d = d, .ptr = win.ptr + off.coef, .len = 2 * R * 24 * 4 }).fill32(0x3e800000, null);
    try (cuda.DeviceBuffer{ .d = d, .ptr = win.ptr + off.cnt, .len = 256 }).fill32(0, null);
    for (&sites) |*s| {
        try s.fnw.fill32(0x3c003c00, null);
        try s.base.fill32(0, null);
        try s.scale.fill32(0x3f800000, null);
        try s.nw.fill32(0x3f800000, null);
    }

    var stream = try cuda.Stream.init(d, false);
    defer stream.deinit();
    const o = k.others(stream);
    var e0 = try cuda.Event.init(d, true);
    defer e0.deinit();
    var e1 = try cuda.Event.init(d, true);
    defer e1.deinit();
    var e2 = try cuda.Event.init(d, true);
    defer e2.deinit();
    const n = windows * SITES;
    const tb = try gpu.gpa.alloc(f32, n);
    defer gpu.gpa.free(tb);
    const tc = try gpu.gpa.alloc(f32, n);
    defer gpu.gpa.free(tc);
    const p = win.ptr;
    for (0..windows + 1) |wi| for (&sites, 0..) |*s, si| {
        const cur: u64 = @intCast(si % 2);
        const a: ops.mhc.Args = .{
            .x = p + off.x,
            .xs = 4 * D,
            .xout = p + off.xout,
            .g = p + off.g,
            .gr = R * D,
            .world = W,
            .gbf16 = 1,
            .post = p + off.coef + cur * R * 24 * 4 + R * 4 * 4,
            .comb = p + off.coef + cur * R * 24 * 4 + R * 8 * 4,
            .pre = p + off.coef + cur * R * 24 * 4,
            .@"fn" = s.fnw.ptr,
            .base = s.base.ptr,
            .scale = s.scale.ptr,
            .part = p + off.part,
            .c = p + off.c,
            .nw = s.nw.ptr,
            .out = p + off.out,
            .opre = p + off.coef + (1 - cur) * R * 24 * 4,
            .opost = p + off.coef + (1 - cur) * R * 24 * 4 + R * 4 * 4,
            .ocomb = p + off.coef + (1 - cur) * R * 24 * 4 + R * 8 * 4,
            .cnt = p + off.cnt,
            .R = R,
            .eps = 1e-6,
            .hc_eps = 1e-6,
            .post_alpha = 2.0,
            .iters = 20,
            .spin = 8000,
            .@"defer" = 1,
        };
        try pollute.fill32(@intCast(si), stream.handle);
        try e0.record(stream);
        try o.mhcBoundary(a, 0, false);
        try e1.record(stream);
        try o.mhcCoef(a);
        try e2.record(stream);
        try e2.synchronize();
        if (wi == 0) continue; // the first window warms the modules and the pages
        const i = (wi - 1) * SITES + si;
        tb[i] = 1e3 * try cuda.Event.elapsedMs(e0, e1);
        tc[i] = 1e3 * try cuda.Event.elapsedMs(e1, e2);
    };
    const b = stats(tb);
    const c = stats(tc);
    const be = if (mode == .@"per-buffer") "cuMemAlloc each" else ar.be.name;
    std.debug.print("mhc-alloc {s:<10} ({s}): boundary {d:.1} us (p10 {d:.1}, p90 {d:.1}), coef {d:.1} us (p10 {d:.1}, p90 {d:.1}), {d} launches\n", .{ @tagName(mode), be, b[0], b[1], b[2], c[0], c[1], c[2], n });
}

pub fn run(gpu: Gpu, k: *const dsv41.Kernels, windows_arg: ?[]const u8) !void {
    const windows: usize = if (windows_arg) |w| try std.fmt.parseInt(usize, w, 10) else 5;
    inline for (.{ Mode.@"per-buffer", Mode.plain, Mode.vmm }) |m| try one(gpu, k, m, @max(windows, 1));
    check.pass("mhc-alloc: {d} windows a mode", .{windows});
}
