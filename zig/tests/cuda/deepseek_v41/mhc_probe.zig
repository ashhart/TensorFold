//! `tf-dsv41-test mhc-probe [windows]`: the 16-row mHC boundary timed under one variable at a time (cold L2, spin...).

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

const Cache = enum { cold_write, cold_read, engine, warm };
const Layout = enum { @"packed", aligned2m, stagger };

/// Byte offsets of the window's buffers in one block.
const Off = struct { x: u64, xout: u64, g: u64, part: u64, c: u64, out: u64, coef: u64, cnt: u64, end: u64 };

fn layout(l: Layout) Off {
    const sizes = [_]u64{ R * 4 * D * 2, R * 4 * D * 2, W * R * D * 2, R * 4 * 40 * 32 * 4, R * D * 2, R * D * 2, 2 * R * 24 * 4, 256 };
    var at: [8]u64 = undefined;
    var cur: u64 = 0;
    for (sizes, 0..) |s, i| {
        at[i] = switch (l) {
            .@"packed" => cur,
            .aligned2m => @as(u64, i) * (2 << 20),
            .stagger => @as(u64, i) * ((1 << 20) + 384 * @as(u64, i)),
        };
        cur = std.mem.alignForward(u64, at[i] + s, 512);
    }
    return .{ .x = at[0], .xout = at[1], .g = at[2], .part = at[3], .c = at[4], .out = at[5], .coef = at[6], .cnt = at[7], .end = cur };
}

const Site = struct { fnw: cuda.DeviceBuffer, base: cuda.DeviceBuffer, scale: cuda.DeviceBuffer, nw: cuda.DeviceBuffer };

const Bench = struct {
    gpu: Gpu,
    k: *const dsv41.Kernels,
    sites: [SITES]Site,
    win: cuda.DeviceBuffer,
    big: cuda.DeviceBuffer, // 512 MiB: the fills and the prefetched "weights"
    table: cuda.DeviceBuffer, // the prefetch's (address, bytes) rows
    stream: cuda.Stream,
    other: cuda.Stream,
    e: [3]cuda.Event,
    host: cuda.HostBuffer, // pinned, mapped: the exchange's slots
    dev2: cuda.DeviceBuffer, // the same bytes in device memory (sysmem's control)

    fn args(b: *const Bench, off: Off, s: *const Site, cur: u64) ops.mhc.Args {
        const p = b.win.ptr;
        const cs: u64 = R * 24 * 4;
        return .{
            .x = p + off.x,
            .xs = 4 * D,
            .xout = p + off.xout,
            .g = p + off.g,
            .gr = R * D,
            .world = W,
            .gbf16 = 1,
            .post = p + off.coef + cur * cs + R * 4 * 4,
            .comb = p + off.coef + cur * cs + R * 8 * 4,
            .pre = p + off.coef + cur * cs,
            .@"fn" = s.fnw.ptr,
            .base = s.base.ptr,
            .scale = s.scale.ptr,
            .part = p + off.part,
            .c = p + off.c,
            .nw = s.nw.ptr,
            .out = p + off.out,
            .opre = p + off.coef + (1 - cur) * cs,
            .opost = p + off.coef + (1 - cur) * cs + R * 4 * 4,
            .ocomb = p + off.coef + (1 - cur) * cs + R * 8 * 4,
            .cnt = p + off.cnt,
            .R = R,
            .eps = 1e-6,
            .hc_eps = 1e-6,
            .post_alpha = 2.0,
            .iters = 20,
            .spin = 8000,
            .@"defer" = 1,
        };
    }

    fn sub(b: *const Bench, at: u64, len: usize) cuda.DeviceBuffer {
        return .{ .d = b.gpu.d, .ptr = b.win.ptr + at, .len = len };
    }

    /// One variant: `windows` x 80 timed launches after one untimed window. `name` overhead: the memset alone.
    fn run(b: *Bench, name: []const u8, l: Layout, cache: Cache, cross: bool, sys: ?bool, windows: usize) !void {
        const off = layout(l);
        try b.win.fill32(0x3f803f80, null); // bf16 1.0 streams / partials
        try b.sub(off.coef, 2 * R * 24 * 4).fill32(0x3e800000, null);
        try b.sub(off.cnt, 256).fill32(0, null);
        const o = b.k.others(b.stream);
        const n = windows * SITES;
        const tb = try b.gpu.gpa.alloc(f32, n);
        defer b.gpu.gpa.free(tb);
        const tc = try b.gpu.gpa.alloc(f32, n);
        defer b.gpu.gpa.free(tc);
        const overhead = std.mem.eql(u8, name, "overhead");
        const gbuf = b.sub(off.g, W * R * D * 2);
        for (0..windows + 1) |wi| for (0..SITES) |si| {
            const s = &b.sites[if (cache == .warm) 0 else si];
            switch (cache) {
                .cold_write => try b.big.fill32(@intCast(si), b.stream.handle),
                .cold_read, .engine => try o.l2Segments(b.table.ptr, 4096, 48, 128, 64 << 10, false),
                .warm => {},
            }
            if (cache == .engine) {
                if (cross) {
                    try b.e[2].record(b.stream);
                    try b.other.wait(b.e[2]);
                    try gbuf.fill32(@intCast(si | 0x3f800000), b.other.handle);
                    try b.e[2].record(b.other);
                    try b.stream.wait(b.e[2]);
                } else try gbuf.fill32(@intCast(si | 0x3f800000), b.stream.handle);
            }
            if (sys) |host| {
                // the exchange's slot traffic: stage 160 KiB out (bf16 from fp32), read 320 KiB of fp32 back into g
                const slot = if (host) try b.host.device() else b.dev2.ptr;
                try o.castBf16(b.big.ptr, slot, W * R * D / 2);
                try o.castBf16(slot + (512 << 10), gbuf.ptr, W * R * D / 2);
            }
            const a = b.args(off, s, @intCast(si % 2));
            try b.e[0].record(b.stream);
            if (overhead) try b.sub(off.cnt + 128, 4).fill32(0, b.stream.handle) else try o.mhcBoundary(a, 0, false);
            try b.e[1].record(b.stream);
            if (!overhead) try o.mhcCoef(a);
            try b.e[2].record(b.stream);
            try b.e[2].synchronize();
            if (wi == 0) continue;
            const i = (wi - 1) * SITES + si;
            tb[i] = 1e3 * try cuda.Event.elapsedMs(b.e[0], b.e[1]);
            tc[i] = 1e3 * try cuda.Event.elapsedMs(b.e[1], b.e[2]);
        };
        const x = stats(tb);
        const y = stats(tc);
        if (overhead) {
            std.debug.print("mhc-probe {s:<11} {d:6.1} us (p10 {d:.1}, p90 {d:.1})\n", .{ name, x[0], x[1], x[2] });
        } else std.debug.print("mhc-probe {s:<11} boundary {d:6.1} us (p10 {d:.1}, p90 {d:.1}), coef {d:5.1} us\n", .{ name, x[0], x[1], x[2], y[0] });
    }
};

fn stats(xs: []f32) [3]f32 {
    std.mem.sort(f32, xs, {}, std.sort.asc(f32));
    return .{ xs[xs.len / 2], xs[xs.len / 10], xs[xs.len * 9 / 10] };
}

var spin_stop: std.atomic.Value(bool) = .init(false);

fn spinner(sink: *u64) void {
    var x: u64 = 1;
    while (!spin_stop.load(.monotonic)) {
        for (0..4096) |_| x = x *% 6364136223846793005 +% 1442695040888963407;
    }
    sink.* = x;
}

/// `engine` with `n` host threads busy-spinning.
fn spun(b: *Bench, n: usize, windows: usize) !void {
    var name_buf: [16]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, "spin-{d}", .{n});
    var sinks: [256]u64 = undefined;
    var ts: [256]std.Thread = undefined;
    const m = @min(n, ts.len);
    spin_stop.store(false, .monotonic);
    for (0..m) |i| ts[i] = try std.Thread.spawn(.{}, spinner, .{&sinks[i]});
    defer {
        spin_stop.store(true, .monotonic);
        for (ts[0..m]) |t| t.join();
    }
    try b.run(name, .@"packed", .engine, false, null, windows);
}

pub fn run(gpu: Gpu, k: *const dsv41.Kernels, windows_arg: ?[]const u8) !void {
    const windows: usize = @max(1, if (windows_arg) |w| try std.fmt.parseInt(usize, w, 10) else 5);
    const d = gpu.d;
    var b: Bench = undefined;
    b.gpu = gpu;
    b.k = k;
    b.win = try cuda.DeviceBuffer.alloc(d, layout(.stagger).end + layout(.aligned2m).end);
    defer b.win.free();
    for (&b.sites) |*s| {
        s.* = .{
            .fnw = try cuda.DeviceBuffer.alloc(d, 24 * 4 * D * 2),
            .base = try cuda.DeviceBuffer.alloc(d, 24 * 4),
            .scale = try cuda.DeviceBuffer.alloc(d, 3 * 4),
            .nw = try cuda.DeviceBuffer.alloc(d, D * 4),
        };
        try s.fnw.fill32(0x3c003c00, null);
        try s.base.fill32(0, null);
        try s.scale.fill32(0x3f800000, null);
        try s.nw.fill32(0x3f800000, null);
    }
    defer for (&b.sites) |*s| {
        s.fnw.free();
        s.base.free();
        s.scale.free();
        s.nw.free();
    };
    b.big = try cuda.DeviceBuffer.alloc(d, 512 << 20);
    defer b.big.free();
    try b.big.fill32(0, null);
    // 4,096 rows of 64 KiB: 256 MiB of the big buffer, read into L2
    var rows: [4096][2]i64 = undefined;
    for (&rows, 0..) |*r, i| r.* = .{ @intCast(b.big.ptr + i * (64 << 10)), 64 << 10 };
    b.table = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(&rows));
    defer b.table.free();
    b.stream = try cuda.Stream.init(d, false);
    defer b.stream.deinit();
    b.other = try cuda.Stream.init(d, true);
    defer b.other.deinit();
    for (&b.e) |*e| e.* = try cuda.Event.init(d, true);
    defer for (&b.e) |*e| e.deinit();
    b.host = try cuda.HostBuffer.allocMapped(d, 1 << 20);
    @memset(b.host.bytes, 0);
    defer b.host.free();
    b.dev2 = try cuda.DeviceBuffer.alloc(d, 1 << 20);
    defer b.dev2.free();
    try b.dev2.fill32(0x3f800000, null);

    try b.run("overhead", .@"packed", .cold_read, false, null, windows);
    try b.run("cold-write", .@"packed", .cold_write, false, null, windows);
    try b.run("cold-read", .@"packed", .cold_read, false, null, windows);
    try b.run("engine", .@"packed", .engine, false, null, windows);
    try b.run("warm", .@"packed", .warm, false, null, windows);
    try b.run("aligned2m", .aligned2m, .engine, false, null, windows);
    try b.run("stagger", .stagger, .engine, false, null, windows);
    try b.run("cross", .@"packed", .engine, true, null, windows);
    try b.run("sysmem", .@"packed", .engine, false, true, windows);
    try b.run("devmem", .@"packed", .engine, false, false, windows);
    const cores = std.Thread.getCpuCount() catch 8;
    for ([_]usize{ 1, @max(cores / 2, 2), cores }) |n| try spun(&b, n, windows);
    check.pass("mhc-probe: {d} windows a variant", .{windows});
}
