//! ggml kernel benchmarks and device probes for xpu-ggml-test, selected by its command-line flags.

const std = @import("std");
const rt = @import("rig.zig");
const gg = @import("xpu").ggml;
const root = @import("ggml.zig");

const Fix = root.Fix;
const gpa = root.gpa;
const nowNs = root.nowNs;
const up = root.up;
const section = root.section;
const tableBuf = root.tableBuf;
const fixFor = root.fixFor;
const randomX = root.randomX;

/// Times `calls` matvecs rotating over nbuf weight buffers so L2 never holds a table; returns ns a call.
pub fn timeMv(r: *rt.Runtime, set: *gg.Set, t: gg.Type, w: []rt.Buffer, x: rt.Buffer, y: rt.Buffer, in: u32, rows: u32) !u64 {
    try set.matvec(t, w[0], x, y, in, 0, rows, false);
    try r.sync();
    var best: u64 = std.math.maxInt(u64);
    for (0..3) |_| {
        const t0 = nowNs();
        for (0..w.len * 2) |i| try set.matvec(t, w[i % w.len], x, y, in, 0, rows, false);
        try r.sync();
        best = @min(best, (nowNs() - t0) / (w.len * 2));
    }
    return best;
}

/// Matvec time on a table of 34816 rows x 5120, GB/s of file-format weight bytes.
pub fn bench(r: *rt.Runtime, set: *gg.Set, f: Fix) !void {
    const rows: u32 = 34816;
    const hdr = try section(u32, f.bytes, 0, 4);
    if (hdr[1] != 5120) return;
    const host = try tableBuf(f, rows, 5120);
    defer gpa.free(host);
    var w = [_]rt.Buffer{ try up(r, host), try up(r, host) };
    const x = try randomX(r, 5120);
    const y = try r.alloc(rows * 4);
    const best = try timeMv(r, set, f.t, &w, x, y, 5120, rows);
    const file_mb = @as(f64, @floatFromInt(@as(u64, rows) * 20 * gg.blockBytes(f.t))) / 1e6;
    std.debug.print("  bench {s}: {d} x 5120, {d:.1} MB, {d:.0} us, {d:.0} GB/s\n", .{ f.key, rows, file_mb, @as(f64, @floatFromInt(best)) / 1e3, file_mb * 1e6 / @as(f64, @floatFromInt(best)) });
    for (&w) |*b| b.free();
}

const Shape = struct { t: gg.Type, rows: u32, in: u32, calls: u32 };

const shapes = [_]Shape{
    .{ .t = .q4_k, .rows = 248320, .in = 5120, .calls = 1 },
    .{ .t = .iq4_xs, .rows = 10240, .in = 5120, .calls = 9 },
    .{ .t = .iq4_xs, .rows = 6144, .in = 5120, .calls = 12 },
    .{ .t = .iq4_xs, .rows = 5120, .in = 6144, .calls = 18 },
    .{ .t = .iq2_xs, .rows = 17408, .in = 5120, .calls = 5 },
    .{ .t = .iq2_xxs, .rows = 17408, .in = 5120, .calls = 3 },
    .{ .t = .iq2_s, .rows = 5120, .in = 17408, .calls = 4 },
    .{ .t = .iq3_s, .rows = 10240, .in = 5120, .calls = 22 },
    .{ .t = .iq3_xxs, .rows = 6144, .in = 5120, .calls = 9 },
    .{ .t = .q4_k, .rows = 5120, .in = 6144, .calls = 9 },
    .{ .t = .iq3_s, .rows = 5120, .in = 17408, .calls = 22 },
    .{ .t = .iq3_xxs, .rows = 10240, .in = 5120, .calls = 13 },
    .{ .t = .iq3_s, .rows = 6144, .in = 5120, .calls = 18 },
    .{ .t = .iq3_xxs, .rows = 17408, .in = 5120, .calls = 39 },
    .{ .t = .iq3_s, .rows = 17408, .in = 5120, .calls = 40 },
    .{ .t = .iq2_xxs, .rows = 12288, .in = 5120, .calls = 1 },
    .{ .t = .iq3_s, .rows = 1024, .in = 5120, .calls = 7 },
    .{ .t = .iq3_s, .rows = 5120, .in = 6144, .calls = 32 },
    .{ .t = .iq3_xxs, .rows = 5120, .in = 17408, .calls = 7 },
    .{ .t = .iq2_s, .rows = 17408, .in = 5120, .calls = 9 },
    .{ .t = .iq4_xs, .rows = 5120, .in = 17408, .calls = 21 },
    .{ .t = .q2_k, .rows = 12288, .in = 5120, .calls = 6 },
    .{ .t = .iq4_xs, .rows = 1024, .in = 5120, .calls = 9 },
    .{ .t = .iq3_xxs, .rows = 5120, .in = 6144, .calls = 5 },
    .{ .t = .q2_k, .rows = 6144, .in = 5120, .calls = 4 },
    .{ .t = .iq2_s, .rows = 1024, .in = 5120, .calls = 1 },
    .{ .t = .iq3_xxs, .rows = 1024, .in = 5120, .calls = 2 },
    .{ .t = .iq2_xs, .rows = 5120, .in = 17408, .calls = 3 },
    .{ .t = .iq1_m, .rows = 17408, .in = 5120, .calls = 1 },
    .{ .t = .iq2_xs, .rows = 10240, .in = 5120, .calls = 1 },
    .{ .t = .q4_k, .rows = 1024, .in = 5120, .calls = 13 },
    .{ .t = .q2_k, .rows = 5120, .in = 17408, .calls = 1 },
    .{ .t = .iq2_s, .rows = 6144, .in = 5120, .calls = 2 },
    .{ .t = .q2_k, .rows = 10240, .in = 5120, .calls = 1 },
    .{ .t = .q4_k, .rows = 6144, .in = 5120, .calls = 3 },
    .{ .t = .iq3_s, .rows = 12288, .in = 5120, .calls = 3 },
    .{ .t = .iq4_xs, .rows = 17408, .in = 5120, .calls = 25 },
    .{ .t = .iq4_xs, .rows = 12288, .in = 5120, .calls = 2 },
    .{ .t = .q2_k, .rows = 17408, .in = 5120, .calls = 1 },
    .{ .t = .q4_k, .rows = 5120, .in = 17408, .calls = 6 },
    .{ .t = .q4_k, .rows = 17408, .in = 5120, .calls = 5 },
    .{ .t = .iq3_xxs, .rows = 12288, .in = 5120, .calls = 3 },
    .{ .t = .q4_k, .rows = 10240, .in = 5120, .calls = 1 },
    .{ .t = .iq2_xxs, .rows = 10240, .in = 5120, .calls = 1 },
    .{ .t = .q4_k, .rows = 12288, .in = 5120, .calls = 1 },
};

/// Matvec time of every projection shape of one GSQ token with cold weights (rotated buffers); GB/s of file bytes.
pub fn benchShapes(r: *rt.Runtime, set: *gg.Set) !void {
    var total_us: f64 = 0;
    var total_bytes: f64 = 0;
    const x = try randomX(r, 17408);
    for (shapes) |sh| {
        const host = try tableBuf(try fixFor(sh.t), sh.rows, sh.in);
        defer gpa.free(host);
        const nbuf: usize = @min(48, @max(3, (160 << 20) / host.len + 1));
        var w: [48]rt.Buffer = undefined;
        for (w[0..nbuf]) |*b| b.* = try up(r, host);
        const y = try r.alloc(@as(usize, sh.rows) * 4);
        const best = try timeMv(r, set, sh.t, w[0..nbuf], x, y, sh.in, sh.rows);
        const bytes: f64 = @floatFromInt(@as(usize, sh.rows) * (sh.in / 256) * gg.blockBytes(sh.t));
        const us = @as(f64, @floatFromInt(best)) / 1e3;
        std.debug.print("  {s:<8} {d:>6} x {d:>5} x{d:<3} {d:7.1} us {d:5.0} GB/s  ks {d}\n", .{ @tagName(sh.t), sh.rows, sh.in, sh.calls, us, bytes / us / 1e3, gg.Set.ksplit(sh.rows, sh.in / 256) });
        total_us += us * @as(f64, @floatFromInt(sh.calls));
        total_bytes += bytes * @as(f64, @floatFromInt(sh.calls));
        for (w[0..nbuf]) |*b| b.free();
    }
    std.debug.print("token total (quantized types only): {d:.2} ms, {d:.0} GB/s average, {d:.2} GB\n", .{ total_us / 1e3, total_bytes / total_us / 1e3, total_bytes / 1e9 });
}

/// One GSQ token of quantized matvecs replayed interleaved (cold weights): sustained clocks and switch effects.
pub fn replay(r: *rt.Runtime, set: *gg.Set) !void {
    const x = try randomX(r, 17408);
    var wbuf: [shapes.len][2]rt.Buffer = undefined;
    var ybuf: [shapes.len]rt.Buffer = undefined;
    for (shapes, 0..) |sh, si| {
        const host = try tableBuf(try fixFor(sh.t), sh.rows, sh.in);
        defer gpa.free(host);
        for (&wbuf[si]) |*b| b.* = try up(r, host);
        ybuf[si] = try r.alloc(@as(usize, sh.rows) * 4);
    }
    var remaining: [shapes.len]u32 = undefined;
    var best: u64 = std.math.maxInt(u64);
    for (0..4) |_| {
        const t0 = nowNs();
        for (0..10) |_| {
            for (shapes, 0..) |sh, si| remaining[si] = sh.calls;
            var left: u32 = 0;
            for (remaining) |c| left += c;
            var parity: usize = 0;
            while (left > 0) {
                for (shapes, 0..) |sh, si| {
                    if (remaining[si] == 0) continue;
                    remaining[si] -= 1;
                    left -= 1;
                    try set.matvec(sh.t, wbuf[si][parity & 1], x, ybuf[si], sh.in, 0, sh.rows, false);
                    parity += 1;
                }
            }
        }
        try r.sync();
        best = @min(best, nowNs() - t0);
    }
    std.debug.print("replay: {d:.2} ms per token (10 tokens a pass, best of 4)\n", .{@as(f64, @floatFromInt(best)) / 1e6 / 10});
}

/// Raw streaming read rate (GB/s) over 512 MB with one probe kernel: the DRAM ceiling the matvecs are compared with.
pub fn probeBandwidth(r: *rt.Runtime) !void {
    var m = try r.module(@import("xpu").kernels.ggml_quant);
    var k = try m.kernel("bw_probe", .{ 64, 1, 1 });
    const bytes: usize = 512 << 20;
    const w = try r.alloc(bytes);
    const out = try r.alloc(64);
    // random contents (fresh device memory is zero and zero data reads faster); PROBE_ZEROS=1 keeps zeros
    if (std.c.getenv("PROBE_ZEROS") == null) {
        const fill = try std.heap.page_allocator.alloc(u8, bytes);
        defer std.heap.page_allocator.free(fill);
        var prng = std.Random.DefaultPrng.init(11);
        prng.random().bytes(fill);
        try r.upload(w, fill);
        try r.sync();
    }
    for ([_]u32{ 1024, 4096, 16384 }) |sgs| {
        // sgs sub-groups, each streaming bytes / sgs
        const per: u32 = @intCast(bytes / 16 / sgs);
        try k.setBuffer(0, w);
        try k.setBuffer(1, out);
        try k.setU32(2, per);
        try k.launch(.{ sgs / 4, 1, 1 });
        try r.sync();
        var best: u64 = std.math.maxInt(u64);
        for (0..5) |_| {
            const t0 = nowNs();
            try k.launch(.{ sgs / 4, 1, 1 });
            try r.sync();
            best = @min(best, nowNs() - t0);
        }
        std.debug.print("  probe {d} sub-groups: {d:.0} GB/s\n", .{ sgs, @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(best)) });
    }
}

/// Fused vs separate launches for the real projection groups (gate+up, GDN qkv+z, attention q+k+v), cold weights.
pub fn benchFuse(r: *rt.Runtime, set: *gg.Set) !void {
    const Case = struct { name: []const u8, t: [3]gg.Type, rows: [3]u32, n: usize };
    const cases = [_]Case{
        .{ .name = "mlp same  ", .t = .{ .iq3_xxs, .iq3_xxs, .iq3_xxs }, .rows = .{ 17408, 17408, 0 }, .n = 2 },
        .{ .name = "mlp iq4+iq3", .t = .{ .iq4_xs, .iq3_s, .iq3_s }, .rows = .{ 17408, 17408, 0 }, .n = 2 },
        .{ .name = "mlp iq3+iq3", .t = .{ .iq3_s, .iq3_s, .iq3_s }, .rows = .{ 17408, 17408, 0 }, .n = 2 },
        .{ .name = "gdn qkv+z  ", .t = .{ .iq3_s, .iq3_xxs, .iq3_s }, .rows = .{ 10240, 6144, 0 }, .n = 2 },
        .{ .name = "attn q+k+v ", .t = .{ .iq2_xxs, .iq3_s, .iq3_s }, .rows = .{ 12288, 1024, 1024 }, .n = 3 },
    };
    const x = try randomX(r, 5120);
    for (cases) |c| {
        var w: [3][4]rt.Buffer = undefined;
        var y: [3]rt.Buffer = undefined;
        for (0..c.n) |i| {
            const host = try tableBuf(try fixFor(c.t[i]), c.rows[i], 5120);
            defer gpa.free(host);
            for (&w[i]) |*b| b.* = try up(r, host);
            y[i] = try r.alloc(@as(usize, c.rows[i]) * 2);
        }
        var sep: u64 = std.math.maxInt(u64);
        var fus: u64 = std.math.maxInt(u64);
        for (0..3) |_| {
            var t0 = nowNs();
            for (0..8) |it| for (0..c.n) |i| try set.matvec(c.t[i], w[i][it % 4], x, y[i], 5120, 0, c.rows[i], false);
            try r.sync();
            sep = @min(sep, (nowNs() - t0) / 8);
            t0 = nowNs();
            for (0..8) |it| {
                var segs: [3]gg.Seg = undefined;
                for (0..c.n) |i| segs[i] = .{ .t = c.t[i], .w = w[i][it % 4], .y = y[i], .rows = c.rows[i], .y_off = 0 };
                try set.matvecMulti(segs[0..c.n], x, 5120);
            }
            try r.sync();
            fus = @min(fus, (nowNs() - t0) / 8);
        }
        std.debug.print("  {s}: separate {d:.1} us, fused {d:.1} us\n", .{ c.name, @as(f64, @floatFromInt(sep)) / 1e3, @as(f64, @floatFromInt(fus)) / 1e3 });
    }
}

pub fn deviceInfo(r: *rt.Runtime) void {
    var props: @import("xpu").abi.DeviceProperties = undefined;
    props.stype = @import("xpu").abi.structure_type_device_properties;
    props.next = null;
    if (r.drv.api.zeDeviceGetProperties(r.device, &props) == 0) {
        std.debug.print("{s}: {d} slices x {d} subslices x {d} EUs, {d} threads an EU, simd {d}, clock {d} MHz\n", .{ std.mem.sliceTo(&props.name, 0), props.num_slices, props.num_subslices_per_slice, props.num_eus_per_subslice, props.num_threads_per_eu, props.physical_eu_simd_width, props.core_clock_rate });
    }
}

/// The matvec load pattern alone (bw_probe2): row groups of nb blocks x 16 lanes in balanced work-groups; time vs size.
pub fn probe2(r: *rt.Runtime) !void {
    var m = try r.module(@import("xpu").kernels.ggml_quant);
    var k = try m.kernel("bw_probe2", .{ 128, 1, 1 });
    const gtot: u32 = 1088;
    for ([_]u32{ 5, 10, 20, 40 }) |nb| {
        const bytes = @as(usize, gtot) * nb * 1792;
        var bufs: [6]rt.Buffer = undefined;
        for (&bufs) |*b| b.* = try r.alloc(bytes);
        const out = try r.alloc(64);
        if (std.c.getenv("PROBE_ZEROS") == null) { // random contents as in probeBandwidth
            const fill = try std.heap.page_allocator.alloc(u8, bytes);
            defer std.heap.page_allocator.free(fill);
            var prng = std.Random.DefaultPrng.init(13);
            prng.random().bytes(fill);
            for (bufs) |b| try r.upload(b, fill);
            try r.sync();
        }
        for ([_]u32{ 160, 256 }) |nwg| {
            try k.setBuffer(0, bufs[0]);
            try k.setBuffer(1, out);
            try k.setU32(2, nb);
            try k.setU32(3, gtot);
            try k.launch(.{ nwg, 1, 1 });
            try r.sync();
            var best: u64 = std.math.maxInt(u64);
            for (0..5) |_| {
                const t0 = nowNs();
                for (0..12) |i| {
                    try k.setBuffer(0, bufs[i % 6]);
                    try k.launch(.{ nwg, 1, 1 });
                }
                try r.sync();
                best = @min(best, (nowNs() - t0) / 12);
            }
            std.debug.print("  probe2 nb {d}, {d} work-groups: {d:.1} us, {d:.0} GB/s of {d:.1} MB\\n", .{ nb, nwg, @as(f64, @floatFromInt(best)) / 1e3, @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(best)), @as(f64, @floatFromInt(bytes)) / 1e6 });
        }
        for (&bufs) |*b| b.free();
    }
}

/// Peak resident work-groups (occ_probe) for several dynamic local memory sizes.
pub fn occProbe(r: *rt.Runtime) !void {
    var m = try r.moduleWith(@import("xpu").kernels.ggml_quant, std.c.getenv("GG_BUILD_FLAGS"));
    var k = try m.kernel("occ_probe", .{ 128, 1, 1 });
    const cnt = try r.alloc(16);
    for ([_]u32{ 0, 4096, 8192, 16384, 24576, 32768 }) |slm| {
        const zeros: [16]u8 = @splat(0);
        try r.upload(cnt, &zeros);
        try r.sync();
        try k.setBuffer(0, cnt);
        try k.setBuffer(1, gg.sub(cnt, 8));
        try k.setU32(2, 200000);
        try k.setU32(3, 1);
        try k.rt.drv.check(k.rt.drv.api.zeKernelSetArgumentValue(k.handle, 4, @max(slm, 4), null), "setLocal");
        try k.launch(.{ 1024, 1, 1 });
        try r.sync();
        var out: [4]u32 = undefined;
        try r.download(std.mem.sliceAsBytes(&out), cnt);
        try r.sync();
        std.debug.print("  dynamic local {d} B: peak resident work-groups {d} ({d} sub-groups)\n", .{ slm, out[2], out[2] * 8 });
    }
}

/// Cost of m = 1..8 rows against the same weights on the big shapes (cold weights), us a call and the ratio to m = 1.
pub fn benchRows(r: *rt.Runtime, set: *gg.Set) !void {
    const Case = struct { t: gg.Type, rows: u32, in: u32 };
    const cases = [_]Case{
        .{ .t = .iq3_s, .rows = 17408, .in = 5120 },  .{ .t = .iq3_s, .rows = 5120, .in = 17408 },  .{ .t = .iq3_s, .rows = 5120, .in = 6144 },
        .{ .t = .iq3_xxs, .rows = 17408, .in = 5120 }, .{ .t = .iq4_xs, .rows = 17408, .in = 5120 }, .{ .t = .iq4_xs, .rows = 5120, .in = 17408 },
        .{ .t = .q4_k, .rows = 17408, .in = 5120 },    .{ .t = .q2_k, .rows = 17408, .in = 5120 },   .{ .t = .iq2_xs, .rows = 17408, .in = 5120 },
        .{ .t = .iq2_xxs, .rows = 17408, .in = 5120 }, .{ .t = .iq2_s, .rows = 17408, .in = 5120 },  .{ .t = .iq1_m, .rows = 17408, .in = 5120 },
        .{ .t = .q4_k, .rows = 248320, .in = 5120 },
    };
    const x = try randomX(r, 8 * 17408);
    for (cases) |c| {
        const host = try tableBuf(try fixFor(c.t), c.rows, c.in);
        defer gpa.free(host);
        const nbuf: usize = @min(24, @max(3, (160 << 20) / host.len + 1));
        var w: [24]rt.Buffer = undefined;
        for (w[0..nbuf]) |*b| b.* = try up(r, host);
        const y = try r.alloc(@as(usize, c.rows) * 8 * 2);
        std.debug.print("  {s:<8} {d:>6} x {d:>5}:", .{ @tagName(c.t), c.rows, c.in });
        var base: f64 = 0;
        for ([_]u32{ 1, 2, 3, 4, 5, 6, 8 }) |m| {
            try set.matvecRows(c.t, w[0], x, m, y, c.in, 0, c.rows);
            try r.sync();
            var best: u64 = std.math.maxInt(u64);
            for (0..3) |_| {
                const t0 = nowNs();
                for (0..nbuf * 2) |i| try set.matvecRows(c.t, w[i % nbuf], x, m, y, c.in, 0, c.rows);
                try r.sync();
                best = @min(best, (nowNs() - t0) / (nbuf * 2));
            }
            const us = @as(f64, @floatFromInt(best)) / 1e3;
            if (m == 1) base = us;
            std.debug.print(" m{d} {d:.0}us x{d:.2}", .{ m, us, us / base });
        }
        std.debug.print("\n", .{});
        for (w[0..nbuf]) |*b| b.free();
    }
}

/// Prefill GEMM speed on the real projection shapes: ms a pass and TFLOP/s (2 * R * N * K) for R = 128, 512, 2048.
pub fn benchPrefill(r: *rt.Runtime, set: *gg.Set) !void {
    const Case = struct { t: gg.Type, rows: u32, in: u32 };
    const cases = [_]Case{
        .{ .t = .iq3_s, .rows = 17408, .in = 5120 }, .{ .t = .iq3_s, .rows = 5120, .in = 17408 }, .{ .t = .iq3_s, .rows = 5120, .in = 6144 },
        .{ .t = .iq3_xxs, .rows = 17408, .in = 5120 }, .{ .t = .iq4_xs, .rows = 17408, .in = 5120 }, .{ .t = .q4_k, .rows = 17408, .in = 5120 },
        .{ .t = .q2_k, .rows = 17408, .in = 5120 }, .{ .t = .iq2_xs, .rows = 17408, .in = 5120 }, .{ .t = .iq2_xxs, .rows = 17408, .in = 5120 },
        .{ .t = .iq2_s, .rows = 17408, .in = 5120 },
    };
    const x = try randomX(r, 2048 * 17408);
    const y = try r.alloc(2048 * 17408 * 2);
    for (cases) |c| {
        const host = try tableBuf(try fixFor(c.t), c.rows, c.in);
        defer gpa.free(host);
        const w = try up(r, host);
        std.debug.print("  {s:<8} {d:>6} x {d:>5}:", .{ @tagName(c.t), c.rows, c.in });
        for ([_]u32{ 128, 512, 2048 }) |R| {
            try set.prefillRows(c.t, w, x, R, y, c.in, 0, c.rows);
            try r.sync();
            var best: u64 = std.math.maxInt(u64);
            for (0..3) |_| {
                const t0 = nowNs();
                for (0..3) |_| try set.prefillRows(c.t, w, x, R, y, c.in, 0, c.rows);
                try r.sync();
                best = @min(best, (nowNs() - t0) / 3);
            }
            const ms = @as(f64, @floatFromInt(best)) / 1e6;
            const flops = 2.0 * @as(f64, @floatFromInt(R)) * @as(f64, @floatFromInt(c.rows)) * @as(f64, @floatFromInt(c.in));
            std.debug.print("  R{d}: {d:.2} ms {d:.1} TF", .{ R, ms, flops / (ms * 1e-3) / 1e12 });
        }
        std.debug.print("\n", .{});
    }
}
