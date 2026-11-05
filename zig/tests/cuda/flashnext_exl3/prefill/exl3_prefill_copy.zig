//! exl3-prefill copy: the expert line copies of the offload pool, timed and checked on one layer.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("../../check.zig");
const exl3e = cuda.exl3_experts;
const Gpu = check.Gpu;
const packs = @import("exl3_prefill_pack.zig");
const experts = @import("exl3_prefill_experts.zig");

const dims = 2560;
const width = 640;
const routed_n = 512;
const table_n = 513;
const slots = 11;

const LayerData = packs.LayerData;
const bf16Bits = packs.bf16Bits;
const genPicks = packs.genPicks;
const SlotsCall = experts.SlotsCall;

// ---------------------------------------------------------------------------------------------------- copies

pub fn copyRun(gpu: Gpu, ex: *const exl3e.Experts, s: cuda.Stream, ld: *const LayerData) !void {
    const a = gpu.gpa;
    // the layer's routed trellis in one pinned bank (what S9 keeps), lines laid out as Lines does
    var total: usize = 0;
    for (ld.routed) |e| for (e) |b| {
        total += b.len;
    };
    var bank = try cuda.HostBuffer.alloc(gpu.d, total);
    defer bank.free();
    const pinned = try a.alloc([3][]const u8, routed_n);
    defer a.free(pinned);
    var at: usize = 0;
    for (ld.routed, pinned) |e, *o| for (e, 0..) |b, p| {
        @memcpy(bank.bytes[at..][0..b.len], b);
        o[p] = bank.bytes[at..][0..b.len];
        at += b.len;
    };
    const n_lines: usize = 64;
    var bytes: [3]usize = .{ 0, 0, 0 };
    var k2: [routed_n][3]i32 = undefined;
    const tiles = (dims / 16) * (width / 16);
    for (pinned, &k2) |e, *w| for (0..3) |p| {
        bytes[p] = @max(bytes[p], e[p].len);
        w[p] = @intCast(e[p].len / (tiles * 16));
    };
    var lines = try cuda.exl3_pool.Lines.init(a, gpu.d, .{ .experts = table_n, .bytes = bytes, .lines = n_lines, .staging = 0 });
    defer lines.deinit();
    var c0 = try cuda.Event.init(gpu.d, true);
    defer c0.deinit();
    var c1 = try cuda.Event.init(gpu.d, true);
    defer c1.deinit();
    var raw = try cuda.DeviceBuffer.alloc(gpu.d, n_lines * lines.layout.line_bytes);
    defer raw.free();
    const moved = blk: {
        var m: usize = 0;
        for (pinned[0..n_lines]) |e| m += e[0].len + e[1].len + e[2].len;
        break :blk m;
    };
    std.debug.print("  {d} expert lines ({d:.2} MiB, {d:.3} MiB a line) from the pinned bank, best of 5:\n", .{ n_lines, @as(f64, @floatFromInt(moved)) / 1048576.0, @as(f64, @floatFromInt(moved)) / 1048576.0 / n_lines });
    var best_load: f64 = std.math.inf(f64);
    var best_raw: f64 = std.math.inf(f64);
    var best_raw1: f64 = std.math.inf(f64);
    for (0..5) |it| {
        try lines.drain();
        try c0.record(lines.copy);
        for (0..n_lines) |i| {
            const id = (it * n_lines + i) % routed_n;
            const prev = (id + routed_n - n_lines) % routed_n;
            try lines.load(i, id, pinned[id], k2[id], if (it == 0) null else prev, true, gpu.io);
        }
        try c1.record(lines.copy);
        try c1.synchronize();
        best_load = @min(best_load, try c0.elapsedMs(c1));
        // the same bytes as three H2D a line, no table writes
        try c0.record(lines.copy);
        for (0..n_lines) |i| for (0..3) |p| {
            const e = pinned[i];
            try gpu.d.check(gpu.d.api.cuMemcpyHtoDAsync_v2(raw.ptr + i * lines.layout.line_bytes + lines.layout.off[p], e[p].ptr, e[p].len, lines.copy.handle), "cuMemcpyHtoDAsync");
        };
        try c1.record(lines.copy);
        try c1.synchronize();
        best_raw = @min(best_raw, try c0.elapsedMs(c1));
        // one H2D of the bank's first n_lines experts (contiguous in the bank)
        try c0.record(lines.copy);
        try gpu.d.check(gpu.d.api.cuMemcpyHtoDAsync_v2(raw.ptr, bank.bytes.ptr, moved, lines.copy.handle), "cuMemcpyHtoDAsync");
        try c1.record(lines.copy);
        try c1.synchronize();
        best_raw1 = @min(best_raw1, try c0.elapsedMs(c1));
    }
    const gbs = struct {
        fn f(b: usize, ms: f64) f64 {
            return @as(f64, @floatFromInt(b)) / ms / 1e6;
        }
    }.f;
    std.debug.print("    Lines.load (3 H2D + 6-9 table memsets + event a line): {d:.3} ms = {d:.3} ms a line, {d:.2} GB/s\n", .{ best_load, best_load / n_lines, gbs(moved, best_load) });
    std.debug.print("    3 bare H2D a line:                                       {d:.3} ms = {d:.3} ms a line, {d:.2} GB/s\n", .{ best_raw, best_raw / n_lines, gbs(moved, best_raw) });
    std.debug.print("    one H2D of the same bytes:                               {d:.3} ms, {d:.2} GB/s\n", .{ best_raw1, gbs(moved, best_raw1) });

    // copies beside compute: a 1024-row experts call on the resident layer while the copy stream moves 64 lines
    const rows: usize = 1024;
    var prng = std.Random.DefaultPrng.init(0xc0b1);
    const rng = prng.random();
    const xh = try a.alloc(u16, rows * dims);
    defer a.free(xh);
    for (xh) |*b| b.* = bf16Bits((rng.float(f32) - 0.5) * 2.0);
    var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xh));
    defer x.free();
    const picks = try a.alloc(i32, rows * slots);
    defer a.free(picks);
    genPicks(rng, 0, rows, picks);
    var pk = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(picks));
    defer pk.free();
    var sbuf = try cuda.DeviceBuffer.alloc(gpu.d, try exl3e.scratchBytes(ld.res, rows, slots));
    defer sbuf.free();
    const sc = try exl3e.scratch(sbuf.ptr, ld.res, rows, slots);
    const call: SlotsCall = .{ .ex = ex, .s = s, .l = ld.res, .x = x.ptr, .picks = pk.ptr, .rows = rows, .sc = sc };
    var k0 = try cuda.Event.init(gpu.d, true);
    defer k0.deinit();
    var k1 = try cuda.Event.init(gpu.d, true);
    defer k1.deinit();
    var alone_c: f64 = std.math.inf(f64);
    var alone_k: f64 = std.math.inf(f64);
    var both_c: f64 = std.math.inf(f64);
    var both_k: f64 = std.math.inf(f64);
    var both_w: f64 = std.math.inf(f64);
    for (0..5) |_| {
        try k0.record(s);
        try call.go();
        try k1.record(s);
        try k1.synchronize();
        alone_k = @min(alone_k, try k0.elapsedMs(k1));
        try c0.record(lines.copy);
        try gpu.d.check(gpu.d.api.cuMemcpyHtoDAsync_v2(raw.ptr, bank.bytes.ptr, moved, lines.copy.handle), "cuMemcpyHtoDAsync");
        try c1.record(lines.copy);
        try c1.synchronize();
        alone_c = @min(alone_c, try c0.elapsedMs(c1));
        // both at once
        try s.synchronize();
        const t0 = check.now(gpu.io);
        try c0.record(lines.copy);
        try gpu.d.check(gpu.d.api.cuMemcpyHtoDAsync_v2(raw.ptr, bank.bytes.ptr, moved, lines.copy.handle), "cuMemcpyHtoDAsync");
        try c1.record(lines.copy);
        try k0.record(s);
        try call.go();
        try k1.record(s);
        try k1.synchronize();
        try c1.synchronize();
        both_w = @min(both_w, @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e6);
        both_c = @min(both_c, try c0.elapsedMs(c1));
        both_k = @min(both_k, try k0.elapsedMs(k1));
    }
    std.debug.print("  overlap: {d}-row experts call alone {d:.3} ms, 64-line H2D alone {d:.3} ms; together: experts {d:.3} ms, H2D {d:.3} ms, wall {d:.3} ms (serial would be {d:.3})\n", .{ rows, alone_k, alone_c, both_k, both_c, both_w, alone_k + alone_c });
    check.pass("layer {d}: copy cost and copy/compute overlap measured", .{ld.layer});
}
