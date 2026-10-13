//! tf-flashnext-widths, decode rows: rows_w's per-row products against float64, each row's bits in every window.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const lib = @import("flashnext_widths_lib.zig");
const lanes = @import("flashnext_widths_lane.zig");
const rows_w = tf.flashnext_rows.rows_w;
const lane_w = tf.flashnext_replay.lane_w;
const pack = tf.flashnext_pack;
const Buf = tf.flashnext_replay.Buf;
const a = lib.a;

const wide = 64; // rows of the reference call: the 32-row call split and the MMA twin's 8-row tiles both cross
const windows = [_]usize{ 1, 2, 3, 4, 5, 8, 9, 16, 17, 33 }; // 1-3 per-row loop, 4+ the MMA twin, 33 crosses 32

fn product(device: mtl.Device, m: lib.Mat) !rows_w.Product {
    const s = try lib.upload(device, std.mem.sliceAsBytes(m.scales));
    const b = try lib.upload(device, std.mem.sliceAsBytes(m.biases));
    return .{ .w = .{ .b = try lib.upload(device, m.words) }, .s = .{ .b = s }, .b = .{ .b = b }, .n = m.n, .k = m.k, .bits = m.bits, .group = @intCast(m.group) };
}

/// Rows [0, wide) of x through `p` in windows of m rows, each window's outputs at its rows.
fn launchWindows(device: mtl.Device, queue: mtl.Queue, cache: *rows_w.Cache, p: rows_w.Product, x: lib.Inputs, m: usize, y: mtl.Buffer, sums: mtl.Buffer) !void {
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    var start: usize = 0;
    while (start + m <= wide) : (start += m) {
        try rows_w.encode(e, a, device, cache, p, .{ .b = x.buf, .off = start * x.k * 2 }, m, .{ .b = y, .off = start * p.n * 2 }, p.n, .{ .b = sums }, .auto, false);
    }
    try lib.commit(cb, e);
}

/// x's bf16 sums of each aligned run of 4, as MLX's 4-bit loop takes them before its fp32 sum.
fn runSums(x: []const f64) ![]f64 {
    const out = try a.alloc(f64, x.len / 4);
    for (out, 0..) |*o, i| {
        var s: f32 = @floatCast(x[i * 4]);
        for (1..4) |t| s = @floatCast(lib.fromBf(lib.toBf(s + @as(f32, @floatCast(x[i * 4 + t])))));
        o.* = s;
    }
    return out;
}

/// Column j's float64 product with x and its magnitude; the bias meets x's bf16 run sums when `runs` is given.
fn column(m: lib.Mat, x: []const f64, runs: ?[]const f64, j: usize) [2]f64 {
    const gn = m.k / m.group;
    var want: f64 = 0;
    var mag: f64 = 0;
    for (0..gn) |g| {
        const s = m.skew * lib.fromBf(m.scales[j * gn + g]);
        const b = lib.fromBf(m.biases[j * gn + g]);
        var dq: f64 = 0;
        var dm: f64 = 0;
        var xs: f64 = 0;
        var xm: f64 = 0;
        for (g * m.group..(g + 1) * m.group) |i| {
            const q: f64 = @floatFromInt(m.code(j, i));
            dq += q * x[i];
            dm += q * @abs(x[i]);
            xs += x[i];
            xm += @abs(x[i]);
        }
        if (runs) |rs| {
            xs = 0;
            for (g * m.group / 4..(g + 1) * m.group / 4) |t| xs += rs[t];
        }
        want += s * dq + b * xs;
        mag += s * dm + @abs(b) * xm;
    }
    return .{ want, mag };
}

/// The columns checked against float64: all up to 1024, else the first and last 128 and 256 strided.
fn columns(n: usize) ![]usize {
    if (n <= 1024) {
        const out = try a.alloc(usize, n);
        for (out, 0..) |*c, i| c.* = i;
        return out;
    }
    const out = try a.alloc(usize, 512);
    for (0..128) |i| {
        out[i] = i;
        out[128 + i] = n - 128 + i;
    }
    for (0..256) |i| out[256 + i] = 128 + i * (n - 256) / 256;
    return out;
}

/// One matrix: 64 rows against float64 (and a 1%-off-scales control), then every window's rows bit for bit.
fn matrix(t: *lib.Tally, device: mtl.Device, queue: mtl.Queue, cache: *rows_w.Cache, n: usize, k: usize, bits: u8, group: usize, seed: u64) !void {
    const m = try lib.Mat.init(n, k, bits, group, seed);
    defer m.deinit();
    defer a.free(m.codes);
    const p = try product(device, m);
    const label = try std.fmt.allocPrint(a, "rows [{d}, {d}] q{d}/g{d}", .{ n, k, bits, group });
    const x = try lib.Inputs.init(device, wide, k, seed ^ 0x5eed);
    const sums = try device.buffer(rows_w.sumsBytes(k, bits, wide), lib.opts);
    const all = try lib.poisoned(device, wide * n * 2);
    try launchWindows(device, queue, cache, p, x, wide, all, sums);
    const got = all.slice(u16, wide * n);
    const terms = rows_w.vpt(bits) + k / (32 * rows_w.vpt(bits)) + 16;
    var worst = lib.Worst{};
    var ctrl = lib.Worst{};
    const wrong = m.skewed(1.01);
    for (0..wide) |r| {
        const runs = if (rows_w.q4(p)) try runSums(x.row(r)) else null;
        defer if (runs) |rs| a.free(rs);
        for (try columns(n)) |j| {
            const g = lib.fromBf(got[r * n + j]);
            const ref = column(m, x.row(r), runs, j);
            worst.add(j, g, ref[0], lib.ratio(g, ref[0], ref[1], terms, true));
            const cref = column(wrong, x.row(r), runs, j);
            ctrl.add(j, g, cref[0], lib.ratio(g, cref[0], cref[1], terms, true));
        }
    }
    worst.report(t, label);
    ctrl.control(t, label);
    var same = true;
    for (windows) |w| {
        const y = try lib.poisoned(device, wide * n * 2);
        defer y.deinit();
        try launchWindows(device, queue, cache, p, x, w, y, sums);
        same = lib.sameRows(t, label, w, all.slice(u8, wide * n * 2), y.slice(u8, wide * n * 2), wide / w * w, n * 2) and same;
    }
    if (same) std.debug.print("RESULT {s}: every row's bits equal in windows of 1, 2, 3, 4, 5, 8, 9, 16, 17 and 33 rows\n", .{label});
    all.deinit();
    sums.deinit();
    x.buf.deinit();
}

/// A fused projection as the engine attaches it: every member's columns equal its own product's, in every window.
fn stack(t: *lib.Tally, device: mtl.Device, queue: mtl.Queue, cache: *rows_w.Cache, widths: []const [3]usize, k: usize, seed: u64) !void {
    const sh = try a.create(rows_w.Shared);
    const sums = rows_w.Shared.max_rows * (rows_w.Shared.max_k / 8) * 4;
    sh.* = .{ .a = a, .device = device, .sums = .{ .{ .b = try device.buffer(sums, lib.opts) }, .{ .b = try device.buffer(sums, lib.opts) } } };
    const members = try a.alloc(rows_w.Member, widths.len);
    var label: std.ArrayList(u8) = .empty;
    try label.print(a, "rows stack K {d}:", .{k});
    var n: usize = 0;
    for (members, widths, 0..) |*mb, wd, i| {
        const m = try lib.Mat.init(wd[0], k, @intCast(wd[1]), wd[2], seed + i);
        mb.* = .{ .p = try product(device, m), .col = n };
        n += wd[0];
        try label.print(a, " {d}x q{d}/g{d}", .{ wd[0], wd[1], wd[2] });
    }
    const d = rows_w.Dense{ .members = members, .n = n, .k = k, .shared = sh };
    const x = try lib.Inputs.init(device, wide, k, seed ^ 0x5eed);
    const alone = try lib.poisoned(device, wide * n * 2); // each member on its own, written at its columns
    {
        const sb = try device.buffer(rows_w.sumsBytes(k, 8, wide), lib.opts);
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        for (members) |mb| try rows_w.encode(e, a, device, cache, mb.p, .{ .b = x.buf }, wide, .{ .b = alone, .off = mb.col * 2 }, n, .{ .b = sb }, .auto, false);
        try lib.commit(cb, e);
    }
    var same = true;
    for (windows) |w| {
        const y = try lib.poisoned(device, wide * n * 2);
        defer y.deinit();
        const cb = queue.commandBuffer();
        const e = cb.compute(.concurrent);
        var start: usize = 0;
        while (start + w <= wide) : (start += w) {
            try rows_w.encodeDense(e, &d, .{ .b = x.buf, .off = start * k * 2 }, w, .{ .b = y, .off = start * n * 2 }, .auto, true);
            e.barrier();
        }
        try lib.commit(cb, e);
        same = lib.sameRows(t, label.items, w, alone.slice(u8, wide * n * 2), y.slice(u8, wide * n * 2), wide / w * w, n * 2) and same;
    }
    if (same) std.debug.print("RESULT {s}: every member's columns equal its own product's in every window\n", .{label.items});
}

pub fn run(t: *lib.Tally, device: mtl.Device, queue: mtl.Queue) !void {
    var cache: rows_w.Cache = .empty;
    var seed: u64 = 0x2a0e;
    for ([_]u8{ 2, 3, 4, 5, 6, 8 }) |bits| for ([_]usize{ 32, 64, 128 }) |group| {
        for ([_][2]usize{ .{ 256, 2560 }, .{ 64, 6144 }, .{ 48, 2560 }, .{ 8, 512 } }) |shape| {
            seed += 16;
            try matrix(t, device, queue, &cache, shape[0], shape[1], bits, group, seed);
        }
    };
    // the decode shapes at the widths the public checkpoints give them; head and draft rows at 8/64
    const real = [_][4]usize{
        .{ 13952, 2560, 6, 64 }, .{ 2560, 6144, 5, 128 },  .{ 10240, 2560, 4, 32 }, .{ 6240, 2560, 5, 128 },
        .{ 16480, 2560, 6, 32 }, .{ 2560, 6144, 6, 32 },   .{ 12800, 2560, 3, 64 }, .{ 2560, 2560, 4, 64 },
        .{ 79616, 2560, 8, 64 }, .{ 248320, 2560, 8, 64 }, .{ 96, 2560, 6, 64 },    .{ 48, 2560, 5, 128 },
    };
    for (real) |c| {
        seed += 16;
        try matrix(t, device, queue, &cache, c[0], c[1], @intCast(c[2]), c[3], seed);
    }
    // the public checkpoints' mixed stacks (rows cut down), two lane steps sharing scratch, and a 64-row window
    const stacks = [_][]const [3]usize{
        &.{ .{ 512, 4, 32 }, .{ 128, 5, 128 }, .{ 48, 5, 128 }, .{ 48, 6, 64 } },
        &.{ .{ 256, 3, 64 }, .{ 32, 3, 64 }, .{ 32, 8, 64 }, .{ 64, 3, 64 } },
        &.{ .{ 256, 6, 64 }, .{ 64, 5, 64 }, .{ 32, 6, 128 }, .{ 64, 2, 32 } },
    };
    for (stacks) |st| {
        seed += 16;
        try stack(t, device, queue, &cache, st, 2560, seed);
    }
}

/// Seconds a launch of `encodeOne` takes over `copies` rotating weight copies: the median of 3 command buffers.
fn timed(queue: mtl.Queue, copies: usize, ctx: anytype, comptime encodeOne: anytype) !f64 {
    const iters = @max(copies, 24);
    var best: [3]f64 = undefined;
    for (&best) |*b| {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        for (0..iters) |i| try encodeOne(ctx, e, i % copies);
        try lib.commit(cb, e);
        b.* = cb.gpuSeconds() / @as(f64, @floatFromInt(iters));
    }
    std.mem.sort(f64, &best, {}, std.sort.asc(f64));
    return best[1];
}

const peak = 800.0; // GB/s: the M3 Ultra's memory bandwidth, the roofline a weight-bound launch is held to

const Case = struct {
    device: mtl.Device,
    rows_cache: *rows_w.Cache,
    lane_cache: *lane_w.Cache,
    prods: []rows_w.Product,
    wides: []lane_w.Wide,
    x: Buf,
    y: Buf,
    sums: Buf,
    md: Buf,
    rows: usize,

    fn rowsOne(c: *const Case, e: mtl.ComputeEncoder, i: usize) !void {
        const p = c.prods[i];
        try rows_w.encode(e, a, c.device, c.rows_cache, p, c.x, c.rows, c.y, p.n, c.sums, .auto, false);
    }

    fn laneOne(c: *const Case, e: mtl.ComputeEncoder, i: usize) !void {
        try lane_w.encode(e, a, c.device, c.lane_cache, &c.wides[i], c.x, c.md, c.rows, c.y);
    }
};

/// rows-time: GPU time a launch of the lane kernel and of the row kernels takes at decode shapes, 1 to 16 rows.
pub fn time(device: mtl.Device, queue: mtl.Queue) !void {
    var rows_cache: rows_w.Cache = .empty;
    var lane_cache: lane_w.Cache = .empty;
    const cases = [_][4]usize{
        .{ 10240, 2560, 4, 32 }, .{ 6240, 2560, 5, 128 },  .{ 2560, 6144, 5, 128 }, .{ 13952, 2560, 6, 64 },
        .{ 2560, 6144, 6, 64 },  .{ 16480, 2560, 6, 32 },  .{ 2560, 6144, 6, 32 },  .{ 13952, 2560, 3, 64 },
        .{ 79616, 2560, 8, 64 }, .{ 248320, 2560, 8, 64 },
    };
    std.debug.print("TIME shape width rows: lane us (GB/s) | rows us (GB/s, % of {d:.0}) | lane/rows\n", .{peak});
    for (cases, 0..) |c, ci| {
        const n = c[0];
        const k = c[1];
        const m = try lib.Mat.init(n, k, @intCast(c[2]), c[3], 0x7100 + ci);
        const bytes = m.words.len + 4 * m.scales.len;
        const copies = @min(256, @max(1, (1 << 30) / bytes + 1)); // past the system cache, as layers are in decode
        var out = pack.Out.init(a);
        try pack.lane.putLane(&out, a, "t", &.{try lanes.q(m)}); // the engine's pack: group 128 read as 64
        const prods = try a.alloc(rows_w.Product, copies);
        const wides = try a.alloc(lane_w.Wide, copies);
        for (prods, wides) |*p, *w| {
            p.* = try product(device, m);
            w.* = try lanes.wideOf(device, &out, k, n);
        }
        a.free(m.codes);
        m.deinit();
        const x = try lib.Inputs.init(device, 16, k, 0x7200 + ci);
        const y = try device.buffer(16 * n * 2, lib.opts);
        const sums = try device.buffer(rows_w.sumsBytes(k, @intCast(c[2]), 16), lib.opts);
        for ([_]usize{ 1, 2, 4, 9, 16 }) |rows| {
            const case = Case{ .device = device, .rows_cache = &rows_cache, .lane_cache = &lane_cache, .prods = prods, .wides = wides, .x = .{ .b = x.buf }, .y = .{ .b = y }, .sums = .{ .b = sums }, .md = try lanes.mdims(device, rows), .rows = rows };
            const tl = try timed(queue, copies, &case, Case.laneOne);
            const tr = try timed(queue, copies, &case, Case.rowsOne);
            const gb = @as(f64, @floatFromInt(bytes)) / 1e9;
            std.debug.print("TIME [{d}, {d}] q{d}/g{d} {d:>2} rows: {d:>8.1} ({d:>5.0}) | {d:>8.1} ({d:>5.0}, {d:>3.0}%) | {d:.3}\n", .{ n, k, c[2], c[3], rows, tl * 1e6, gb / tl, tr * 1e6, gb / tr, 100 * gb / tr / peak, tl / tr });
        }
        for (prods, wides) |p, w| {
            for ([_]Buf{ p.w, p.s, p.b, w.runs[0].wq, w.runs[0].sbt }) |b| b.b.deinit();
        }
        x.buf.deinit();
        y.deinit();
        sums.deinit();
    }
}
