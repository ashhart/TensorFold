//! tf-flashnext-widths, prompt products: qmmw_nax at each oQ width, dense, split-K, members and expert gathers.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const lib = @import("flashnext_widths_lib.zig");
const fz = tf.flashnext_replay;
const pw = fz.prompt_w;
const Buf = fz.Buf;
const Run = fz.Run;
const a = lib.a;

const wide = 129; // two 64-row tiles and one row more
const windows = [_]usize{ 1, 2, 3, 8, 16, 17, 33, 64, 65 };

/// A prompt with only what the any-width launches read: the run and the expert offsets.
fn prompt(r: *Run, offsets: Buf) fz.Prompt {
    var p: fz.Prompt = undefined;
    p.r = r;
    p.skip = 0;
    p.b.off = offsets;
    return p;
}

fn weights(device: mtl.Device, m: lib.Mat) ![3]Buf {
    return .{ .{ .b = try lib.upload(device, m.words) }, .{ .b = try lib.upload(device, std.mem.sliceAsBytes(m.scales)) }, .{ .b = try lib.upload(device, std.mem.sliceAsBytes(m.biases)) } };
}

/// Rows [start, start + m) of x through the dense product at `w`'s member widths, every window of m rows.
fn dense(p: *fz.Prompt, wd: *pw.Wide, w: pw.Weight, ws: [3]Buf, x: lib.Inputs, n: usize, m: usize, y: mtl.Buffer) !void {
    const cb = p.r.queue.commandBuffer();
    p.r.enc = cb.compute(.serial);
    var start: usize = 0;
    while (start + m <= wide) : (start += m) pw.qmm(p, wd, w, .{ .b = x.buf, .off = start * x.k * 2 }, ws, x.k, n, m, .{ .b = y, .off = start * n * 2 }, 0);
    try lib.commit(cb, p.r.enc);
}

/// A fused projection of `members` (rows, bits, group): float64 per member, then every window's rows bit for bit.
fn denseCheck(t: *lib.Tally, p: *fz.Prompt, wd: *pw.Wide, members: []const [3]usize, k: usize, seed: u64) !void {
    var label: std.ArrayList(u8) = .empty;
    try label.print(a, "prompt dense K {d}:", .{k});
    var mats = try a.alloc(lib.Mat, members.len);
    var parts = try a.alloc(pw.Part, members.len);
    var words: std.ArrayList(u8) = .empty;
    var scales: std.ArrayList(u8) = .empty;
    var biases: std.ArrayList(u8) = .empty;
    var n: usize = 0;
    for (members, 0..) |mb, i| {
        mats[i] = try lib.Mat.init(mb[0], k, @intCast(mb[1]), mb[2], seed + i);
        parts[i] = .{ .rows = mb[0], .bits = @intCast(mb[1]), .group = @intCast(mb[2]) };
        try words.appendSlice(a, mats[i].words);
        try scales.appendSlice(a, std.mem.sliceAsBytes(mats[i].scales));
        try biases.appendSlice(a, std.mem.sliceAsBytes(mats[i].biases));
        try label.print(a, " {d}x q{d}/g{d}", .{ mb[0], mb[1], mb[2] });
        n += mb[0];
        const kk = @as(u32, @intCast(mb[1])) << 16 | @as(u32, @intCast(mb[2]));
        if (!wd.pipes.contains(kk)) try wd.pipes.put(a, kk, try pw.compile(p.r, @intCast(mb[1]), @intCast(mb[2])));
    }
    const d = p.r.device;
    const ws: [3]Buf = .{ .{ .b = try lib.upload(d, words.items) }, .{ .b = try lib.upload(d, scales.items) }, .{ .b = try lib.upload(d, biases.items) } };
    const w: pw.Weight = if (members.len == 1) .{ .bits = parts[0].bits, .group = parts[0].group } else .{ .parts = parts };
    const x = try lib.Inputs.init(d, wide, k, seed ^ 0x9e);
    const all = try lib.poisoned(d, wide * n * 2);
    try dense(p, wd, w, ws, x, n, wide, all);
    var worst = lib.Worst{};
    var ctrl = lib.Worst{};
    const got = all.slice(u16, wide * n);
    var col: usize = 0;
    for (mats) |m| {
        // each weight rounds to bf16 before its fp32 product (qmmw's dequantw): the reference rounds it the same way
        const terms = try m.denseBf();
        defer for (terms) |v| a.free(v);
        const wrong = try m.skewed(1.01).denseBf();
        defer for (wrong) |v| a.free(v);
        for (0..m.n) |j| for (0..wide) |r| {
            const g = lib.fromBf(got[r * n + col + j]);
            const ref = lib.dot3(x.row(r), terms, j);
            worst.add(col + j, g, ref[0], lib.ratio3(g, ref, k + 64, true));
            const cref = lib.dot3(x.row(r), wrong, j);
            ctrl.add(col + j, g, cref[0], lib.ratio3(g, cref, k + 64, true));
        };
        col += m.n;
    }
    worst.report(t, label.items);
    ctrl.control(t, label.items);
    var same = true;
    for (windows) |m| {
        const y = try lib.poisoned(d, wide * n * 2);
        try dense(p, wd, w, ws, x, n, m, y);
        same = lib.sameRows(t, label.items, m, all.slice(u8, wide * n * 2), y.slice(u8, wide * n * 2), wide / m * m, n * 2) and same;
    }
    if (same) std.debug.print("RESULT {s}: every row's bits equal in windows of 1, 2, 3, 8, 16, 17, 33, 64 and 65 rows\n", .{label.items});
}

/// The generic text at 6/32 against the recorded tf_qmm6_t_nax on the same weights: the same bits.
fn recordedTwin(t: *lib.Tally, p: *fz.Prompt, wd: *pw.Wide) !void {
    const d = p.r.device;
    const m = try lib.Mat.init(192, 2560, 6, 32, 0x632);
    const ws = try weights(d, m);
    const x = try lib.Inputs.init(d, wide, 2560, 0x633);
    try wd.pipes.put(a, 6 << 16 | 32, try pw.compile(p.r, 6, 32));
    const gen = try lib.poisoned(d, wide * 192 * 2);
    try dense(p, wd, .{ .bits = 6, .group = 32 }, ws, x, 192, wide, gen);
    const text = try tf.flashnext_replay.frags.source(d, a, pw.ks.flashnext_qmm6);
    const rlib = try mtl.Library.fromSource(d, text, mtl.CompileOptions.mlx());
    const pipe = try mtl.Pipeline.init(d, rlib, "tf_qmm6_t_nax", false);
    const rec = try lib.poisoned(d, wide * 192 * 2);
    const cb = p.r.queue.commandBuffer();
    p.r.enc = cb.compute(.serial);
    p.bind(pipe, &.{ ws[0], ws[1], ws[2], .{ .b = x.buf } });
    const prm = [4]i32{ 2560, 192, wide, 0 };
    p.r.enc.setBytes(std.mem.asBytes(&prm), 4);
    p.r.enc.setBuffer(rec, 0, 5);
    p.r.enc.dispatchThreads(mtl.Size.of(3 * 128, 3, 1), mtl.Size.of(128, 1, 1));
    try lib.commit(cb, p.r.enc);
    if (lib.sameRows(t, "prompt recorded qmm6 [192, 2560] against the generic text at 6/32", wide, rec.slice(u8, wide * 384), gen.slice(u8, wide * 384), wide, 384))
        std.debug.print("RESULT prompt qmm6 [192, 2560]: the generic text's bits equal the recorded kernel's\n", .{});
}

/// The hyper-connection's down projection as the prompt splits it: 8 K parts of fp32 partials, summed against float64.
fn splitCheck(t: *lib.Tally, p: *fz.Prompt, wd: *pw.Wide, bits: u8, group: usize, seed: u64) !void {
    var label_buf: [96]u8 = undefined;
    const label = try std.fmt.bufPrint(&label_buf, "prompt split-K [324, 10240] q{d}/g{d}", .{ bits, group });
    const n = 324;
    const k = 10240;
    const d = p.r.device;
    const m = try lib.Mat.init(n, k, bits, group, seed);
    defer m.deinit();
    const ws = try weights(d, m);
    const x = try lib.Inputs.init(d, wide, k, seed + 1);
    const kk = @as(u32, bits) << 16 | @as(u32, @intCast(group));
    if (!wd.pipes.contains(kk)) try wd.pipes.put(a, kk, try pw.compile(p.r, bits, @intCast(group)));
    const part = try lib.poisoned(d, 8 * wide * n * 4);
    const cb = p.r.queue.commandBuffer();
    p.r.enc = cb.compute(.serial);
    p.bind(wd.of(bits, @intCast(group)).splitk, &.{ ws[0], ws[1], ws[2], .{ .b = x.buf } });
    const prm = [4]i32{ k, n, wide, 8 };
    p.r.enc.setBytes(std.mem.asBytes(&prm), 4);
    p.r.enc.setBuffer(part, 0, 5);
    p.r.enc.dispatchThreads(mtl.Size.of(((n + 63) / 64) * 128, (wide + 63) / 64, 8), mtl.Size.of(128, 1, 1));
    try lib.commit(cb, p.r.enc);
    const terms = try m.denseBf();
    defer for (terms) |v| a.free(v);
    const wrong = try m.skewed(1.01).denseBf();
    defer for (wrong) |v| a.free(v);
    const got = part.slice(f32, 8 * wide * n);
    var worst = lib.Worst{};
    var ctrl = lib.Worst{};
    for (0..wide) |r| for (0..n) |j| {
        var sum: f64 = 0;
        for (0..8) |z| sum += got[(z * wide + r) * n + j];
        // each part is an fp32 sum of K / 8 products (any order: at most that many steps), the parts added here exactly
        worst.add(j, sum, lib.dot3(x.row(r), terms, j)[0], lib.ratio3(sum, lib.dot3(x.row(r), terms, j), k / 8 + 64, false));
        ctrl.add(j, sum, lib.dot3(x.row(r), wrong, j)[0], lib.ratio3(sum, lib.dot3(x.row(r), wrong, j), k / 8 + 64, false));
    };
    worst.report(t, label);
    ctrl.control(t, label);
}

/// Sorted-expert gathers (rows grouped by expert, 8 experts used): the product and the fused gate/up activation.
fn gatherCheck(t: *lib.Tally, p: *fz.Prompt, wd: *pw.Wide, bits: u8, group: usize, seed: u64) !void {
    var label_buf: [96]u8 = undefined;
    const label = try std.fmt.bufPrint(&label_buf, "prompt expert gathers [8 x 128, 2560] q{d}/g{d}", .{ bits, group });
    const e_n = 8;
    const n = 128;
    const k = 2560;
    const d = p.r.device;
    const gm = try lib.Mat.init(e_n * n, k, bits, group, seed);
    const um = try lib.Mat.init(e_n * n, k, bits, group, seed + 1);
    defer gm.deinit();
    defer um.deinit();
    const gw = try weights(d, gm);
    const uw = try weights(d, um);
    const x = try lib.Inputs.init(d, wide, k, seed + 2);
    const off = p.b.off.b.slice(i32, 512);
    var expert: [wide]usize = undefined;
    for (&expert, 0..) |*e, i| e.* = i * e_n / wide; // rows sorted by expert
    for (off, 0..) |*o, e| {
        o.* = wide;
        for (expert, 0..) |x_e, i| if (x_e >= e) {
            o.* = @intCast(i);
            break;
        };
    }
    const kk = @as(u32, bits) << 16 | @as(u32, @intCast(group));
    if (!wd.pipes.contains(kk)) try wd.pipes.put(a, kk, try pw.compile(p.r, bits, @intCast(group)));
    const pipes = wd.of(bits, @intCast(group));
    const y = try lib.poisoned(d, wide * n * 2);
    const act = try lib.poisoned(d, wide * n * 2);
    const cb = p.r.queue.commandBuffer();
    p.r.enc = cb.compute(.serial);
    pw.gather(p, pipes, 32, .{ .b = x.buf }, &gw, k, n, wide, .{ .b = y });
    pw.gatherGU(p, pipes, 32, .{ .b = x.buf }, &gw, &uw, k, n, wide, .{ .b = act });
    try lib.commit(cb, p.r.enc);
    var worst = [2]lib.Worst{ .{}, .{} }; // product, gate/up activation
    var ctrl = [2]lib.Worst{ .{}, .{} };
    for ([_]f64{ 1, 1.01 }, 0..) |skew, which| {
        const gt = try gm.skewed(skew).denseBf();
        defer for (gt) |v| a.free(v);
        const ut = try um.skewed(skew).denseBf();
        defer for (ut) |v| a.free(v);
        const w = if (which == 0) &worst else &ctrl;
        for (0..wide) |r| for (0..n) |j| {
            const g = lib.dot3(x.row(r), gt, expert[r] * n + j);
            const u = lib.dot3(x.row(r), ut, expert[r] * n + j);
            const got = lib.fromBf(y.slice(u16, wide * n)[r * n + j]);
            w[0].add(j, got, g[0], lib.ratio3(got, g, k + 64, true));
            const sig = 1 / (1 + @exp(-g[0]));
            const want = g[0] * sig * u[0];
            // the sums' fp32 and tie bounds, then the gate's bf16 rounding; silu's slope is below 1.1
            const dg = @as(f64, k + 64) * lib.unit * g[1] + g[2] + 0x1p-8 * @abs(g[0]);
            const du = @as(f64, k + 64) * lib.unit * u[1] + u[2] + 0x1p-8 * @abs(u[0]);
            const tol = 1.1 * @abs(u[0]) * dg + @abs(g[0] * sig) * du + 0x1p-7 * @abs(want) + 1e-6;
            const ga = lib.fromBf(act.slice(u16, wide * n)[r * n + j]);
            w[1].add(j, ga, want, @abs(ga - want) / tol);
        };
    }
    var act_label: [128]u8 = undefined;
    const gu = try std.fmt.bufPrint(&act_label, "{s} gate/up", .{label});
    worst[0].report(t, label);
    ctrl[0].control(t, label);
    worst[1].report(t, gu);
    ctrl[1].control(t, gu);
}

pub fn run(t: *lib.Tally, r: *Run) !void {
    const offsets = try r.buffer(512 * 4);
    var p = prompt(r, .{ .b = offsets });
    const wd = try a.create(pw.Wide);
    wd.* = .{};
    try recordedTwin(t, &p, wd);
    var seed: u64 = 0x9000;
    for ([_]usize{ 2, 3, 4, 5, 6, 8 }) |bits| for ([_]usize{ 32, 64, 128 }) |group| {
        seed += 8;
        try denseCheck(t, &p, wd, &.{.{ 192, bits, group }}, 2560, seed);
        try denseCheck(t, &p, wd, &.{.{ 64, bits, group }}, 640, seed + 4);
        try splitCheck(t, &p, wd, @intCast(bits), group, seed + 5);
        try gatherCheck(t, &p, wd, @intCast(bits), group, seed + 6);
    };
    // the oQ fused stacks, members at their own widths (q, k, v, index_qk and the DeltaNet inputs, rows cut down)
    try denseCheck(t, &p, wd, &.{ .{ 256, 3, 64 }, .{ 32, 3, 64 }, .{ 32, 8, 64 }, .{ 64, 3, 64 } }, 2560, 0x9a01);
    try denseCheck(t, &p, wd, &.{ .{ 256, 4, 64 }, .{ 64, 5, 128 }, .{ 32, 5, 128 }, .{ 32, 6, 64 } }, 2560, 0x9a02);
    try denseCheck(t, &p, wd, &.{ .{ 128, 6, 64 }, .{ 32, 6, 64 }, .{ 32, 6, 128 }, .{ 64, 6, 64 } }, 2560, 0x9a03);
    // oQ4e layer 21's DeltaNet inputs: the 5-bit run's b and the 6-bit a keep their 48 rows
    try denseCheck(t, &p, wd, &.{ .{ 256, 4, 64 }, .{ 64, 5, 128 }, .{ 48, 5, 128 }, .{ 48, 6, 64 } }, 2560, 0x9a04);
    // the prompt's other shapes: HC up (K 320), N 10240, and att.o and gdn.out (K 6144) at their oQ widths
    for ([_]u8{ 2, 4, 8 }, 0..) |bits, i| try denseCheck(t, &p, wd, &.{.{ 256, bits, 64 }}, 320, 0x9b00 + i);
    for ([_][2]usize{ .{ 3, 64 }, .{ 4, 64 }, .{ 5, 128 }, .{ 8, 64 } }, 0..) |w, i| try denseCheck(t, &p, wd, &.{.{ 64, w[0], w[1] }}, 6144, 0x9b10 + i);
    try denseCheck(t, &p, wd, &.{.{ 10240, 3, 64 }}, 320, 0x9b20);
}
