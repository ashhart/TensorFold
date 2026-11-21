//! tf-glm-prompt-select: prompt index scores against main's, key lists against mark and merge: bits and GPU time.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const tf = @import("tensorfold");
const kernels = tf.glm.kernels;
const st = tf.glm.state;

const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
const a = std.heap.page_allocator;

const HI = 32;
const DI = 128;
const QR = 20480; // qr_proj rows: 64 heads' q_nope, then the indexer's queries
const QOFF = 64 * 256;
const TOP = 2048;
const KPOOL = 4;
const WIDTH = TOP + KPOOL - 1;
const ROWS = 256; // prompt.zig's select_rows
/// Main's glm_index_nax.metal: 64-row tiles of 4 simdgroups, 32 rows x 32 blocks each.
const main_index = @embedFile("contract/baselines/glm_index_nax_main.metal.txt");

const ScoreArgs = extern struct { p0: u32, q_stride: u32, w_stride: u32, s_stride: u32, rows: u32 };
const SelectArgs = extern struct { p0: u32, top: u32, width: u32, s_stride: u32, i_stride: u32 };

fn buf(device: mtl.Device, bytes: usize) !mtl.Buffer {
    return device.buffer(@max(bytes, 16), opts);
}

fn toBf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

fn fillBf(b: mtl.Buffer, n: usize, seed: u64, amp: f32) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    for (b.slice(u16, n)) |*v| v.* = toBf((r.float(f32) - 0.5) * 2 * amp);
}

const Rig = struct {
    queue: mtl.Queue,
    k: *const kernels.Kernels,
    main_scores: mtl.Pipeline,
    cap: u32,
    s_stride: u32,
    words: u32,
    pool: mtl.Buffer,
    iq: mtl.Buffer,
    iw: mtl.Buffer,
    scores: mtl.Buffer,
    indices: mtl.Buffer,
    bits: mtl.Buffer,
};

const Op = enum { scores, rows64, rows128, main, old, mark, merge };

fn encode(e: mtl.ComputeEncoder, r: *const Rig, op: Op, p0: u32) void {
    const k = r.k;
    const most = (p0 + ROWS) / KPOOL;
    const sel = SelectArgs{ .p0 = p0, .top = TOP / KPOOL, .width = WIDTH, .s_stride = r.s_stride, .i_stride = WIDTH };
    switch (op) {
        .scores, .rows64, .rows128, .main => {
            e.setPipeline(if (op == .main) r.main_scores else k.index_scores_nax);
            e.setBuffer(r.iq, QOFF * 2, 0);
            e.setBuffer(r.iw, 0, 1);
            e.setBuffer(r.pool, 0, 2);
            e.setBuffer(r.scores, 0, 3);
            e.setValue([5]u32{ p0, QR, HI, r.s_stride, ROWS }, 4);
            const tr: u32 = switch (op) { // prompt.zig's tiles, either height, main's
                .scores => tf.glm.prompt.scoreRows(most, ROWS),
                .rows128 => 128,
                else => 64,
            };
            e.dispatchGroups(mtl.Size.of((most + 63) / 64, (ROWS + tr - 1) / tr, 1), mtl.Size.of(2 * tr, 1, 1));
        },
        .old => {
            e.setPipeline(k.index_select);
            e.setBuffer(r.scores, 0, 0);
            e.setBuffer(r.indices, 0, 1);
            e.setValue(sel, 2);
            e.dispatchGroups(mtl.Size.of(ROWS, 1, 1), mtl.Size.of(1024, 1, 1));
        },
        .mark => if (most > 3072) {
            e.setPipeline(k.select_mark);
            e.setBuffer(r.scores, 0, 0);
            e.setBuffer(r.bits, 0, 1);
            e.setValue(sel, 2);
            e.setValue(r.words, 3);
            e.dispatchGroups(mtl.Size.of((most + 4095) / 4096, ROWS, 1), mtl.Size.of(256, 1, 1));
        },
        .merge => {
            e.setPipeline(k.select_merge);
            e.setBuffer(r.scores, 0, 0);
            e.setBuffer(r.indices, 0, 1);
            e.setValue(sel, 2);
            e.setValue(r.words, 3);
            e.setBuffer(r.bits, 0, 4);
            e.dispatchGroups(mtl.Size.of(ROWS, 1, 1), mtl.Size.of(512, 1, 1));
        },
    }
}

fn submit(r: *const Rig, ops: []const Op, p0: u32, count: usize) !f64 {
    const cb = r.queue.commandBuffer();
    const e = cb.compute(.serial);
    for (0..count) |_| for (ops) |op| encode(e, r, op, p0);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("GPU failed: {s}\n", .{msg});
        return error.GpuFailed;
    }
    return cb.gpuSeconds() * 1e6 / @as(f64, @floatFromInt(count));
}

/// Median GPU microseconds of `ops` once.
fn time(r: *const Rig, ops: []const Op, p0: u32) !f64 {
    var t: [8]f64 = undefined;
    for (&t) |*v| v.* = try submit(r, ops, p0, 6);
    std.mem.sort(f64, t[1..], {}, std.sort.asc(f64));
    return t[4];
}

pub fn main() !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const k = try kernels.load(a, device);
    const main_src = try tf.frags.source(device, a, main_index);
    const main_lib = try mtl.Library.fromSource(device, main_src, mtl.CompileOptions.mlx());
    const cap: u32 = 131072;
    const s_stride = cap / KPOOL + 1;
    const r = Rig{
        .queue = queue,
        .k = k,
        .main_scores = try mtl.Pipeline.init(device, main_lib, "glm_index_scores_nax", false),
        .cap = cap,
        .s_stride = s_stride,
        .words = st.selectWords(cap, KPOOL),
        .pool = try buf(device, @as(usize, s_stride) * DI * 2),
        .iq = try buf(device, @as(usize, ROWS) * QR * 2),
        .iw = try buf(device, @as(usize, ROWS) * HI * 2),
        .scores = try buf(device, @as(usize, ROWS) * s_stride * 4),
        .indices = try buf(device, @as(usize, ROWS) * WIDTH * 4),
        .bits = try buf(device, @as(usize, ROWS) * st.selectWords(cap, KPOOL) * 4),
    };
    fillBf(r.pool, @as(usize, s_stride) * DI, 1, 1.0);
    fillBf(r.iq, @as(usize, ROWS) * QR, 2, 1.0);
    fillBf(r.iw, @as(usize, ROWS) * HI, 3, 0.2);
    var failures: usize = 0;
    var pit = std.mem.tokenizeScalar(u8, std.mem.span(platform.c.getenv("GLM_POS") orelse "2048,4096,12288,16384,32768,65536,90000,120000"), ',');
    const n = @as(usize, ROWS) * WIDTH;
    const want = try a.alloc(i32, n);
    defer a.free(want);
    while (pit.next()) |pv| {
        const p0 = try std.fmt.parseInt(u32, pv, 10);
        if (p0 + ROWS + 1 > cap) continue;
        const sn = @as(usize, ROWS) * r.s_stride;
        @memset(r.scores.slice(u32, sn), 0x7fc00001);
        _ = try submit(&r, &.{.main}, p0, 1);
        const theirs = try a.dupe(u32, r.scores.slice(u32, sn));
        defer a.free(theirs);
        @memset(r.scores.slice(u32, sn), 0x7fc00001);
        _ = try submit(&r, &.{ .scores, .old }, p0, 1);
        const same_scores = std.mem.eql(u32, theirs, r.scores.slice(u32, sn));
        if (!same_scores) failures += 1;
        @memcpy(want, r.indices.slice(i32, n));
        @memset(r.indices.slice(i32, n), 0x5a5a5a5a);
        _ = try submit(&r, &.{ .mark, .merge }, p0, 1);
        const ok = std.mem.eql(i32, want, r.indices.slice(i32, n));
        if (!ok) failures += 1;
        const t_main = try time(&r, &.{.main}, p0);
        const t_new = try time(&r, &.{.scores}, p0);
        const t64 = try time(&r, &.{.rows64}, p0);
        const t128 = try time(&r, &.{.rows128}, p0);
        const to = try time(&r, &.{.old}, p0);
        const tm = try time(&r, &.{.mark}, p0);
        const tg = try time(&r, &.{.merge}, p0);
        std.debug.print("pos {d:6}, {d} rows: lists {s}, scores {s}; scores {d:7.1} -> {d:7.1} us (64 rows {d:7.1}, 128 rows {d:7.1}), select {d:7.1} -> {d:7.1} us (mark {d:6.1}, merge {d:6.1})\n", .{ p0, ROWS, if (ok) "equal" else "DIFFER", if (same_scores) "equal" else "DIFFER", t_main, t_new, t64, t128, to, tm + tg, tm, tg });
    }
    if (failures > 0) std.process.exit(1);
}
