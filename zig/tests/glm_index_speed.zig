//! tf-glm-index-speed: GLM's decode indexer and sparse attention, new kernels against the old bit for bit, GPU times.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const tf = @import("tensorfold");
const kernels = tf.glm.kernels;
const st = tf.glm.state;

const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
const a = std.heap.page_allocator;
const reps = 16;
const runs = 7;

const HI = 32;
const DI = 128;
const QR = 20480; // qr_proj rows: 64 heads' q_nope, then the indexer's queries
const QOFF = 64 * 256;
const RANK = 512;
const H = 64;
const TOP = 2048;
const KPOOL = 4;
const WIDTH = TOP + KPOOL - 1;
const SPAN = 128;
const BLOCKS = (WIDTH + SPAN - 1) / SPAN;
const MAX_ROWS = 16;

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
    cap: u32,
    s_stride: u32,
    pool: mtl.Buffer,
    iq: mtl.Buffer,
    iw: mtl.Buffer,
    scores: mtl.Buffer,
    indices: mtl.Buffer,
    bits: mtl.Buffer,
    keys: mtl.Buffer,
    ql: mtl.Buffer,
    part: mtl.Buffer,
    out: mtl.Buffer,
};

const Shape = struct { p0: u32, rows: u32 };

const Op = enum { scores, select_old, select_new, attend, mark, merge };

fn encode(e: mtl.ComputeEncoder, r: *const Rig, op: Op, s: Shape) void {
    const k = r.k;
    const most = (s.p0 + s.rows) / KPOOL;
    const sel = SelectArgs{ .p0 = s.p0, .top = TOP / KPOOL, .width = WIDTH, .s_stride = r.s_stride, .i_stride = WIDTH };
    switch (op) {
        .scores => {
            e.setPipeline(k.index_decode);
            e.setBuffer(r.iq, QOFF * 2, 0);
            e.setBuffer(r.iw, 0, 1);
            e.setBuffer(r.pool, 0, 2);
            e.setBuffer(r.scores, 0, 3);
            e.setValue(ScoreArgs{ .p0 = s.p0, .q_stride = QR, .w_stride = HI, .s_stride = r.s_stride, .rows = s.rows }, 4);
            e.dispatchGroups(mtl.Size.of((most + 127) / 128, s.rows, 1), mtl.Size.of(128, 1, 1));
        },
        .select_old => {
            e.setPipeline(k.index_select);
            e.setBuffer(r.scores, 0, 0);
            e.setBuffer(r.indices, 0, 1);
            e.setValue(sel, 2);
            e.dispatchGroups(mtl.Size.of(s.rows, 1, 1), mtl.Size.of(1024, 1, 1));
        },
        .select_new, .mark, .merge => { // forward.zig's selectKeys after its scores
            const words = st.selectWords(r.cap, KPOOL);
            if (most > 3072 and op != .merge) {
                e.setPipeline(k.select_mark);
                e.setBuffer(r.scores, 0, 0);
                e.setBuffer(r.bits, 0, 1);
                e.setValue(sel, 2);
                e.setValue(words, 3);
                e.dispatchGroups(mtl.Size.of((most + 4095) / 4096, s.rows, 1), mtl.Size.of(256, 1, 1));
            }
            if (op == .mark) return;
            e.setPipeline(k.select_merge);
            e.setBuffer(r.scores, 0, 0);
            e.setBuffer(r.indices, 0, 1);
            e.setValue(sel, 2);
            e.setValue(words, 3);
            e.setBuffer(r.bits, 0, 4);
            e.dispatchGroups(mtl.Size.of(s.rows, 1, 1), mtl.Size.of(512, 1, 1));
        },
        .attend => {
            const vals = @as(usize, s.rows) * BLOCKS * H * RANK * 4;
            const marks = @as(usize, s.rows) * BLOCKS * H * 4;
            e.setPipeline(k.sparse_split);
            e.setBuffer(r.ql, 0, 0);
            e.setBuffer(r.keys, 0, 1);
            e.setBuffer(r.indices, 0, 2);
            e.setValue(@as(f32, 1.0 / 16.0), 3);
            e.setValue([4]i32{ WIDTH, @intCast(s.p0 + s.rows), SPAN, 0 }, 4);
            e.setBuffer(r.part, 0, 5);
            e.setBuffer(r.part, vals, 6);
            e.setBuffer(r.part, vals + marks, 7);
            e.dispatchGroups(mtl.Size.of(H / 16, BLOCKS, s.rows), mtl.Size.of(128, 1, 1));
            e.setPipeline(k.sparse_combine);
            e.setBuffer(r.part, 0, 0);
            e.setBuffer(r.part, vals, 1);
            e.setBuffer(r.part, vals + marks, 2);
            e.setValue(@as(i32, BLOCKS), 3);
            e.setBuffer(r.out, 0, 4);
            e.dispatchGroups(mtl.Size.of(H, s.rows, 1), mtl.Size.of(128, 1, 1));
        },
    }
}

fn submit(r: *const Rig, op: Op, s: Shape, count: usize) !f64 {
    const cb = r.queue.commandBuffer();
    const e = cb.compute(.serial);
    for (0..count) |_| encode(e, r, op, s);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("GPU failed: {s}\n", .{msg});
        return error.GpuFailed;
    }
    return cb.gpuSeconds() * 1e6 / @as(f64, @floatFromInt(count));
}

/// Median GPU time of one launch of `op`, in microseconds.
fn time(r: *const Rig, op: Op, s: Shape) !f64 {
    var t: [runs + 1]f64 = undefined;
    for (&t) |*v| v.* = try submit(r, op, s, reps);
    const sl = t[1..];
    std.mem.sort(f64, sl, {}, std.sort.asc(f64));
    return sl[runs / 2];
}

/// Score rows that stress the select: ties, NaNs, signed zeros, infinities, ramps and one hot span.
const Fill = enum { real, uniform, ties, specials, ascending, descending, hot_span, constant, sampled_high };

fn fillScores(r: *const Rig, f: Fill, s: Shape, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    for (0..s.rows) |row| {
        const n = (s.p0 + @as(u32, @intCast(row)) + 1) / KPOOL;
        const v = r.scores.slice(f32, MAX_ROWS * @as(usize, r.s_stride))[row * r.s_stride ..][0..n];
        for (v, 0..) |*x, i| x.* = switch (f) {
            .real => x.*,
            .uniform => rnd.float(f32) * 8 - 2,
            .ties => @floatFromInt(rnd.uintLessThan(u32, 6)),
            .specials => switch (rnd.uintLessThan(u32, 16)) {
                0 => std.math.nan(f32),
                1 => -std.math.nan(f32),
                2 => 0.0,
                3 => -0.0,
                4 => std.math.inf(f32),
                5 => -std.math.inf(f32),
                else => rnd.float(f32) - 0.5,
            },
            .ascending => @floatFromInt(i / 3),
            .descending => @floatFromInt(n - i / 2),
            .hot_span => if (i >= n / 3 and i < n / 3 + 700) 5.0 + rnd.float(f32) else rnd.float(f32),
            .constant => 1.5,
            .sampled_high => if (sampled(i, n)) 3.0 + rnd.float(f32) else rnd.float(f32), // the floor overshoots
        };
    }
}

/// Whether glm_select.metal's floor samples block i of n (256 runs of 8 at even spacing).
fn sampled(i: usize, n: u32) bool {
    var q: u64 = 0;
    while (q < 2048) : (q += 8) if (i >= q / 8 * n / 256 / 8 * 8 and i < q / 8 * n / 256 / 8 * 8 + 8) return true;
    return false;
}

/// The new select's key lists against the old kernel's on the same scores: true when every entry is the same.
fn same(r: *const Rig, s: Shape) !bool {
    const n = @as(usize, s.rows) * WIDTH;
    _ = try submit(r, .select_old, s, 1);
    const want = try a.dupe(i32, r.indices.slice(i32, n));
    defer a.free(want);
    @memset(r.indices.slice(i32, n), 0x5a5a5a5a);
    _ = try submit(r, .select_new, s, 1);
    return std.mem.eql(i32, want, r.indices.slice(i32, n));
}

pub fn main() !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const k = try kernels.load(a, device);
    const cap: u32 = 131072;
    const s_stride = cap / KPOOL + 1;
    const r = Rig{
        .queue = queue,
        .k = k,
        .cap = cap,
        .s_stride = s_stride,
        .pool = try buf(device, @as(usize, s_stride) * DI * 2),
        .iq = try buf(device, @as(usize, MAX_ROWS) * QR * 2),
        .iw = try buf(device, @as(usize, MAX_ROWS) * HI * 2),
        .scores = try buf(device, @as(usize, MAX_ROWS) * s_stride * 4),
        .indices = try buf(device, @as(usize, MAX_ROWS) * WIDTH * 4),
        .bits = try buf(device, @as(usize, MAX_ROWS) * st.selectWords(cap, KPOOL) * 4),
        .keys = try buf(device, @as(usize, cap) * RANK * 2),
        .ql = try buf(device, @as(usize, MAX_ROWS) * H * RANK * 2),
        .part = try buf(device, @as(usize, MAX_ROWS) * BLOCKS * H * (RANK + 2) * 4),
        .out = try buf(device, @as(usize, MAX_ROWS) * H * RANK * 2),
    };
    fillBf(r.pool, @as(usize, s_stride) * DI, 1, 1.0);
    fillBf(r.iq, @as(usize, MAX_ROWS) * QR, 2, 1.0);
    fillBf(r.iw, @as(usize, MAX_ROWS) * HI, 3, 0.2);
    fillBf(r.keys, @as(usize, cap) * RANK, 4, 1.0);
    fillBf(r.ql, @as(usize, MAX_ROWS) * H * RANK, 5, 1.0);
    var failures: usize = 0;
    var cases: usize = 0;
    // every fill at positions across the select's paths: all blocks, one group's whole row, the marked candidates
    const positions = [_]u32{ 2047, 2048, 2051, 2100, 8191, 12280, 16380, 16384, 16387, 16400, 20000, 40000, 90000, 120000, 131000 };
    for (positions) |p0| for ([_]u32{ 1, 3, 16 }) |rows| {
        if (p0 + rows + 1 > cap) continue;
        const s = Shape{ .p0 = p0, .rows = rows };
        for (std.enums.values(Fill), 0..) |f, fi| {
            if (f == .real) _ = try submit(&r, .scores, s, 1);
            fillScores(&r, f, s, p0 * 31 + rows * 7 + fi);
            cases += 1;
            if (!try same(&r, s)) {
                failures += 1;
                std.debug.print("DIFFER: position {d} rows {d} fill {t}\n", .{ p0, rows, f });
            }
        }
    };
    std.debug.print("select: {d} of {d} cases give the old kernel's key lists\n", .{ cases - failures, cases });
    if (platform.c.getenv("GLM_INDEX_TIMES") != null) {
        var pit = std.mem.tokenizeScalar(u8, std.mem.span(platform.c.getenv("GLM_POS") orelse "4096,16384,32768,65536,90000,120000"), ',');
        while (pit.next()) |pv| {
            const p0 = try std.fmt.parseInt(u32, pv, 10);
            for ([_]u32{ 1, 4 }) |rows| {
                const s = Shape{ .p0 = p0, .rows = rows };
                _ = try submit(&r, .scores, s, 1);
                const ts = try time(&r, .scores, s);
                const t_old = try time(&r, .select_old, s);
                const t_new = try time(&r, .select_new, s);
                const ta = try time(&r, .attend, s);
                const tm = try time(&r, .mark, s);
                const tg = try time(&r, .merge, s);
                std.debug.print("pos {d:6} rows {d:2}: scores {d:6.1} us  select {d:6.1} -> {d:6.1} us (mark {d:5.1}, merge {d:5.1})  sparse attention {d:6.1} us\n", .{ p0, rows, ts, t_old, t_new, tm, tg, ta });
            }
        }
    }
    if (failures > 0) std.process.exit(1);
}
