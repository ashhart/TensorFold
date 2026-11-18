//! tf-glm-index-speed: GLM-5.3-Flash's decode indexer and sparse attention at a long position, GPU time a launch.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const tf = @import("tensorfold");
const kernels = tf.glm.kernels;

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

const ScoreArgs = extern struct { p0: u32, q_stride: u32, w_stride: u32, s_stride: u32, rows: u32 };
const SelectArgs = extern struct { p0: u32, top: u32, width: u32, s_stride: u32, i_stride: u32 };

fn env(name: [:0]const u8, default: u32) u32 {
    const v = platform.c.getenv(name) orelse return default;
    return std.fmt.parseInt(u32, std.mem.span(v), 10) catch default;
}

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

const Bufs = struct {
    pool: mtl.Buffer,
    iq: mtl.Buffer,
    iw: mtl.Buffer,
    scores: mtl.Buffer,
    indices: mtl.Buffer,
    keys: mtl.Buffer,
    ql: mtl.Buffer,
    part: mtl.Buffer,
    out: mtl.Buffer,
};

const Shape = struct { p0: u32, rows: u32, s_stride: u32 };

fn score(e: mtl.ComputeEncoder, k: *const kernels.Kernels, b: *const Bufs, s: Shape) void {
    const most = (s.p0 + s.rows) / KPOOL;
    e.setPipeline(k.index_decode);
    e.setBuffer(b.iq, QOFF * 2, 0);
    e.setBuffer(b.iw, 0, 1);
    e.setBuffer(b.pool, 0, 2);
    e.setBuffer(b.scores, 0, 3);
    e.setValue(ScoreArgs{ .p0 = s.p0, .q_stride = QR, .w_stride = HI, .s_stride = s.s_stride, .rows = s.rows }, 4);
    e.dispatchGroups(mtl.Size.of((most + 127) / 128, s.rows, 1), mtl.Size.of(128, 1, 1));
}

fn select(e: mtl.ComputeEncoder, k: *const kernels.Kernels, b: *const Bufs, s: Shape) void {
    e.setPipeline(k.index_select);
    e.setBuffer(b.scores, 0, 0);
    e.setBuffer(b.indices, 0, 1);
    e.setValue(SelectArgs{ .p0 = s.p0, .top = TOP / KPOOL, .width = WIDTH, .s_stride = s.s_stride, .i_stride = WIDTH }, 2);
    e.dispatchGroups(mtl.Size.of(s.rows, 1, 1), mtl.Size.of(1024, 1, 1));
}

fn attend(e: mtl.ComputeEncoder, k: *const kernels.Kernels, b: *const Bufs, s: Shape) void {
    const vals = @as(usize, s.rows) * BLOCKS * H * RANK * 4;
    const marks = @as(usize, s.rows) * BLOCKS * H * 4;
    e.setPipeline(k.sparse_split);
    e.setBuffer(b.ql, 0, 0);
    e.setBuffer(b.keys, 0, 1);
    e.setBuffer(b.indices, 0, 2);
    e.setValue(@as(f32, 1.0 / 16.0), 3);
    e.setValue([4]i32{ WIDTH, @intCast(s.p0 + s.rows), SPAN, 0 }, 4);
    e.setBuffer(b.part, 0, 5);
    e.setBuffer(b.part, vals, 6);
    e.setBuffer(b.part, vals + marks, 7);
    e.dispatchGroups(mtl.Size.of(H / 16, BLOCKS, s.rows), mtl.Size.of(128, 1, 1));
    e.setPipeline(k.sparse_combine);
    e.setBuffer(b.part, 0, 0);
    e.setBuffer(b.part, vals, 1);
    e.setBuffer(b.part, vals + marks, 2);
    e.setValue(@as(i32, BLOCKS), 3);
    e.setBuffer(b.out, 0, 4);
    e.dispatchGroups(mtl.Size.of(H, s.rows, 1), mtl.Size.of(128, 1, 1));
}

fn time(queue: mtl.Queue, comptime f: anytype, k: *const kernels.Kernels, b: *const Bufs, s: Shape) !f64 {
    var t: [runs + 1]f64 = undefined;
    for (&t) |*v| {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        for (0..reps) |_| f(e, k, b, s);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("GPU failed: {s}\n", .{msg});
            return error.GpuFailed;
        }
        v.* = cb.gpuSeconds() * 1e6 / reps;
    }
    const sl = t[1..];
    std.mem.sort(f64, sl, {}, std.sort.asc(f64));
    return sl[runs / 2];
}

fn once(queue: mtl.Queue, comptime f: anytype, k: *const kernels.Kernels, b: *const Bufs, s: Shape) !void {
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    f(e, k, b, s);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("GPU failed: {s}\n", .{msg});
        return error.GpuFailed;
    }
}

pub fn main() !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const k = try kernels.load(a, device);
    const cap = env("GLM_CAP", 131072);
    const s_stride = cap / KPOOL + 1;
    const max_rows: u32 = 16;
    const b = Bufs{
        .pool = try buf(device, @as(usize, s_stride) * DI * 2),
        .iq = try buf(device, @as(usize, max_rows) * QR * 2),
        .iw = try buf(device, @as(usize, max_rows) * HI * 2),
        .scores = try buf(device, @as(usize, max_rows) * s_stride * 4),
        .indices = try buf(device, @as(usize, max_rows) * WIDTH * 4),
        .keys = try buf(device, @as(usize, cap) * RANK * 2),
        .ql = try buf(device, @as(usize, max_rows) * H * RANK * 2),
        .part = try buf(device, @as(usize, max_rows) * BLOCKS * H * (RANK + 2) * 4),
        .out = try buf(device, @as(usize, max_rows) * H * RANK * 2),
    };
    fillBf(b.pool, @as(usize, s_stride) * DI, 1, 1.0);
    fillBf(b.iq, @as(usize, max_rows) * QR, 2, 1.0);
    fillBf(b.iw, @as(usize, max_rows) * HI, 3, 0.2);
    fillBf(b.keys, @as(usize, cap) * RANK, 4, 1.0);
    fillBf(b.ql, @as(usize, max_rows) * H * RANK, 5, 1.0);
    var pit = std.mem.tokenizeScalar(u8, std.mem.span(platform.c.getenv("GLM_POS") orelse "4096,16384,32768,65536,90000,120000"), ',');
    while (pit.next()) |pv| {
        const p0 = try std.fmt.parseInt(u32, pv, 10);
        for ([_]u32{ 1, 4 }) |rows| {
            const s = Shape{ .p0 = p0, .rows = rows, .s_stride = s_stride };
            try once(queue, score, k, &b, s);
            const ts = try time(queue, score, k, &b, s);
            const tsel = try time(queue, select, k, &b, s);
            const ta = try time(queue, attend, k, &b, s);
            std.debug.print("pos {d:6} rows {d:2}: scores {d:7.1} us  select {d:7.1} us  sparse attention {d:7.1} us\n", .{ p0, rows, ts, tsel, ta });
        }
    }
}
