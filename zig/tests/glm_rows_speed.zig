//! tf-glm-rows-speed: GLM-5.3-Flash's decode products at the checkpoints' widths and shapes, GPU time and GB/s.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const wide = tf.glm.wide;
const affine = tf.glm.affine;
const kernels = tf.glm.kernels;
const Ref = affine.Ref;
const cand = @import("glm_rows_speed_cand.zig");
const speed_prompt = @import("glm_rows_speed_prompt.zig");
const speed_rows = @import("glm_rows_speed_rows.zig");
const Affine = affine.Affine;

const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
pub const a = std.heap.page_allocator;
const hot = 768 << 20; // a class's matrices rotate over at least this many bytes: past the GPU's caches
pub const reps = 32; // launches a command buffer
const runs = 7; // command buffers a timing (median)

pub const H = 64; // KDA and MLA heads
const KD = 128;
pub const W = H * KD;
pub const PS = 3 * W + 2 * KD + H; // KDA's stacked projection
const XP = 2208; // MLA x_proj rows
pub const QR = 20480; // qr_proj rows
pub const N = 2048; // an expert's intermediate rows
pub const D = 4096;
pub const E = 64; // experts in a pool
pub const TOPK = 8;

pub fn buf(device: mtl.Device, bytes: usize) !mtl.Buffer {
    return device.buffer(@max(bytes, 16), opts);
}

fn fill(b: mtl.Buffer, bytes: usize, seed: u64) void {
    var s = seed | 1;
    for (b.slice(u64, bytes / 8)) |*v| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        v.* = s;
    }
}

/// `copies` matrices [n, k] at bits/64 (experts > 1: each one stacked [experts, n, k]); codes random, tables small.
pub const Pool = struct {
    q: []Affine,
    bufs: []mtl.Buffer,
    bytes: usize, // one copy's

    pub fn init(device: mtl.Device, n: usize, k: usize, bits: u8, experts: usize, min_copies: usize) !Pool {
        const w: affine.Width = .{ .bits = bits, .group = 64 };
        const one = affine.bytes(n, k, w) * experts;
        const copies = if (experts > 1) min_copies else @max(
            min_copies,
            @min(64, (hot + one - 1) / one),
        ); // a pool's picks walk its experts
        const p = Pool{ .q = try a.alloc(Affine, copies), .bufs = try a.alloc(mtl.Buffer, 3 * copies), .bytes = one };
        for (0..copies) |c| {
            const wb = affine.wBytes(n, k, w) * experts;
            const sb = affine.sBytes(n, k, w) * experts;
            p.bufs[3 * c] = try buf(device, wb);
            fill(p.bufs[3 * c], wb, 0x9e37 + c * 7919 + n + k);
            for (1..3) |j| {
                p.bufs[3 * c + j] = try buf(device, sb);
                @memset(p.bufs[3 * c + j].slice(u16, sb / 2), if (j == 1) 0x3c00 else 0xbc00); // 2^-7, -2^-7
            }
            p.q[c] = .{
                .w = .{ .buf = p.bufs[3 * c] },
                .s = .{ .buf = p.bufs[3 * c + 1] },
                .b = .{ .buf = p.bufs[3 * c + 2] },
                .n = @intCast(n),
                .k = @intCast(k),
                .bits = bits,
                .group = 64,
            };
        }
        return p;
    }

    pub fn deinit(p: Pool) void {
        for (p.bufs) |b| b.deinit();
        a.free(p.q);
        a.free(p.bufs);
    }

    pub fn at(p: Pool, i: usize) Affine {
        return p.q[i % p.q.len];
    }
};

/// Activations and outputs every launch shares (wide enough for any class).
pub const Io = struct {
    x: mtl.Buffer,
    y: mtl.Buffer,
    z: mtl.Buffer,
    ids: mtl.Buffer,
    uids: mtl.Buffer,
    umem: mtl.Buffer,
    ucount: mtl.Buffer,
};

pub const Bench = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    k: *kernels.Kernels,
    io: Io,
    reps: usize = reps,

    /// Median GPU microseconds a launch of `f(e, i, args)` over `reps` launches i in one command buffer.
    pub fn time(b: *const Bench, comptime f: anytype, args: anytype) !f64 {
        var t: [runs + 1]f64 = undefined;
        for (&t) |*v| {
            const cb = b.queue.commandBuffer();
            const e = cb.compute(.serial);
            for (0..b.reps) |i| @call(.auto, f, .{ e, i } ++ args);
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| {
                std.debug.print("GPU failed: {s}\n", .{msg});
                return error.GpuFailed;
            }
            v.* = cb.gpuSeconds() * 1e6 / @as(f64, @floatFromInt(b.reps));
        }
        const s = t[1..]; // the first warms the pipelines
        std.mem.sort(f64, s, {}, std.sort.asc(f64));
        return s[runs / 2];
    }

    /// One launch of `f(e, 0, args)`, waited on.
    pub fn run(b: *const Bench, comptime f: anytype, args: anytype) !void {
        const cb = b.queue.commandBuffer();
        const e = cb.compute(.serial);
        @call(.auto, f, .{ e, @as(usize, 0) } ++ args);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("GPU failed: {s}\n", .{msg});
            return error.GpuFailed;
        }
    }
};

pub fn bind(e: mtl.ComputeEncoder, first: usize, refs: anytype) void {
    inline for (refs, 0..) |r, i| e.setBuffer(r.buf, r.off, first + i);
}
fn shape(e: mtl.ComputeEncoder, index: usize, dims: anytype) void {
    var v: [dims.len]i32 = undefined;
    inline for (dims, 0..) |d, i| v[i] = @intCast(d);
    e.setBytes(std.mem.asBytes(&v), index);
}
pub fn rf(b: mtl.Buffer) Ref {
    return .{ .buf = b };
}

// Uniform 4/64: the generated kernels' launches (forward.zig, moe.zig).
fn qmv(e: mtl.ComputeEncoder, i: usize, pipe: mtl.Pipeline, p: Pool, io: Io) void {
    const q = p.at(i);
    e.setPipeline(pipe);
    bind(e, 0, .{ rf(io.x), q.w, q.s, q.b, rf(io.y) });
    e.dispatchThreads(mtl.Size.of(32, q.n / 4, 1), mtl.Size.of(32, 1, 1));
}
fn absorb(e: mtl.ComputeEncoder, i: usize, k: *const kernels.Kernels, p: Pool, io: Io) void {
    const q = p.at(i);
    e.setPipeline(k.absorb);
    bind(e, 0, .{ q.w, q.s, q.b, rf(io.x), rf(io.y) });
    e.setValue(@as(u32, QR), 5);
    e.setValue(@as(u32, H), 6);
    e.dispatchGroups(mtl.Size.of(1, 512 / 64, H), mtl.Size.of(64, 1, 1));
}
fn unabsorb(e: mtl.ComputeEncoder, i: usize, k: *const kernels.Kernels, p: Pool, io: Io) void {
    const q = p.at(i);
    e.setPipeline(k.unabsorb);
    bind(e, 0, .{ q.w, q.s, q.b, rf(io.x), rf(io.y) });
    e.setValue(@as(u32, H), 5);
    e.dispatchGroups(mtl.Size.of(1, 256 / 4, H), mtl.Size.of(32, 1, 1));
}
/// The routed experts and the shared one in one gather (moe.zig `both`, one row: two outputs a simdgroup).
fn moeBoth(
    e: mtl.ComputeEncoder,
    i: usize,
    k: *const kernels.Kernels,
    gu: [2]Pool,
    down: Pool,
    sh: [2]Pool,
    io: Io,
) void {
    const g = gu[0].at(i);
    const u = gu[1].at(i);
    const s = sh[0].at(i);
    const ids = rf(io.uids).at((i % reps) * 128 * 4);
    e.setPipeline(k.moe_gateup_0_one);
    bind(e, 0, .{rf(io.x)});
    shape(e, 1, .{ 1, D });
    bind(e, 2, .{ g.w, g.s, g.b, u.w, u.s, u.b, s.w, s.s, s.b, ids, rf(io.umem), rf(io.ucount) });
    e.setValue(@as(f32, 10.0), 14);
    bind(e, 15, .{rf(io.z)});
    e.dispatchThreads(mtl.Size.of(32, N / 2, TOPK + 1), mtl.Size.of(32, 1, 1));
    const d = down.at(i);
    const sd = sh[1].at(i);
    e.setPipeline(k.moe_down_0_one);
    bind(e, 0, .{rf(io.z)});
    shape(e, 1, .{ 1, TOPK + 1, N });
    bind(e, 2, .{ d.w, d.s, d.b, sd.w, sd.s, sd.b, ids, rf(io.umem), rf(io.ucount), rf(io.y) });
    e.dispatchThreads(mtl.Size.of(32, D / 2, TOPK + 1), mtl.Size.of(32, 1, 1));
}
/// oQ4's routed experts: the generated 4/64 gathers without the shared slot (moe.zig `experts` part 2).
fn moeRouted4(e: mtl.ComputeEncoder, i: usize, k: *const kernels.Kernels, gu: [2]Pool, down: Pool, io: Io) void {
    const g = gu[0].at(i);
    const u = gu[1].at(i);
    const ids = rf(io.uids).at((i % reps) * 128 * 4);
    e.setPipeline(k.moe_gateup_2);
    bind(e, 0, .{rf(io.x)});
    shape(e, 1, .{ 1, D });
    bind(e, 2, .{ g.w, g.s, g.b, u.w, u.s, u.b, g.w, g.s, g.b, ids, rf(io.umem), rf(io.ucount) });
    e.setValue(@as(f32, 10.0), 14);
    bind(e, 15, .{rf(io.z)});
    e.dispatchThreads(mtl.Size.of(32, N / 4, TOPK), mtl.Size.of(32, 1, 1));
    const d = down.at(i);
    e.setPipeline(k.moe_down_2);
    bind(e, 0, .{rf(io.z)});
    shape(e, 1, .{ 1, TOPK, N });
    bind(e, 2, .{ d.w, d.s, d.b, d.w, d.s, d.b, ids, rf(io.umem), rf(io.ucount), rf(io.y) });
    e.dispatchThreads(mtl.Size.of(32, D / 4, TOPK), mtl.Size.of(32, 1, 1));
}

// Other widths: wide.zig's launches as the engine makes them.
pub fn rows(e: mtl.ComputeEncoder, i: usize, p: *const wide.Pipes, pool: Pool, io: Io) void {
    const q = pool.at(i);
    wide.dense(p, e, q, .{ .in = rf(io.x), .ldx = q.k, .out = rf(io.y), .ldo = q.n, .rows = 1 });
}
fn perHead(e: mtl.ComputeEncoder, i: usize, p: *const wide.Pipes, pool: Pool, io: Io, ldx: u32, hsx: u32) void {
    const q = pool.at(i);
    wide.perHead(p, e, q, H, .{ .in = rf(io.x), .ldx = ldx, .out = rf(io.y), .ldo = q.n, .rows = 1 }, hsx, q.n / H);
}
fn shared(e: mtl.ComputeEncoder, i: usize, p: *const wide.Pipes, k: *const kernels.Kernels, sh: [2]Pool, io: Io) void {
    _ = k;
    wide.glu(p, e, sh[0].at(i), N, rf(io.x), rf(io.z).at(2 * N * 2), 1, 10.0); // moe.zig sharedRows: one launch
    const dq = sh[1].at(i);
    wide.dense(p, e, dq, .{ .in = rf(io.z).at(2 * N * 2), .ldx = N, .out = rf(io.y), .ldo = D, .rows = 1 });
}
/// A KDA layer's step state (16 copies: each launch its own, as each layer has).
const Kda = struct {
    p: mtl.Buffer,
    cs: [16]mtl.Buffer,
    cw: mtl.Buffer,
    a: mtl.Buffer,
    dtb: mtl.Buffer,
    st: [16]mtl.Buffer,
    st_out: [16]mtl.Buffer,
    onw: mtl.Buffer,
    fg: mtl.Buffer, // f_b's and g_b's products made apart (.pre)

    fn init(device: mtl.Device) !Kda {
        var k: Kda = undefined;
        k.p = try buf(device, PS * 2);
        @memset(k.p.slice(u16, PS), 0x3c00);
        k.cw = try buf(device, 4 * 3 * W * 4);
        @memset(k.cw.slice(f32, 4 * 3 * W), 0.25);
        k.a = try buf(device, H * 4);
        @memset(k.a.slice(f32, H), 0.5);
        k.dtb = try buf(device, W * 4);
        @memset(k.dtb.slice(f32, W), 0);
        k.onw = try buf(device, KD * 4);
        @memset(k.onw.slice(f32, KD), 1);
        k.fg = try buf(device, 2 * 16 * W * 2);
        for (&k.cs, &k.st, &k.st_out) |*c, *s, *o| {
            c.* = try buf(device, 3 * 3 * W * 2);
            @memset(c.slice(u16, 3 * 3 * W), 0);
            s.* = try buf(device, H * KD * KD * 4);
            @memset(s.slice(f32, H * KD * KD), 0);
            o.* = try buf(device, H * KD * KD * 4);
        }
        return k;
    }
};

/// kda_rows (generated dots, or `.pre`: f_b, g_b made apart first) on launch i's state and gates.
fn kdaStep(
    e: mtl.ComputeEncoder,
    i: usize,
    pipe: mtl.Pipeline,
    s: *const Kda,
    gates: [2]Pool,
    pre: ?*const wide.Pipes,
    io: Io,
) void {
    const f = gates[0].at(i);
    const g = gates[1].at(i);
    var fb = [3]Ref{ f.w, f.s, f.b };
    var gb = [3]Ref{ g.w, g.s, g.b };
    if (pre) |p| {
        const fo = rf(s.fg);
        const go = fo.at(16 * W * 2);
        wide.dense(p, e, f, .{ .in = rf(s.p).at(3 * W * 2), .ldx = PS, .out = fo, .ldo = W, .rows = 1 });
        wide.dense(p, e, g, .{ .in = rf(s.p).at((3 * W + KD) * 2), .ldx = PS, .out = go, .ldo = W, .rows = 1 });
        fb = .{ fo, fo, fo };
        gb = .{ go, go, go };
    }
    const c = i % 16;
    e.setPipeline(pipe);
    bind(e, 0, .{rf(s.p)});
    shape(e, 1, .{ 1, PS });
    bind(e, 2, .{
        rf(s.cs[c]),
        rf(s.cw),
        fb[0],
        fb[1],
        fb[2],
        gb[0],
        gb[1],
        gb[2],
        rf(s.a),
        rf(s.dtb),
        rf(s.st[c]),
        rf(s.onw),
    });
    e.setValue(@as(f32, -5.0), 14);
    e.setValue(@as(f32, 1e-5), 15);
    bind(e, 16, .{ rf(io.y), rf(s.st_out[c]), rf(s.cs[(c + 1) % 16]) });
    e.dispatchThreads(mtl.Size.of(32, 32, H), mtl.Size.of(32, 32, 1));
}

/// oQ2's MoE: routed by pick and the shared expert, a launch for gate|up and one for down (wide.moe).
pub fn moeTwo(
    e: mtl.ComputeEncoder,
    i: usize,
    p: *const wide.Pipes,
    gu: [2]Pool,
    down: Pool,
    sh: [2]Pool,
    io: Io,
) void {
    const m: wide.Moe = .{
        .gate = gu[0].at(i),
        .up = gu[1].at(i),
        .down = down.at(i),
        .sh_gate_up = sh[0].at(i),
        .sh_down = sh[1].at(i),
    };
    const z = rf(io.z);
    wide.moe(p, e, m, E, TOPK, 10.0, .{
        .x = rf(io.x),
        .ids = rf(io.ids).at((i % reps) * TOPK * 4),
        .act = z,
        .acts = z.at(TOPK * N * 2),
        .ye = rf(io.y),
        .ys = rf(io.y).at(TOPK * D * 2),
    }, 1);
}

fn routedRows(e: mtl.ComputeEncoder, i: usize, p: *const wide.Pipes, gu: [2]Pool, down: Pool, io: Io) void {
    const ids = rf(io.ids).at((i % reps) * TOPK * 4);
    wide.byIds(
        p,
        e,
        .{ .w = gu[0].at(i), .up = gu[1].at(i), .experts = E, .repeat = TOPK, .lim = 10.0 },
        rf(io.x),
        ids,
        rf(io.z),
        TOPK,
    );
    wide.byIds(p, e, .{ .w = down.at(i), .experts = E, .repeat = 1 }, rf(io.z), ids, rf(io.y), TOPK);
}

/// A token's GPU time by class: launches a token times the median launch.
const Row = struct { name: []const u8, us: f64, count: f64, bytes: f64 };
const Sheet = struct {
    rows: [24]Row = undefined,
    n: usize = 0,

    fn add(s: *Sheet, name: []const u8, us: f64, count: f64, bytes: usize) void {
        s.rows[s.n] = .{ .name = name, .us = us, .count = count, .bytes = @floatFromInt(bytes) };
        s.n += 1;
        std.debug.print("  {s:<26} {d:9.1} us  {d:7.1} MB  {d:6.1} GB/s  x{d:<5.0} {d:7.3} ms/token\n", .{
            name,
            us,
            @as(f64, @floatFromInt(bytes)) / 1e6,
            @as(f64, @floatFromInt(bytes)) / us / 1e3,
            count,
            us * count / 1e3,
        });
    }

    fn total(s: *const Sheet, label: []const u8) void {
        var ms: f64 = 0;
        var gb: f64 = 0;
        for (s.rows[0..s.n]) |x| {
            ms += x.us * x.count / 1e3;
            gb += x.bytes * x.count / 1e9;
        }
        std.debug.print(
            "  {s}: {d:.3} ms a token over the products, {d:.3} GB read ({d:.0} GB/s)\n",
            .{ label, ms, gb, gb / ms * 1e3 },
        );
    }
};

pub fn main() !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const k = try kernels.load(a, device);
    defer k.deinit();
    try k.widen(a, device);
    const p = &k.wide.?;
    const io = Io{
        .x = try buf(device, 160 << 20),
        .y = try buf(device, 160 << 20),
        .z = try buf(device, 64 << 20),
        .ids = try buf(device, reps * TOPK * 4),
        .uids = try buf(device, reps * 128 * 4),
        .umem = try buf(device, 128 * 16 * 4),
        .ucount = try buf(device, 4),
    };
    @memset(io.x.slice(u16, 32 << 20), 0x3c00);
    for (0..reps) |i| for (0..TOPK) |j| { // launch i's picks: distinct experts, the pool walked in order
        io.ids.slice(i32, reps * TOPK)[i * TOPK + j] = @intCast((i * TOPK + j) % E);
        io.uids.slice(i32, reps * 128)[i * 128 + j] = @intCast((i * TOPK + j) % E);
    };
    @memset(io.umem.slice(i32, 128 * 16), -1);
    for (0..TOPK) |u| io.umem.slice(i32, 128 * 16)[u * 16] = @intCast(u); // unique id u: row 0's pick u
    io.ucount.slice(i32, 1)[0] = TOPK;
    var b = Bench{ .device = device, .queue = queue, .k = k, .io = io };
    if (std.c.getenv("GLM_SPEED_ROWS") != null) return speed_rows.multi(&b, p);
    std.debug.print(
        "GPU {s}, tensor units {}; one row, {d} launches a command buffer, median of {d}\n",
        .{ device.name(), device.tensorUnits(), reps, runs },
    );
    const shapes = [_]struct {
        name: []const u8,
        n: usize,
        k: usize,
        uni: ?mtl.Pipeline,
        count: [3]f64,
    }{ // counts: uniform, oQ2, oQ4
        .{ .name = "kda in", .n = PS, .k = D, .uni = k.qmv_kda_in, .count = .{ 34, 34, 34 } },
        .{ .name = "kda f_b/g_b", .n = W, .k = KD, .uni = null, .count = .{ 0, 0, 0 } }, // inside the KDA step's time
        .{ .name = "kda out", .n = D, .k = W, .uni = k.qmv_kda_out, .count = .{ 34, 34, 34 } },
        .{ .name = "mla x_proj", .n = XP, .k = D, .uni = k.qmv_x, .count = .{ 11, 11, 11 } },
        .{ .name = "mla qr_proj", .n = QR, .k = 1536, .uni = k.qmv_qr, .count = .{ 11, 11, 11 } },
        .{ .name = "mla out", .n = D, .k = 16384, .uni = k.qmv_mla_out, .count = .{ 11, 11, 11 } },
        .{ .name = "dense gate|up", .n = 24576, .k = D, .uni = k.qmv_dense_gu, .count = .{ 3, 3, 3 } },
        .{ .name = "dense down", .n = D, .k = 12288, .uni = k.qmv_dense_down, .count = .{ 3, 3, 3 } },
        .{ .name = "lm_head", .n = 154880, .k = D, .uni = k.qmv_head, .count = .{ 1, 1, 1 } },
    };
    var uni = Sheet{};
    var oq2 = Sheet{};
    var oq4 = Sheet{};
    std.debug.print("generated 4/64 (uniform) and glm_rows at 8, 6, 5 bits:\n", .{});
    for (shapes) |s| {
        if (s.uni) |pipe| {
            const m = try Pool.init(device, s.n, s.k, 4, 1, 1);
            defer m.deinit();
            uni.add(s.name, try b.time(qmv, .{ pipe, m, io }), s.count[0], m.bytes);
        }
        for ([_]u8{ 8, 6, 5 }) |bits| {
            const m = try Pool.init(device, s.n, s.k, bits, 1, 1);
            defer m.deinit();
            const us = try b.time(rows, .{ p, m, io });
            var name_buf: [40]u8 = undefined;
            const name = try std.fmt.bufPrint(&name_buf, "{s} q{d}", .{ s.name, bits });
            if (bits == 8) {
                oq2.add(try a.dupe(u8, name), us, s.count[1], m.bytes);
                oq4.add(
                    try a.dupe(u8, name),
                    us,
                    s.count[2],
                    m.bytes,
                ); // oQ4's 5/6-bit members priced at 8 (an upper bound)
            } else std.debug.print(
                "  {s:<26} {d:9.1} us  {d:7.1} MB  {d:6.1} GB/s\n",
                .{ name, us, @as(f64, @floatFromInt(m.bytes)) / 1e6, @as(f64, @floatFromInt(m.bytes)) / us / 1e3 },
            );
        }
    }
    { // the KDA step: generated 4/4 dots (uniform), generated 8/8 dots, f_b and g_b made apart at 8 and 5 bits (.pre)
        const st = try Kda.init(device);
        const g4 = [2]Pool{ try Pool.init(device, W, KD, 4, 1, 16), try Pool.init(device, W, KD, 4, 1, 16) };
        const g8 = [2]Pool{ try Pool.init(device, W, KD, 8, 1, 16), try Pool.init(device, W, KD, 8, 1, 16) };
        const g5 = [2]Pool{ try Pool.init(device, W, KD, 5, 1, 16), try Pool.init(device, W, KD, 5, 1, 16) };
        const step_bytes = 2 * H * KD * KD * 4; // the state read and written
        uni.add(
            "kda step (dots 4/4)",
            try b.time(kdaStep, .{ k.kda_rows, &st, g4, null, io }),
            34,
            step_bytes + g4[0].bytes * 2,
        );
        const us_pre = try b.time(kdaStep, .{ p.kda_pre, &st, g8, p, io });
        std.debug.print("  {s:<26} {d:9.1} us\n", .{ "kda step (.pre q8)", us_pre });
        const us_pre5 = try b.time(kdaStep, .{ p.kda_pre, &st, g5, p, io });
        std.debug.print("  {s:<26} {d:9.1} us\n", .{ "kda step (.pre q5)", us_pre5 });
        const f8 = wide.kdaGate(p, g8[0].at(0), g8[1].at(0)).fused;
        const us_f8 = try b.time(kdaStep, .{ f8, &st, g8, null, io });
        oq2.add("kda step (fused q8)", us_f8, 34, step_bytes + g8[0].bytes * 2);
        oq4.add("kda step (fused q8)", us_f8, 34, step_bytes + g8[0].bytes * 2);
        const f5 = wide.kdaGate(p, g5[0].at(0), g5[1].at(0)).fused;
        std.debug.print(
            "  {s:<26} {d:9.1} us\n",
            .{ "kda step (fused q5)", try b.time(kdaStep, .{ f5, &st, g5, null, io }) },
        );
    }
    { // MLA absorb / unabsorb: kv_b's halves (uniform) or embed_q / unembed_out a head at 8 bits
        const kv = try Pool.init(device, H * 512, 512, 4, 1, 1);
        defer kv.deinit();
        uni.add("mla absorb", try b.time(absorb, .{ k, kv, io }), 11, kv.bytes / 2);
        uni.add("mla unabsorb", try b.time(unabsorb, .{ k, kv, io }), 11, kv.bytes / 2);
        const eq = try Pool.init(device, H * 512, 256, 8, 1, 1);
        defer eq.deinit();
        const us_eq = try b.time(perHead, .{ p, eq, io, @as(u32, QR), @as(u32, 256) });
        oq2.add("mla embed_q q8", us_eq, 11, eq.bytes);
        oq4.add("mla embed_q q8", us_eq, 11, eq.bytes);
        const uo = try Pool.init(device, H * 256, 512, 8, 1, 1);
        defer uo.deinit();
        const us_uo = try b.time(perHead, .{ p, uo, io, @as(u32, H * 512), @as(u32, 512) });
        oq2.add("mla unembed_out q8", us_uo, 11, uo.bytes);
        oq4.add("mla unembed_out q8", us_uo, 11, uo.bytes);
    }
    { // MoE: the routed experts (8 picks) with the shared expert
        const gu4 = [2]Pool{ try Pool.init(device, N, D, 4, E, 1), try Pool.init(device, N, D, 4, E, 1) };
        defer for (gu4) |x| x.deinit();
        const dn4 = try Pool.init(device, D, N, 4, E, 1);
        defer dn4.deinit();
        const routed4 = 3 * affine.bytes(N, D, .{ .bits = 4, .group = 64 }) * TOPK;
        const sh4 = [2]Pool{ try Pool.init(device, 2 * N, D, 4, 1, 2), try Pool.init(device, D, N, 4, 1, 2) };
        defer for (sh4) |x| x.deinit();
        const shared4 = sh4[0].bytes + sh4[1].bytes;
        uni.add("moe routed+shared 4", try b.time(moeBoth, .{ k, gu4, dn4, sh4, io }), 42, routed4 + shared4);
        oq4.add("moe routed 4", try b.time(moeRouted4, .{ k, gu4, dn4, io }), 42, routed4);
        const sh8 = [2]Pool{ try Pool.init(device, 2 * N, D, 8, 1, 2), try Pool.init(device, D, N, 8, 1, 2) };
        defer for (sh8) |x| x.deinit();
        const us_sh = try b.time(shared, .{ p, k, sh8, io });
        std.debug.print("  {s:<26} {d:9.1} us (oQ2: in the routed launches below)\n", .{ "moe shared q8", us_sh });
        oq4.add("moe shared q8", us_sh, 42, sh8[0].bytes + sh8[1].bytes);
        const alt = try b.time(moeTwo, .{ p, gu4, dn4, sh8, io }); // oQ4's routed 4/64 on glm_rows sums with the shared
        std.debug.print(
            "  {s:<26} {d:9.1} us (not taken: changes routed bits)\n",
            .{ "moe routed 4 + shared q8", alt },
        );
        const gu2 = [2]Pool{ try Pool.init(device, N, D, 2, E, 1), try Pool.init(device, N, D, 2, E, 1) };
        defer for (gu2) |x| x.deinit();
        const dn2 = try Pool.init(device, D, N, 2, E, 1);
        defer dn2.deinit();
        std.debug.print(
            "  {s:<26} {d:9.1} us (apart)\n",
            .{ "moe routed q2 (ids)", try b.time(routedRows, .{ p, gu2, dn2, io }) },
        );
        oq2.add(
            "moe routed q2 + shared q8",
            try b.time(moeTwo, .{ p, gu2, dn2, sh8, io }),
            42,
            3 * affine.bytes(N, D, .{ .bits = 2, .group = 64 }) * TOPK + sh8[0].bytes + sh8[1].bytes,
        );
        if (platform.c.getenv("GLM_SPEED_CANDIDATES") != null) {
            const lib = try cand.candidateLib(device);
            defer lib.deinit();
            try cand.moeCandidates(&b, lib, p, gu2, dn2, sh8);
        }
    }
    if (platform.c.getenv("GLM_SPEED_CANDIDATES") != null) {
        const lib = try cand.candidateLib(device);
        defer lib.deinit();
        std.debug.print("candidates at 8 bits (bits checked == glm_rows):\n", .{});
        for (shapes) |s| {
            const m = try Pool.init(device, s.n, s.k, 8, 1, 1);
            defer m.deinit();
            try cand.candidates(&b, lib, p, m, s.name);
        }
    }
    if (platform.c.getenv("GLM_SPEED_PROMPT") != null) try speed_prompt.prompt(&b, k, p);
    std.debug.print("uniform 4/64:\n", .{});
    for (uni.rows[0..uni.n]) |x| std.debug.print("  {s:<26} {d:7.3} ms/token\n", .{ x.name, x.us * x.count / 1e3 });
    uni.total("uniform 4/64");
    std.debug.print("oQ2:\n", .{});
    for (oq2.rows[0..oq2.n]) |x| std.debug.print("  {s:<26} {d:7.3} ms/token\n", .{ x.name, x.us * x.count / 1e3 });
    oq2.total("oQ2");
    std.debug.print("oQ4 (5/6-bit members priced at 8 bits):\n", .{});
    for (oq4.rows[0..oq4.n]) |x| std.debug.print("  {s:<26} {d:7.3} ms/token\n", .{ x.name, x.us * x.count / 1e3 });
    oq4.total("oQ4");
}
