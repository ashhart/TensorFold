//! gdn_prep.cu against core/gdn_prep_ref.zig: random conv windows over two streams, bits at every row count.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");

const ref = core.gdn_prep_ref;
const counts = [_]usize{ 1, 3, 17, 40 };
const most = counts[counts.len - 1];

fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn bf(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}

fn bfs(a: std.mem.Allocator, rng: std.Random, n: usize, scale: f32) ![]u16 {
    const v = try a.alloc(u16, n);
    for (v) |*x| x.* = bf16(rng.floatNorm(f32) * scale);
    return v;
}

const Bufs = struct {
    gpu: check.Gpu,
    list: std.ArrayList(cuda.DeviceBuffer) = .empty,

    fn up(b: *Bufs, bytes: []const u8) !u64 {
        const d = try cuda.DeviceBuffer.fromHost(b.gpu.d, bytes);
        try b.list.append(b.gpu.gpa, d);
        return d.ptr;
    }

    fn out(b: *Bufs, s: cuda.Stream, n: usize) !u64 {
        const d = try cuda.DeviceBuffer.alloc(b.gpu.d, @max(n, 16));
        try b.list.append(b.gpu.gpa, d);
        try d.fill8(0xff, s.handle);
        return d.ptr;
    }

    fn back(b: *Bufs, a: std.mem.Allocator, comptime T: type, ptr: u64, n: usize) ![]T {
        const o = try a.alloc(T, n);
        const d: cuda.DeviceBuffer = .{ .d = b.gpu.d, .ptr = ptr, .len = n * @sizeOf(T) };
        try d.download(0, std.mem.sliceAsBytes(o));
        return o;
    }
};

fn case(gpu: check.Gpu, a: std.mem.Allocator, s: cuda.Stream, g: *const cuda.gdn_prep.GdnPrep, rng: std.Random, sh: cuda.gdn_prep.Shape, worst: *f64) !void {
    var b: Bufs = .{ .gpu = gpu };
    defer {
        for (b.list.items) |*d| d.free();
        b.list.deinit(gpu.gpa);
    }
    const c = (2 * sh.kh + sh.vh) * sh.dk;
    const taps = sh.keep + 1;
    const ld = c + 64; // stacked rows: something after the channels
    const ab_ld = sh.vh + 3;
    const qkv = try bfs(a, rng, most * ld, 1);
    const state = try bfs(a, rng, 2 * sh.keep * c, 1);
    const conv = try bfs(a, rng, c * taps, 0.4);
    const av = try bfs(a, rng, most * ab_ld, 2);
    const bv = try bfs(a, rng, most * ab_ld, 2);
    const a_log = try a.alloc(f32, sh.vh);
    const dtb = try a.alloc(f32, sh.vh);
    for (a_log, dtb) |*x, *y| {
        x.* = rng.float(f32) * 2 - 1;
        y.* = rng.floatNorm(f32);
    }
    const sid = try a.alloc(i32, most);
    const win = try a.alloc(i32, most * taps);
    for (sid, 0..) |*x, i| x.* = @intCast(i % 2);
    for (win, 0..) |*w, i| { // kept rows or any qkv row (a tree's window is any rows)
        w.* = if (rng.uintLessThan(u32, 3) == 0) @intCast(rng.uintLessThan(usize, sh.keep)) else @intCast(sh.keep + rng.uintLessThan(usize, most));
        if (i % taps == taps - 1) w.* = @intCast(sh.keep + i / taps); // the row itself last
    }
    const base: cuda.gdn_prep.Io = .{ .qkv = try b.up(std.mem.sliceAsBytes(qkv)), .qkv_ld = @intCast(ld), .state = try b.up(std.mem.sliceAsBytes(state)), .conv = try b.up(std.mem.sliceAsBytes(conv)), .win = try b.up(std.mem.sliceAsBytes(win)), .sid = try b.up(std.mem.sliceAsBytes(sid)), .a = try b.up(std.mem.sliceAsBytes(av)), .a_ld = @intCast(ab_ld), .b = try b.up(std.mem.sliceAsBytes(bv)), .b_ld = @intCast(ab_ld), .a_log = try b.up(std.mem.sliceAsBytes(a_log)), .dt_bias = try b.up(std.mem.sliceAsBytes(dtb)), .q = 0, .k = 0, .v = 0, .g = 0, .beta = 0 };
    var ios: [counts.len]cuda.gdn_prep.Io = undefined;
    for (counts, &ios) |m, *io| {
        io.* = base;
        io.q = try b.out(s, m * sh.kh * sh.dk * 2);
        io.k = try b.out(s, m * sh.kh * sh.dk * 2);
        io.v = try b.out(s, m * sh.vh * sh.dk * 2);
        io.g = try b.out(s, m * sh.vh * 4);
        io.beta = try b.out(s, m * sh.vh * 4);
        try g.run(s, io.*, m, sh);
    }
    try s.synchronize();
    const io = ios[counts.len - 1];
    const q = try b.back(a, u16, io.q, most * sh.kh * sh.dk);
    const k = try b.back(a, u16, io.k, most * sh.kh * sh.dk);
    const v = try b.back(a, u16, io.v, most * sh.vh * sh.dk);
    const gg = try b.back(a, f32, io.g, most * sh.vh);
    const be = try b.back(a, f32, io.beta, most * sh.vh);
    var name: [64]u8 = undefined;
    const label = try std.fmt.bufPrint(&name, "gdn prep kh {d} vh {d} dk {d} taps {d}", .{ sh.kh, sh.vh, sh.dk, taps });
    const rows = try a.alloc([]f64, taps);
    for (rows) |*x| x.* = try a.alloc(f64, sh.dk);
    const xs = try a.alloc([]const f64, taps);
    const w = try a.alloc(f64, sh.dk * taps);
    const cv = try a.alloc(f64, sh.dk);
    const err = try a.alloc(f64, sh.dk);
    const want = try a.alloc(f64, sh.dk);
    var most_q: f64 = 0;
    const rel = 3 * std.math.pow(f64, 2, -8) + std.math.pow(f64, 2, -18); // c rounded, the norm through it, the output
    for (0..most) |r| for (0..2 * sh.kh + sh.vh) |hd| {
        for (rows, xs, 0..) |x, *p, t| {
            const src: usize = @intCast(win[r * taps + t]);
            for (x, 0..) |*o, i| o.* = if (src < sh.keep) bf(state[(@as(usize, @intCast(sid[r])) * sh.keep + src) * c + hd * sh.dk + i]) else bf(qkv[(src - sh.keep) * ld + hd * sh.dk + i]);
            p.* = x;
        }
        for (0..sh.dk) |i| for (0..taps) |t| {
            w[i * taps + t] = bf(conv[(hd * sh.dk + i) * taps + t]);
        };
        ref.conv(xs, w, taps, cv, err);
        if (hd < 2 * sh.kh) {
            const is_q = hd < sh.kh;
            const f = ref.norm(cv, is_q, sh.eps, want);
            const got = if (is_q) q[(r * sh.kh + hd) * sh.dk ..] else k[(r * sh.kh + hd - sh.kh) * sh.dk ..];
            for (want, err, 0..) |x, e, i| most_q = @max(most_q, @abs(bf(got[i]) - x) / (@abs(x) * rel + f * e + 1e-30));
        } else {
            const hv = hd - 2 * sh.kh;
            for (cv, err, 0..) |x, e, i| most_q = @max(most_q, @abs(bf(v[(r * sh.vh + hv) * sh.dk + i]) - x) / (@abs(x) * (std.math.pow(f64, 2, -8) + std.math.pow(f64, 2, -20)) + e + 1e-30));
            const x = bf(av[r * ab_ld + hv]) + dtb[hv];
            const sp = if (x > 20) x else std.math.log1p(@exp(x));
            const wg = ref.gate(bf(av[r * ab_ld + hv]), dtb[hv], a_log[hv]);
            most_q = @max(most_q, @abs(gg[r * sh.vh + hv] - wg) / (@abs(wg) * std.math.pow(f64, 2, -20) * (4 + 4 * @exp(@as(f64, a_log[hv])) * sp) + 1e-30));
            const wb = 1 / (1 + @exp(-bf(bv[r * ab_ld + hv])));
            most_q = @max(most_q, @abs(be[r * sh.vh + hv] - wb) / (wb * std.math.pow(f64, 2, -19) + 1e-30));
        }
    };
    worst.* = @max(worst.*, most_q);
    try check.expect(most_q <= 1, "{s}: within float64 tolerance (worst {d:.3})", .{ label, most_q });
    for (counts[0 .. counts.len - 1], ios[0 .. counts.len - 1]) |m, o| {
        try check.sameBytes(label, std.mem.sliceAsBytes(try b.back(a, u16, o.q, m * sh.kh * sh.dk)), std.mem.sliceAsBytes(q[0 .. m * sh.kh * sh.dk]));
        try check.sameBytes(label, std.mem.sliceAsBytes(try b.back(a, u16, o.v, m * sh.vh * sh.dk)), std.mem.sliceAsBytes(v[0 .. m * sh.vh * sh.dk]));
        try check.sameBytes(label, std.mem.sliceAsBytes(try b.back(a, f32, o.g, m * sh.vh)), std.mem.sliceAsBytes(gg[0 .. m * sh.vh]));
    }
}

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var g = try cuda.gdn_prep.GdnPrep.load(gpu.d);
    defer g.unload();
    var prng = std.Random.DefaultPrng.init(48);
    var worst: f64 = 0;
    try case(gpu, arena.allocator(), s, &g, prng.random(), .{ .kh = 16, .vh = 48, .dk = 128, .keep = 3 }, &worst); // Qwen3.8-27B
    try case(gpu, arena.allocator(), s, &g, prng.random(), .{ .kh = 2, .vh = 3, .dk = 64, .keep = 3 }, &worst);
    try case(gpu, arena.allocator(), s, &g, prng.random(), .{ .kh = 1, .vh = 2, .dk = 256, .keep = 1 }, &worst);
    check.pass("gdn prep: conv windows, q/k norms, v, g and beta within float64 tolerance (worst {d:.3}), bits at every row count", .{worst});
}
