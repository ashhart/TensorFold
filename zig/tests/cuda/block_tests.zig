//! block_attention.cu against core/attention_ref.zig: streams' blocks over windowed contexts, a stream's bits alone.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const dump = @import("byte_dump.zig");

const D = 128;

fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn bf(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
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
        const d = try cuda.DeviceBuffer.alloc(b.gpu.d, n);
        try b.list.append(b.gpu.gpa, d);
        try d.fill8(0xff, s.handle);
        return d.ptr;
    }

    fn back(b: *Bufs, a: std.mem.Allocator, ptr: u64, n: usize) ![]u16 {
        const o = try a.alloc(u16, n);
        const d: cuda.DeviceBuffer = .{ .d = b.gpu.d, .ptr = ptr, .len = n * 2 };
        try d.download(0, std.mem.sliceAsBytes(o));
        return o;
    }
};

/// Streams `which` of the case: their blocks packed head-major, their contexts; the output rows.
fn call(b: *Bufs, a: std.mem.Allocator, s: cuda.Stream, g: *const cuda.block_attention.BlockAttention, which: []const usize, q: []const u16, kb: []const u16, vb: []const u16, ctx_k: []const u64, ctx_v: []const u64, lens: []const i32, h: usize, hk: usize, L: usize, all_rows: usize, window: usize, causal: bool) ![]u16 {
    const rows = which.len * L;
    const pq = try a.alloc(u16, h * rows * D);
    const pk = try a.alloc(u16, hk * rows * D);
    const pv = try a.alloc(u16, hk * rows * D);
    const table = try a.alloc(i64, 2 * which.len);
    const ls = try a.alloc(i32, which.len);
    for (which, 0..) |j, i| {
        for (0..h) |hd| @memcpy(pq[(hd * rows + i * L) * D ..][0 .. L * D], q[(hd * all_rows + j * L) * D ..][0 .. L * D]);
        for (0..hk) |hd| {
            @memcpy(pk[(hd * rows + i * L) * D ..][0 .. L * D], kb[(hd * all_rows + j * L) * D ..][0 .. L * D]);
            @memcpy(pv[(hd * rows + i * L) * D ..][0 .. L * D], vb[(hd * all_rows + j * L) * D ..][0 .. L * D]);
        }
        table[2 * i] = @intCast(ctx_k[j]);
        table[2 * i + 1] = @intCast(ctx_v[j]);
        ls[i] = lens[j];
    }
    const o = try b.out(s, rows * h * D * 2);
    const sc: f32 = @floatCast(1 / @sqrt(@as(f64, D)));
    try g.run(s, try b.up(std.mem.sliceAsBytes(pq)), try b.up(std.mem.sliceAsBytes(pk)), try b.up(std.mem.sliceAsBytes(pv)), try b.up(std.mem.sliceAsBytes(table)), try b.up(std.mem.sliceAsBytes(ls)), o, .{ .h = h, .hk = hk, .rows = rows, .block = L, .window = window, .scale = sc, .causal = causal });
    try s.synchronize();
    const got = try b.back(a, o, rows * h * D);
    try dump.output(a, "block", "attention", "rows", &.{ rows, h, D }, .bf16, .{ .hk = hk, .block = L, .window = window, .causal = causal, .streams = which }, std.mem.sliceAsBytes(got));
    return got;
}

fn case(gpu: check.Gpu, a: std.mem.Allocator, s: cuda.Stream, g: *const cuda.block_attention.BlockAttention, rng: std.Random, L: usize, causal: bool, worst: *f64) !void {
    var b: Bufs = .{ .gpu = gpu };
    defer {
        for (b.list.items) |*d| d.free();
        b.list.deinit(gpu.gpa);
    }
    const h = 32;
    const hk = 8;
    const window = 2048;
    const lens = [_]i32{ 5, 700, 2100 };
    const S = lens.len;
    const rows = S * L;
    const q = try a.alloc(u16, h * rows * D);
    const kb = try a.alloc(u16, hk * rows * D);
    const vb = try a.alloc(u16, hk * rows * D);
    for (q) |*x| x.* = bf16(rng.floatNorm(f32));
    for (kb) |*x| x.* = bf16(rng.floatNorm(f32));
    for (vb) |*x| x.* = bf16(rng.floatNorm(f32));
    var ck: [S][]u16 = undefined;
    var cv: [S][]u16 = undefined;
    var pk: [S]u64 = undefined;
    var pv: [S]u64 = undefined;
    for (lens, 0..) |n, j| {
        const m: usize = @intCast(n);
        ck[j] = try a.alloc(u16, hk * m * D);
        cv[j] = try a.alloc(u16, hk * m * D);
        for (ck[j]) |*x| x.* = bf16(rng.floatNorm(f32));
        for (cv[j]) |*x| x.* = bf16(rng.floatNorm(f32));
        pk[j] = try b.up(std.mem.sliceAsBytes(ck[j]));
        pv[j] = try b.up(std.mem.sliceAsBytes(cv[j]));
    }
    const all = [_]usize{ 0, 1, 2 };
    const got = try call(&b, a, s, g, &all, q, kb, vb, &pk, &pv, &lens, h, hk, L, rows, window, causal);
    const scale = 1 / @sqrt(@as(f64, D));
    const kd = try a.alloc(f64, (window + 1 + L) * D);
    const vd = try a.alloc(f64, kd.len);
    const keys = try a.alloc(usize, window + 1 + L);
    const qv = try a.alloc(f64, D);
    const want = try a.alloc(f64, D);
    const tol = try a.alloc(f64, D);
    var most: f64 = 0;
    for (0..S) |j| for (0..L) |r| for (0..h) |hd| {
        const kh = hd / (h / hk);
        const n: usize = @intCast(lens[j]);
        var nk: usize = 0;
        for (0..n) |k| if (n + r - k <= window) {
            for (0..D) |e| {
                kd[nk * D + e] = bf(ck[j][(kh * n + k) * D + e]);
                vd[nk * D + e] = bf(cv[j][(kh * n + k) * D + e]);
            }
            keys[nk] = nk;
            nk += 1;
        };
        for (0..L) |c| if (!causal or c <= r) {
            for (0..D) |e| {
                kd[nk * D + e] = bf(kb[(kh * rows + j * L + c) * D + e]);
                vd[nk * D + e] = bf(vb[(kh * rows + j * L + c) * D + e]);
            }
            keys[nk] = nk;
            nk += 1;
        };
        for (qv, 0..) |*x, e| x.* = bf(q[(hd * rows + j * L + r) * D + e]);
        core.attention_ref.head(qv, kd, vd, keys[0..nk], D, scale, want, tol);
        for (want, tol, 0..) |w, t, e| most = @max(most, @abs(bf(got[((j * L + r) * h + hd) * D + e]) - w) / t);
    };
    worst.* = @max(worst.*, most);
    try check.expect(most <= 1, "block attention L {d}{s}: within float64 tolerance (worst {d:.3})", .{ L, if (causal) " causal" else "", most });
    const one = [_]usize{2};
    const alone = try call(&b, a, s, g, &one, q, kb, vb, &pk, &pv, &lens, h, hk, L, rows, window, causal);
    try check.sameBytes("block attention: one stream alone", std.mem.sliceAsBytes(alone), std.mem.sliceAsBytes(got[2 * L * h * D ..][0 .. L * h * D]));
}

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var g = try cuda.block_attention.BlockAttention.load(gpu.d);
    defer g.unload();
    var prng = std.Random.DefaultPrng.init(16);
    var worst: f64 = 0;
    try case(gpu, arena.allocator(), s, &g, prng.random(), 16, false, &worst); // Qwen3.8-27B's DFlash2: 32/8 heads, a 2,048 window
    try case(gpu, arena.allocator(), s, &g, prng.random(), 7, true, &worst);
    check.pass("block attention: windowed contexts then the block, within float64 tolerance (worst {d:.3}), a stream's bits alone", .{worst});
}
