//! The sink merge at Hy4-flash's shapes: tf_hy4_attend_merge and the mla image's tf_mla_merge_sink, byte for byte.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

const lw = cuda.mla.latent_width;

/// Sliding layers: 64 heads in one 512-key chunk; DSA layers: 32 in four (2,048 keys); and the heads a rank holds.
const Form = struct { heads: usize, chunks: usize };
const forms = [_]Form{ .{ .heads = 64, .chunks = 1 }, .{ .heads = 32, .chunks = 1 }, .{ .heads = 16, .chunks = 1 }, .{ .heads = 32, .chunks = 4 }, .{ .heads = 16, .chunks = 4 }, .{ .heads = 8, .chunks = 4 } };
const row_counts = [_]usize{ 1, 17, 64 };

fn bf(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}

/// Chunk partials as the chunk tiles leave them (o [chunks][rows][heads][512], m and l [chunks][rows][heads]), sinks.
const Parts = struct {
    o: []f32,
    m: []f32,
    l: []f32,
    sinks: []f32,

    /// Maxima -4 to 4, masses 1 to 512, o within its mass; row 0's last chunk empty (-inf, 0) when there are several.
    fn init(a: std.mem.Allocator, r: std.Random, rows: usize, f: Form) !Parts {
        const n = f.chunks * rows * f.heads;
        const p: Parts = .{ .o = try a.alloc(f32, n * lw), .m = try a.alloc(f32, n), .l = try a.alloc(f32, n), .sinks = try a.alloc(f32, f.heads) };
        for (p.m, p.l, 0..) |*m, *l, i| {
            m.* = -4 + 8 * r.float(f32);
            l.* = 1 + 511 * r.float(f32);
            for (p.o[i * lw ..][0..lw]) |*o| o.* = l.* * (r.float(f32) - 0.5);
        }
        for (p.sinks) |*s| s.* = -3 + 9 * r.float(f32);
        if (f.chunks > 1) for (0..f.heads) |h| {
            const at = ((f.chunks - 1) * rows) * f.heads + h;
            p.m[at] = -std.math.inf(f32);
            p.l[at] = 0;
            @memset(p.o[at * lw ..][0..lw], 0);
        };
        return p;
    }

    fn deinit(p: Parts, a: std.mem.Allocator) void {
        for ([_][]f32{ p.o, p.m, p.l, p.sinks }) |x| a.free(x);
    }
};

/// One merge launch on the partials: block (row, head) of 128, as Hy4-flash's attention and Mla.mergeSink call it.
fn merge(gpu: Gpu, s: cuda.Stream, k: cuda.Function, p: Parts, rows: usize, f: Form) ![]u8 {
    var bufs: [4]cuda.DeviceBuffer = undefined;
    for (&bufs, [_][]f32{ p.o, p.m, p.l, p.sinks }) |*b, x| b.* = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x));
    defer for (&bufs) |*b| b.free();
    var out = try cuda.DeviceBuffer.alloc(gpu.d, rows * f.heads * lw * 2);
    defer out.free();
    var a: cuda.Args = .{};
    inline for (.{ bufs[0].ptr, bufs[1].ptr, bufs[2].ptr, bufs[3].ptr, out.ptr, @as(i64, @intCast(rows)), @as(i64, @intCast(f.heads)), @as(i32, @intCast(f.chunks)) }) |v| a.add(v);
    try cuda.launch.launch(k, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(f.heads) }, .block = .{ .x = 128 } }, s, &a);
    try s.synchronize();
    return check.download(gpu, out);
}

/// float64: o = sum o_i exp(m_i - top) / (exp(sink - top) + sum l_i exp(m_i - top)), top the largest of sink and m_i.
fn worst(p: Parts, rows: usize, f: Form, got: []const u8) f64 {
    const y = std.mem.bytesAsSlice(u16, got);
    var bad: f64 = 0;
    for (0..rows) |r| for (0..f.heads) |h| {
        const rh = r * f.heads + h;
        var top: f64 = p.sinks[h];
        for (0..f.chunks) |c| top = @max(top, @as(f64, p.m[c * rows * f.heads + rh]));
        var den = @exp(@as(f64, p.sinks[h]) - top);
        for (0..f.chunks) |c| {
            const at = c * rows * f.heads + rh;
            den += @as(f64, p.l[at]) * @exp(@as(f64, p.m[at]) - top);
        }
        for (0..lw) |j| {
            var num: f64 = 0;
            var mag: f64 = 0;
            for (0..f.chunks) |c| {
                const at = c * rows * f.heads + rh;
                const t = @as(f64, p.o[at * lw + j]) * @exp(@as(f64, p.m[at]) - top);
                num += t;
                mag += @abs(t);
            }
            const want = num / den;
            const allow = @abs(want) * 0x1p-8 + mag / den * 0x1p-20 + 1e-30; // bf16 out; fp32 exp and sums
            bad = @max(bad, @abs(bf(y[rh * lw + j]) - want) / allow);
        }
    };
    return bad;
}

/// `old`: hy4_mla_attention's fatbin from another build (main's), whose tf_hy4_attend_merge must give the same bytes.
pub fn run(gpu: Gpu, old: ?[]const u8) !void {
    const a = gpu.gpa;
    var hy4 = try cuda.Module.load(gpu.d, cuda.kernels.hy4_mla_attention);
    defer hy4.unload();
    var mla = try cuda.Module.load(gpu.d, cuda.kernels.mla);
    defer mla.unload();
    const image: ?[]align(16) u8 = if (old) |path| try std.Io.Dir.cwd().readFileAllocOptions(gpu.io, path, a, .limited(1 << 26), .@"16", null) else null;
    defer if (image) |x| a.free(x);
    var before: ?cuda.Module = if (image) |x| try cuda.Module.load(gpu.d, x) else null;
    defer if (before) |*m| m.unload();
    const ks = [_]cuda.Function{ try hy4.function("tf_hy4_attend_merge"), try mla.function("tf_mla_merge_sink") };
    const old_k: ?cuda.Function = if (before) |m| try m.function("tf_hy4_attend_merge") else null;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0x51c4);
    var cases: usize = 0;
    var most: f64 = 0;
    for (forms) |f| for (row_counts) |rows| {
        const p = try Parts.init(a, prng.random(), rows, f);
        defer p.deinit(a);
        const got = try merge(gpu, s, ks[0], p, rows, f);
        defer a.free(got);
        const core_ = try merge(gpu, s, ks[1], p, rows, f);
        defer a.free(core_);
        try check.sameBytes("tf_mla_merge_sink against tf_hy4_attend_merge", core_, got);
        if (old_k) |k| {
            const was = try merge(gpu, s, k, p, rows, f);
            defer a.free(was);
            try check.sameBytes("tf_hy4_attend_merge against the other build's", got, was);
        }
        const w = worst(p, rows, f, got);
        try check.expect(w <= 1, "sink merge at {d} rows x {d} heads x {d} chunks: {e} of its float64 bound", .{ rows, f.heads, f.chunks, w });
        most = @max(most, w);
        cases += 1;
    };
    check.pass("sink merge: {d} shapes, tf_mla_merge_sink gives tf_hy4_attend_merge's bytes; at most {e} of the float64 bound", .{ cases, most });
    if (old_k != null) check.pass("sink merge: tf_hy4_attend_merge gives the other build's bytes at all {d} shapes", .{cases});
}
