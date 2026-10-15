//! norms.cu's bf16-residual and gated rows and embed_rows.cu's token rows against float64, bits at every row count.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");

const nref = core.norms_ref;
const aref = core.act_ref;
const counts = [_]usize{ 1, 3, 17, 64 };
const most = counts[counts.len - 1];
const eps: f32 = 1e-6;

fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn bf(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}

const Rig = struct {
    gpu: check.Gpu,
    a: std.mem.Allocator,
    s: cuda.Stream,
    rng: std.Random,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    worst: f64 = 0,

    fn dev(r: *Rig, bytes: []const u8) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(r.gpu.d, bytes);
        try r.bufs.append(r.gpu.gpa, b);
        return b.ptr;
    }

    fn out(r: *Rig, bytes: usize) !u64 {
        const b = try cuda.DeviceBuffer.alloc(r.gpu.d, bytes);
        try r.bufs.append(r.gpu.gpa, b);
        try b.fill8(0xff, r.s.handle); // a value the kernel misses reads NaN
        return b.ptr;
    }

    fn back(r: *Rig, comptime T: type, ptr: u64, n: usize) ![]T {
        const o = try r.a.alloc(T, n);
        try r.s.synchronize();
        const b: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = ptr, .len = n * @sizeOf(T) };
        try b.download(0, std.mem.sliceAsBytes(o));
        return o;
    }

    /// bf16 values: mostly within [-scale, scale), one in 64 twenty times larger.
    fn bfs(r: *Rig, n: usize, scale: f32) []u16 {
        const v = r.a.alloc(u16, n) catch unreachable;
        for (v, 0..) |*x, i| x.* = bf16((r.rng.float(f32) * 2 - 1) * scale * (if (i % 64 == 7) @as(f32, 20) else 1));
        return v;
    }

    /// Weights near 1.
    fn weights(r: *Rig, n: usize) []f32 {
        const v = r.a.alloc(f32, n) catch unreachable;
        for (v) |*x| x.* = 0.5 + r.rng.float(f32);
        return v;
    }

    fn within(r: *Rig, what: []const u8, got: f64, want: f64, tol: f64) !void {
        const q = @abs(got - want) / tol;
        if (!(q <= 1)) {
            try check.expect(false, "{s}: within float64 tolerance (got {d}, want {d}, {d:.3} tolerances)", .{ what, got, want, q });
        }
        r.worst = @max(r.worst, q);
    }

    /// Group sums `xs` of the stored bf16 rows `y` ([rows, k]).
    fn sums(r: *Rig, what: []const u8, y: []const u16, xs: []const f32, rows: usize, k: usize) !void {
        var row: [64]f64 = undefined;
        for (0..rows) |i| for (0..k / 64) |j| {
            for (&row, 0..) |*v, e| v.* = bf(y[i * k + 64 * j + e]);
            var want: [1]f64 = undefined;
            nref.groupSums(&row, &want);
            try r.within(what, xs[i * (k / 64) + j], want[0], nref.groupSumTolerance(&row));
        };
    }
};

fn addRmsH(r: *Rig, n: *const cuda.norms.Norms, width: usize, residual: bool, wk: cuda.norms.Kind) !void {
    const x = r.bfs(most * width, 1);
    const rv = r.bfs(most * width, 1);
    const wf = r.weights(width);
    const wb = try r.a.alloc(u16, width);
    for (wb, wf) |*o, v| o.* = bf16(v);
    const wvals = try r.a.alloc(f64, width);
    for (wvals, wf, wb) |*o, f, b| o.* = if (wk == .bf16) bf(b) else f;
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    const dr = try r.dev(std.mem.sliceAsBytes(rv));
    const dw = try r.dev(if (wk == .bf16) std.mem.sliceAsBytes(wb) else std.mem.sliceAsBytes(wf));
    var hs: [counts.len]u64 = undefined;
    var ys: [counts.len]u64 = undefined;
    var xss: [counts.len]u64 = undefined;
    for (counts, 0..) |m, c| {
        hs[c] = try r.out(m * width * 2);
        ys[c] = try r.out(m * width * 2);
        xss[c] = try r.out(m * width / 64 * 4);
        try n.addRmsH(r.s, dx, if (residual) dr else null, dw, wk, hs[c], ys[c], xss[c], m, width, eps);
    }
    var name: [96]u8 = undefined;
    const label = try std.fmt.bufPrint(&name, "add_rms_h width {d}{s}, {t} weights", .{ width, if (residual) "" else ", no residual", wk });
    const h = try r.back(u16, hs[counts.len - 1], most * width);
    const y = try r.back(u16, ys[counts.len - 1], most * width);
    const xs = try r.back(f32, xss[counts.len - 1], most * width / 64);
    const hv = try r.a.alloc(f64, width);
    const want = try r.a.alloc(f64, width);
    for (0..most) |row| {
        for (hv, 0..) |*v, i| {
            const at = row * width + i;
            if (residual) {
                const sum = bf(x[at]) + bf(rv[at]);
                try r.within(label, bf(h[at]), sum, nref.tolerance(sum, .bf16));
                v.* = bf(h[at]);
            } else v.* = bf(x[at]);
        }
        nref.rms(hv, wvals, eps, want);
        for (want, 0..) |w, i| try r.within(label, bf(y[row * width + i]), w, nref.tolerance(w, .bf16));
    }
    try r.sums(label, y, xs, most, width);
    for (counts[0 .. counts.len - 1], 0..) |m, c| {
        try check.sameBytes(label, std.mem.sliceAsBytes(try r.back(u16, ys[c], m * width)), std.mem.sliceAsBytes(y[0 .. m * width]));
        try check.sameBytes(label, std.mem.sliceAsBytes(try r.back(f32, xss[c], m * width / 64)), std.mem.sliceAsBytes(xs[0 .. m * width / 64]));
        if (residual) try check.sameBytes(label, std.mem.sliceAsBytes(try r.back(u16, hs[c], m * width)), std.mem.sliceAsBytes(h[0 .. m * width]));
    }
}

fn gatedRms(r: *Rig, n: *const cuda.norms.Norms, heads: usize, dv: usize) !void {
    const width = heads * dv;
    const y_ld = width + 64; // a stacked row with something after the heads
    const z_ld = 2 * width;
    const y = r.bfs(most * y_ld, 1);
    const z = r.bfs(most * z_ld, 3);
    const wb = try r.a.alloc(u16, dv);
    for (wb, r.weights(dv)) |*o, v| o.* = bf16(v);
    const wvals = try r.a.alloc(f64, dv);
    for (wvals, wb) |*o, b| o.* = bf(b);
    const dy = try r.dev(std.mem.sliceAsBytes(y));
    const dz = try r.dev(std.mem.sliceAsBytes(z));
    const dw = try r.dev(std.mem.sliceAsBytes(wb));
    var os: [counts.len]u64 = undefined;
    var xss: [counts.len]u64 = undefined;
    for (counts, 0..) |m, c| {
        os[c] = try r.out(m * width * 2);
        xss[c] = try r.out(m * width / 64 * 4);
        try n.gatedRms(r.s, dy, y_ld, dz, z_ld, dw, os[c], xss[c], m, heads, dv, eps);
    }
    var name: [96]u8 = undefined;
    const label = try std.fmt.bufPrint(&name, "gated rms {d} heads of {d}", .{ heads, dv });
    const o = try r.back(u16, os[counts.len - 1], most * width);
    const xs = try r.back(f32, xss[counts.len - 1], most * width / 64);
    const yv = try r.a.alloc(f64, dv);
    const zv = try r.a.alloc(f64, dv);
    const want = try r.a.alloc(f64, dv);
    for (0..most) |row| for (0..heads) |hd| {
        for (yv, zv, 0..) |*a, *b, e| {
            a.* = bf(y[row * y_ld + hd * dv + e]);
            b.* = bf(z[row * z_ld + hd * dv + e]);
        }
        nref.gatedRms(yv, zv, wvals, eps, want);
        for (want, 0..) |w, e| try r.within(label, bf(o[(row * heads + hd) * dv + e]), w, aref.tolerance(w));
    };
    try r.sums(label, o, xs, most, width);
    for (counts[0 .. counts.len - 1], 0..) |m, c| {
        try check.sameBytes(label, std.mem.sliceAsBytes(try r.back(u16, os[c], m * width)), std.mem.sliceAsBytes(o[0 .. m * width]));
        try check.sameBytes(label, std.mem.sliceAsBytes(try r.back(f32, xss[c], m * width / 64)), std.mem.sliceAsBytes(xs[0 .. m * width / 64]));
    }
}

fn embed(r: *Rig, d: usize) !void {
    var e = try cuda.embed_rows.EmbedRows.load(r.gpu.d);
    defer e.unload();
    const vocab = 300;
    const w = try r.a.alloc(u32, vocab * d / 8);
    for (w) |*x| x.* = r.rng.int(u32);
    const sc = r.bfs(vocab * d / 64, 0.1);
    const bi = r.bfs(vocab * d / 64, 0.5);
    const tab = r.bfs(vocab * d, 1);
    const ids = try r.a.alloc(i32, most);
    for (ids) |*x| x.* = @intCast(r.rng.uintLessThan(usize, vocab));
    const did = try r.dev(std.mem.sliceAsBytes(ids));
    const o4 = try r.out(most * d * 2);
    const o16 = try r.out(most * d * 2);
    try e.affine4(r.s, did, try r.dev(std.mem.sliceAsBytes(w)), try r.dev(std.mem.sliceAsBytes(sc)), try r.dev(std.mem.sliceAsBytes(bi)), o4, most, d);
    try e.bf16(r.s, did, try r.dev(std.mem.sliceAsBytes(tab)), o16, most, d);
    const g4 = try r.back(u16, o4, most * d);
    const g16 = try r.back(u16, o16, most * d);
    for (0..most) |row| {
        const t: usize = @intCast(ids[row]);
        for (0..d) |i| {
            const q: u4 = @truncate(w[t * d / 8 + i / 8] >> @intCast(4 * (i % 8)));
            const want = aref.affine4(q, bf(sc[t * d / 64 + i / 64]), bf(bi[t * d / 64 + i / 64]));
            try r.within("embed affine4", bf(g4[row * d + i]), want, nref.tolerance(want, .bf16));
        }
        try check.sameBytes("embed bf16", std.mem.sliceAsBytes(g16[row * d ..][0..d]), std.mem.sliceAsBytes(tab[t * d ..][0..d]));
    }
}

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(27);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = try cuda.Stream.init(gpu.d, false), .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
        r.s.deinit();
    }
    var n = try cuda.norms.Norms.load(gpu.d);
    defer n.unload();
    try addRmsH(&r, &n, 5120, true, .bf16); // Qwen3.8-27B's residual
    try addRmsH(&r, &n, 5120, false, .bf16); // the first layer
    try addRmsH(&r, &n, 2048, true, .f32);
    try addRmsH(&r, &n, 2688, false, .bf16);
    try addRmsH(&r, &n, 2688, true, .bf16); // Nemotron 3.5 Lightning's hidden
    try gatedRms(&r, &n, 48, 128); // the 27B's GDN output
    try gatedRms(&r, &n, 5, 64);
    try gatedRms(&r, &n, 3, 256);
    try embed(&r, 5120); // the 27B's hidden
    try embed(&r, 2688);
    try embed(&r, 192);
    check.pass("norm rows: add_rms_h, gated rms, embed rows within float64 tolerance (worst {d:.3}), group sums of the stored outputs, every row count's bits equal", .{r.worst});
}
