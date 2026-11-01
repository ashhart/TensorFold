//! norms.cu's bf16-residual and gated rows and embed_rows.cu's token rows against float64, bits at every row count.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");

const nref = core.norms_ref;
const aref = core.act_ref;
pub const counts = [_]usize{ 1, 3, 17, 64 };
pub const most = counts[counts.len - 1];
const eps: f32 = 1e-6;

pub fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

pub fn bf(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}

pub const Rig = struct {
    gpu: check.Gpu,
    a: std.mem.Allocator,
    s: cuda.Stream,
    rng: std.Random,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    worst: f64 = 0,

    pub fn dev(r: *Rig, bytes: []const u8) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(r.gpu.d, bytes);
        try r.bufs.append(r.gpu.gpa, b);
        return b.ptr;
    }

    pub fn out(r: *Rig, bytes: usize) !u64 {
        const b = try cuda.DeviceBuffer.alloc(r.gpu.d, bytes);
        try r.bufs.append(r.gpu.gpa, b);
        try b.fill8(0xff, r.s.handle); // a value the kernel misses reads NaN
        return b.ptr;
    }

    pub fn back(r: *Rig, comptime T: type, ptr: u64, n: usize) ![]T {
        const o = try r.a.alloc(T, n);
        try r.s.synchronize();
        const b: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = ptr, .len = n * @sizeOf(T) };
        try b.download(0, std.mem.sliceAsBytes(o));
        return o;
    }

    /// bf16 values: mostly within [-scale, scale), one in 64 twenty times larger.
    pub fn bfs(r: *Rig, n: usize, scale: f32) []u16 {
        const v = r.a.alloc(u16, n) catch unreachable;
        for (v, 0..) |*x, i| x.* = bf16((r.rng.float(f32) * 2 - 1) * scale * (if (i % 64 == 7) @as(f32, 20) else 1));
        return v;
    }

    /// Weights near 1.
    pub fn weights(r: *Rig, n: usize) []f32 {
        const v = r.a.alloc(f32, n) catch unreachable;
        for (v) |*x| x.* = 0.5 + r.rng.float(f32);
        return v;
    }

    pub fn within(r: *Rig, what: []const u8, got: f64, want: f64, tol: f64) !void {
        const q = @abs(got - want) / tol;
        if (!(q <= 1)) {
            try check.expect(false, "{s}: within float64 tolerance (got {d}, want {d}, {d:.3} tolerances)", .{ what, got, want, q });
        }
        r.worst = @max(r.worst, q);
    }

    /// Group sums `xs` of the stored bf16 rows `y` ([rows, k]).
    pub fn sums(r: *Rig, what: []const u8, y: []const u16, xs: []const f32, rows: usize, k: usize) !void {
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

fn embed(r: *Rig, d: usize, group: usize, copies: usize) !void {
    var e = try cuda.embed_rows.EmbedRows.load(r.gpu.d);
    defer e.unload();
    const vocab = 300;
    const w = try r.a.alloc(u32, vocab * d / 8);
    for (w) |*x| x.* = r.rng.int(u32);
    const sc = r.bfs(vocab * d / group, 0.1);
    const bi = r.bfs(vocab * d / group, 0.5);
    const tab = try r.a.alloc(u16, vocab * d);
    for (tab, 0..) |*x, i| x.* = @truncate(i); // every BF16 bit pattern, including NaN payloads
    const ids = try r.a.alloc(i32, most);
    for (ids) |*x| x.* = @intCast(r.rng.uintLessThan(usize, vocab));
    const did = try r.dev(std.mem.sliceAsBytes(ids));
    const o4 = try r.out(most * copies * d * 2);
    const o16 = try r.out(most * d * 2);
    const dw = try r.dev(std.mem.sliceAsBytes(w));
    const ds = try r.dev(std.mem.sliceAsBytes(sc));
    const db = try r.dev(std.mem.sliceAsBytes(bi));
    if (group == 64 and copies == 1) try e.affine4(r.s, did, dw, ds, db, o4, most, d) else try e.affine4Groups(r.s, did, dw, ds, db, o4, most, d, group, copies);
    const dt = try r.dev(std.mem.sliceAsBytes(tab));
    try e.bf16(r.s, did, dt, o16, most, d);
    const oc = try r.out(most * copies * d * 2);
    try e.bf16Copies(r.s, did, dt, oc, 0, d, copies);
    try std.testing.expectError(error.RowShape, e.bf16Copies(r.s, did, dt, oc, most, d, 0));
    try std.testing.expectError(error.RowShape, e.bf16Copies(r.s, did, dt, oc + 2, most, d, copies));
    try std.testing.expectError(error.RowShape, e.bf16Copies(r.s, did, dt, oc, most, d - 1, copies));
    for (counts) |rows| {
        try e.bf16Copies(r.s, did, dt, oc, rows, d, copies);
        const got = try r.back(u16, oc, rows * copies * d);
        for (0..rows) |row| {
            const token: usize = @intCast(ids[row]);
            for (0..copies) |c| try check.sameBytes("embed BF16 exact copies", std.mem.sliceAsBytes(got[(row * copies + c) * d ..][0..d]), std.mem.sliceAsBytes(tab[token * d ..][0..d]));
        }
    }
    const g4 = try r.back(u16, o4, most * copies * d);
    const g16 = try r.back(u16, o16, most * d);
    for (0..most) |row| {
        const t: usize = @intCast(ids[row]);
        for (0..d) |i| {
            const q: u4 = @truncate(w[t * d / 8 + i / 8] >> @intCast(4 * (i % 8)));
            const want = aref.affine4(q, bf(sc[t * d / group + i / group]), bf(bi[t * d / group + i / group]));
            try r.within("embed affine4", bf(g4[row * copies * d + i]), want, nref.tolerance(want, .bf16));
        }
        for (1..copies) |c| try check.sameBytes("embed affine4 copies", std.mem.sliceAsBytes(g4[(row * copies + c) * d ..][0..d]), std.mem.sliceAsBytes(g4[row * copies * d ..][0..d]));
        try check.sameBytes("embed bf16", std.mem.sliceAsBytes(g16[row * d ..][0..d]), std.mem.sliceAsBytes(tab[t * d ..][0..d]));
    }
}

/// RMS rows `x_ld` apart with fp32 weights, and the stored outputs' sums of each `group`.
fn rmsSums(r: *Rig, n: *const cuda.norms.Norms, width: usize, x_ld: usize, group: usize) !void {
    const x = r.bfs(most * x_ld, 1);
    const wf = r.weights(width);
    const wvals = try r.a.alloc(f64, width);
    for (wvals, wf) |*o, v| o.* = v;
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    const dw = try r.dev(std.mem.sliceAsBytes(wf));
    var ys: [counts.len]u64 = undefined;
    var xss: [counts.len]u64 = undefined;
    for (counts, 0..) |m, c| {
        ys[c] = try r.out(m * width * 2);
        xss[c] = try r.out(m * width / group * 4);
        try n.rmsSums(r.s, dx, x_ld, dw, ys[c], xss[c], m, width, group, eps);
    }
    const y = try r.back(u16, ys[counts.len - 1], most * width);
    const xs = try r.back(f32, xss[counts.len - 1], most * width / group);
    const xv = try r.a.alloc(f64, width);
    const want = try r.a.alloc(f64, width);
    for (0..most) |row| {
        for (xv, 0..) |*v, i| v.* = bf(x[row * x_ld + i]);
        nref.rms(xv, wvals, eps, want);
        for (want, 0..) |w, i| try r.within("rms sums", bf(y[row * width + i]), w, nref.tolerance(w, .bf16));
        for (0..width / group) |j| {
            var t: f64 = 0;
            var abs: f64 = 0;
            for (y[row * width + j * group ..][0..group]) |v| {
                t += bf(v);
                abs += @abs(bf(v));
            }
            try r.within("rms sums group sums", xs[row * (width / group) + j], t, abs * @as(f64, @floatFromInt(group)) * std.math.pow(f64, 2, -24) + 1e-30);
        }
    }
    for (counts[0 .. counts.len - 1], 0..) |m, c| {
        try check.sameBytes("rms sums", std.mem.sliceAsBytes(try r.back(u16, ys[c], m * width)), std.mem.sliceAsBytes(y[0 .. m * width]));
        try check.sameBytes("rms sums group sums", std.mem.sliceAsBytes(try r.back(f32, xss[c], m * width / group)), std.mem.sliceAsBytes(xs[0 .. m * width / group]));
    }
}

/// rms_tiles.cuh at tiles 1, 10, 40 and rows 1-17 against tf_rms_rows_sums' bytes; odd rows, NaN between rows.
fn rmsSumsTiles(r: *Rig, n: *const cuda.norms.Norms) !void {
    const width = cuda.norms.tiles_width;
    const x_ld = width + 64;
    const rows = 17;
    const groups = width / 32;
    const x = r.bfs(rows * x_ld, 1);
    for (0..rows) |row| @memset(x[row * x_ld + width ..][0..64], 0x7fc0);
    @memset(x[0..width], 0);
    for (0..width) |i| { // row 1 spans 2^-20 to 2^19; row 5's squares overflow fp32
        const e: i32 = @intCast(i % 200);
        x[x_ld + i] = bf16(std.math.pow(f32, 2, @floatFromInt(@rem(e, 40) - 20)));
        x[5 * x_ld + i] = bf16(std.math.pow(f32, 2, @floatFromInt(e - 100)));
    }
    x[2 * x_ld + 77] = 0x7fc0;
    x[3 * x_ld + 5000] = 0x7f80;
    x[4 * x_ld + 9] = 0xff80;
    x[4 * x_ld + 10230] = 0x7f80;
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    const dw = try r.dev(std.mem.sliceAsBytes(r.weights(width)));
    var base = n.*;
    base.sm = 0; // no table row: tf_rms_rows_sums
    const yb = try r.out(rows * width * 2);
    const xb = try r.out(rows * groups * 4);
    try base.rmsSums(r.s, dx, x_ld, dw, yb, xb, rows, width, 32, eps);
    const want_y = try r.back(u16, yb, rows * width);
    const want_xs = try r.back(f32, xb, rows * groups);
    for ([_]u32{ 1, 10, 40 }) |tiles| for (1..rows + 1) |m| {
        const y = try r.out(m * width * 2);
        const xs = try r.out(m * groups * 4);
        try n.rmsSumsTiles(r.s, dx, x_ld, dw, y, xs, m, tiles, eps);
        try check.sameBytes("rms tiles", std.mem.sliceAsBytes(try r.back(u16, y, m * width)), std.mem.sliceAsBytes(want_y[0 .. m * width]));
        try check.sameBytes("rms tiles group sums", std.mem.sliceAsBytes(try r.back(f32, xs, m * groups)), std.mem.sliceAsBytes(want_xs[0 .. m * groups]));
    };
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
    try embed(&r, 5120, 64, 1); // the 27B's hidden
    try embed(&r, 2688, 64, 1);
    try embed(&r, 192, 64, 1);
    try embed(&r, 2560, 32, 4); // Flash Next: group 32, a copy a hyper-connection stream
    try embed(&r, 96, 32, 1);
    try rmsSums(&r, &n, 2560, 10240, 32); // Flash Next's MTP head: the embedding, its first stream of four
    try rmsSums(&r, &n, 10240, 10240, 32); // and the four streams normed as one row
    try rmsSumsTiles(&r, &n);
    check.pass("norm rows: add_rms_h, gated rms, rms with group sums, embed rows within float64 tolerance (worst {d:.3}), group sums of the stored outputs, every row count's bits equal, the MTP tiles' bits tf_rms_rows_sums' at tiles 1, 10, 40 and rows 1-17", .{r.worst});
}
